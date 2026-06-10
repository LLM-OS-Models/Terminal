#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"

CURRENT_RUN_ID="${CURRENT_RUN_ID:-run_20260609T101140Z_echo_public1500_prepared_only_resume380_dockerseed_pathfix2_long_setsid_vllm4_train2}"
CURRENT_RUN_DIR="${CURRENT_RUN_DIR:-/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/$CURRENT_RUN_ID}"
CURRENT_OUTPUT_DIR="${CURRENT_OUTPUT_DIR:-/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_$CURRENT_RUN_ID}"
CURRENT_LAUNCHER_PID="${CURRENT_LAUNCHER_PID:-1921085}"

MODEL_PATH="${MODEL_PATH:-LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch}"
PREPARED_JSONL="${PREPARED_JSONL:-/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl}"
PREPARED_ROWS="${PREPARED_ROWS:-1500}"
PREPARED_SOURCE_OPENTHOUGHTS_AGENT_V1_RL="${PREPARED_SOURCE_OPENTHOUGHTS_AGENT_V1_RL:-728}"
PREPARED_SOURCE_ENDLESS_TERMINALS="${PREPARED_SOURCE_ENDLESS_TERMINALS:-772}"

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8123/v1,http://127.0.0.1:8124/v1,http://127.0.0.1:8125/v1,http://127.0.0.1:8126/v1}"
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MAX_STEPS="${MAX_STEPS:-100000}"
NEXT_MAX_WALL_TIME_HOURS="${NEXT_MAX_WALL_TIME_HOURS:-47.5}"
PROMPTS_PER_RANK="${PROMPTS_PER_RANK:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-12288}"
MAX_TURNS="${MAX_TURNS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-20}"
VERIFIER_TIMEOUT="${VERIFIER_TIMEOUT:-90}"
MAX_TERMINAL_OUTPUT_CHARS="${MAX_TERMINAL_OUTPUT_CHARS:-8000}"
WORLD_MODEL_COEFF="${WORLD_MODEL_COEFF:-0.03}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.15}"
COMMAND_BONUS="${COMMAND_BONUS:-0.0}"
FORMAT_PENALTY="${FORMAT_PENALTY:-0.05}"
REWARD_SUCCESS_BONUS="${REWARD_SUCCESS_BONUS:-0.2}"
SAVE_STEPS="${SAVE_STEPS:-10}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

HF_ENV_FILE="${HF_ENV_FILE:-$ROOT_DIR/.env}"
HF_ROLLOUT_REPO="${HF_ROLLOUT_REPO:-LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts}"
HF_ADAPTER_REPO="${HF_ADAPTER_REPO:-LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-GRPO-Adapters}"

SLEEP_SEC="${SLEEP_SEC:-60}"
POST_EXIT_GRACE_SEC="${POST_EXIT_GRACE_SEC:-120}"
AUTOPILOT_DIR="${AUTOPILOT_DIR:-/home/work/.data/liquid_cli_sft/autopilot}"
LOCK_FILE="${LOCK_FILE:-$AUTOPILOT_DIR/${CURRENT_RUN_ID}.continue.lock}"
LOG_FILE="${LOG_FILE:-$AUTOPILOT_DIR/${CURRENT_RUN_ID}.continue.log}"

mkdir -p "$AUTOPILOT_DIR"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE"
}

checkpoint_step() {
  basename "$1" | awk -F- '{print $NF}'
}

latest_checkpoint() {
  find "$CURRENT_OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%p\n' \
    | sort -V \
    | tail -1
}

healthcheck_vllm() {
  local first_url="${VLLM_BASE_URL%%,*}"
  curl -fsS "$first_url/models" >/dev/null
}

if [[ -e "$LOCK_FILE" ]]; then
  log "lock exists; refusing duplicate scheduler lock=$LOCK_FILE"
  exit 0
fi

printf '%s\n' "$$" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

cd "$ROOT_DIR"
log "scheduler armed current_pid=$CURRENT_LAUNCHER_PID current_run=$CURRENT_RUN_ID"
log "policy: vllm_gpus=$VLLM_GPUS train_gpus=$TRAIN_GPUS ignore_gpus=6,7"

while kill -0 "$CURRENT_LAUNCHER_PID" 2>/dev/null; do
  latest="$(latest_checkpoint || true)"
  if [[ -n "${latest:-}" ]]; then
    log "current run alive latest_checkpoint=$(basename "$latest")"
  else
    log "current run alive latest_checkpoint=none"
  fi
  sleep "$SLEEP_SEC"
done

log "current run exited; waiting ${POST_EXIT_GRACE_SEC}s for final checkpoint flush"
sleep "$POST_EXIT_GRACE_SEC"

RESUME_ADAPTER="$(latest_checkpoint || true)"
if [[ -z "$RESUME_ADAPTER" || ! -d "$RESUME_ADAPTER" ]]; then
  log "no checkpoint found under $CURRENT_OUTPUT_DIR; cannot continue"
  exit 1
fi

if ! healthcheck_vllm; then
  log "vLLM endpoint is not healthy; refusing to start continuation without rollout servers"
  exit 1
fi

RESUME_STEP="$(checkpoint_step "$RESUME_ADAPTER")"
NEXT_RUN_ID="${NEXT_RUN_ID:-run_$(date -u +%Y%m%dT%H%M%SZ)_echo_public1500_continue_from_${RESUME_STEP}_vllm4_train2}"
NEXT_RUN_DIR="${NEXT_RUN_DIR:-/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/$NEXT_RUN_ID}"
NEXT_OUTPUT_DIR="${NEXT_OUTPUT_DIR:-/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_$NEXT_RUN_ID}"
NEXT_TRACE_DIR="$NEXT_RUN_DIR/traces"
NEXT_SANDBOX_ROOT="$NEXT_RUN_DIR/sandboxes"

mkdir -p "$NEXT_RUN_DIR/sync_logs" "$NEXT_TRACE_DIR" "$NEXT_SANDBOX_ROOT" "$NEXT_OUTPUT_DIR"

cat > "$NEXT_RUN_DIR/run.env" <<EOF
RUN_ID=$NEXT_RUN_ID
MODEL_PATH=$MODEL_PATH
RESUME_ADAPTER=$RESUME_ADAPTER
PREPARED_JSONL=$PREPARED_JSONL
PREPARED_ROWS=$PREPARED_ROWS
PREPARED_SOURCE_OPENTHOUGHTS_AGENT_V1_RL=$PREPARED_SOURCE_OPENTHOUGHTS_AGENT_V1_RL
PREPARED_SOURCE_ENDLESS_TERMINALS=$PREPARED_SOURCE_ENDLESS_TERMINALS
VLLM_GPUS=$VLLM_GPUS
TRAIN_GPUS=$TRAIN_GPUS
VLLM_BASE_URL=$VLLM_BASE_URL
NO_DOCKER=true
ROLL_OUT_BACKEND=vllm_http
WORLD_SIZE=$NPROC_PER_NODE
NPROC_PER_NODE=$NPROC_PER_NODE
PROMPTS_PER_RANK=$PROMPTS_PER_RANK
NUM_GENERATIONS=$NUM_GENERATIONS
GLOBAL_ROLLOUTS_PER_STEP=$((NPROC_PER_NODE * PROMPTS_PER_RANK * NUM_GENERATIONS))
ROLLOUT_WORKERS=$ROLLOUT_WORKERS
MAX_STEPS=$MAX_STEPS
MAX_WALL_TIME_HOURS=$NEXT_MAX_WALL_TIME_HOURS
MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH
MAX_TURNS=$MAX_TURNS
MAX_NEW_TOKENS=$MAX_NEW_TOKENS
COMMAND_TIMEOUT=$COMMAND_TIMEOUT
VERIFIER_TIMEOUT=$VERIFIER_TIMEOUT
MAX_TERMINAL_OUTPUT_CHARS=$MAX_TERMINAL_OUTPUT_CHARS
WORLD_MODEL_COEFF=$WORLD_MODEL_COEFF
LEARNING_RATE=$LEARNING_RATE
WEIGHT_DECAY=$WEIGHT_DECAY
WARMUP_STEPS=$WARMUP_STEPS
MAX_GRAD_NORM=$MAX_GRAD_NORM
COMMAND_BONUS=$COMMAND_BONUS
FORMAT_PENALTY=$FORMAT_PENALTY
REWARD_SUCCESS_BONUS=$REWARD_SUCCESS_BONUS
SAVE_STEPS=$SAVE_STEPS
GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING
PARENT_RUN_ID=$CURRENT_RUN_ID
PARENT_OUTPUT_DIR=$CURRENT_OUTPUT_DIR
NOTES=Auto-continuation after parent max-wall-time; ECHO-style GRPO with terminal observation CE loss; no-Docker sandbox; GPUs 6,7 intentionally unused.
EOF

log "starting continuation run_id=$NEXT_RUN_ID resume=$RESUME_ADAPTER"

setsid bash -lc "
  cd '$ROOT_DIR'
  env \
    MODEL_PATH='$MODEL_PATH' \
    SFT_ADAPTER_PATH='$RESUME_ADAPTER' \
    VLLM_BASE_URL='$VLLM_BASE_URL' \
    TRAIN_GPUS='$TRAIN_GPUS' \
    NPROC_PER_NODE='$NPROC_PER_NODE' \
    MAX_STEPS='$MAX_STEPS' \
    MAX_WALL_TIME_HOURS='$NEXT_MAX_WALL_TIME_HOURS' \
    PROMPTS_PER_RANK='$PROMPTS_PER_RANK' \
    NUM_GENERATIONS='$NUM_GENERATIONS' \
    ROLLOUT_WORKERS='$ROLLOUT_WORKERS' \
    MAX_SEQ_LENGTH='$MAX_SEQ_LENGTH' \
    MAX_TURNS='$MAX_TURNS' \
    MAX_NEW_TOKENS='$MAX_NEW_TOKENS' \
    COMMAND_TIMEOUT='$COMMAND_TIMEOUT' \
    VERIFIER_TIMEOUT='$VERIFIER_TIMEOUT' \
    MAX_TERMINAL_OUTPUT_CHARS='$MAX_TERMINAL_OUTPUT_CHARS' \
    WORLD_MODEL_COEFF='$WORLD_MODEL_COEFF' \
    LEARNING_RATE='$LEARNING_RATE' \
    WEIGHT_DECAY='$WEIGHT_DECAY' \
    WARMUP_STEPS='$WARMUP_STEPS' \
    MAX_GRAD_NORM='$MAX_GRAD_NORM' \
    COMMAND_BONUS='$COMMAND_BONUS' \
    FORMAT_PENALTY='$FORMAT_PENALTY' \
    REWARD_SUCCESS_BONUS='$REWARD_SUCCESS_BONUS' \
    SAVE_STEPS='$SAVE_STEPS' \
    GRADIENT_CHECKPOINTING='$GRADIENT_CHECKPOINTING' \
    OUTPUT_DIR='$NEXT_OUTPUT_DIR' \
    TRACE_DIR='$NEXT_TRACE_DIR' \
    SANDBOX_ROOT='$NEXT_SANDBOX_ROOT' \
    PREPARED_ONLY=1 \
    PREPARED_JSONL='$PREPARED_JSONL' \
    LAUNCHER=file \
    bash Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh \
      >> '$NEXT_RUN_DIR/train.log' 2>&1
" &
TRAIN_LAUNCHER_PID="$!"
printf '%s\n' "$TRAIN_LAUNCHER_PID" > "$NEXT_RUN_DIR/launcher.pid"
log "continuation launcher_pid=$TRAIN_LAUNCHER_PID"

setsid bash -lc "
  cd '$ROOT_DIR'
  '$TRAIN_ENV/bin/python' Liquid-CLI/scripts/sync_echo_rollouts_to_hf_dataset.py \
    --repo-id '$HF_ROLLOUT_REPO' \
    --run-dir '$NEXT_RUN_DIR' \
    --output-dir '$NEXT_OUTPUT_DIR' \
    --path-in-repo 'runs/$NEXT_RUN_ID' \
    --env-file '$HF_ENV_FILE' \
    --interval-sec 300 \
    --loop \
    >> '$NEXT_RUN_DIR/sync_logs/rollouts_sync.log' 2>&1
" &
ROLLOUT_SYNC_PID="$!"
printf '%s\n' "$ROLLOUT_SYNC_PID" > "$NEXT_RUN_DIR/rollout_sync.pid"

setsid bash -lc "
  cd '$ROOT_DIR'
  '$TRAIN_ENV/bin/python' Liquid-CLI/scripts/sync_echo_adapter_checkpoints_to_hf_model.py \
    --repo-id '$HF_ADAPTER_REPO' \
    --run-dir '$NEXT_RUN_DIR' \
    --output-dir '$NEXT_OUTPUT_DIR' \
    --env-file '$HF_ENV_FILE' \
    --interval-sec 900 \
    --loop \
    >> '$NEXT_RUN_DIR/sync_logs/adapter_sync.log' 2>&1
" &
ADAPTER_SYNC_PID="$!"
printf '%s\n' "$ADAPTER_SYNC_PID" > "$NEXT_RUN_DIR/adapter_sync.pid"

log "sync_pids rollout=$ROLLOUT_SYNC_PID adapter=$ADAPTER_SYNC_PID"
log "continuation scheduled and launched; next_run_dir=$NEXT_RUN_DIR"
