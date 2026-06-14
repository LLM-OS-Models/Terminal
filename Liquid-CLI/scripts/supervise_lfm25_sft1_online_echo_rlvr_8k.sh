#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
RUN_DIR="${RUN_DIR:-/home/work/.data/liquid_cli_sft/live_terminal_echo_online/run_20260613T014646Z_sft1_online_vllm_lora_sync_g4_t4_sync5_wm005}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__online_echo_grpo_vllm_lora_sync_run_20260613T014646Z_sft1_online_vllm_lora_sync_g4_t4_sync5_wm005}"
PREPARED_JSONL="${PREPARED_JSONL:-/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl}"
TARGET_MAX_STEPS="${TARGET_MAX_STEPS:-8900}"
MIN_START_STEP="${MIN_START_STEP:-900}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
POLL_SECONDS="${POLL_SECONDS:-120}"
TRAIN_LOG="${TRAIN_LOG:-$RUN_DIR/logs/train_restart_8k_from900.log}"
SUPERVISOR_LOG="${SUPERVISOR_LOG:-$RUN_DIR/logs/train_supervisor_8k_from900.log}"
PID_FILE="${PID_FILE:-$RUN_DIR/train_supervisor_8k_from900.pid}"

mkdir -p "$RUN_DIR/logs"
echo "$$" > "$PID_FILE"

log() {
  echo "[$(TZ=Asia/Seoul date -Is)] $*" | tee -a "$SUPERVISOR_LOG"
}

latest_step() {
  find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
    | sed 's/.*checkpoint-//' \
    | awk 'BEGIN {m=0} /^[0-9]+$/ {if ($1 > m) m=$1} END {print m}'
}

train_pids() {
  ps -eo pid=,args= \
    | grep -F 'Liquid-CLI/train_lfm_terminal_echo_live_grpo.py' \
    | grep -F -- "--output-dir $OUTPUT_DIR" \
    | grep -v grep \
    | awk '{print $1}'
}

train_count() {
  train_pids | wc -l | tr -d ' '
}

cleanup_partial_train() {
  local pids
  pids="$(train_pids | tr '\n' ' ')"
  if [[ -n "$pids" ]]; then
    log "cleanup_partial_train pids=$pids"
    kill $pids 2>/dev/null || true
    sleep 20
    kill -9 $pids 2>/dev/null || true
  fi
  pkill -f "run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh" 2>/dev/null || true
}

start_train() {
  local step="$1"
  local adapter="$OUTPUT_DIR/checkpoint-$step"
  if [[ ! -d "$adapter" ]]; then
    log "missing_adapter adapter=$adapter"
    return 1
  fi

  cd "$ROOT_DIR" || return 1
  log "start_train start_step=$step max_steps=$TARGET_MAX_STEPS adapter=$adapter"
  setsid bash -lc '
    set -euo pipefail
    source "$0/run_env.sh"
    export PREPARED_ONLY=1
    export PREPARED_JSONL="$1"
    export SFT_ADAPTER_PATH="$2"
    export START_STEP="$3"
    export MAX_STEPS="$4"
    export MAX_WALL_TIME_HOURS=0
    export TRAIN_GPUS=4,5
    export NPROC_PER_NODE=2
    export PROMPTS_PER_RANK=1
    export NUM_GENERATIONS=4
    export ROLLOUT_WORKERS=8
    export MAX_TURNS=4
    export MAX_NEW_TOKENS=256
    export COMMAND_TIMEOUT=8
    export VERIFIER_TIMEOUT=40
    export SAVE_STEPS=25
    export VLLM_LORA_SYNC_STEPS=5
    export WORLD_MODEL_COEFF=0.05
    export LEARNING_RATE=1e-6
    export WEIGHT_DECAY=0.01
    export WARMUP_STEPS=10
    export MAX_GRAD_NORM=0.2
    export GRADIENT_CHECKPOINTING=1
    export LAUNCHER=file
    echo "[$(TZ=Asia/Seoul date -Is)] supervised_restart start_step=$START_STEP max_steps=$MAX_STEPS adapter=$SFT_ADAPTER_PATH"
    exec Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh
  ' "$RUN_DIR" "$PREPARED_JSONL" "$adapter" "$step" "$TARGET_MAX_STEPS" >> "$TRAIN_LOG" 2>&1 &
  echo "$!" > "$RUN_DIR/train_restart_8k_from900.pid"
  log "started_launcher pid=$(cat "$RUN_DIR/train_restart_8k_from900.pid")"
}

log "supervisor_start target_max_steps=$TARGET_MAX_STEPS min_start_step=$MIN_START_STEP poll_seconds=$POLL_SECONDS"

while true; do
  step="$(latest_step)"
  if [[ -z "$step" || "$step" -lt "$MIN_START_STEP" ]]; then
    step="$MIN_START_STEP"
  fi

  if [[ "$step" -ge "$TARGET_MAX_STEPS" ]]; then
    log "target_reached latest_checkpoint=$step"
    exit 0
  fi

  count="$(train_count)"
  if [[ "$count" -ge "$NPROC_PER_NODE" ]]; then
    log "train_running count=$count latest_checkpoint=$step next_checkpoint=$((step + 25))"
    sleep "$POLL_SECONDS"
    continue
  fi

  if [[ "$count" -gt 0 ]]; then
    log "partial_train_detected count=$count expected=$NPROC_PER_NODE latest_checkpoint=$step"
    cleanup_partial_train
  else
    log "train_not_running latest_checkpoint=$step"
  fi

  start_train "$step" || true
  sleep "$POLL_SECONDS"
done
