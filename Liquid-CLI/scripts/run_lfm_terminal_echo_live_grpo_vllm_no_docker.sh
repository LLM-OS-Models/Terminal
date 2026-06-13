#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
TRAIN_ENV="${TRAIN_ENV:-$ROOT_DIR/.liquid-sft-env}"
MODEL_PATH="${MODEL_PATH:-LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8123/v1}"
VLLM_SERVED_MODEL="${VLLM_SERVED_MODEL:-$MODEL_PATH}"
VLLM_LORA_NAME="${VLLM_LORA_NAME:-}"
VLLM_LORA_SYNC_STEPS="${VLLM_LORA_SYNC_STEPS:-0}"
VLLM_LORA_SYNC_DIR="${VLLM_LORA_SYNC_DIR:-}"
VLLM_LORA_LOAD_INPLACE="${VLLM_LORA_LOAD_INPLACE:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MAX_STEPS="${MAX_STEPS:-100}"
MAX_WALL_TIME_HOURS="${MAX_WALL_TIME_HOURS:-0}"
PROMPTS_PER_RANK="${PROMPTS_PER_RANK:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-0}"
MAX_TURNS="${MAX_TURNS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
WORLD_MODEL_COEFF="${WORLD_MODEL_COEFF:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.2}"
SAVE_STEPS="${SAVE_STEPS:-25}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-32768}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-20}"
VERIFIER_TIMEOUT="${VERIFIER_TIMEOUT:-60}"
MAX_TERMINAL_OUTPUT_CHARS="${MAX_TERMINAL_OUTPUT_CHARS:-12000}"
COMMAND_BONUS="${COMMAND_BONUS:-0.02}"
FORMAT_PENALTY="${FORMAT_PENALTY:-0.05}"
REWARD_SUCCESS_BONUS="${REWARD_SUCCESS_BONUS:-0.0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32}"
TRACE_DIR="${TRACE_DIR:-/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/traces}"
SANDBOX_ROOT="${SANDBOX_ROOT:-/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/sandboxes}"
LAUNCHER="${LAUNCHER:-file}"
PREPARED_JSONL="${PREPARED_JSONL:-}"
PREPARED_ONLY="${PREPARED_ONLY:-0}"

cd "$ROOT_DIR"
FIRST_VLLM_BASE_URL="${VLLM_BASE_URL%%,*}"
curl -fsS "$FIRST_VLLM_BASE_URL/models" >/dev/null

TORCH_LIB="$TRAIN_ENV/lib/python3.12/site-packages/torch/lib"
NVIDIA_ROOT="$TRAIN_ENV/lib/python3.12/site-packages/nvidia"
NVIDIA_LIBS=""
if [[ -d "$NVIDIA_ROOT" ]]; then
  NVIDIA_LIBS="$(find "$NVIDIA_ROOT" -path '*/lib' -type d | paste -sd: -)"
fi

TRAIN_ARGS=(
  Liquid-CLI/train_lfm_terminal_echo_live_grpo.py
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --trace-dir "$TRACE_DIR"
  --sandbox-root "$SANDBOX_ROOT"
  --rollout-backend vllm_http
  --vllm-base-url "$VLLM_BASE_URL"
  --vllm-served-model "$VLLM_SERVED_MODEL"
  --vllm-lora-sync-steps "$VLLM_LORA_SYNC_STEPS"
  --max-steps "$MAX_STEPS"
  --max-wall-time-hours "$MAX_WALL_TIME_HOURS"
  --prompts-per-rank "$PROMPTS_PER_RANK"
  --num-generations "$NUM_GENERATIONS"
  --rollout-workers "$ROLLOUT_WORKERS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --max-turns "$MAX_TURNS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --command-timeout "$COMMAND_TIMEOUT"
  --verifier-timeout "$VERIFIER_TIMEOUT"
  --max-terminal-output-chars "$MAX_TERMINAL_OUTPUT_CHARS"
  --world-model-coeff "$WORLD_MODEL_COEFF"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --warmup-steps "$WARMUP_STEPS"
  --max-grad-norm "$MAX_GRAD_NORM"
  --command-bonus "$COMMAND_BONUS"
  --format-penalty "$FORMAT_PENALTY"
  --reward-success-bonus "$REWARD_SUCCESS_BONUS"
  --save-steps "$SAVE_STEPS"
  --logging-steps 1
  "$@"
)

if [[ -n "$VLLM_LORA_NAME" ]]; then
  TRAIN_ARGS+=(--vllm-lora-name "$VLLM_LORA_NAME")
fi

if [[ -n "$VLLM_LORA_SYNC_DIR" ]]; then
  TRAIN_ARGS+=(--vllm-lora-sync-dir "$VLLM_LORA_SYNC_DIR")
fi

if [[ "$VLLM_LORA_LOAD_INPLACE" == "1" ]]; then
  TRAIN_ARGS+=(--vllm-lora-load-inplace)
else
  TRAIN_ARGS+=(--no-vllm-lora-load-inplace)
fi

if [[ "$PREPARED_ONLY" == "1" ]]; then
  TRAIN_ARGS+=(
    --no-include-openthoughts-rl
    --no-include-endless
    --no-include-tb-dev
    --no-include-tblite-train
  )
else
  TRAIN_ARGS+=(
    --include-openthoughts-rl
    --include-endless
    --include-tb-dev
    --include-tblite-train
  )
fi

if [[ -n "$PREPARED_JSONL" ]]; then
  IFS=',' read -r -a PREPARED_FILES <<< "$PREPARED_JSONL"
  for prepared_file in "${PREPARED_FILES[@]}"; do
    TRAIN_ARGS+=(--prepared-jsonl "$prepared_file")
  done
fi

if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  TRAIN_ARGS+=(--gradient-checkpointing)
else
  TRAIN_ARGS+=(--no-gradient-checkpointing)
fi

if [[ -n "$SFT_ADAPTER_PATH" ]]; then
  TRAIN_ARGS+=(--sft-adapter-path "$SFT_ADAPTER_PATH")
fi

COMMON_ENV=(
  -u PYTHONPATH
  PYTHONNOUSERSITE=1
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
)

if [[ "$LAUNCHER" == "torchrun" ]]; then
  exec env \
    "${COMMON_ENV[@]}" \
    "$TRAIN_ENV/bin/python" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$NPROC_PER_NODE" \
      "${TRAIN_ARGS[@]}"
fi

if [[ "$LAUNCHER" != "file" ]]; then
  echo "Unknown LAUNCHER=$LAUNCHER; expected file or torchrun" >&2
  exit 2
fi

mkdir -p "$TRACE_DIR" "$SANDBOX_ROOT"
INIT_FILE="${DIST_INIT_FILE:-$TRACE_DIR/dist_init_${NPROC_PER_NODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
rm -f "$INIT_FILE"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for ((local_rank=0; local_rank<NPROC_PER_NODE; local_rank++)); do
  env \
    "${COMMON_ENV[@]}" \
    WORLD_SIZE="$NPROC_PER_NODE" \
    RANK="$local_rank" \
    LOCAL_RANK="$local_rank" \
    DIST_INIT_METHOD="file://$INIT_FILE" \
    "$TRAIN_ENV/bin/python" -u "${TRAIN_ARGS[@]}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
