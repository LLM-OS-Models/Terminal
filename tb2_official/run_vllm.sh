#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/ubuntu/Terminal/tb2_official}"
VLLM_BIN="${VLLM_BIN:-/home/ubuntu/.venvs/tb2-vllm/bin/vllm}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
API_KEY="${API_KEY:-}"

BASE_MODEL="$ROOT_DIR/models/sft1"
STATIC_ADAPTER="$ROOT_DIR/models/static-parent-610/checkpoints/checkpoint-610"
ONLINE_ADAPTER="$ROOT_DIR/models/online-425/checkpoints/checkpoint-425"

required_files=(
  "$BASE_MODEL/model.safetensors"
  "$STATIC_ADAPTER/adapter_model.safetensors"
  "$ONLINE_ADAPTER/adapter_model.safetensors"
)
for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "missing required model file: $required_file" >&2
    exit 1
  fi
done

api_args=()
if [[ -n "$API_KEY" ]]; then
  api_args+=(--api-key "$API_KEY")
fi

exec env -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
  "$VLLM_BIN" serve "$BASE_MODEL" \
    --served-model-name lfm25-sft1 \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enforce-eager \
    --enable-lora \
    --max-lora-rank 32 \
    --max-loras 2 \
    --max-cpu-loras 2 \
    --lora-modules \
      "lfm25-static-610=$STATIC_ADAPTER" \
      "lfm25-online-425=$ONLINE_ADAPTER" \
    "${api_args[@]}"
