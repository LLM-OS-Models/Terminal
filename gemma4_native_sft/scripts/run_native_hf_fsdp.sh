#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH=""
while (($#)); do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 --config path/to/config.env" >&2
  exit 2
fi

source .liquid-sft-env/bin/activate
source "$CONFIG_PATH"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/.data/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/work/.data/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/work/.data/huggingface/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"

if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_OVERRIDE"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29531}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG_PATH" .env)}"
LOG_ROOT="${LOG_ROOT:-/home/work/.data/gemma4_native_sft/logs}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "$LOG_ROOT" "$(dirname "$OUTPUT_DIR")"

echo "config_path=$CONFIG_PATH"
echo "run_name=$RUN_NAME"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all}"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "master_port=$MASTER_PORT"
echo "model_id=$MODEL_ID"
echo "dataset_path=$DATASET_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "log_path=$LOG_PATH"

args=(
  --model-id "$MODEL_ID"
  --dataset-path "$DATASET_PATH"
  --output-dir "$OUTPUT_DIR"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
  --save-strategy "$SAVE_STRATEGY"
  --logging-steps "$LOGGING_STEPS"
  --warmup-ratio "$WARMUP_RATIO"
)

if [[ -n "${TEMPLATE_MODEL_ID:-}" ]]; then
  args+=(--template-model-id "$TEMPLATE_MODEL_ID")
fi
if [[ -n "${FSDP:-}" ]]; then
  args+=(--fsdp "$FSDP")
fi
if [[ -n "${FSDP_CONFIG:-}" ]]; then
  args+=(--fsdp-config "$FSDP_CONFIG")
fi
if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
  args+=(--gradient-checkpointing)
fi
if [[ "${SAVE_ONLY_MODEL:-0}" == "1" ]]; then
  args+=(--save-only-model)
fi
if [[ -n "${SAVE_TOTAL_LIMIT:-}" ]]; then
  args+=(--save-total-limit "$SAVE_TOTAL_LIMIT")
fi
if [[ "${SKIP_FINAL_SAVE:-0}" == "1" ]]; then
  args+=(--skip-final-save)
fi
if [[ -n "${SAVE_STEPS:-}" ]]; then
  args+=(--save-steps "$SAVE_STEPS")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  args+=(--max-steps "$MAX_STEPS")
fi

torchrun \
  --nnodes 1 \
  --nproc_per_node "$NPROC_PER_NODE" \
  --rdzv_backend c10d \
  --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
  --rdzv_id "$RUN_NAME" \
  gemma4_native_sft/scripts/train_native_hf_fsdp.py \
  "${args[@]}" \
  2>&1 | tee "$LOG_PATH"
