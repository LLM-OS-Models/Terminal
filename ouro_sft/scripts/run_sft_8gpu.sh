#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="ouro_sft/configs/sft_ouro_1_4b_8gpu.env"
DRY_RUN=0
POSITIONAL=()
while (($#)); do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

source .liquid-sft-env/bin/activate
source "$CONFIG_PATH"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export HF_HOME=/home/work/.data/huggingface
export HF_HUB_CACHE=/home/work/.data/huggingface/hub
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG_PATH" .env)}"
LOG_PATH="/home/work/.data/ouro_sft/logs/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"
LATEST_LOG_PATH="/home/work/.data/ouro_sft/logs/${RUN_NAME}_latest.log"

EXTRA_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--gradient-checkpointing)
fi
if [[ "${PUSH_TO_HUB:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--push-to-hub)
fi

mkdir -p \
  /home/work/.data/ouro_sft/datasets \
  /home/work/.data/ouro_sft/models \
  /home/work/.data/ouro_sft/logs \
  "$OUTPUT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "torchrun --standalone --nproc_per_node $NPROC_PER_NODE ouro_sft/scripts/train_sft_hf_ddp.py \\"
  echo "  --model-path $MODEL_PATH --output-dir $OUTPUT_DIR ..."
  exit 0
fi

echo "config_path=$CONFIG_PATH"
echo "run_name=$RUN_NAME"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "model_path=$MODEL_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "log_path=$LOG_PATH"
printf '%s\n' "$LOG_PATH" > "$LATEST_LOG_PATH"

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
  ouro_sft/scripts/train_sft_hf_ddp.py \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
  --processed-data-path "$PROCESSED_DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --save-strategy "$SAVE_STRATEGY" \
  --logging-steps "$LOGGING_STEPS" \
  --warmup-ratio "$WARMUP_RATIO" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"
