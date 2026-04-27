#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="qwen_sft/configs/sft_qwen35_27b_hf_fsdp.env"
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

source .liquid-sft-env/bin/activate
source "$CONFIG_PATH"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export HF_HOME=/home/work/.data/liquid_cli_sft/cache/hf_home
export HF_HUB_CACHE=/home/work/.data/liquid_cli_sft/cache/hub
export TRANSFORMERS_CACHE=/home/work/.data/liquid_cli_sft/cache/transformers
export HF_DATASETS_CACHE=/home/work/.data/liquid_cli_sft/cache/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG_PATH" .env)}"
LOG_PATH="/home/work/.data/qwen_sft/logs/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p \
  /home/work/.data/qwen_sft/datasets \
  /home/work/.data/qwen_sft/models \
  /home/work/.data/qwen_sft/logs

echo "config_path=$CONFIG_PATH"
echo "run_name=$RUN_NAME"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "model_path=$MODEL_PATH"
echo "processed_data_path=$PROCESSED_DATA_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "log_path=$LOG_PATH"

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
  qwen_sft/scripts/train_qwen_hf_fsdp.py \
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
  --fsdp "$FSDP" \
  --fsdp-config "$FSDP_CONFIG" \
  ${GRADIENT_CHECKPOINTING:+--gradient-checkpointing} \
  2>&1 | tee "$LOG_PATH"
