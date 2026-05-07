#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="Liquid-CLI/configs/sft_h200_7gpu_processed.env"
DRY_RUN=0

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
      echo "Unknown argument: $1" >&2
      exit 1
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
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS=32
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"

LOG_PATH="/home/work/.data/liquid_cli_sft/logs/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"

EXTRA_ARGS=()
if [[ "${PUSH_TO_HUB:-0}" == "1" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required when PUSH_TO_HUB=1." >&2
    exit 1
  fi
  EXTRA_ARGS+=(--push-to-hub --hub-model-id "$HUB_MODEL_ID")
fi
if [[ -n "${DATASET_PATH:-}" ]]; then
  EXTRA_ARGS+=(--dataset-path "$DATASET_PATH")
fi
if [[ -n "${HOLDOUT_PATH:-}" ]]; then
  EXTRA_ARGS+=(--holdout-path "$HOLDOUT_PATH")
fi
if [[ "${OVERWRITE_PROCESSED_DATA:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite-processed-data)
fi
if [[ "${OVERWRITE_TRAIN_READY_DATA:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite-train-ready-data)
fi

mkdir -p \
  /home/work/.data/liquid_cli_sft/cache/hf_home \
  /home/work/.data/liquid_cli_sft/cache/hub \
  /home/work/.data/liquid_cli_sft/cache/transformers \
  /home/work/.data/liquid_cli_sft/cache/datasets \
  /home/work/.data/liquid_cli_sft/logs \
  "$(dirname "$PROCESSED_DATA_PATH")" \
  "$(dirname "$TRAIN_READY_DATA_PATH")" \
  "$OUTPUT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  cat <<EOF
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" Liquid-CLI/train_unsloth_processed.py \
  --model-path "$MODEL_PATH" \
  --dataset-name "$DATASET_NAME" \
  --dataset-split "$DATASET_SPLIT" \
  --processed-data-path "$PROCESSED_DATA_PATH" \
  --train-ready-data-path "$TRAIN_READY_DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --save-strategy "$SAVE_STRATEGY" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --warmup-ratio "$WARMUP_RATIO" \
  --dataset-num-proc "$DATASET_NUM_PROC"
EOF
  if ((${#EXTRA_ARGS[@]})); then
    printf '  %q' "${EXTRA_ARGS[@]}"
    printf '\n'
  fi
  exit 0
fi

echo "config_path=$CONFIG_PATH"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "model_path=$MODEL_PATH"
echo "dataset_name=$DATASET_NAME"
echo "dataset_path=${DATASET_PATH:-}"
echo "dataset_split=$DATASET_SPLIT"
echo "processed_data_path=$PROCESSED_DATA_PATH"
echo "train_ready_data_path=$TRAIN_READY_DATA_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "epochs=$NUM_TRAIN_EPOCHS"
echo "save_strategy=$SAVE_STRATEGY"
echo "log_path=$LOG_PATH"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
  Liquid-CLI/train_unsloth_processed.py \
  --model-path "$MODEL_PATH" \
  --dataset-name "$DATASET_NAME" \
  --dataset-split "$DATASET_SPLIT" \
  --processed-data-path "$PROCESSED_DATA_PATH" \
  --train-ready-data-path "$TRAIN_READY_DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --save-strategy "$SAVE_STRATEGY" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --warmup-ratio "$WARMUP_RATIO" \
  --dataset-num-proc "$DATASET_NUM_PROC" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"
