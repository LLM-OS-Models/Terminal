#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="gemma4_sft/configs/sft_gemma4_31b_8gpu.env"
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
export HF_HOME=/home/work/.data/liquid_cli_sft/cache/hf_home
export HF_HUB_CACHE=/home/work/.data/liquid_cli_sft/cache/hub
export TRANSFORMERS_CACHE=/home/work/.data/liquid_cli_sft/cache/transformers
export HF_DATASETS_CACHE=/home/work/.data/liquid_cli_sft/cache/datasets
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG_PATH" .env)}"
LOG_PATH="/home/work/.data/gemma4_sft/logs/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"

EXTRA_ARGS=()
if [[ "${PUSH_TO_HUB:-0}" == "1" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required when PUSH_TO_HUB=1."
    exit 1
  fi
  EXTRA_ARGS+=(--push-to-hub)
  if [[ -n "${HUB_MODEL_ID:-}" ]]; then
    EXTRA_ARGS+=(--hub-model-id "$HUB_MODEL_ID")
  fi
fi

mkdir -p \
  /home/work/.data/gemma4_sft/datasets \
  /home/work/.data/gemma4_sft/models \
  /home/work/.data/gemma4_sft/logs \
  /home/work/.data/liquid_cli_sft/cache/hf_home \
  /home/work/.data/liquid_cli_sft/cache/hub \
  /home/work/.data/liquid_cli_sft/cache/transformers \
  /home/work/.data/liquid_cli_sft/cache/datasets \
  "$OUTPUT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  cat <<EOF
torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" gemma4_sft/scripts/train_sft_unsloth_ddp.py \
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
  --warmup-ratio "$WARMUP_RATIO"
EOF
  if [[ "$SAVE_STRATEGY" == "steps" ]]; then
    printf '  --save-steps %q\n' "$SAVE_STEPS"
  fi
  if ((${#EXTRA_ARGS[@]})); then
    printf '  %q' "${EXTRA_ARGS[@]}"
    printf '\n'
  fi
  exit 0
fi

echo "config_path=$CONFIG_PATH"
echo "run_name=$RUN_NAME"
echo "nproc_per_node=$NPROC_PER_NODE"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
fi
if [[ -n "${MODEL_NAME:-}" ]]; then
  echo "model_name=$MODEL_NAME"
fi
echo "model_path=$MODEL_PATH"
echo "data_path=$DATA_PATH"
echo "processed_data_path=$PROCESSED_DATA_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "epochs=$NUM_TRAIN_EPOCHS"
if [[ "$SAVE_STRATEGY" == "steps" ]]; then
  echo "checkpoints=every_${SAVE_STEPS}_steps_no_delete"
else
  echo "checkpoints=every_epoch_no_delete"
fi
echo "log_path=$LOG_PATH"

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
  gemma4_sft/scripts/train_sft_unsloth_ddp.py \
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
