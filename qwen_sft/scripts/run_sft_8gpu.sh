#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="qwen_sft/configs/sft_qwen35_2b_8gpu.env"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
if [[ -n "${UNSLOTH_RETURN_LOGITS:-}" ]]; then
  export UNSLOTH_RETURN_LOGITS
fi
if [[ -n "${UNSLOTH_FUSED_CE_COMPILE_DISABLE:-}" ]]; then
  export UNSLOTH_FUSED_CE_COMPILE_DISABLE
fi
if [[ -n "${UNSLOTH_DISABLE_FUSED_CE:-}" ]]; then
  export UNSLOTH_DISABLE_FUSED_CE
fi
if [[ -n "${UNSLOTH_COMPILE_LOCATION:-}" ]]; then
  export UNSLOTH_COMPILE_LOCATION
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG_PATH" .env)}"
LOG_PATH="/home/work/.data/qwen_sft/logs/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"
LATEST_LOG_PATH="/home/work/.data/qwen_sft/logs/${RUN_NAME}_latest.log"

EXTRA_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--gradient-checkpointing)
fi
if [[ "${LOAD_IN_4BIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load-in-4bit)
fi
if [[ -n "${FSDP:-}" ]]; then
  EXTRA_ARGS+=(--fsdp "$FSDP")
fi
if [[ -n "${FSDP_CONFIG:-}" ]]; then
  EXTRA_ARGS+=(--fsdp-config "$FSDP_CONFIG")
fi
if [[ "${UNSLOTH_RETURN_LOGITS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--return-logits)
fi
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
  /home/work/.data/qwen_sft/datasets \
  /home/work/.data/qwen_sft/models \
  /home/work/.data/qwen_sft/logs \
  /home/work/.data/liquid_cli_sft/cache/hf_home \
  /home/work/.data/liquid_cli_sft/cache/hub \
  /home/work/.data/liquid_cli_sft/cache/transformers \
  /home/work/.data/liquid_cli_sft/cache/datasets \
  "$OUTPUT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  cat <<EOF
torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" qwen_sft/scripts/train_sft_unsloth_ddp.py \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
  --processed-data-path "$PROCESSED_DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --conversation-mode "${CONVERSATION_MODE:-turn_pairs}" \
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
echo "train_mode=${TRAIN_MODE:-full}"
echo "epochs=$NUM_TRAIN_EPOCHS"
if [[ "$SAVE_STRATEGY" == "steps" ]]; then
  echo "checkpoints=every_${SAVE_STEPS}_steps_no_delete"
else
  echo "checkpoints=every_epoch_no_delete"
fi
echo "log_path=$LOG_PATH"
printf '%s\n' "$LOG_PATH" > "$LATEST_LOG_PATH"

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
  qwen_sft/scripts/train_sft_unsloth_ddp.py \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
  --processed-data-path "$PROCESSED_DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --conversation-mode "${CONVERSATION_MODE:-turn_pairs}" \
  --train-mode "${TRAIN_MODE:-full}" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --save-strategy "$SAVE_STRATEGY" \
  --logging-steps "$LOGGING_STEPS" \
  --warmup-ratio "$WARMUP_RATIO" \
  --lora-r "${LORA_R:-16}" \
  --lora-alpha "${LORA_ALPHA:-16}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"
