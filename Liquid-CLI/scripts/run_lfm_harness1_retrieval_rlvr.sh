#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="Liquid-CLI/configs/rlvr_h200_8gpu_lfm25_8b_a1b_harness1_retrieval_grpo.env"
DRY_RUN=0
OVERWRITE_DATASET=0

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
    --overwrite-dataset)
      OVERWRITE_DATASET=1
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
export UNSLOTH_MOE_BACKEND="${UNSLOTH_MOE_BACKEND:-grouped_mm}"
export UNSLOTH_COMPILE_DISABLE="${UNSLOTH_COMPILE_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

mkdir -p /home/work/.data/liquid_cli_sft/logs "$(dirname "$DATASET_PATH")" "$OUTPUT_DIR"
LOG_PATH="/home/work/.data/liquid_cli_sft/logs/${RUN_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"

BUILD_ARGS=()
if [[ "$OVERWRITE_DATASET" == "1" ]]; then
  BUILD_ARGS+=(--overwrite)
fi
if [[ "${ALLOW_GOLD_ONLY_CANDIDATES:-0}" == "1" ]]; then
  BUILD_ARGS+=(--allow-gold-only-candidates)
fi

EXTRA_ARGS=()
if [[ -n "${SFT_ADAPTER_PATH:-}" ]]; then
  EXTRA_ARGS+=(--sft-adapter-path "$SFT_ADAPTER_PATH")
fi
if [[ "${USE_VLLM:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use-vllm)
fi
if [[ "${PUSH_TO_HUB:-0}" == "1" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required when PUSH_TO_HUB=1." >&2
    exit 1
  fi
  EXTRA_ARGS+=(--push-to-hub --hub-model-id "$HUB_MODEL_ID")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cat <<EOF
CONFIG_PATH=$CONFIG_PATH
DATASET_PATH=$DATASET_PATH
OUTPUT_DIR=$OUTPUT_DIR
LOG_PATH=$LOG_PATH

python Liquid-CLI/scripts/build_harness1_retrieval_rlvr_dataset.py \\
  --dataset-kind "${DATASET_KIND:-browsecomp}" \\
  --input-jsonl "${INPUT_JSONL:-${BROWSECOMP_JSONL:-}}" \\
  --hf-dataset "${HF_DATASET:-kellyhongg/1_18_sec_train}" \\
  --hf-split "${HF_SPLIT:-train}" \\
  --split-ids-path "${SPLIT_IDS_PATH:-}" \\
  --split-id-key "${SPLIT_ID_KEY:-rl_query_ids}" \\
  --output-path "$DATASET_PATH" \\
  --max-doc-chars "$MAX_DOC_CHARS" \\
  --max-gold-docs "$MAX_GOLD_DOCS" \\
  --max-evidence-docs "$MAX_EVIDENCE_DOCS" \\
  --max-negative-docs "$MAX_NEGATIVE_DOCS" ${BUILD_ARGS[*]}

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" Liquid-CLI/train_lfm_retrieval_rlvr_grpo.py \\
  --model-path "$MODEL_PATH" \\
  --dataset-path "$DATASET_PATH" \\
  --output-dir "$OUTPUT_DIR" \\
  --max-seq-length "$MAX_SEQ_LENGTH" \\
  --max-prompt-length "$MAX_PROMPT_LENGTH" \\
  --max-completion-length "$MAX_COMPLETION_LENGTH" \\
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \\
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \\
  --num-generations "$NUM_GENERATIONS" \\
  --max-steps "$MAX_STEPS" \\
  --learning-rate "$LEARNING_RATE" \\
  --warmup-ratio "$WARMUP_RATIO" \\
  --save-steps "$SAVE_STEPS" \\
  --logging-steps "$LOGGING_STEPS" \\
  --beta "$BETA" \\
  --temperature "$TEMPERATURE" \\
  --top-p "$TOP_P" \\
  --min-p "$MIN_P" \\
  --lora-rank "$LORA_RANK" \\
  --lora-alpha "$LORA_ALPHA" \\
  --lora-dropout "$LORA_DROPOUT" \\
  --target-modules "$TARGET_MODULES" \\
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-0}" ${EXTRA_ARGS[*]}
EOF
  exit 0
fi

if [[ ! -f "${DATASET_PATH}.ready" || "$OVERWRITE_DATASET" == "1" ]]; then
  python Liquid-CLI/scripts/build_harness1_retrieval_rlvr_dataset.py \
    --dataset-kind "${DATASET_KIND:-browsecomp}" \
    --input-jsonl "${INPUT_JSONL:-${BROWSECOMP_JSONL:-}}" \
    --hf-dataset "${HF_DATASET:-kellyhongg/1_18_sec_train}" \
    --hf-split "${HF_SPLIT:-train}" \
    --split-ids-path "${SPLIT_IDS_PATH:-}" \
    --split-id-key "${SPLIT_ID_KEY:-rl_query_ids}" \
    --output-path "$DATASET_PATH" \
    --max-doc-chars "$MAX_DOC_CHARS" \
    --max-gold-docs "$MAX_GOLD_DOCS" \
    --max-evidence-docs "$MAX_EVIDENCE_DOCS" \
    --max-negative-docs "$MAX_NEGATIVE_DOCS" \
    "${BUILD_ARGS[@]}"
else
  echo "dataset_ready=$DATASET_PATH"
fi

echo "config_path=$CONFIG_PATH"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "model_path=$MODEL_PATH"
echo "sft_adapter_path=${SFT_ADAPTER_PATH:-}"
echo "dataset_path=$DATASET_PATH"
echo "dataset_kind=${DATASET_KIND:-browsecomp}"
echo "output_dir=$OUTPUT_DIR"
echo "max_steps=$MAX_STEPS"
echo "num_generations=$NUM_GENERATIONS"
echo "effective_batch_size=$((NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "log_path=$LOG_PATH"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
  Liquid-CLI/train_lfm_retrieval_rlvr_grpo.py \
  --model-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --max-steps "$MAX_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-ratio "$WARMUP_RATIO" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --beta "$BETA" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --min-p "$MIN_P" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --target-modules "$TARGET_MODULES" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-0}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"

if [[ "${EVAL_AFTER_TRAIN:-1}" == "1" ]]; then
  ADAPTER_PATH="$OUTPUT_DIR/final_lora"
  EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-$DATASET_PATH}"
  EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$OUTPUT_DIR/eval}"
  EVAL_JSONL="$EVAL_OUTPUT_DIR/predictions.jsonl"
  EVAL_METRICS="$EVAL_OUTPUT_DIR/metrics.json"
  mkdir -p "$EVAL_OUTPUT_DIR"
  echo "eval_adapter_path=$ADAPTER_PATH"
  echo "eval_dataset_path=$EVAL_DATASET_PATH"
  echo "eval_metrics=$EVAL_METRICS"
  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}" python Liquid-CLI/scripts/eval_harness1_retrieval_curation.py \
    --model-path "$MODEL_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --dataset-path "$EVAL_DATASET_PATH" \
    --output-jsonl "$EVAL_JSONL" \
    --metrics-json "$EVAL_METRICS" \
    --split-limit "${EVAL_SPLIT_LIMIT:-0}" \
    --max-prompt-length "${EVAL_MAX_PROMPT_LENGTH:-$MAX_PROMPT_LENGTH}" \
    --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-$MAX_COMPLETION_LENGTH}" \
    --batch-size "${EVAL_BATCH_SIZE:-1}" \
    2>&1 | tee -a "$LOG_PATH"
fi
