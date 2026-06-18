#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

load_env_assignments() {
  local env_file="$1"
  local tmp_file

  [ -f "$env_file" ] || return 0
  tmp_file="$(mktemp)"
  grep -E '^[[:space:]]*(export[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=' "$env_file" > "$tmp_file" || true
  if [ -s "$tmp_file" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$tmp_file"
    set +a
  fi
  rm -f "$tmp_file"
}

load_env_assignments ../.env
load_env_assignments .env.local
if [ -n "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
  export HUGGINGFACE_TOKEN="$HF_TOKEN"
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_PATH="${MODEL_PATH:-LiquidAI/LFM2.5-8B-A1B}"
DATASET_KIND="${DATASET_KIND:-sec_hf}"
DATA_ROOT="${DATA_ROOT:-/home/work/.data/harness1/lfm25_local_rlvr}"
DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/${DATASET_KIND}_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/LFM2.5-8B-A1B__local_harness_rlvr_lora_${DATASET_KIND}_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-/home/work/.data/harness1/logs}"
mkdir -p "$LOG_DIR" "$(dirname "$DATASET_PATH")" "$OUTPUT_DIR"
LOG_PATH="$LOG_DIR/lfm25_local_rlvr_${DATASET_KIND}_${RUN_ID}.log"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_STEPS="${MAX_STEPS:-300}"
SAVE_STEPS="${SAVE_STEPS:-25}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/work/.data/liquid_cli_sft/cache/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/.data/liquid_cli_sft/cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/work/.data/liquid_cli_sft/cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/work/.data/liquid_cli_sft/cache/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-128}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES

BUILD_ARGS=(
  --dataset-kind "$DATASET_KIND"
  --output-path "$DATASET_PATH"
  --limit "${LIMIT:-0}"
  --max-doc-chars "${MAX_DOC_CHARS:-900}"
  --max-gold-docs "${MAX_GOLD_DOCS:-12}"
  --max-evidence-docs "${MAX_EVIDENCE_DOCS:-12}"
  --max-negative-docs "${MAX_NEGATIVE_DOCS:-18}"
  --min-candidates "${MIN_CANDIDATES:-8}"
)

if [ "$DATASET_KIND" = "browsecomp" ]; then
  BUILD_ARGS+=(--input-jsonl "${BROWSECOMP_JSONL:-/home/work/.data/harness1/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl}")
fi
if [ "$DATASET_KIND" = "sec_jsonl" ]; then
  BUILD_ARGS+=(--input-jsonl "${SEC_JSONL:-/home/work/.data/harness1/sec/sec_rlvr_candidates.jsonl}")
fi
if [ "$DATASET_KIND" = "sec_hf" ]; then
  BUILD_ARGS+=(--hf-dataset "${HF_DATASET:-kellyhongg/1_18_sec_train}")
  BUILD_ARGS+=(--hf-split "${HF_SPLIT:-train}")
  BUILD_ARGS+=(--split-ids-path "${SPLIT_IDS_PATH:-datagen/splits/sec_splits.json}")
  BUILD_ARGS+=(--split-id-key "${SPLIT_ID_KEY:-rl_query_ids}")
fi
if [ "${ALLOW_GOLD_ONLY_CANDIDATES:-0}" = "1" ]; then
  BUILD_ARGS+=(--allow-gold-only-candidates)
fi
if [ "${OVERWRITE_DATASET:-0}" = "1" ]; then
  BUILD_ARGS+=(--overwrite)
fi

echo "run_id=$RUN_ID"
echo "model_path=$MODEL_PATH"
echo "dataset_kind=$DATASET_KIND"
echo "dataset_path=$DATASET_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "log_path=$LOG_PATH"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "max_steps=$MAX_STEPS"

if [ ! -f "${DATASET_PATH}.ready" ] || [ "${OVERWRITE_DATASET:-0}" = "1" ]; then
  python training/build_lfm25_local_rlvr_dataset.py "${BUILD_ARGS[@]}"
else
  echo "dataset_ready=$DATASET_PATH"
fi

EXTRA_ARGS=()
if [ -n "${SFT_ADAPTER_PATH:-}" ]; then
  EXTRA_ARGS+=(--sft-adapter-path "$SFT_ADAPTER_PATH")
fi
if [ "${USE_VLLM:-0}" = "1" ]; then
  EXTRA_ARGS+=(--use-vllm)
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
  training/train_lfm25_local_rlvr_grpo.py \
  --model-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-prompt-length "${MAX_PROMPT_LENGTH:-8192}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH:-512}" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --max-steps "$MAX_STEPS" \
  --learning-rate "${LEARNING_RATE:-2e-6}" \
  --warmup-ratio "${WARMUP_RATIO:-0.05}" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "${LOGGING_STEPS:-1}" \
  --beta "${BETA:-0.02}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --top-p "${TOP_P:-0.95}" \
  --min-p "${MIN_P:-0.05}" \
  --lora-rank "${LORA_RANK:-32}" \
  --lora-alpha "${LORA_ALPHA:-64}" \
  --lora-dropout "${LORA_DROPOUT:-0.0}" \
  --target-modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS:-0}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"
