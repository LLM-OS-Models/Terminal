#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL_PATH="${MODEL_PATH:-LiquidAI/LFM2.5-8B-A1B}"
DATASET_PATH="${DATASET_PATH:-/home/work/.data/harness1/lfm25_local_rlvr/browsecomp_lfm25_harness1_browsecomp_fallback_20260618T103645Z}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/work/.data/harness1/sft_data/lfm25_agentic_sft_${RUN_ID}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/harness1/models/LFM2.5-8B-A1B__agentic_sft_lora_${RUN_ID}}"
TRAIN_ENV="${TRAIN_ENV:-$REPO_ROOT/.liquid-sft-env}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

"$TRAIN_ENV/bin/python" harness-1/training/build_lfm25_agentic_sft.py \
  --dataset-path "$DATASET_PATH" \
  --output-jsonl "$TRAIN_JSONL" \
  --limit "${LIMIT:-0}" \
  --max-search-results "${MAX_SEARCH_RESULTS:-8}" \
  --max-review-docs "${MAX_REVIEW_DOCS:-3}" \
  "${INCLUDE_VERIFY_FLAG:---include-verify}"

TORCH_LIB="$TRAIN_ENV/lib/python3.12/site-packages/torch/lib"
NVIDIA_ROOT="$TRAIN_ENV/lib/python3.12/site-packages/nvidia"
NVIDIA_LIBS=""
if [ -d "$NVIDIA_ROOT" ]; then
  NVIDIA_LIBS="$(find "$NVIDIA_ROOT" -path '*/lib' -type d | paste -sd: -)"
fi

ADAPTER_ARGS=()
if [ -n "${SFT_ADAPTER_PATH:-}" ]; then
  ADAPTER_ARGS=(--sft-adapter-path "$SFT_ADAPTER_PATH")
fi

env -u PYTHONPATH \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}" \
  LD_LIBRARY_PATH="$TORCH_LIB${NVIDIA_LIBS:+:$NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$TRAIN_ENV/bin/python" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    harness-1/training/train_lfm25_rlvr_json_sft.py \
      --model-path "$MODEL_PATH" \
      "${ADAPTER_ARGS[@]}" \
      --train-jsonl "$TRAIN_JSONL" \
      --output-dir "$OUTPUT_DIR" \
      --max-seq-length "${MAX_SEQ_LENGTH:-8192}" \
      --epochs "${EPOCHS:-2}" \
      --learning-rate "${LEARNING_RATE:-1e-4}" \
      --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
      --save-steps "${SAVE_STEPS:-50}" \
      --lora-rank "${LORA_RANK:-32}" \
      --lora-alpha "${LORA_ALPHA:-64}" \
      --lora-dropout "${LORA_DROPOUT:-0.0}" \
      --target-modules "${TARGET_MODULES:-q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate}"
