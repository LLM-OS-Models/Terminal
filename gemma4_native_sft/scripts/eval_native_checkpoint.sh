#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH=""
MODEL_SHORT=""
GPU="${GPU:-0}"
TP="${TP:-1}"
EVAL_PATH="${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/work/.data/tb2_lite_eval/gemma4_native_sft_20260508}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

while (($#)); do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-short)
      MODEL_SHORT="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --eval-path)
      EVAL_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL_PATH" || -z "$MODEL_SHORT" ]]; then
  echo "Usage: $0 --model-path PATH --model-short NAME [--gpu 0] [--tp 1]" >&2
  exit 2
fi

source .liquid-sft-env/bin/activate

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/.data/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/work/.data/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/work/.data/huggingface/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" python tb2_lite/scripts/replay_eval.py \
  --model "$MODEL_PATH" \
  --tokenizer-path "$MODEL_PATH" \
  --model-short "$MODEL_SHORT" \
  --gpu "$GPU" \
  --eval-path "$EVAL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --tp "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-tokens "$MAX_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --temperature 0 \
  --top-p 1 \
  --language-model-only \
  --thinking-mode off \
  --strip-thinking-history on \
  --gemma4-empty-thought-channel auto \
  --skip-if-exists
