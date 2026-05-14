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
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
LIMIT="${LIMIT:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
THINKING_MODE="${THINKING_MODE:-off}"
STRIP_THINKING_HISTORY="${STRIP_THINKING_HISTORY:-on}"
GEMMA4_EMPTY_THOUGHT_CHANNEL="${GEMMA4_EMPTY_THOUGHT_CHANNEL:-auto}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"
PYTHON_BIN="${VLLM_PYTHON:-}"

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
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --cpu-offload-gb)
      CPU_OFFLOAD_GB="$2"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --enforce-eager)
      ENFORCE_EAGER=1
      shift
      ;;
    --disable-custom-all-reduce)
      DISABLE_CUSTOM_ALL_REDUCE=1
      shift
      ;;
    --thinking-mode)
      THINKING_MODE="$2"
      shift 2
      ;;
    --strip-thinking-history)
      STRIP_THINKING_HISTORY="$2"
      shift 2
      ;;
    --gemma4-empty-thought-channel)
      GEMMA4_EMPTY_THOUGHT_CHANNEL="$2"
      shift 2
      ;;
    --no-skip-if-exists)
      SKIP_IF_EXISTS=0
      shift
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

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".vllm-eval-cu129/bin/python" ]]; then
    PYTHON_BIN=".vllm-eval-cu129/bin/python"
  elif [[ -x ".vllm-0_19_1/bin/python" ]]; then
    PYTHON_BIN=".vllm-0_19_1/bin/python"
  elif [[ -x ".vllm-uv-env/bin/python" ]]; then
    PYTHON_BIN=".vllm-uv-env/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME
VENV_ROOT="$(cd "$(dirname "$PYTHON_BIN")/.." 2>/dev/null && pwd || true)"
if [[ -n "$VENV_ROOT" ]]; then
  for libdir in \
    "$VENV_ROOT/lib/python3.12/site-packages/nvidia/cu13/lib" \
    "$VENV_ROOT/lib/python3.12/site-packages/nvidia/cuda_runtime/lib" \
    "$VENV_ROOT/lib/python3.12/site-packages/torch/lib"; do
    if [[ -d "$libdir" ]]; then
      export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
    fi
  done
fi
export HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/work/.data/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/work/.data/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/work/.data/huggingface/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

mkdir -p "$OUTPUT_DIR"

ARGS=(
  tb2_lite/scripts/replay_eval.py
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
  --thinking-mode "$THINKING_MODE" \
  --strip-thinking-history "$STRIP_THINKING_HISTORY" \
  --gemma4-empty-thought-channel "$GEMMA4_EMPTY_THOUGHT_CHANNEL"
)

if [[ "$CPU_OFFLOAD_GB" != "0" && "$CPU_OFFLOAD_GB" != "0.0" ]]; then
  ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi
if [[ -n "$MAX_NUM_SEQS" ]]; then
  ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
  ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi
if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
  ARGS+=(--disable-custom-all-reduce)
fi
if [[ "$SKIP_IF_EXISTS" == "1" ]]; then
  ARGS+=(--skip-if-exists)
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" "${ARGS[@]}"
