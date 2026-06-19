#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HARNESS_ROOT/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-LiquidAI/LFM2.5-8B-A1B}"
DATASET_PATH="${DATASET_PATH:-/home/work/.data/harness1/lfm25_local_rlvr/browsecomp_lfm25_harness1_browsecomp_fallback_20260618T103645Z}"
ADAPTER_PATH="${ADAPTER_PATH:?ADAPTER_PATH is required}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${RESULT_DIR:-/home/work/.data/harness1/evals/$RUN_ID}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-lfm25_harness1_rlvr_eval}"
VLLM_GPUS="${VLLM_GPUS:-7}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8133}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_BASE_PORT}/v1}"
LIMIT="${LIMIT:-120}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TRAIN_ENV="${TRAIN_ENV:-$REPO_ROOT/.liquid-sft-env}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-$RESULT_DIR/vllm_logs}"

mkdir -p "$RESULT_DIR" "$VLLM_LOG_DIR"

VLLM_PID=""
cleanup() {
  if [ -n "$VLLM_PID" ]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

MODEL_PATH="$MODEL_PATH" \
  SERVED_MODEL_NAME="$MODEL_PATH" \
  VLLM_GPUS="$VLLM_GPUS" \
  BASE_PORT="$VLLM_BASE_PORT" \
  MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}" \
  GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}" \
  ENABLE_LORA=1 \
  MAX_LORA_RANK="${MAX_LORA_RANK:-32}" \
  LOG_DIR="$VLLM_LOG_DIR" \
  bash Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh \
  > "$VLLM_LOG_DIR/launcher.log" 2>&1 &
VLLM_PID="$!"

for _attempt in $(seq 1 "${VLLM_READY_TIMEOUT_SEC:-600}"); do
  if curl -fsS "$VLLM_BASE_URL/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM eval server exited before ready" >&2
    tail -n 200 "$VLLM_LOG_DIR/launcher.log" >&2 || true
    exit 1
  fi
  sleep 1
done
curl -fsS "$VLLM_BASE_URL/models" >/dev/null

curl -fsS -X POST "$VLLM_BASE_URL/load_lora_adapter" \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"$SERVED_MODEL_NAME\",\"lora_path\":\"$ADAPTER_PATH\",\"load_inplace\":false}" \
  > "$RESULT_DIR/load_lora_response.txt"

"$TRAIN_ENV/bin/python" harness-1/eval_scripts/eval_lfm25_rlvr_retrieval_vllm.py \
  --dataset-path "$DATASET_PATH" \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --vllm-base-url "$VLLM_BASE_URL" \
  --limit "$LIMIT" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --max-tokens "$MAX_TOKENS" \
  --output-jsonl "$RESULT_DIR/predictions.jsonl" \
  --summary-json "$RESULT_DIR/summary.json"

echo "summary=$RESULT_DIR/summary.json"
echo "predictions=$RESULT_DIR/predictions.jsonl"

