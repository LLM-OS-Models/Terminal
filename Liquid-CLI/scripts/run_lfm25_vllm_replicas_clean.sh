#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"
MODEL_PATH="${MODEL_PATH:-LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8123}"
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LOG_DIR="${LOG_DIR:-/tmp/lfm25_vllm_replicas}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-300}"
START_STAGGER_SEC="${START_STAGGER_SEC:-0}"

VENV_SITE="$VLLM_ENV/lib/python3.12/site-packages"
VLLM_LD_LIBRARY_PATH="$VENV_SITE/torch/lib:$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cu13/lib:$VENV_SITE/nvidia/cublas/lib:$VENV_SITE/nvidia/cudnn/lib:$VENV_SITE/nvidia/nccl/lib:$VENV_SITE/nvidia/cusparselt/lib:$VENV_SITE/nvidia/nvshmem/lib:/usr/local/cuda/compat/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "$VLLM_GPUS"
PIDS=()
URLS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  port="$((BASE_PORT + i))"
  url="http://$HOST:$port/v1"
  URLS+=("$url")
  env -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$VLLM_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
      --host "$HOST" \
      --port "$port" \
      --model "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --trust-remote-code \
      --dtype bfloat16 \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --enforce-eager \
      > "$LOG_DIR/vllm_gpu${gpu}_port${port}.log" 2>&1 &
  PIDS+=("$!")
  echo "started gpu=$gpu port=$port pid=${PIDS[-1]}"
  ready=0
  for _attempt in $(seq 1 "$READY_TIMEOUT_SEC"); do
    if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
      echo "vLLM replica exited before ready gpu=$gpu port=$port pid=${PIDS[-1]}" >&2
      tail -n 200 "$LOG_DIR/vllm_gpu${gpu}_port${port}.log" >&2 || true
      exit 1
    fi
    if curl -fsS "$url/models" >/dev/null 2>&1; then
      echo "ready $url"
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != "1" ]]; then
    echo "vLLM replica did not become ready in ${READY_TIMEOUT_SEC}s gpu=$gpu port=$port pid=${PIDS[-1]}" >&2
    tail -n 200 "$LOG_DIR/vllm_gpu${gpu}_port${port}.log" >&2 || true
    exit 1
  fi
  if [[ "$START_STAGGER_SEC" != "0" ]]; then
    sleep "$START_STAGGER_SEC"
  fi
done

printf '%s\n' "${PIDS[@]}" > "$LOG_DIR/pids.txt"
printf '%s\n' "${URLS[@]}" > "$LOG_DIR/urls.txt"
printf '%s\n' "${URLS[*]// /,}" > "$LOG_DIR/vllm_base_urls.txt"
echo "VLLM_BASE_URLS=$(paste -sd, "$LOG_DIR/urls.txt")"

wait
