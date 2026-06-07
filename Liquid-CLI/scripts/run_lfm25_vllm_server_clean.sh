#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"
MODEL_PATH="${MODEL_PATH:-LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8123}"
VLLM_GPUS="${VLLM_GPUS:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

VENV_SITE="$VLLM_ENV/lib/python3.12/site-packages"
VLLM_LD_LIBRARY_PATH="$VENV_SITE/torch/lib:$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cublas/lib:$VENV_SITE/nvidia/cudnn/lib:$VENV_SITE/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT_DIR"
exec env -u PYTHONPATH \
  LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH" \
  CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
  "$VLLM_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --enforce-eager
