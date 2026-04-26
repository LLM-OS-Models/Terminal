#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
DEEPSEEK_ENV="${DEEPSEEK_ENV:-.deepseek-env/bin/activate}"
source "$DEEPSEEK_ENV"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_deepseek_pro}"
RESULTS_DIR="tb2_lite/results/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

PRO_MODEL="${PRO_MODEL:-unsloth/DeepSeek-V4-Pro}"
PRO_CKPT="${PRO_CKPT:-/home/work/deepseek_models/DeepSeek-V4-Pro-mp8}"
PRO_CONFIG="${PRO_CONFIG:-/home/work/.data/huggingface/models--unsloth--DeepSeek-V4-Pro/snapshots/baeea9247452e63fcbcea672a588b6db36edd378/inference/config.json}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --master-port 29611 --nproc-per-node 8 \
  tb2_lite/scripts/deepseek_replay_eval.py \
  --model-name "$PRO_MODEL" \
  --ckpt-path "$PRO_CKPT" \
  --config-path "$PRO_CONFIG" \
  --output-dir "$RESULTS_DIR" \
  --batch-size 1 \
  --max-model-len 4096 \
  --max-new-tokens 1024 \
  --temperature 1.0 \
  --thinking-mode thinking \
  >"${RESULTS_DIR}/DeepSeek-V4-Pro.log" 2>&1

echo "Saved DeepSeek Pro results to ${RESULTS_DIR}"
