#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source eval/env.sh

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_deepseek_flash}"
RESULTS_DIR="tb2_lite/results/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

FLASH_MODEL="${FLASH_MODEL:-unsloth/DeepSeek-V4-Flash}"
FLASH_CKPT="${FLASH_CKPT:-/home/work/deepseek_models/DeepSeek-V4-Flash-mp4}"
FLASH_CONFIG="${FLASH_CONFIG:-/home/work/.data/huggingface/models--unsloth--DeepSeek-V4-Flash/snapshots/bc486f653513c9179e20a970587dcbe928bf7b96/inference/config.json}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --master-port 29601 --nproc-per-node 4 \
  tb2_lite/scripts/deepseek_replay_eval.py \
  --model-name "$FLASH_MODEL" \
  --ckpt-path "$FLASH_CKPT" \
  --config-path "$FLASH_CONFIG" \
  --output-dir "$RESULTS_DIR" \
  --batch-size 1 \
  --shard-count 2 \
  --shard-index 0 \
  --max-model-len 4096 \
  --max-new-tokens 1024 \
  --temperature 1.0 \
  --thinking-mode thinking \
  >"${RESULTS_DIR}/DeepSeek-V4-Flash.part00.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --master-port 29602 --nproc-per-node 4 \
  tb2_lite/scripts/deepseek_replay_eval.py \
  --model-name "$FLASH_MODEL" \
  --ckpt-path "$FLASH_CKPT" \
  --config-path "$FLASH_CONFIG" \
  --output-dir "$RESULTS_DIR" \
  --batch-size 1 \
  --shard-count 2 \
  --shard-index 1 \
  --max-model-len 4096 \
  --max-new-tokens 1024 \
  --temperature 1.0 \
  --thinking-mode thinking \
  >"${RESULTS_DIR}/DeepSeek-V4-Flash.part01.log" 2>&1 &

wait

python tb2_lite/scripts/merge_deepseek_shards.py \
  --results-dir "$RESULTS_DIR" \
  --model-short "DeepSeek-V4-Flash"

echo "Saved DeepSeek Flash results to ${RESULTS_DIR}"
