#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
DEEPSEEK_ENV="${DEEPSEEK_ENV:-.deepseek-env/bin/activate}"
source "$DEEPSEEK_ENV"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_deepseek_flash}"
RESULTS_DIR="tb2_lite/results/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

FLASH_MODEL="${FLASH_MODEL:-unsloth/DeepSeek-V4-Flash}"
FLASH_CKPT="${FLASH_CKPT:-/home/work/deepseek_models/DeepSeek-V4-Flash-mp4}"
FLASH_CONFIG="${FLASH_CONFIG:-/home/work/.data/huggingface/models--unsloth--DeepSeek-V4-Flash/snapshots/bc486f653513c9179e20a970587dcbe928bf7b96/inference/config.json}"
FLASH_EVAL_PATH="${FLASH_EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
FLASH_BATCH_SIZE="${FLASH_BATCH_SIZE:-1}"
FLASH_MAX_MODEL_LEN="${FLASH_MAX_MODEL_LEN:-4096}"
FLASH_MAX_NEW_TOKENS="${FLASH_MAX_NEW_TOKENS:-1024}"
FLASH_PROGRESS_EVERY="${FLASH_PROGRESS_EVERY:-8}"
FLASH_TEMPERATURE="${FLASH_TEMPERATURE:-1.0}"
FLASH_THINKING_MODE="${FLASH_THINKING_MODE:-thinking}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --master-port 29601 --nproc-per-node 4 \
  tb2_lite/scripts/deepseek_replay_eval.py \
  --model-name "$FLASH_MODEL" \
  --ckpt-path "$FLASH_CKPT" \
  --config-path "$FLASH_CONFIG" \
  --eval-path "$FLASH_EVAL_PATH" \
  --output-dir "$RESULTS_DIR" \
  --batch-size "$FLASH_BATCH_SIZE" \
  --shard-count 2 \
  --shard-index 0 \
  --max-model-len "$FLASH_MAX_MODEL_LEN" \
  --max-new-tokens "$FLASH_MAX_NEW_TOKENS" \
  --temperature "$FLASH_TEMPERATURE" \
  --thinking-mode "$FLASH_THINKING_MODE" \
  --progress-every "$FLASH_PROGRESS_EVERY" \
  >"${RESULTS_DIR}/DeepSeek-V4-Flash.part00.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --master-port 29602 --nproc-per-node 4 \
  tb2_lite/scripts/deepseek_replay_eval.py \
  --model-name "$FLASH_MODEL" \
  --ckpt-path "$FLASH_CKPT" \
  --config-path "$FLASH_CONFIG" \
  --eval-path "$FLASH_EVAL_PATH" \
  --output-dir "$RESULTS_DIR" \
  --batch-size "$FLASH_BATCH_SIZE" \
  --shard-count 2 \
  --shard-index 1 \
  --max-model-len "$FLASH_MAX_MODEL_LEN" \
  --max-new-tokens "$FLASH_MAX_NEW_TOKENS" \
  --temperature "$FLASH_TEMPERATURE" \
  --thinking-mode "$FLASH_THINKING_MODE" \
  --progress-every "$FLASH_PROGRESS_EVERY" \
  >"${RESULTS_DIR}/DeepSeek-V4-Flash.part01.log" 2>&1 &

wait

python tb2_lite/scripts/merge_deepseek_shards.py \
  --results-dir "$RESULTS_DIR" \
  --model-short "DeepSeek-V4-Flash"

echo "Saved DeepSeek Flash results to ${RESULTS_DIR}"
