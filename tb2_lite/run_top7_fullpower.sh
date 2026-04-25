#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source eval/env.sh

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="tb2_lite/results/${RUN_ID}"
mkdir -p "$RESULTS_DIR"

MODELS=(
  "google/gemma-4-E2B-it"
  "LiquidAI/LFM2-8B-A1B"
  "nvidia/Nemotron-Terminal-14B"
  "Qwen/Qwen3.6-35B-A3B-FP8"
  "google/gemma-4-31B-it"
  "LiquidAI/LFM2-24B-A2B"
  "gyung/LFM2-8B-Terminal-SFT-Unsloth"
)

EXTRA_ARGS=(
  ""
  ""
  ""
  "--gdn-triton"
  ""
  ""
  ""
)

GPUS=(${TB2_LITE_GPUS:-1 2 3 4 5 6 7})

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  gpu="${GPUS[$idx]}"
  log_file="${RESULTS_DIR}/$(basename "$model").log"
  extra="${EXTRA_ARGS[$idx]}"
  CUDA_VISIBLE_DEVICES="$gpu" python tb2_lite/scripts/replay_eval.py \
    --model "$model" \
    --gpu "$gpu" \
    --eval-path tb2_lite/data/replay_full.jsonl \
    --output-dir "$RESULTS_DIR" \
    --max-tokens 768 \
    ${extra} \
    >"$log_file" 2>&1 &
done

wait

python tb2_lite/scripts/summarize_replay_results.py \
  --results-dir "$RESULTS_DIR" \
  --output-path "${RESULTS_DIR}/SUMMARY.md"

echo "Saved results to ${RESULTS_DIR}"
