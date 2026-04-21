#!/bin/bash
# Run all models in parallel, one per GPU
# Usage: bash run_all.sh [--max-samples N]
#
# Models and GPU assignment:
#   GPU 0: Qwen/Qwen3.5-2B
#   GPU 1: Qwen/Qwen3.5-4B
#   GPU 2: Qwen/Qwen3.5-9B
#   GPU 3: google/gemma-4-E2B-it
#   GPU 4: google/gemma-4-E4B-it
#   GPU 5: OBLITERATUS/gemma-4-E4B-it-OBLITERATED
#   GPU 6: Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled
#   GPU 7: Qwen/Qwen3.6-35B-A3B-FP8

set -e
cd "$(dirname "$0")"

MAX_SAMPLES=""
if [ "$1" = "--max-samples" ] && [ -n "$2" ]; then
    MAX_SAMPLES="--max-samples $2"
fi

declare -A MODELS
MODELS[0]="Qwen/Qwen3.5-2B"
MODELS[1]="Qwen/Qwen3.5-4B"
MODELS[2]="Qwen/Qwen3.5-9B"
MODELS[3]="google/gemma-4-E2B-it"
MODELS[4]="google/gemma-4-E4B-it"
MODELS[5]="OBLITERATUS/gemma-4-E4B-it-OBLITERATED"
MODELS[6]="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
MODELS[7]="Qwen/Qwen3.6-35B-A3B-FP8"

PIDS=()

for gpu in 0 1 2 3 4 5 6 7; do
    model="${MODELS[$gpu]}"
    echo "[GPU $gpu] Starting: $model"
    CUDA_VISIBLE_DEVICES=$gpu python3 run_eval.py \
        --model "$model" \
        --gpu 0 \
        --output-dir results \
        $MAX_SAMPLES \
        > "logs/gpu${gpu}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All 8 models running in parallel. PIDs: ${PIDS[*]}"
echo "Logs: eval/logs/gpu{0-7}.log"
echo ""

# Wait for all and report
FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[GPU $i] DONE"
    else
        echo "[GPU $i] FAILED (exit $?)"
        FAIL=1
    fi
done

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "All evaluations complete! Results in eval/results/"
    echo "Run: python3 summarize.py"
else
    echo ""
    echo "Some evaluations failed. Check logs/ for details."
fi
