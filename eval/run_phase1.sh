#!/bin/bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EVAL_DIR"

source "$EVAL_DIR/env.sh"

LOG_DIR="${LOG_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "=== Phase 1: $(date) ==="

PIDS=()
LABELS=()

run_model() {
    local gpu=$1
    local model=$2
    local logfile="$LOG_DIR/phase1_$(echo "$model" | tr '/' '_' | tr '[:upper:]' '[:lower:]').log"
    echo "[GPU $gpu] $model"
    CUDA_VISIBLE_DEVICES=$gpu python3 vllm_eval.py \
        --model "$model" --gpu 0 --output-dir "$RESULTS_DIR" \
        > "$logfile" 2>&1
}

# Phase 1: docs/NEXT_EVAL_PLAN.md text-only models + Nemotron + gemma-26B
run_model 0 "principled-intelligence/Qwen3.5-2B-text-only" &
PIDS+=($!); LABELS+=("GPU0 Qwen3.5-2B-text-only")
run_model 1 "principled-intelligence/gemma-4-E2B-it-text-only" &
PIDS+=($!); LABELS+=("GPU1 gemma-E2B-text-only")
run_model 2 "principled-intelligence/Qwen3.5-4B-text-only" &
PIDS+=($!); LABELS+=("GPU2 Qwen3.5-4B-text-only")
run_model 3 "principled-intelligence/gemma-4-E4B-it-text-only" &
PIDS+=($!); LABELS+=("GPU3 gemma-E4B-text-only")
run_model 4 "principled-intelligence/Qwen3.5-9B-text-only" &
PIDS+=($!); LABELS+=("GPU4 Qwen3.5-9B-text-only")
run_model 5 "nvidia/Nemotron-Terminal-8B" &
PIDS+=($!); LABELS+=("GPU5 Nemotron-8B")
run_model 6 "nvidia/Nemotron-Terminal-14B" &
PIDS+=($!); LABELS+=("GPU6 Nemotron-14B")
run_model 7 "google/gemma-4-26B-A4B-it" &
PIDS+=($!); LABELS+=("GPU7 gemma-26B-MoE")

echo "PIDs: ${PIDS[*]}"
echo "Waiting for Phase 1..."

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[OK] ${LABELS[$i]}"
    else
        echo "[FAIL] ${LABELS[$i]}"
        FAIL=$((FAIL+1))
    fi
done

echo "=== Phase 1 done at $(date) (failures: $FAIL / ${#PIDS[@]}) ==="
