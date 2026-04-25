#!/bin/bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$EVAL_DIR"

source "$EVAL_DIR/env.sh"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-logs/$RUN_ID}"
RESULTS_DIR="${RESULTS_DIR:-results/$RUN_ID}"
RESERVED_GPU="${RESERVED_GPU:-7}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

PIDS=()
LABELS=()

safe_name() {
    echo "$1" | tr '/' '_' | tr '[:upper:]' '[:lower:]'
}

run_model() {
    local gpus=$1
    local model=$2
    local tp=$3
    shift 3
    local extra_args=("$@")
    local logfile="$LOG_DIR/$(safe_name "$model").log"

    echo "[GPU $gpus] $model (TP=$tp ${extra_args[*]})"
    CUDA_VISIBLE_DEVICES="$gpus" python3 "$EVAL_DIR/vllm_eval.py" \
        --model "$model" \
        --gpu 0 \
        --tp "$tp" \
        --output-dir "$RESULTS_DIR" \
        "${extra_args[@]}" \
        > "$logfile" 2>&1 &
    PIDS+=($!)
    LABELS+=("GPU $gpus :: $model")
}

wait_batch() {
    local batch_name=$1
    local fail=0
    echo "=== Waiting for $batch_name ==="
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "[OK] ${LABELS[$i]}"
        else
            echo "[FAIL] ${LABELS[$i]}"
            fail=$((fail + 1))
        fi
    done
    echo "=== $batch_name complete (failures: $fail / ${#PIDS[@]}) ==="
    PIDS=()
    LABELS=()
}

echo "=== NEXT eval (7 GPUs, reserving GPU $RESERVED_GPU): $(date -u) ==="
echo "Logs: $LOG_DIR"
echo "Results: $RESULTS_DIR"

# Batch 1: docs/NEXT_EVAL_PLAN.md phase 1 subset (7 GPUs in use, GPU 7 reserved)
run_model "0" "principled-intelligence/Qwen3.5-2B-text-only" 1
run_model "1" "principled-intelligence/gemma-4-E2B-it-text-only" 1
run_model "2" "principled-intelligence/Qwen3.5-4B-text-only" 1
run_model "3" "principled-intelligence/gemma-4-E4B-it-text-only" 1
run_model "4" "principled-intelligence/Qwen3.5-9B-text-only" 1
run_model "5" "nvidia/Nemotron-Terminal-8B" 1
run_model "6" "nvidia/Nemotron-Terminal-14B" 1
wait_batch "batch1-small"

# Batch 2: remaining single-GPU models from phase 1/2/3 + gyung LFM terminal SFT
run_model "0" "google/gemma-4-26B-A4B-it" 1
run_model "1" "Qwen/Qwen3.6-35B-A3B-FP8" 1 --gdn-triton
run_model "2" "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled" 1
run_model "3" "LiquidAI/LFM2-24B-A2B" 1
run_model "4" "LiquidAI/LFM2-8B-A1B" 1
run_model "5" "LiquidAI/LFM2-2.6B" 1
run_model "6" "LiquidAI/LFM2.5-1.2B-Instruct" 1
wait_batch "batch2-single-gpu"

# Batch 3: 2-GPU large models + LFM terminal SFT + one expected-failure verification
run_model "0,1" "nvidia/Nemotron-Terminal-32B" 2
run_model "2,3" "google/gemma-4-31B-it" 2
run_model "4" "gyung/LFM2-8B-Terminal-SFT-Unsloth" 1
run_model "5" "Jiunsong/supergemma4-26b-abliterated-multimodal" 1
wait_batch "batch3-large-and-lfm"

python3 "$EVAL_DIR/summarize.py" --results-dir "$RESULTS_DIR" | tee "$RESULTS_DIR/summary.txt"

echo "=== Eval finished: $(date -u) ==="
