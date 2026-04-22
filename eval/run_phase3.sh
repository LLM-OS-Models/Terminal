#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p logs results

source /home/work/.projects/Terminal/.eval-env/bin/activate

echo "=== Phase 3 (LFM): $(date) ==="

PIDS=()
LABELS=()

run_model() {
    local gpus=$1
    local model=$2
    local tp=${3:-1}
    local logfile="logs/phase3_$(echo "$model" | tr '/' '_' | tr '[:upper:]' '[:lower:]').log"
    echo "[GPU $gpus] $model (TP=$tp)"
    CUDA_VISIBLE_DEVICES=$gpus python3 vllm_eval.py \
        --model "$model" --gpu 0 --tp $tp --output-dir results \
        > "$logfile" 2>&1
}

# LFM models evaluation (Phase 1/2 완료 후 실행)
#   0: LFM2-24B-A2B (23.84B MoE)
#   1: LFM2-8B-A1B (8.34B MoE)
#   2: LFM2-2.6B (2.57B Dense)
#   3: LFM2.5-1.2B-Instruct (1.17B Dense, instruction-tuned)

run_model "0" "LiquidAI/LFM2-24B-A2B" 1 &
PIDS+=($!); LABELS+=("GPU0 LFM2-24B-A2B")
run_model "1" "LiquidAI/LFM2-8B-A1B" 1 &
PIDS+=($!); LABELS+=("GPU1 LFM2-8B-A1B")
run_model "2" "LiquidAI/LFM2-2.6B" 1 &
PIDS+=($!); LABELS+=("GPU2 LFM2-2.6B")
run_model "3" "LiquidAI/LFM2.5-1.2B-Instruct" 1 &
PIDS+=($!); LABELS+=("GPU3 LFM2.5-1.2B-Instruct")

echo "PIDs: ${PIDS[*]}"
echo "Waiting for Phase 3..."

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[OK] ${LABELS[$i]}"
    else
        echo "[FAIL] ${LABELS[$i]}"
        FAIL=$((FAIL+1))
    fi
done

echo "=== Phase 3 done at $(date) (failures: $FAIL / ${#PIDS[@]}) ==="
