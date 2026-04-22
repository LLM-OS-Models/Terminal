#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p logs results

source /home/work/.projects/Terminal/.eval-env/bin/activate

echo "=== Phase 2: $(date) ==="

PIDS=()
LABELS=()

run_model() {
    local gpus=$1
    local model=$2
    local tp=${3:-1}
    local logfile="logs/phase2_$(echo "$model" | tr '/' '_' | tr '[:upper:]' '[:lower:]').log"
    echo "[GPU $gpus] $model (TP=$tp)"
    CUDA_VISIBLE_DEVICES=$gpus python3 vllm_eval.py \
        --model "$model" --gpu 0 --tp $tp --output-dir results \
        > "$logfile" 2>&1
}

# GPU layout:
#   0,1: Nemotron-Terminal-32B (TP=2)
#   2,3: gemma-4-31B-it (TP=2)
#   4:   Qwen3.6-35B-A3B-FP8 (single)
#   5:   supergemma4-26b-abliterated (single)
#   6:   Qwen3.6-27B (single)
#   7:   Qwen3.5-27B-Distilled (single)

run_model "0,1" "nvidia/Nemotron-Terminal-32B" 2 &
PIDS+=($!); LABELS+=("GPU0,1 Nemotron-32B")
run_model "2,3" "google/gemma-4-31B-it" 2 &
PIDS+=($!); LABELS+=("GPU2,3 gemma-31B")
run_model "4" "Qwen/Qwen3.6-35B-A3B-FP8" 1 &
PIDS+=($!); LABELS+=("GPU4 Qwen3.6-35B")
run_model "5" "Jiunsong/supergemma4-26b-abliterated-multimodal" 1 &
PIDS+=($!); LABELS+=("GPU5 supergemma4-26b-abliterated")
run_model "6" "Qwen/Qwen3.6-27B" 1 &
PIDS+=($!); LABELS+=("GPU6 Qwen3.6-27B")
run_model "7" "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled" 1 &
PIDS+=($!); LABELS+=("GPU7 Qwen3.5-27B-Distilled")

echo "PIDs: ${PIDS[*]}"
echo "Waiting for Phase 2..."

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[OK] ${LABELS[$i]}"
    else
        echo "[FAIL] ${LABELS[$i]}"
        FAIL=$((FAIL+1))
    fi
done

echo "=== Phase 2 done at $(date) (failures: $FAIL / ${#PIDS[@]}) ==="
