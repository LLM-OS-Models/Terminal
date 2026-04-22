#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p logs results

source /home/work/.projects/Terminal/.eval-env/bin/activate

echo "=== Jiunsong GGUF models: $(date) ==="

PIDS=()
LABELS=()

run_model() {
    local gpu=$1
    local model=$2
    local logfile="logs/jiunsong_$(echo "$model" | tr '/' '_' | tr '[:upper:]' '[:lower:]').log"
    echo "[GPU $gpu] $model"
    CUDA_VISIBLE_DEVICES=$gpu python3 vllm_eval.py \
        --model "$model" --gpu 0 --output-dir results \
        > "$logfile" 2>&1
}

run_model 0 "Jiunsong/supergemma4-26b-uncensored-gguf-v2" &
PIDS+=($!); LABELS+=("GPU0 uncensored-gguf-v2")
run_model 1 "Jiunsong/supergemma4-26b-abliterated-multimodal-gguf-8bit" &
PIDS+=($!); LABELS+=("GPU1 multi-gguf-8bit")
run_model 2 "Jiunsong/SuperGemma4-31b-abliterated-GGUF" &
PIDS+=($!); LABELS+=("GPU2 31b-abliterated-GGUF")

echo "PIDs: ${PIDS[*]}"
echo "Waiting..."

for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[OK] ${LABELS[$i]}"
    else
        echo "[FAIL] ${LABELS[$i]}"
    fi
done

echo "=== Done at $(date) ==="
