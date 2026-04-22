#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p logs results gguf_cache

source /home/work/.projects/Terminal/.eval-env/bin/activate

echo "=== Jiunsong GGUF download+eval: $(date) ==="

# Download GGUF files from HuggingFace
echo "[1/2] Downloading GGUF files..."
huggingface-cli download Jiunsong/supergemma4-26b-uncensored-gguf-v2 supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf --local-dir gguf_cache --local-dir-use-symlinks False 2>&1 | tail -3 &
DL1=$!

huggingface-cli download Jiunsong/SuperGemma4-31b-abliterated-GGUF SuperGemma4-31b-abliterated.Q4_K_M.gguf --local-dir gguf_cache --local-dir-use-symlinks False 2>&1 | tail -3 &
DL2=$!

wait $DL1 && echo "DL1 done" || echo "DL1 failed"
wait $DL2 && echo "DL2 done" || echo "DL2 failed"

echo "[2/2] Running evaluations..."

PIDS=()
LABELS=()

run_model() {
    local gpu=$1
    local model_path=$2
    local logfile="logs/jiunsong_$(basename "$model_path" .gguf).log"
    echo "[GPU $gpu] $model_path"
    CUDA_VISIBLE_DEVICES=$gpu python3 vllm_eval.py \
        --model "$model_path" --gpu 0 --output-dir results \
        > "$logfile" 2>&1
}

run_model 0 "gguf_cache/supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf" &
PIDS+=($!); LABELS+=("GPU0 uncensored-Q4")
run_model 1 "gguf_cache/SuperGemma4-31b-abliterated.Q4_K_M.gguf" &
PIDS+=($!); LABELS+=("GPU1 31b-abliterated-Q4")

for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[OK] ${LABELS[$i]}"
    else
        echo "[FAIL] ${LABELS[$i]}"
    fi
done

echo "=== Done at $(date) ==="
