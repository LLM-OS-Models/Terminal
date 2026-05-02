#!/bin/bash
# run_all_remaining_evals.sh — TB2-lite remaining evals (GPU pool management)
# Uses all 8 H200 GPUs, prioritizes fast evals, skips completed ones.
set -uo pipefail

cd /home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/scripts

EVAL_DATA="/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/data/replay_full.jsonl"
OUTPUT_DIR="/home/work/.data/tb2_lite_eval/20260502T012745Z"
MODELS="/home/work/.data/qwen_sft/models"
LOG_DIR="/tmp/tb2_eval_logs"
mkdir -p "$LOG_DIR"

V019="../../.vllm-0_19_1/bin/python"
V019_PP="../../.vllm-0_19_1/lib/python3.12/site-packages"

echo "[$(date +%T)] === TB2-lite Remaining Evals Started ==="
echo "[$(date +%T)] Output: $OUTPUT_DIR"

# ============================================
# Skip already-completed evals
# ============================================
skip_if_done() {
    local name="$1"
    if [ -f "$OUTPUT_DIR/${name}.json" ]; then
        echo "SKIP:$name"
        return 0
    fi
    echo "TODO:$name"
    return 1
}

# ============================================
# Job queue: fastest first
# Format: TYPE|MODEL_PATH|NAME|ESTIMATED_SECONDS
# ============================================
declare -a JOBS=()
declare -a JOBS_DONE=()

# --- Already completed in this batch (skip) ---
ALREADY_DONE=(
    gemma4_26b_a4b_e1_chat gemma4_26b_a4b_e2_chat
    gemma4_26b_a4b_e1_rp105 gemma4_26b_a4b_e2_rp105
    gemma4_31b_e1_chat gemma4_31b_e2_chat gemma4_31b_e2_vllm020_cu129
    gemma4_e2b_e1_hf_fsdp gemma4_e2b_e2_hf_fsdp
    gemma4_e2b_e1_rp105 gemma4_e2b_e2_rp105
    lfm2_24b_a2b_e1_chat lfm2_24b_a2b_e2_chat
    lfm2_24b_a2b_e1_rp105 lfm2_24b_a2b_e2_rp105
    gemma4_31b_e1 gemma4_31b_e2  # vLLM garbage results
    lfm2_24b_a2b_e1 lfm2_24b_a2b_e2
    gemma4_26b_a4b_e1 gemma4_26b_a4b_e2
)

# --- Running right now (PID-managed by previous session, skip re-launch) ---
RUNNING_NOW=(
    gemma4_31b_e2_transformers  # GPU 0
    gemma4_31b_e1_transformers  # GPU 1
    gemma4_e4b_e1_hf_fsdp       # GPU 2
    gemma4_e4b_e2_hf_fsdp       # GPU 3
)

is_already_done() {
    local name="$1"
    for d in "${ALREADY_DONE[@]}"; do
        [ "$d" = "$name" ] && return 0
    done
    for d in "${RUNNING_NOW[@]}"; do
        [ "$d" = "$name" ] && return 0
    done
    # Also check file
    [ -f "$OUTPUT_DIR/${name}.json" ] && return 0
    return 1
}

# Build job queue — ordered fast→slow
# Qwen3.5-2B multimodal (~30s)
if ! is_already_done "qwen3_5_2b_fullft_samecount"; then
    JOBS+=("TF_MM|$MODELS/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/checkpoint-110|qwen3_5_2b_fullft_samecount|30")
fi

# Qwen3.5-4B multimodal (~60s)
if ! is_already_done "qwen3_5_4b_fullft_2bdata"; then
    JOBS+=("TF_MM|$MODELS/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata/checkpoint-1920|qwen3_5_4b_fullft_2bdata|60")
fi

# Gemma4-E2B DDP e1 (~30s)
if ! is_already_done "gemma4_e2b_e1_ddp4gpu"; then
    JOBS+=("TF|$MODELS/google__gemma-4-E2B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-367|gemma4_e2b_e1_ddp4gpu|30")
fi

# Gemma4-E2B DDP e2 (~30s)
if ! is_already_done "gemma4_e2b_e2_ddp4gpu"; then
    JOBS+=("TF|$MODELS/google__gemma-4-E2B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-734|gemma4_e2b_e2_ddp4gpu|30")
fi

# Qwen3.5-9B multimodal (~180s)
if ! is_already_done "qwen3_5_9b_fullft_2bdata"; then
    JOBS+=("TF_MM|$MODELS/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata/checkpoint-2193|qwen3_5_9b_fullft_2bdata|180")
fi

# Qwen3.6-35B-A3B MoE (~600s with vLLM, ~1200s with transformers)
if ! is_already_done "qwen3_6_35b_a3b_hf_fsdp"; then
    JOBS+=("TF|$MODELS/Qwen__Qwen3.6-35B-A3B__terminal_sft_2epoch_hf_fsdp/checkpoint-2934|qwen3_6_35b_a3b_hf_fsdp|1200")
fi

# Gemma4-E4B DDP e1 (~2400s)
if ! is_already_done "gemma4_e4b_e1_ddp4gpu"; then
    JOBS+=("TF|$MODELS/google__gemma-4-E4B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-1467|gemma4_e4b_e1_ddp4gpu|2400")
fi

# Gemma4-E4B DDP e2 (~2400s)
if ! is_already_done "gemma4_e4b_e2_ddp4gpu"; then
    JOBS+=("TF|$MODELS/google__gemma-4-E4B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-2934|gemma4_e4b_e2_ddp4gpu|2400")
fi

# Qwen3.5-27B (~1800s with transformers)
if ! is_already_done "qwen3_5_27b_hf_fsdp"; then
    JOBS+=("TF|$MODELS/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-1917|qwen3_5_27b_hf_fsdp|1800")
fi

TOTAL_JOBS=${#JOBS[@]}
if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "[$(date +%T)] No remaining jobs. All done!"
    exit 0
fi

echo "[$(date +%T)] Job queue: $TOTAL_JOBS jobs"
for i in "${!JOBS[@]}"; do
    IFS='|' read -r TYPE MODEL NAME EST <<< "${JOBS[$i]}"
    echo "  [$((i+1))/$TOTAL_JOBS] $NAME (~${EST}s, $TYPE)"
done

# ============================================
# GPU pool management
# ============================================
NUM_GPUS=8
declare -A GPU_PID     # gpu_id -> pid
declare -A GPU_JOB     # gpu_id -> job_name
declare -A GPU_LOG     # gpu_id -> log_path

# Detect GPUs currently busy (check nvidia-smi memory usage > 5GB)
detect_busy_gpus() {
    declare -gA BUSY_GPUS
    for g in $(seq 0 $((NUM_GPUS-1))); do
        mem=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")
        mem=$(echo "$mem" | tr -d ' ')
        if [ "$mem" -gt 5000 ]; then
            BUSY_GPUS[$g]=1
        else
            BUSY_GPUS[$g]=0
        fi
    done
}

find_free_gpu() {
    detect_busy_gpus
    for g in $(seq 0 $((NUM_GPUS-1))); do
        # Not busy from external process AND not in our pool
        if [ "${BUSY_GPUS[$g]:-0}" -eq 0 ] && [ -z "${GPU_PID[$g]:-}" ]; then
            echo "$g"
            return
        fi
    done
    echo ""
}

cleanup_finished() {
    for g in "${!GPU_PID[@]}"; do
        pid="${GPU_PID[$g]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            name="${GPU_JOB[$g]}"
            log="${GPU_LOG[$g]}"
            echo "[$(date +%T)] GPU $g FINISHED: $name"
            # Check result
            if [ -f "$OUTPUT_DIR/${name}.json" ]; then
                score=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/${name}.json')); print(d.get('aggregate',{}).get('next_action_score','N/A'))" 2>/dev/null || echo "ERROR")
                echo "[$(date +%T)]   → score=$score"
                JOBS_DONE+=("$name")
            else
                echo "[$(date +%T)]   → NO RESULT FILE (check $log)"
            fi
            unset GPU_PID[$g]
            unset GPU_JOB[$g]
            unset GPU_LOG[$g]
        fi
    done
}

wait_for_free_gpu() {
    while true; do
        cleanup_finished
        gpu=$(find_free_gpu)
        if [ -n "$gpu" ]; then
            echo "$gpu"
            return
        fi
        sleep 10
    done
}

wait_all_done() {
    while [ ${#GPU_PID[@]} -gt 0 ]; do
        cleanup_finished
        sleep 10
    done
}

# ============================================
# Launch a job on a specific GPU
# ============================================
launch_job() {
    local gpu="$1"
    local type="$2"
    local model="$3"
    local name="$4"
    local log="$LOG_DIR/${name}.log"

    echo "[$(date +%T)] GPU $gpu: STARTING $name ($type)" | tee -a "$LOG_DIR/master.log"

    case "$type" in
        TF)
            CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 \
                PYTHONPATH=$V019_PP $V019 replay_eval_transformers.py \
                --model "$model" --model-short "$name" \
                --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
                --dtype bfloat16 --max-model-len 8192 --max-tokens 1024 --temperature 0.0 \
                &>"$log" &
            ;;
        TF_MM)
            CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 \
                PYTHONPATH=$V019_PP $V019 replay_eval_universal.py \
                --model "$model" --model-short "$name" \
                --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
                --dtype bfloat16 --max-model-len 8192 --max-tokens 1024 --temperature 0.0 \
                &>"$log" &
            ;;
        *)
            echo "[$(date +%T)] Unknown job type: $type" | tee -a "$LOG_DIR/master.log"
            return 1
            ;;
    esac

    GPU_PID[$gpu]=$!
    GPU_JOB[$gpu]=$name
    GPU_LOG[$gpu]=$log
    echo "[$(date +%T)]   PID=${GPU_PID[$gpu]}, log=$log"
}

# ============================================
# Main execution loop
# ============================================
JOB_IDX=0
echo "" | tee -a "$LOG_DIR/master.log"
echo "[$(date +%T)] === Starting job dispatch ===" | tee -a "$LOG_DIR/master.log"

for entry in "${JOBS[@]}"; do
    JOB_IDX=$((JOB_IDX+1))
    IFS='|' read -r TYPE MODEL NAME EST <<< "$entry"

    gpu=$(wait_for_free_gpu)
    echo "[$(date +%T)] [$JOB_IDX/$TOTAL_JOBS] Assigning $NAME to GPU $gpu" | tee -a "$LOG_DIR/master.log"
    launch_job "$gpu" "$TYPE" "$MODEL" "$NAME"
done

echo "[$(date +%T)] All jobs dispatched. Waiting for completion..." | tee -a "$LOG_DIR/master.log"
wait_all_done

echo "" | tee -a "$LOG_DIR/master.log"
echo "[$(date +%T)] === ALL EVALS COMPLETE ===" | tee -a "$LOG_DIR/master.log"
echo "[$(date +%T)] Results: $OUTPUT_DIR" | tee -a "$LOG_DIR/master.log"

# Final summary
echo "" | tee -a "$LOG_DIR/master.log"
echo "=== Final Results ===" | tee -a "$LOG_DIR/master.log"
python3 -c "
import json, glob
results = []
for f in sorted(glob.glob('$OUTPUT_DIR/*.json')):
    name = f.split('/')[-1].replace('.json','')
    if name == 'summary': continue
    try:
        d = json.load(open(f))
        agg = d.get('aggregate', {})
        score = agg.get('next_action_score', 'N/A')
        f1 = agg.get('avg_command_f1', 'N/A')
        exact = agg.get('first_cmd_exact_pct', 'N/A')
        vjson = agg.get('valid_json_pct', 'N/A')
        backend = d.get('backend', '?')
        results.append((score, name, f1, exact, vjson, backend))
    except Exception as e:
        results.append((0, name, '?', '?', '?', f'ERR:{e}'))

results.sort(key=lambda x: float(x[0]) if isinstance(x[0], (int, float)) else 0, reverse=True)
for i, (score, name, f1, exact, vjson, backend) in enumerate(results, 1):
    print(f'{i:3d}. [{score:>6}] {name:45s} F1={f1}  Exact={exact}%  ValidJSON={vjson}%  [{backend}]')
" | tee -a "$LOG_DIR/master.log"
