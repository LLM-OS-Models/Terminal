#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${TB2_RUN_ID:-corrected_readme_models_vllm}"
EVAL_PATH="${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
OUT_DIR="${OUT_DIR:-/home/work/.data/tb2_lite_eval/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
REPORT_PATH="${REPORT_PATH:-README.md}"
VLLM_PY="${VLLM_PY:-$ROOT_DIR/.vllm-0_19_1/bin/python}"
VLLM_SITE="${VLLM_SITE:-$ROOT_DIR/.vllm-0_19_1/lib/python3.12/site-packages}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
GPU_COUNT="${GPU_COUNT:-8}"
HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
POLL_SECONDS="${POLL_SECONDS:-20}"

read_env_value() {
  local key="$1"
  local line
  line="$(grep -E "^(export[[:space:]]+)?${key}=" "$ROOT_DIR/.env" 2>/dev/null | tail -n 1 || true)"
  line="${line#export }"
  printf '%s' "${line#*=}" | sed -E "s/^['\"]//; s/['\"]$//"
}

if [[ -f "$ROOT_DIR/.env" ]]; then
  hf_token_value="$(read_env_value HF_TOKEN)"
  hub_token_value="$(read_env_value HUGGINGFACE_HUB_TOKEN)"
  if [[ -n "$hf_token_value" ]]; then
    export HF_TOKEN="$hf_token_value"
  fi
  if [[ -n "$hub_token_value" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$hub_token_value"
  elif [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  fi
fi

export HF_HOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VLLM_SITE${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_DIR/runner.log"
}

summarize() {
  "$ROOT_DIR/.liquid-sft-env/bin/python" tb2_lite/scripts/summarize_corrected_results.py \
    --results-dir "$OUT_DIR" \
    --output-path "$OUT_DIR/SUMMARY.md" \
    --title "README 모델 corrected TB2-lite vLLM 재평가 (${RUN_ID})" >/dev/null
  cp "$OUT_DIR/SUMMARY.md" "$REPORT_PATH"
  log "summary updated: $OUT_DIR/SUMMARY.md and $REPORT_PATH"
}

copy_existing_results() {
  local src
  for src in \
    "/home/work/.data/tb2_lite_eval/20260507T_lfm_family_corrected_after_24b_eval" \
    "/home/work/.data/tb2_lite_eval/20260508_corrected_readme1_lfm"
  do
    if [[ -d "$src" ]]; then
      cp -n "$src"/*.json "$OUT_DIR"/ 2>/dev/null || true
      log "copied existing results from $src"
    fi
  done
}

declare -a JOBS=()

add_job() {
  local short="$1"
  local model="$2"
  local extra="${3:-}"
  JOBS+=("${short}|${model}|${extra}")
}

# 1. README 상위권과 빠른 LFM/Qwen/Nemotron 기준 모델.
add_job "gyung_lfm2_8b_terminal_sft_unsloth" "gyung/LFM2-8B-Terminal-SFT-Unsloth"
add_job "nemotron_terminal_8b" "nvidia/Nemotron-Terminal-8B"
add_job "qwen35_2b_base" "Qwen/Qwen3.5-2B"
add_job "qwen35_4b_base" "Qwen/Qwen3.5-4B"
add_job "qwen35_9b_base" "Qwen/Qwen3.5-9B"
add_job "liquid_lfm25_1p2b_instruct_base" "LiquidAI/LFM2.5-1.2B-Instruct"
add_job "liquid_lfm2_2p6b_base" "LiquidAI/LFM2-2.6B"
add_job "liquid_lfm2_8b_a1b_base" "LiquidAI/LFM2-8B-A1B"

# 2. README 상위권 학습 소형/중형 모델.
add_job "qwen35_2b_sft_samecount_e2" "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/checkpoint-110-fixed"
add_job "qwen35_2b_sft_samecount_e1" "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/checkpoint-55-vllmfix3"
add_job "qwen35_4b_sft_2bdata_e1" "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata/checkpoint-960-vllmfix4"
add_job "qwen35_4b_sft_2bdata_e2" "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata/checkpoint-1920-fixed"
add_job "qwen35_9b_sft_2bdata_e2" "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata/checkpoint-4386"
add_job "qwen35_9b_sft_2bdata_e1" "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata/checkpoint-2193-fixed"
add_job "lfm25_1p2b_sft_unsloth_e2" "LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth"
add_job "lfm2_8b_sft_unsloth_e2" "LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth"
add_job "lfm2_2p6b_sft_unsloth_e2" "LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth"
add_job "qwen35_2b_sft_unsloth_e2" "LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth"

# 3. README 대형 base 모델.
add_job "nemotron_terminal_14b" "nvidia/Nemotron-Terminal-14B"
add_job "nemotron_terminal_32b" "nvidia/Nemotron-Terminal-32B"
add_job "qwen35_27b_base" "Qwen/Qwen3.5-27B"
add_job "qwen36_27b_base" "Qwen/Qwen3.6-27B"
add_job "qwen36_35b_a3b_fp8_base" "Qwen/Qwen3.6-35B-A3B-FP8"
add_job "gemma4_26b_a4b_it_base" "google/gemma-4-26B-A4B-it"
add_job "gemma4_31b_it_base" "google/gemma-4-31B-it"
add_job "liquid_lfm2_24b_a2b_base" "LiquidAI/LFM2-24B-A2B"
add_job "jackrong_qwen35_27b_claude_opus_distill" "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"

# 4. README 대형 SFT 모델.
add_job "qwen36_27b_sft_hf_fsdp_e2" "LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData"
add_job "qwen36_27b_sft_hf_fsdp_e1" "LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData"
add_job "qwen35_27b_sft_hf_fsdp_e2" "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-3834"
add_job "qwen35_27b_sft_hf_fsdp_e1" "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-1917"
add_job "qwen35_35b_a3b_sft_hf_fsdp_e2" "LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData"
add_job "qwen35_35b_a3b_sft_hf_fsdp_e1" "LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData"
add_job "qwen36_35b_a3b_sft_hf_fsdp_e2" "/home/work/.data/qwen_sft/models/Qwen__Qwen3.6-35B-A3B__terminal_sft_2epoch_hf_fsdp/checkpoint-5868"
add_job "qwen36_35b_a3b_sft_hf_fsdp_e1" "/home/work/.data/qwen_sft/models/Qwen__Qwen3.6-35B-A3B__terminal_sft_2epoch_hf_fsdp/checkpoint-2934"
add_job "gemma4_26b_a4b_sft_hf_fsdp_e2" "/home/work/.data/qwen_sft/models/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-1468"
add_job "gemma4_26b_a4b_sft_hf_fsdp_e1" "/home/work/.data/qwen_sft/models/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-734"
add_job "lfm2_24b_a2b_sft_hf_fsdp_e2" "/home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-1468"
add_job "lfm2_24b_a2b_sft_hf_fsdp_e1" "/home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-734"

# 5. README 하위권/형식 점검 대상. 위 큐가 끝나는 대로 남는 GPU에 자동 투입된다.
add_job "gemma4_e4b_it_base" "google/gemma-4-E4B-it"
add_job "gemma4_e2b_it_base" "google/gemma-4-E2B-it"
add_job "gemma4_e2b_sft_ddp_e2" "/home/work/.data/qwen_sft/models/google__gemma-4-E2B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-734"
add_job "gemma4_e2b_sft_ddp_e1" "/home/work/.data/qwen_sft/models/google__gemma-4-E2B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-367"
add_job "gemma4_e2b_sft_hf_fsdp_e2" "/home/work/.data/qwen_sft/models/google__gemma-4-E2B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-5868"
add_job "gemma4_e2b_sft_hf_fsdp_e1" "/home/work/.data/qwen_sft/models/google__gemma-4-E2B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-2934"
add_job "gemma4_e4b_sft_ddp_e2" "/home/work/.data/qwen_sft/models/google__gemma-4-E4B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-2934"
add_job "gemma4_e4b_sft_ddp_e1" "/home/work/.data/qwen_sft/models/google__gemma-4-E4B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-1467"
add_job "gemma4_31b_sft_hf_fsdp_e2" "/home/work/.data/qwen_sft/models/google__gemma-4-31B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-2934"
add_job "gemma4_31b_sft_hf_fsdp_e1" "/home/work/.data/qwen_sft/models/google__gemma-4-31B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-1467"
add_job "ouro_2p6b_terminal_sft" "LLM-OS-Models/Ouro-2.6B-Terminal-SFT"
add_job "ouro_1p4b_terminal_sft" "LLM-OS-Models/Ouro-1.4B-Terminal-SFT"
add_job "bytedance_ouro_1p4b_base" "ByteDance/Ouro-1.4B"
add_job "bytedance_ouro_2p6b_thinking" "ByteDance/Ouro-2.6B-Thinking"
add_job "bytedance_ouro_2p6b_base" "ByteDance/Ouro-2.6B"
add_job "bytedance_ouro_1p4b_thinking" "ByteDance/Ouro-1.4B-Thinking"
add_job "ouro_1p4b_thinking_terminal_sft" "LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT"
add_job "ouro_2p6b_thinking_terminal_sft" "LLM-OS-Models/Ouro-2.6B-Thinking-Terminal-SFT"

launch_job() {
  local gpu="$1"
  local job="$2"
  local short model extra output_json log_path
  IFS='|' read -r short model extra <<< "$job"
  output_json="$OUT_DIR/${short}.json"
  log_path="$LOG_DIR/${short}.log"
  if [[ -s "$output_json" ]]; then
    log "skip existing gpu=$gpu short=$short"
    return 1
  fi
  log "launch gpu=$gpu short=$short model=$model"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$VLLM_PY" tb2_lite/scripts/replay_eval.py \
      --model "$model" \
      --model-short "$short" \
      --gpu "$gpu" \
      --eval-path "$EVAL_PATH" \
      --output-dir "$OUT_DIR" \
      --dtype bfloat16 \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-tokens "$MAX_TOKENS" \
      --temperature 0.0 \
      --top-p 1.0 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --language-model-only \
      --skip-if-exists \
      $extra
  ) >"$log_path" 2>&1 &
  PIDS[$gpu]="$!"
  SHORTS[$gpu]="$short"
  LOGS[$gpu]="$log_path"
  return 0
}

copy_existing_results
summarize

log "queue start jobs=${#JOBS[@]} gpu_count=$GPU_COUNT out_dir=$OUT_DIR"

declare -a PIDS=()
declare -a SHORTS=()
declare -a LOGS=()
next_idx=0
active=0
failed=0
completed=0

while (( next_idx < ${#JOBS[@]} || active > 0 )); do
  changed=0

  for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
    pid="${PIDS[$gpu]:-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        log "done gpu=$gpu short=${SHORTS[$gpu]}"
      else
        status="$?"
        failed=$((failed + 1))
        log "failed status=$status gpu=$gpu short=${SHORTS[$gpu]} log=${LOGS[$gpu]}"
      fi
      PIDS[$gpu]=""
      SHORTS[$gpu]=""
      LOGS[$gpu]=""
      active=$((active - 1))
      completed=$((completed + 1))
      changed=1
    fi
  done

  for ((gpu = 0; gpu < GPU_COUNT && next_idx < ${#JOBS[@]}; gpu++)); do
    while [[ -z "${PIDS[$gpu]:-}" && next_idx < ${#JOBS[@]} ]]; do
      if launch_job "$gpu" "${JOBS[$next_idx]}"; then
        active=$((active + 1))
      fi
      next_idx=$((next_idx + 1))
      changed=1
    done
  done

  if (( changed )); then
    summarize
    log "queue progress completed=$completed active=$active next=$next_idx total=${#JOBS[@]} failed=$failed"
  fi

  if (( active > 0 )); then
    sleep "$POLL_SECONDS"
  fi
done

summarize
log "queue finished completed=$completed failed=$failed out_dir=$OUT_DIR"
exit 0
