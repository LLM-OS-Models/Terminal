#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${TB2_RUN_ID:-20260508_corrected_top_models}"
EVAL_PATH="${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
OUT_DIR="${OUT_DIR:-/home/work/.data/tb2_lite_eval/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-/home/work/.data/tb2_lite_eval/${RUN_ID}/logs}"
REPORT_PATH="${REPORT_PATH:-MODEL_EVAL_RESCORING_REPORT_2026-05-08.md}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.vllm-0_19_1/bin/python}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
HF_HOME="${HF_HOME:-/home/work/.data/huggingface}"
export HF_HOME
export PYTHONNOUSERSITE=1

mkdir -p "$OUT_DIR" "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_DIR/runner.log"
}

copy_lfm_results() {
  local src="/home/work/.data/tb2_lite_eval/20260507T_lfm_family_corrected_after_24b_eval"
  if [[ -d "$src" ]]; then
    cp -n "$src"/*.json "$OUT_DIR"/ 2>/dev/null || true
    log "copied trained LFM checkpoint results from $src"
  else
    log "missing trained LFM result source: $src"
  fi
}

run_one() {
  local gpu="$1"
  local short="$2"
  local model="$3"
  local extra="${4:-}"
  local output_json="$OUT_DIR/${short}.json"
  local log_path="$LOG_DIR/${short}.log"

  log "launch gpu=$gpu short=$short model=$model"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" tb2_lite/scripts/replay_eval.py \
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
      $extra
  ) >"$log_path" 2>&1 &
}

wait_batch() {
  local batch_name="$1"
  local failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if [[ "$failed" -ne 0 ]]; then
    log "batch completed with failures: $batch_name"
  else
    log "batch completed: $batch_name"
  fi
  summarize
}

summarize() {
  "$ROOT_DIR/.liquid-sft-env/bin/python" tb2_lite/scripts/summarize_corrected_results.py \
    --results-dir "$OUT_DIR" \
    --output-path "$OUT_DIR/SUMMARY.md" \
    --title "Corrected TB2-lite 상위권 모델 재평가 (${RUN_ID})" >/dev/null
  cp "$OUT_DIR/SUMMARY.md" "$REPORT_PATH"
  log "summary updated: $OUT_DIR/SUMMARY.md and $REPORT_PATH"
}

run_batch() {
  local batch_name="$1"
  shift
  PIDS=()
  log "batch start: $batch_name"
  local gpu=0
  while (($#)); do
    local short="$1"
    local model="$2"
    local extra="${3:-}"
    shift 3
    if [[ -s "$OUT_DIR/${short}.json" ]]; then
      log "skip existing result: $short"
    else
      run_one "$gpu" "$short" "$model" "$extra"
      PIDS+=("$!")
      gpu=$((gpu + 1))
    fi
  done
  wait_batch "$batch_name"
}

copy_lfm_results
summarize

# 1차: 작은 모델과 기존 상위권을 먼저 돌려 빠르게 비교한다.
run_batch "wave1_small_and_top_baselines" \
  "gyung_lfm2_8b_terminal_sft_unsloth" "gyung/LFM2-8B-Terminal-SFT-Unsloth" "" \
  "nemotron_terminal_8b" "nvidia/Nemotron-Terminal-8B" "" \
  "qwen35_2b_base" "Qwen/Qwen3.5-2B" "" \
  "qwen35_4b_base" "Qwen/Qwen3.5-4B" "" \
  "qwen35_9b_base" "Qwen/Qwen3.5-9B" "" \
  "gemma4_e2b_it" "google/gemma-4-E2B-it" "" \
  "gemma4_e4b_it" "google/gemma-4-E4B-it" "" \
  "nemotron_terminal_14b" "nvidia/Nemotron-Terminal-14B" ""

# 2차: 중대형 기존 상위권/대형 기준 모델.
run_batch "wave2_large_baselines" \
  "nemotron_terminal_32b" "nvidia/Nemotron-Terminal-32B" "" \
  "qwen35_27b_base" "Qwen/Qwen3.5-27B" "" \
  "qwen36_27b_base" "Qwen/Qwen3.6-27B" "" \
  "qwen36_35b_a3b_fp8_base" "Qwen/Qwen3.6-35B-A3B-FP8" "" \
  "gemma4_26b_a4b_it" "google/gemma-4-26B-A4B-it" "" \
  "gemma4_31b_it" "google/gemma-4-31B-it" "" \
  "liquid_lfm2_24b_a2b_base" "LiquidAI/LFM2-24B-A2B" "" \
  "jackrong_qwen35_27b_claude_opus_distill" "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled" ""

# 3차: 기존 README 상위권 학습 Qwen 모델들. 필요하면 HF에서 받아서 평가한다.
run_batch "wave3_trained_qwen_small_mid" \
  "qwen35_2b_sft_samecount_e1" "LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount" "" \
  "qwen35_2b_sft_samecount_e2" "LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount" "" \
  "qwen35_4b_sft_2bdata_e1" "LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData" "" \
  "qwen35_4b_sft_2bdata_e2" "LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData" "" \
  "qwen35_9b_sft_2bdata_e1" "LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-1Epoch-FullFT-2BData" "" \
  "qwen35_9b_sft_2bdata_e2" "LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData" "" \
  "qwen36_27b_sft_hf_fsdp_e1" "LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData" "" \
  "qwen36_27b_sft_hf_fsdp_e2" "LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData" ""

# 4차: 기존에 문제 지적이 있었던 대형 학습 모델들. 새 corrected 프로토콜로 다시 본다.
run_batch "wave4_large_trained_models" \
  "qwen35_27b_sft_hf_fsdp_e1" "LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData" "" \
  "qwen35_27b_sft_hf_fsdp_e2" "LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData" "" \
  "qwen35_35b_a3b_sft_hf_fsdp_e1" "LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData" "" \
  "qwen35_35b_a3b_sft_hf_fsdp_e2" "LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData" "" \
  "qwen36_35b_a3b_sft_hf_fsdp_e1" "LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData" "" \
  "qwen36_35b_a3b_sft_hf_fsdp_e2" "LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData" "" \
  "gemma4_26b_a4b_sft_hf_fsdp_e1" "LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData" "" \
  "gemma4_26b_a4b_sft_hf_fsdp_e2" "LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData" ""

log "all evaluation waves completed"
