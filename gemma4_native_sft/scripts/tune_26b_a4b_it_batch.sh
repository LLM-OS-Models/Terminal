#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="/home/work/.data/gemma4_native_sft/logs"
SUMMARY_DIR="/home/work/.data/gemma4_native_sft/tuning"
SUMMARY_PATH="$SUMMARY_DIR/gemma4_26b_a4b_it_batch_tuning_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$LOG_DIR" "$SUMMARY_DIR"
export BF16="${BF16:-0}"

configs=(
  gemma4_native_sft/configs/tune_gemma4_26b_a4b_it_native_4gpu_bsz16_acc1_actckpt.env
  gemma4_native_sft/configs/tune_gemma4_26b_a4b_it_native_4gpu_bsz14_acc1_actckpt.env
  gemma4_native_sft/configs/tune_gemma4_26b_a4b_it_native_4gpu_bsz12_acc1_actckpt.env
  gemma4_native_sft/configs/tune_gemma4_26b_a4b_it_native_4gpu_bsz10_acc1_actckpt.env
  gemma4_native_sft/configs/tune_gemma4_26b_a4b_it_native_4gpu_bsz8_acc1_actckpt.env
  gemma4_native_sft/configs/tune_gemma4_26b_a4b_it_native_4gpu_bsz6_acc1_actckpt.env
  gemma4_native_sft/configs/tune_gemma4_26b_a4b_it_native_4gpu_bsz4_acc2_actckpt.env
  gemma4_native_sft/configs/tune_gemma4_26b_a4b_it_native_4gpu_bsz3_acc3_actckpt.env
  gemma4_native_sft/configs/tune_gemma4_26b_a4b_it_native_4gpu_bsz2_acc4_actckpt.env
)

log() {
  echo "[$(TZ=Asia/Seoul date '+%F %T KST')] $*" | tee -a "$SUMMARY_PATH"
}

latest_log_for_run() {
  local run_name="$1"
  ls -1t "$LOG_DIR/${run_name}_"*.log 2>/dev/null | head -n 1 || true
}

summarize_run() {
  local config="$1"
  local status="$2"
  local run_name
  run_name="$(grep '^RUN_NAME=' "$config" | cut -d= -f2-)"
  local per_device grad_acc nproc effective latest
  per_device="$(grep '^PER_DEVICE_TRAIN_BATCH_SIZE=' "$config" | cut -d= -f2-)"
  grad_acc="$(grep '^GRADIENT_ACCUMULATION_STEPS=' "$config" | cut -d= -f2-)"
  nproc="$(grep '^NPROC_PER_NODE=' "$config" | cut -d= -f2-)"
  effective=$((per_device * grad_acc * nproc))
  latest="$(latest_log_for_run "$run_name")"
  log "RESULT status=$status run=$run_name per_device=$per_device grad_acc=$grad_acc nproc=$nproc effective_batch=$effective bf16=$BF16 log=$latest"
  if [[ -n "$latest" ]]; then
    if rg -i "out of memory|cuda.*oom|CUDA out of memory|cublas.*alloc|torch.OutOfMemoryError" "$latest" >/dev/null; then
      log "OOM_DETECTED run=$run_name"
    fi
    tail -n 18 "$latest" | sed 's/^/[tail] /' | tee -a "$SUMMARY_PATH"
  fi
}

log "starting Gemma 4 26B-A4B-it 4GPU batch tuning"
for config in "${configs[@]}"; do
  run_name="$(grep '^RUN_NAME=' "$config" | cut -d= -f2-)"
  log "START config=$config run=$run_name"
  rm -rf "$(grep '^OUTPUT_DIR=' "$config" | cut -d= -f2-)"
  set +e
  bash gemma4_native_sft/scripts/run_native_hf_fsdp.sh --config "$config"
  code=$?
  set -e
  summarize_run "$config" "exit_$code"
  if [[ "$code" == "0" ]]; then
    log "SUCCESS_FIRST_STABLE_CONFIG=$config"
    break
  fi
  log "continuing after failed config=$config"
  sleep 20
done
log "batch tuning complete summary=$SUMMARY_PATH"
