#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TB2_RUN_ID="${TB2_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_lfm_corrected_full_eval}"
RESULTS_DIR="/home/work/.data/tb2_lite_eval/${TB2_RUN_ID}"
LOG_DIR="/home/work/.data/liquid_cli_sft/logs"
EVAL_PATH="${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-49152}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-1024}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
VLLM_PY="${VLLM_PY:-$ROOT_DIR/.vllm-0_19_1/bin/python}"
VLLM_SITE="${VLLM_SITE:-$ROOT_DIR/.vllm-0_19_1/lib/python3.12/site-packages}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

CONFIGS=(
  "Liquid-CLI/configs/sft_h200_7gpu_processed.env"
  "Liquid-CLI/configs/sft_h200_8gpu_lfm25_1p2b_processed.env"
  "Liquid-CLI/configs/sft_h200_8gpu_lfm2_2p6b_processed.env"
  "qwen_sft/configs/sft_lfm2_24b_a2b_hf_fsdp.env"
)

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  echo "[$(timestamp_utc)] $*"
}

source_config_value() {
  local config_path="$1"
  local key="$2"
  (
    source "$config_path"
    eval "printf '%s' \"\${$key}\""
  )
}

safe_name() {
  local value="$1"
  value="${value//\//__}"
  value="${value//-/_}"
  value="${value//./p}"
  echo "$value"
}

first_and_last_checkpoints() {
  local output_dir="$1"
  mapfile -t checkpoints < <(
    find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
      | sort -t- -k2,2n
  )
  if ((${#checkpoints[@]} == 0)); then
    return 1
  fi
  local last_idx=$(( ${#checkpoints[@]} - 1 ))
  echo "${checkpoints[0]}"
  if [[ "${checkpoints[$last_idx]}" != "${checkpoints[0]}" ]]; then
    echo "${checkpoints[$last_idx]}"
  fi
}

declare -a MODEL_PATHS=()
declare -a MODEL_SHORTS=()

for config in "${CONFIGS[@]}"; do
  output_dir="$(source_config_value "$config" OUTPUT_DIR)"
  run_name="$(source_config_value "$config" RUN_NAME)"
  if [[ ! -d "$output_dir" ]]; then
    log "missing output_dir for config=$config output_dir=$output_dir"
    exit 1
  fi
  mapfile -t selected < <(first_and_last_checkpoints "$output_dir")
  if ((${#selected[@]} == 0)); then
    log "no checkpoints found in output_dir=$output_dir"
    exit 1
  fi
  for checkpoint_name in "${selected[@]}"; do
    MODEL_PATHS+=("$output_dir/$checkpoint_name")
    MODEL_SHORTS+=("$(safe_name "${run_name}_${checkpoint_name}_lfm_eval")")
  done
done

if ((${#MODEL_PATHS[@]} != 8)); then
  log "expected 8 eval jobs, got ${#MODEL_PATHS[@]} jobs"
  printf '%s\n' "${MODEL_PATHS[@]}"
  exit 1
fi

log "starting 8-way vLLM LFM eval run_id=$TB2_RUN_ID eval_path=$EVAL_PATH"
declare -a PIDS=()
declare -a JOB_LOGS=()

for idx in "${!MODEL_PATHS[@]}"; do
  gpu="$idx"
  model_path="${MODEL_PATHS[$idx]}"
  model_short="${MODEL_SHORTS[$idx]}"
  log_path="$LOG_DIR/tb2_lfm_eval_${model_short}_$(date -u +%Y%m%dT%H%M%SZ).log"
  JOB_LOGS+=("$log_path")
  extra_args=()
  if [[ "$SKIP_IF_EXISTS" == "1" ]]; then
    extra_args+=(--skip-if-exists)
  fi
  log "eval launch gpu=$gpu model_short=$model_short model_path=$model_path"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_SITE" "$VLLM_PY" \
      tb2_lite/scripts/replay_eval_lfm_vllm.py \
      --model "$model_path" \
      --model-short "$model_short" \
      --gpu "$gpu" \
      --eval-path "$EVAL_PATH" \
      --output-dir "$RESULTS_DIR" \
      --dtype bfloat16 \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --max-tokens "$VLLM_MAX_TOKENS" \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --temperature 0.0 \
      "${extra_args[@]}"
  ) >"$log_path" 2>&1 &
  PIDS+=("$!")
done

status=0
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then
    log "eval done model_short=${MODEL_SHORTS[$idx]}"
  else
    job_status="$?"
    log "eval failed status=$job_status model_short=${MODEL_SHORTS[$idx]} log=${JOB_LOGS[$idx]}"
    status=1
  fi
done

"$ROOT_DIR/.liquid-sft-env/bin/python" tb2_lite/scripts/summarize_replay_results.py \
  --results-dir "$RESULTS_DIR" \
  --output-path "$RESULTS_DIR/SUMMARY.md" \
  >/dev/null

log "summary written: $RESULTS_DIR/SUMMARY.md"
exit "$status"
