#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CURRENT_8B_CONFIG="Liquid-CLI/configs/sft_h200_7gpu_processed.env"
SMALL_MODEL_CONFIGS=(
  "Liquid-CLI/configs/sft_h200_8gpu_lfm25_1p2b_processed.env"
  "Liquid-CLI/configs/sft_h200_8gpu_lfm2_2p6b_processed.env"
)
FSDP_24B_CONFIG="qwen_sft/configs/sft_lfm2_24b_a2b_hf_fsdp.env"
TB2_RUN_ID="${TB2_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_liquid_family_autopilot}"
TB2_RESULTS_DIR="/home/work/.data/tb2_lite_eval/${TB2_RUN_ID}"
LOG_DIR="/home/work/.data/liquid_cli_sft/logs"
POLL_SECONDS="${POLL_SECONDS:-30}"
QUIET_MEM_MB="${QUIET_MEM_MB:-5000}"
CURRENT_8B_PID="${CURRENT_8B_PID:-}"
RETRAIN_CURRENT_8B="${RETRAIN_CURRENT_8B:-0}"
SKIP_EXISTING_FINAL="${SKIP_EXISTING_FINAL:-0}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-49152}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-1024}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"

VLLM_PY="$ROOT_DIR/.vllm-0_19_1/bin/python"
VLLM_SITE="$ROOT_DIR/.vllm-0_19_1/lib/python3.12/site-packages"

mkdir -p "$TB2_RESULTS_DIR" "$LOG_DIR"

while (($#)); do
  case "$1" in
    --current-8b-pid)
      CURRENT_8B_PID="$2"
      shift 2
      ;;
    --tb2-run-id)
      TB2_RUN_ID="$2"
      TB2_RESULTS_DIR="/home/work/.data/tb2_lite_eval/${TB2_RUN_ID}"
      mkdir -p "$TB2_RESULTS_DIR"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  echo "[$(timestamp_utc)] $*"
}

detect_current_8b_pid() {
  ps -eo pid,cmd | awk '/torchrun --standalone --nproc_per_node 7 Liquid-CLI\/train_unsloth_processed.py/ && !/awk/ {print $1; exit}'
}

wait_for_pid_exit() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  log "waiting for pid=$pid to exit"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  log "pid=$pid has exited"
}

gpu_is_quiet() {
  local gpu="$1"
  local mem
  mem="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  [[ -n "$mem" && "$mem" -lt "$QUIET_MEM_MB" ]]
}

wait_for_all_gpus() {
  local gpus=("$@")
  while true; do
    local all_ready=1
    for gpu in "${gpus[@]}"; do
      if ! gpu_is_quiet "$gpu"; then
        all_ready=0
        break
      fi
    done
    if [[ "$all_ready" == "1" ]]; then
      log "gpus ready: ${gpus[*]}"
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

source_config_value() {
  local config_path="$1"
  local key="$2"
  (
    source "$config_path"
    eval "printf '%s' \"\${$key}\""
  )
}

start_upload_watch() {
  local config_path="$1"
  local wait_pid="$2"
  local upload_subdir="$3"
  local run_name
  run_name="$(source_config_value "$config_path" RUN_NAME)"
  local log_path="$LOG_DIR/upload_watch_${run_name}_$(date -u +%Y%m%dT%H%M%SZ).log"
  nohup bash Liquid-CLI/scripts/watch_upload_after_train.sh \
    --config "$config_path" \
    --wait-pid "$wait_pid" \
    --upload-subdir "$upload_subdir" \
    --poll-seconds "$POLL_SECONDS" \
    >"$log_path" 2>&1 &
  echo "$!"
}

sorted_checkpoints() {
  local output_dir="$1"
  find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
    | sort -t- -k2,2n
}

safe_model_short_part() {
  local value="$1"
  value="${value//-/_}"
  value="${value//./p}"
  echo "$value"
}

run_tb2_eval() {
  local model_path="$1"
  local model_short="$2"
  local gpu="$3"
  local log_path="$LOG_DIR/tb2_${model_short}_$(date -u +%Y%m%dT%H%M%SZ).log"

  if [[ ! -d "$model_path" ]]; then
    log "skip eval missing model_path=$model_path"
    return 0
  fi

  log "tb2_lite eval start: model_short=$model_short gpu=$gpu model_path=$model_path"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 PYTHONPATH="$VLLM_SITE" "$VLLM_PY" \
    tb2_lite/scripts/replay_eval.py \
    --model "$model_path" \
    --model-short "$model_short" \
    --gpu "$gpu" \
    --eval-path tb2_lite/data/replay_full.jsonl \
    --output-dir "$TB2_RESULTS_DIR" \
    --dtype bfloat16 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-tokens "$VLLM_MAX_TOKENS" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --temperature 0.0 \
    >"$log_path" 2>&1
  log "tb2_lite eval done: model_short=$model_short"
}

evaluate_output_dir_checkpoints() {
  local config_path="$1"
  local eval_gpu="$2"
  local output_dir
  output_dir="$(source_config_value "$config_path" OUTPUT_DIR)"
  local run_name
  run_name="$(source_config_value "$config_path" RUN_NAME)"

  local checkpoint_name
  while IFS= read -r checkpoint_name; do
    [[ -n "$checkpoint_name" ]] || continue
    run_tb2_eval \
      "$output_dir/$checkpoint_name" \
      "${run_name}_$(safe_model_short_part "$checkpoint_name")" \
      "$eval_gpu"
  done < <(sorted_checkpoints "$output_dir")

  if [[ -e "$output_dir/final" ]]; then
    run_tb2_eval "$output_dir/final" "${run_name}_final" "$eval_gpu"
  fi
  summarize_tb2
}

summarize_tb2() {
  "$ROOT_DIR/.liquid-sft-env/bin/python" tb2_lite/scripts/summarize_replay_results.py \
    --results-dir "$TB2_RESULTS_DIR" \
    --output-path "$TB2_RESULTS_DIR/SUMMARY.md" \
    >/dev/null
}

run_liquid_cli_training() {
  local config_path="$1"
  local upload_subdir="$2"
  local run_name
  run_name="$(source_config_value "$config_path" RUN_NAME)"
  local output_dir
  output_dir="$(source_config_value "$config_path" OUTPUT_DIR)"
  local launcher_log="$LOG_DIR/launcher_${run_name}_$(date -u +%Y%m%dT%H%M%SZ).log"

  if [[ "$SKIP_EXISTING_FINAL" == "1" && -e "$output_dir/final" ]]; then
    log "training skip existing final: config=$config_path output_dir=$output_dir"
    return 0
  fi

  log "training start: config=$config_path"
  nohup bash Liquid-CLI/scripts/run_sft_7gpu_processed.sh --config "$config_path" >"$launcher_log" 2>&1 &
  local train_pid="$!"
  local watch_pid
  watch_pid="$(start_upload_watch "$config_path" "$train_pid" "$upload_subdir")"
  log "training pid=$train_pid upload_watch_pid=$watch_pid"

  local train_status=0
  if wait "$train_pid"; then
    train_status=0
  else
    train_status="$?"
  fi
  log "training exit status=$train_status config=$config_path"

  wait "$watch_pid" || true
  return "$train_status"
}

run_fsdp_training() {
  local config_path="$1"
  local upload_subdir="$2"
  local run_name
  run_name="$(source_config_value "$config_path" RUN_NAME)"
  local output_dir
  output_dir="$(source_config_value "$config_path" OUTPUT_DIR)"
  local launcher_log="$LOG_DIR/launcher_${run_name}_$(date -u +%Y%m%dT%H%M%SZ).log"

  if [[ "$SKIP_EXISTING_FINAL" == "1" && -e "$output_dir/final" ]]; then
    log "fsdp training skip existing final: config=$config_path output_dir=$output_dir"
    return 0
  fi

  log "fsdp training start: config=$config_path"
  nohup bash qwen_sft/scripts/run_hf_fsdp_8gpu.sh --config "$config_path" >"$launcher_log" 2>&1 &
  local train_pid="$!"
  local watch_pid
  watch_pid="$(start_upload_watch "$config_path" "$train_pid" "$upload_subdir")"
  log "fsdp training pid=$train_pid upload_watch_pid=$watch_pid"

  local train_status=0
  if wait "$train_pid"; then
    train_status=0
  else
    train_status="$?"
  fi
  log "fsdp training exit status=$train_status config=$config_path"

  wait "$watch_pid" || true
  return "$train_status"
}

evaluate_liquid_cli_outputs() {
  local config_path="$1"
  local eval_gpu="$2"
  evaluate_output_dir_checkpoints "$config_path" "$eval_gpu"
}

evaluate_fsdp_24b_outputs() {
  local config_path="$1"
  local eval_gpu="$2"
  evaluate_output_dir_checkpoints "$config_path" "$eval_gpu"
}

if [[ -z "$CURRENT_8B_PID" ]]; then
  CURRENT_8B_PID="$(detect_current_8b_pid || true)"
fi

if [[ -n "$CURRENT_8B_PID" ]]; then
  log "detected current 8b pid=$CURRENT_8B_PID"
  start_upload_watch "$CURRENT_8B_CONFIG" "$CURRENT_8B_PID" "final" >/dev/null
  wait_for_pid_exit "$CURRENT_8B_PID"
  wait_for_all_gpus 1
  evaluate_liquid_cli_outputs "$CURRENT_8B_CONFIG" 1
elif [[ "$RETRAIN_CURRENT_8B" == "1" ]]; then
  wait_for_all_gpus 1 2 3 4 5 6 7
  run_liquid_cli_training "$CURRENT_8B_CONFIG" "final"
  wait_for_all_gpus 1
  evaluate_liquid_cli_outputs "$CURRENT_8B_CONFIG" 1
fi

wait_for_all_gpus 0 1 2 3 4 5 6 7

for config_path in "${SMALL_MODEL_CONFIGS[@]}"; do
  run_liquid_cli_training "$config_path" "final"
  wait_for_all_gpus 1
  evaluate_liquid_cli_outputs "$config_path" 1
  wait_for_all_gpus 0 1 2 3 4 5 6 7
done

run_fsdp_training "$FSDP_24B_CONFIG" "final"
wait_for_all_gpus 1
evaluate_fsdp_24b_outputs "$FSDP_24B_CONFIG" 1

log "autopilot complete"
log "tb2_lite results dir=$TB2_RESULTS_DIR"
