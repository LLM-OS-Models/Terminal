#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WAIT_PID="${WAIT_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-120}"
STATUS_SECONDS="${STATUS_SECONDS:-7200}"
CHAIN_RUN_ID="${CHAIN_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_lfm_2p6b_24b_chain}"
LOG_DIR="/home/work/.data/liquid_cli_sft/logs"
CHAIN_LOG="$LOG_DIR/chain_${CHAIN_RUN_ID}.log"
FSDP_24B_CONFIG="${FSDP_24B_CONFIG:-qwen_sft/configs/sft_lfm2_24b_a2b_hf_fsdp.env}"
EVAL_RUN_ID="${EVAL_RUN_ID:-${CHAIN_RUN_ID}_eval}"

mkdir -p "$LOG_DIR"

while (($#)); do
  case "$1" in
    --wait-pid)
      WAIT_PID="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --status-seconds)
      STATUS_SECONDS="$2"
      shift 2
      ;;
    --eval-run-id)
      EVAL_RUN_ID="$2"
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
  echo "[$(timestamp_utc)] $*" | tee -a "$CHAIN_LOG"
}

source_config_value() {
  local config_path="$1"
  local key="$2"
  (
    source "$config_path"
    eval "printf '%s' \"\${$key}\""
  )
}

checkpoint_count() {
  local output_dir="$1"
  find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | wc -l
}

latest_train_tail() {
  local pattern="$1"
  local log_path
  log_path="$(ls -1t $pattern 2>/dev/null | head -n 1 || true)"
  if [[ -n "$log_path" ]]; then
    log "latest_log=$log_path"
    tail -n 20 "$log_path" | sed 's/^/[tail] /' | tee -a "$CHAIN_LOG"
  fi
}

wait_for_pid_exit() {
  local pid="$1"
  local label="$2"
  local last_status
  last_status="$(date +%s)"
  log "waiting for $label pid=$pid"
  while true; do
    stat="$(ps -p "$pid" -o stat= 2>/dev/null | tr -d ' ' || true)"
    if [[ -z "$stat" || "$stat" == Z* ]]; then
      break
    fi
    now="$(date +%s)"
    if ((now - last_status >= STATUS_SECONDS)); then
      log "$label still running; gpu snapshot follows"
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$CHAIN_LOG"
      latest_train_tail "/home/work/.data/liquid_cli_sft/logs/sft_h200_8gpu_lfm2_2p6b_processed_template_holdout_*.log"
      last_status="$now"
    fi
    sleep "$POLL_SECONDS"
  done
  log "$label pid=$pid exited"
}

wait_for_quiet_gpus() {
  while true; do
    busy=0
    for gpu in 0 1 2 3 4 5 6 7; do
      used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      if [[ -n "$used" && "$used" -gt 5000 ]]; then
        busy=1
      fi
    done
    if [[ "$busy" == "0" ]]; then
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

if [[ -z "$WAIT_PID" ]]; then
  WAIT_PID="$(pgrep -f 'run_sft_7gpu_processed.sh --config Liquid-CLI/configs/sft_h200_8gpu_lfm2_2p6b_processed.env' | head -n 1 || true)"
fi

if [[ -z "$WAIT_PID" ]]; then
  log "2.6B launcher pid not found; refusing to start 24B blindly"
  exit 1
fi

LFM_2P6B_OUTPUT="$(source_config_value Liquid-CLI/configs/sft_h200_8gpu_lfm2_2p6b_processed.env OUTPUT_DIR)"
wait_for_pid_exit "$WAIT_PID" "LFM2-2.6B training"

count_2p6b="$(checkpoint_count "$LFM_2P6B_OUTPUT")"
if ((count_2p6b < 2)); then
  log "2.6B did not produce two epoch checkpoints: output=$LFM_2P6B_OUTPUT count=$count_2p6b"
  exit 1
fi
log "2.6B checkpoints ready: output=$LFM_2P6B_OUTPUT count=$count_2p6b"

wait_for_quiet_gpus

RUN_NAME_24B="$(source_config_value "$FSDP_24B_CONFIG" RUN_NAME)"
OUT_24B="$(source_config_value "$FSDP_24B_CONFIG" OUTPUT_DIR)"
LAUNCHER_24B_LOG="$LOG_DIR/launcher_${RUN_NAME_24B}_$(date -u +%Y%m%dT%H%M%SZ).log"

log "starting 24B training config=$FSDP_24B_CONFIG output=$OUT_24B"
(
  bash qwen_sft/scripts/run_hf_fsdp_8gpu.sh --config "$FSDP_24B_CONFIG"
) >"$LAUNCHER_24B_LOG" 2>&1 &
PID_24B="$!"
log "24B launcher pid=$PID_24B log=$LAUNCHER_24B_LOG"
wait_for_pid_exit "$PID_24B" "LFM2-24B training"

count_24b="$(checkpoint_count "$OUT_24B")"
if ((count_24b < 2)); then
  log "24B did not produce two epoch checkpoints: output=$OUT_24B count=$count_24b"
  exit 1
fi
log "24B checkpoints ready: output=$OUT_24B count=$count_24b"

wait_for_quiet_gpus
log "starting final 8-way evaluation run_id=$EVAL_RUN_ID"
TB2_RUN_ID="$EVAL_RUN_ID" bash Liquid-CLI/scripts/evaluate_lfm_family_after_24b.sh 2>&1 | tee -a "$CHAIN_LOG"
log "chain complete"
