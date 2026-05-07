#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="Liquid-CLI/configs/sft_h200_7gpu_processed.env"
WAIT_PID=""
UPLOAD_SUBDIR="final"
POLL_SECONDS=30

while (($#)); do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --wait-pid)
      WAIT_PID="$2"
      shift 2
      ;;
    --upload-subdir)
      UPLOAD_SUBDIR="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

source .liquid-sft-env/bin/activate
source "$CONFIG_PATH"

if [[ -z "${HF_TOKEN:-}" && -f "$ROOT_DIR/.env" ]]; then
  set -a
  source <(grep '^export HF_TOKEN=' "$ROOT_DIR/.env")
  set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not available for upload." >&2
  exit 1
fi

UPLOAD_DIR="$OUTPUT_DIR/$UPLOAD_SUBDIR"
READY_MARKER="$OUTPUT_DIR/.hf_upload_complete"
CONFIG_FILE="$UPLOAD_DIR/config.json"
TOKENIZER_FILE="$UPLOAD_DIR/tokenizer_config.json"

if [[ -n "$WAIT_PID" ]]; then
  echo "waiting for training pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
fi

has_weight_artifact() {
  compgen -G "$UPLOAD_DIR/model*.safetensors" >/dev/null || \
    compgen -G "$UPLOAD_DIR/pytorch_model*.bin" >/dev/null
}

marker_is_current() {
  [[ -f "$READY_MARKER" ]] || return 1
  [[ "$READY_MARKER" -nt "$CONFIG_FILE" && "$READY_MARKER" -nt "$TOKENIZER_FILE" ]] || return 1
  local weight_path
  for weight_path in "$UPLOAD_DIR"/model*.safetensors "$UPLOAD_DIR"/pytorch_model*.bin; do
    [[ -e "$weight_path" ]] || continue
    if [[ "$weight_path" -nt "$READY_MARKER" ]]; then
      return 1
    fi
  done
  return 0
}

echo "waiting for final artifacts in $UPLOAD_DIR"
until [[ -f "$CONFIG_FILE" && -f "$TOKENIZER_FILE" ]] && has_weight_artifact; do
  sleep "$POLL_SECONDS"
done

if marker_is_current; then
  echo "upload already completed for current artifacts: $READY_MARKER"
  exit 0
fi

python Liquid-CLI/scripts/upload_model_to_hub.py \
  --output-dir "$OUTPUT_DIR" \
  --hub-model-id "$HUB_MODEL_ID" \
  --upload-subdir "$UPLOAD_SUBDIR" \
  --commit-message "Upload final model for $RUN_NAME"

date -u +%Y-%m-%dT%H:%M:%SZ > "$READY_MARKER"
echo "upload complete: $HUB_MODEL_ID"
