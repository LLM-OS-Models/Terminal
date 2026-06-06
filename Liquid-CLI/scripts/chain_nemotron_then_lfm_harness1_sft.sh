#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RESULT_PATH="tb2_lite/results/20260606T_nemotron3_ultra_550b_a55b_gguf_q4km_eval/nvidia-nemotron-3-ultra-550b-a55b-ud-q4-k-m.json"
TRAIN_CONFIG="Liquid-CLI/configs/sft_h200_8gpu_lfm25_8b_a1b_harness1_lora.env"
POLL_SECONDS=60
GENERATE_TRAJECTORIES=0
SKIP_TRAIN=0

while (($#)); do
  case "$1" in
    --result-path)
      RESULT_PATH="$2"
      shift 2
      ;;
    --train-config)
      TRAIN_CONFIG="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --generate-trajectories)
      GENERATE_TRAJECTORIES=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

LOG_PATH="/home/work/.data/liquid_cli_sft/logs/chain_nemotron_then_lfm_harness1_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "chain_log=$LOG_PATH"
echo "waiting_result=$RESULT_PATH"
echo "train_config=$TRAIN_CONFIG"

while true; do
  status_json="$(python - "$RESULT_PATH" <<'PY'
import datetime
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

path = Path(sys.argv[1])
now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
if not path.exists():
    print(json.dumps({"exists": False, "now_kst": now.strftime("%Y-%m-%d %H:%M:%S KST")}))
    raise SystemExit

obj = json.loads(path.read_text())
steps = obj.get("generated_steps") or len(obj.get("per_step", []))
total = obj.get("total_input_steps") or 303
sec = obj.get("avg_sec_per_step") or 0.0
eta = now + datetime.timedelta(seconds=max(0, total - steps) * sec)
agg = obj.get("aggregate", {})
print(json.dumps({
    "exists": True,
    "complete": bool(obj.get("complete")),
    "now_kst": now.strftime("%Y-%m-%d %H:%M:%S KST"),
    "steps": steps,
    "total": total,
    "score": agg.get("next_action_score"),
    "valid_json": agg.get("valid_json_pct"),
    "avg_sec_per_step": round(sec, 3),
    "eta_kst": eta.strftime("%Y-%m-%d %H:%M:%S KST"),
}, ensure_ascii=False))
PY
)"
  echo "$status_json"
  if python - "$status_json" <<'PY'
import json
import sys
sys.exit(0 if json.loads(sys.argv[1]).get("complete") else 1)
PY
  then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "nemotron_eval_complete=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "skip_train=1"
  exit 0
fi

if [[ "$GENERATE_TRAJECTORIES" == "1" ]]; then
  bash Liquid-CLI/scripts/run_lfm_harness1_lora_sft.sh \
    --config "$TRAIN_CONFIG" \
    --generate-trajectories
else
  bash Liquid-CLI/scripts/run_lfm_harness1_lora_sft.sh \
    --config "$TRAIN_CONFIG"
fi
