#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/ubuntu/Terminal/tb2_official}"
HARBOR_BIN="${HARBOR_BIN:-/home/ubuntu/.tools/bin/harbor}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
MODEL_ID="${1:-}"

if ! docker info >/dev/null 2>&1 \
  && [[ "${TB2_DOCKER_GROUP_REEXEC:-0}" != "1" ]] \
  && getent group docker | grep -Eq "[:,]${USER}(,|$)"; then
  printf -v script_and_args '%q ' "$0" "$@"
  exec sg docker -c "TB2_DOCKER_GROUP_REEXEC=1 $script_and_args"
fi

if [[ -z "$MODEL_ID" ]]; then
  echo "usage: $0 <lfm25-sft1|lfm25-static-610|lfm25-online-425>" >&2
  exit 1
fi
case "$MODEL_ID" in
  lfm25-sft1|lfm25-static-610|lfm25-online-425) ;;
  *)
    echo "unknown model id: $MODEL_ID" >&2
    exit 1
    ;;
esac

TB2_ATTEMPTS="${TB2_ATTEMPTS:-1}"
TB2_CONCURRENCY="${TB2_CONCURRENCY:-1}"
TB2_LIMIT="${TB2_LIMIT-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TB2_SCOPE="${TB2_LIMIT:-full89}"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
JOB_NAME="${JOB_NAME:-tb2-${MODEL_ID}-${TB2_SCOPE}-k${TB2_ATTEMPTS}-n${TB2_CONCURRENCY}-${RUN_TIMESTAMP}}"

"$ROOT_DIR/docker_preflight.sh"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
MODEL_LIST="$(curl -fsS \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  "$API_BASE/models")"
python3 - "$MODEL_ID" "$MODEL_LIST" <<'PY'
import json
import sys

expected = sys.argv[1]
payload = json.loads(sys.argv[2])
available = {item["id"] for item in payload.get("data", [])}
if expected not in available:
    raise SystemExit(f"model {expected!r} is not served; available={sorted(available)}")
print(f"vLLM endpoint ready: {expected}")
PY

args=(
  "$HARBOR_BIN" run
  -d terminal-bench/terminal-bench-2@1
  -a terminus-2
  -m "openai/$MODEL_ID"
  --ak "api_base=$API_BASE"
  --ak parser_name=json
  --ak "temperature=$TEMPERATURE"
  --ak 'model_info={"max_input_tokens":32768,"max_output_tokens":4096,"input_cost_per_token":0,"output_cost_per_token":0}'
  --ak 'llm_kwargs={"max_tokens":4096}'
  -n "$TB2_CONCURRENCY"
  -k "$TB2_ATTEMPTS"
  --yes
  --job-name "$JOB_NAME"
  --jobs-dir "$ROOT_DIR/jobs"
)

if [[ -n "$TB2_LIMIT" ]]; then
  args+=( -l "$TB2_LIMIT" )
fi

exec "${args[@]}"
