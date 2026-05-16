#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}
cd "$ROOT_DIR"

HF_HOME=${HF_HOME:-/home/work/.data/huggingface}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}
export PYTHONNOUSERSITE=1
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-12.9}

EVAL_PY=${EVAL_PY:-$ROOT_DIR/.eval-env/bin/python}
VLLM_PY=${VLLM_PY:-$ROOT_DIR/.vllm-eval-cu129-strict/bin/python}
SGLANG_PY=${SGLANG_PY:-$ROOT_DIR/.sglang-eval/bin/python}
EVAL_PATH=${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_post_deepseek_flash_queue}
OUT_DIR=${OUT_DIR:-tb2_lite/results/$RUN_ID}
LOG_DIR=${LOG_DIR:-tb2_lite/logs/post_deepseek_flash_queue/$RUN_ID}
mkdir -p "$OUT_DIR" "$LOG_DIR"

STRICT_SITE="$ROOT_DIR/.vllm-eval-cu129-strict/lib/python3.12/site-packages"
STRICT_LIBS=$(find "$STRICT_SITE/nvidia" -path '*/lib' -type d 2>/dev/null | sort | paste -sd ':' -)
if [[ -n "$STRICT_LIBS" ]]; then
  export LD_LIBRARY_PATH="$STRICT_LIBS:/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH:-}"
fi
SGLANG_SITE="$ROOT_DIR/.sglang-eval/lib/python3.12/site-packages"
SGLANG_LIBS=$(find "$SGLANG_SITE/nvidia" -path '*/lib' -type d 2>/dev/null | sort | paste -sd ':' -)
if [[ -n "$SGLANG_LIBS" ]]; then
  export LD_LIBRARY_PATH="$SGLANG_LIBS:${LD_LIBRARY_PATH:-}"
fi

log() {
  printf '[%s KST] %s\n' "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_DIR/queue.log"
}

wait_local_snapshot() {
  local repo="$1"
  while true; do
    if python - "$repo" <<'PY' >/dev/null 2>&1
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    cache_dir="/home/work/.data/huggingface/hub",
    local_files_only=True,
)
PY
    then
      log "download ready repo=$repo"
      return 0
    fi
    log "download not ready yet repo=$repo; retry in 300s"
    sleep 300
  done
}

wait_deepseek_finish() {
  log "waiting for active DeepSeek-V4-Flash run to finish"
  while pgrep -f 'tb2_lite/(run_deepseek_flash_full8.sh|scripts/deepseek_replay_eval.py)' >/dev/null 2>&1; do
    python - <<'PY' 2>/dev/null | tee -a "$LOG_DIR/queue.log" || true
import datetime, glob, json, os
now = datetime.datetime.now(datetime.timezone.utc)
kst = datetime.timezone(datetime.timedelta(hours=9))
files = sorted(
    glob.glob("tb2_lite/results/20260515T114312Z_deepseek_flash_official_mp4_full8_len49152_bs8_retry/*.status.*.json"),
    key=os.path.getmtime,
)
done = total = 0
latest_finish = None
for path in files:
    data = json.load(open(path))
    d = int(data.get("completed_steps") or 0)
    t = int(data.get("total_steps") or 0)
    done += d
    total += t
    eta = data.get("eta_sec")
    finish = "unknown"
    if eta is not None:
        finish_dt = (now + datetime.timedelta(seconds=float(eta))).astimezone(kst)
        finish = finish_dt.strftime("%Y-%m-%d %H:%M KST")
        latest_finish = max(latest_finish or finish_dt, finish_dt)
    print(
        f"[{datetime.datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')}] "
        f"deepseek {os.path.basename(path)} {d}/{t} remaining={max(t-d, 0)} finish={finish}",
        flush=True,
    )
if total:
    print(
        f"[{datetime.datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')}] "
        f"deepseek total {done}/{total} remaining={max(total-done, 0)} "
        f"finish={latest_finish.strftime('%Y-%m-%d %H:%M KST') if latest_finish else 'unknown'}",
        flush=True,
    )
PY
    sleep 300
  done
  log "DeepSeek process finished; continuing queue"
}

run_step_flash() {
  wait_local_snapshot "stepfun-ai/Step-3.5-Flash-FP8"
  log "starting Step-3.5-Flash-FP8 SGLang server tp=8 ep=8 fp8"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "$SGLANG_PY" -m sglang.launch_server \
    --model-path stepfun-ai/Step-3.5-Flash-FP8 \
    --served-model-name step3p5-flash \
    --tp-size 8 \
    --ep-size 8 \
    --trust-remote-code \
    --mem-fraction-static 0.9 \
    --attention-backend cutlass_mla \
    --tool-call-parser step3 \
    --reasoning-parser step3 \
    --quantization fp8 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --enable-multi-layer-eagle \
    --host 127.0.0.1 \
    --port 30001 \
    >"$LOG_DIR/step_3_5_flash_fp8_sglang_server.log" 2>&1 &
  local server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true' RETURN
  wait_http "http://127.0.0.1:30001/v1/models" "Step-3.5-Flash-FP8"

  log "starting Step-3.5-Flash-FP8 replay evaluation through SGLang OpenAI API"
  "$EVAL_PY" -u tb2_lite/scripts/replay_eval_openai.py \
    --model step3p5-flash \
    --tokenizer-path stepfun-ai/Step-3.5-Flash-FP8 \
    --model-short step_3_5_flash_fp8_sglang_tp8_ep8 \
    --base-url http://127.0.0.1:30001/v1 \
    --eval-path "$EVAL_PATH" \
    --output-dir "$OUT_DIR" \
    --max-model-len 65536 \
    --max-tokens 1024 \
    --temperature 1.0 \
    --top-p 0.95 \
    --concurrency 24 \
    --request-timeout 900 \
    --retries 2 \
    --allow-raw-fallback \
    --skip-if-exists \
    >"$LOG_DIR/step_3_5_flash_fp8_sglang_tp8_ep8.eval.log" 2>&1

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - RETURN
  log "finished Step-3.5-Flash-FP8 evaluation result=$OUT_DIR/step_3_5_flash_fp8_sglang_tp8_ep8.json"
}

wait_http() {
  local url="$1"
  local name="$2"
  for _ in $(seq 1 240); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "$name server ready url=$url"
      return 0
    fi
    sleep 5
  done
  log "$name server did not become ready in time"
  return 1
}

run_llada_flash() {
  wait_local_snapshot "inclusionAI/LLaDA2.1-flash"
  log "starting LLaDA2.1-flash SGLang dLLM server tp=4 JointThreshold"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$SGLANG_PY" -m sglang.launch_server \
    --model-path inclusionAI/LLaDA2.1-flash \
    --dllm-algorithm JointThreshold \
    --tp-size 4 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --max-running-requests 1 \
    --attention-backend flashinfer \
    --host 127.0.0.1 \
    --port 30000 \
    >"$LOG_DIR/llada2_1_flash_sglang_server.log" 2>&1 &
  local server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true' RETURN
  wait_http "http://127.0.0.1:30000/v1/models" "LLaDA2.1-flash"

  log "starting LLaDA2.1-flash replay evaluation through SGLang OpenAI API"
  "$EVAL_PY" -u tb2_lite/scripts/replay_eval_openai.py \
    --model inclusionAI/LLaDA2.1-flash \
    --tokenizer-path inclusionAI/LLaDA2.1-flash \
    --model-short llada2_1_flash_sglang_tp4_speed \
    --base-url http://127.0.0.1:30000/v1 \
    --eval-path "$EVAL_PATH" \
    --output-dir "$OUT_DIR" \
    --max-model-len 32768 \
    --max-tokens 1024 \
    --temperature 0.0 \
    --top-p 1.0 \
    --concurrency 1 \
    --request-timeout 1800 \
    --retries 1 \
    --extra-body-json '{"gen_length":1024,"block_length":32,"threshold":0.5,"editing_threshold":0.0,"max_post_steps":16}' \
    --allow-raw-fallback \
    --skip-if-exists \
    >"$LOG_DIR/llada2_1_flash_sglang_tp4_speed.eval.log" 2>&1

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - RETURN
  log "finished LLaDA2.1-flash evaluation result=$OUT_DIR/llada2_1_flash_sglang_tp4_speed.json"
}

wait_deepseek_finish
run_step_flash
run_llada_flash
log "queue finished out_dir=$OUT_DIR"
