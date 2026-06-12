#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch}"
CONT_OUTPUT_DIR="${CONT_OUTPUT_DIR:-/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260611T094438Z_echo_public1500_continue_from_1880_vllm4_train2}"
EVAL_PATH="${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612}"
SHORT_PREFIX="${SHORT_PREFIX:-lfm25-echo-rlvr-continue-checkpoint-}"
GPU="${GPU:-6}"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
POLL_SECONDS="${POLL_SECONDS:-300}"
EVAL_STRIDE="${EVAL_STRIDE:-10}"
EVAL_RECENT="${EVAL_RECENT:-5}"
EVAL_EARLY_UNTIL="${EVAL_EARLY_UNTIL:-150}"
EVAL_EARLY_STRIDE="${EVAL_EARLY_STRIDE:-10}"
EVAL_FOCUS_START="${EVAL_FOCUS_START:-200}"
EVAL_FOCUS_END="${EVAL_FOCUS_END:-300}"
EVAL_FOCUS_STRIDE="${EVAL_FOCUS_STRIDE:-10}"
EVAL_ALL="${EVAL_ALL:-0}"
EVAL_ORDER="${EVAL_ORDER:-latest_first}"
WAIT_PID_FILE="${WAIT_PID_FILE:-/home/work/.data/liquid_cli_sft/eval_pids/echo_gpu6_eval.pid}"

mkdir -p "$RESULTS_DIR/logs" /home/work/.data/liquid_cli_sft/eval_pids
LOG_FILE="$RESULTS_DIR/logs/eval_gpu${GPU}_watch.log"
PID_FILE="/home/work/.data/liquid_cli_sft/eval_pids/echo_gpu${GPU}_watch.pid"
echo "$$" > "$PID_FILE"

VENV_SITE="$VLLM_ENV/lib/python3.12/site-packages"
VLLM_LD_LIBRARY_PATH="$VENV_SITE/torch/lib:$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cublas/lib:$VENV_SITE/nvidia/cudnn/lib:$VENV_SITE/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-12.9:/usr/local/cuda-12.9/include:/usr/include/x86_64-linux-gnu:/opt/hpcx/ucc/lib:/usr/local/cuda/lib64"

log() {
  echo "[$(date -Is)] $*" | tee -a "$LOG_FILE"
}

wait_for_existing_eval() {
  if [[ ! -s "$WAIT_PID_FILE" ]]; then
    return
  fi
  local pid
  pid="$(cat "$WAIT_PID_FILE" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    return
  fi
  while kill -0 "$pid" 2>/dev/null; do
    log "waiting_for_existing_gpu${GPU}_eval pid=$pid"
    sleep 60
  done
}

wait_for_gpu_eval_idle() {
  while pgrep -af "tb2_lite/scripts/replay_eval.py" | awk -v gpu="$GPU" '$0 ~ "--gpu " gpu {print $1}' | grep -q .; do
    log "waiting_for_gpu${GPU}_replay_eval_to_finish"
    sleep 30
  done
}

write_readme() {
  "$VLLM_ENV/bin/python" - "$RESULTS_DIR" "$ROOT_DIR/README.md" <<'PY'
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

results_dir = pathlib.Path(sys.argv[1])
root_readme = pathlib.Path(sys.argv[2])

baseline = [
    {
        "name": "LFM2.5-8B-A1B Terminal ToolBench Full SFT 1Epoch",
        "score": 52.30,
        "cmd_f1": 0.5230,
        "first_cmd": "49.5%",
        "valid_json": "76.9%",
        "path": "tb2_lite/results/20260605T_live_hrm_lora_lfm_epoch1/LFM2.5-8B-A1B-terminal-toolbench-full-1epoch-checkpoint-1542.json",
    },
    {
        "name": "LFM2.5-8B-A1B Terminal ToolBench Full SFT 2Epoch",
        "score": 50.48,
        "cmd_f1": 0.5048,
        "first_cmd": "49.2%",
        "valid_json": "74.9%",
        "path": "tb2_lite/results/20260605T_all_idle_eval/LFM2.5-8B-A1B-terminal-toolbench-full-2epoch-final.json",
    },
    {
        "name": "LiquidAI/LFM2.5-8B-A1B Base",
        "score": 36.53,
        "cmd_f1": 0.3653,
        "first_cmd": "39.9%",
        "valid_json": "59.1%",
        "path": "/home/work/.data/tb2_lite_eval/corrected_readme_models_vllm/liquid_lfm25_8b_a1b_base.json",
    },
]

rows = []
for p in sorted(results_dir.glob("*.json")):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    agg = d.get("aggregate", {})
    if not agg:
        continue
    rows.append(
        {
            "name": d.get("model_short") or d.get("model") or p.stem,
            "score": float(agg.get("next_action_score", 0.0)),
            "cmd_f1": float(agg.get("avg_command_f1", 0.0)),
            "first_cmd": f'{agg.get("first_cmd_exact_pct", 0.0)}%',
            "valid_json": f'{agg.get("valid_json_pct", 0.0)}%',
            "path": str(p),
        }
    )

all_rows = sorted(baseline + rows, key=lambda x: x["score"], reverse=True)
best = all_rows[0] if all_rows else None
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

lines = [
    "# LFM2.5 ECHO RLVR GPU6 TB2-lite Evaluation",
    "",
    f"Updated: `{now}`",
    "",
    "This directory tracks TB2-lite full replay evaluations for ECHO-style RLVR LoRA checkpoints.",
    "",
    "Score 기준: `100 * avg_command_f1`.",
    "",
]
if best:
    lines += [f"Current best in this comparison: `{best['name']}` Score `{best['score']:.2f}`.", ""]
lines += [
    "| Rank | Model / checkpoint | Score | Cmd F1 | First Cmd | Valid JSON | Result |",
    "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
]
for i, row in enumerate(all_rows, 1):
    lines.append(
        f"| {i} | `{row['name']}` | {row['score']:.2f} | {row['cmd_f1']:.4f} | {row['first_cmd']} | {row['valid_json']} | `{row['path']}` |"
    )
lines.append("")
lines.append("Note: the GPU6 watcher evaluates stride checkpoints plus the most recent checkpoints as they appear.")
lines.append("")
(results_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

root_text = root_readme.read_text(encoding="utf-8", errors="replace") if root_readme.exists() else ""
section_title = "## 진행 중: LFM2.5 ECHO RLVR GPU6 평가"
section = "\n".join(
    [
        section_title,
        "",
        "2026-06-12 기준 GPU 6번에서 ECHO RLVR LoRA checkpoint TB2-lite 평가를 계속 진행한다.",
        "",
        f"- 상세 기록: [`docs/ECHO_RLVR_GPU6_EVAL_20260612.md`](docs/ECHO_RLVR_GPU6_EVAL_20260612.md)",
        f"- 결과 디렉터리: `{results_dir}`",
        f"- 현재 비교 최고점: `{best['name'] if best else 'pending'}` Score `{best['score']:.2f}`" if best else "- 현재 비교 최고점: pending",
        f"- RLVR 평가 완료 개수: `{len(rows)}`",
        "",
    ]
)
if section_title in root_text:
    root_text = re.sub(rf"{re.escape(section_title)}\n.*?(?=\n## |\Z)", section.rstrip() + "\n", root_text, flags=re.S)
else:
    marker = "## 전체 순위"
    if marker in root_text:
        root_text = root_text.replace(marker, section + "\n" + marker, 1)
    else:
        root_text = root_text.rstrip() + "\n\n" + section
root_readme.write_text(root_text, encoding="utf-8")
PY
}

result_exists() {
  local short="$1"
  [[ -s "$RESULTS_DIR/${short}.json" ]]
}

run_eval() {
  local adapter="$1"
  local short="$2"
  if result_exists "$short"; then
    log "skip_existing short=$short"
    return
  fi
  wait_for_gpu_eval_idle
  log "eval_start short=$short adapter=$adapter"
  env -u PYTHONPATH \
    LD_LIBRARY_PATH="$VLLM_LD_LIBRARY_PATH" \
    CUDA_VISIBLE_DEVICES="$GPU" \
    "$VLLM_ENV/bin/python" tb2_lite/scripts/replay_eval.py \
      --model "$BASE_MODEL" \
      --tokenizer-path "$BASE_MODEL" \
      --model-short "$short" \
      --gpu "$GPU" \
      --eval-path "$EVAL_PATH" \
      --output-dir "$RESULTS_DIR" \
      --dtype bfloat16 \
      --tp 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --max-tokens "$MAX_TOKENS" \
      --temperature 0.0 \
      --top-p 1.0 \
      --language-model-only \
      --allow-raw-fallback \
      --lora-path "$adapter" \
      --lora-name "$short" \
      --lora-id 1 \
      --max-lora-rank 32 \
      --skip-if-exists \
      2>&1 | tee -a "$LOG_FILE"
  write_readme
  log "eval_done short=$short"
}

list_candidate_checkpoints() {
  "$VLLM_ENV/bin/python" - \
    "$CONT_OUTPUT_DIR" \
    "$EVAL_STRIDE" \
    "$EVAL_RECENT" \
    "$EVAL_EARLY_UNTIL" \
    "$EVAL_EARLY_STRIDE" \
    "$EVAL_FOCUS_START" \
    "$EVAL_FOCUS_END" \
    "$EVAL_FOCUS_STRIDE" \
    "$EVAL_ALL" \
    "$EVAL_ORDER" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
stride = int(sys.argv[2])
recent = int(sys.argv[3])
early_until = int(sys.argv[4])
early_stride = int(sys.argv[5])
focus_start = int(sys.argv[6])
focus_end = int(sys.argv[7])
focus_stride = int(sys.argv[8])
eval_all = str(sys.argv[9]).lower() in {"1", "true", "yes", "on", "all"}
eval_order = str(sys.argv[10]).lower()

items = []
for path in root.glob("checkpoint-*"):
    try:
        step = int(path.name.rsplit("-", 1)[1])
    except Exception:
        continue
    items.append((step, path))
items.sort()

if eval_all:
    selected = {step: path for step, path in items}
else:
    selected = {step: path for step, path in items if step % stride == 0}
    for step, path in items:
        if step <= early_until and step % early_stride == 0:
            selected[step] = path
        if focus_start <= step <= focus_end and step % focus_stride == 0:
            selected[step] = path
recent_steps = []
if recent > 0:
    for step, path in items[-recent:]:
        if not eval_all:
            selected[step] = path
        recent_steps.append(step)

seen = set()
if eval_order in {"latest_first", "recent_first", "desc"}:
    for step in reversed(recent_steps):
        path = selected[step]
        seen.add(step)
        print(f"{step}\t{path}")
if eval_order in {"desc", "descending"}:
    iterator = sorted(selected.items(), reverse=True)
else:
    iterator = sorted(selected.items())
for step, path in iterator:
    if step in seen:
        continue
    print(f"{step}\t{path}")
PY
}

log "watch_start gpu=$GPU cont_dir=$CONT_OUTPUT_DIR short_prefix=$SHORT_PREFIX eval_all=$EVAL_ALL order=$EVAL_ORDER stride=$EVAL_STRIDE recent=$EVAL_RECENT early=${EVAL_EARLY_STRIDE}<=${EVAL_EARLY_UNTIL} focus=${EVAL_FOCUS_START}-${EVAL_FOCUS_END}/${EVAL_FOCUS_STRIDE}"
wait_for_existing_eval
wait_for_gpu_eval_idle
write_readme

while true; do
  while IFS=$'\t' read -r step adapter; do
    [[ -n "$step" && -n "$adapter" ]] || continue
    run_eval "$adapter" "${SHORT_PREFIX}${step}"
  done < <(list_candidate_checkpoints)
  write_readme
  log "watch_sleep seconds=$POLL_SECONDS"
  sleep "$POLL_SECONDS"
done
