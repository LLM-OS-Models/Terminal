#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/work/.projects/LLM-OS-Models/Terminal}"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch}"
PARENT_OUTPUT_DIR="${PARENT_OUTPUT_DIR:-/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260609T101140Z_echo_public1500_prepared_only_resume380_dockerseed_pathfix2_long_setsid_vllm4_train2}"
CONT_OUTPUT_DIR="${CONT_OUTPUT_DIR:-/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260611T094438Z_echo_public1500_continue_from_1880_vllm4_train2}"
EVAL_PATH="${EVAL_PATH:-tb2_lite/data/replay_full.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612}"
GPU="${GPU:-6}"
VLLM_ENV="${VLLM_ENV:-$ROOT_DIR/.vllm-lfm-cu12}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"

mkdir -p "$RESULTS_DIR/logs"
LOG_FILE="$RESULTS_DIR/logs/eval_gpu${GPU}.log"

VENV_SITE="$VLLM_ENV/lib/python3.12/site-packages"
VLLM_LD_LIBRARY_PATH="$VENV_SITE/torch/lib:$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cublas/lib:$VENV_SITE/nvidia/cudnn/lib:$VENV_SITE/nvidia/nccl/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-12.9:/usr/local/cuda-12.9/include:/usr/include/x86_64-linux-gnu:/opt/hpcx/ucc/lib:/usr/local/cuda/lib64"

latest_checkpoint() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%p\n' | sort -V | tail -1
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
    lines += [
        f"Current best in this comparison: `{best['name']}` Score `{best['score']:.2f}`.",
        "",
    ]
lines += [
    "| Rank | Model / checkpoint | Score | Cmd F1 | First Cmd | Valid JSON | Result |",
    "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
]
for i, row in enumerate(all_rows, 1):
    lines.append(
        f"| {i} | `{row['name']}` | {row['score']:.2f} | {row['cmd_f1']:.4f} | {row['first_cmd']} | {row['valid_json']} | `{row['path']}` |"
    )
lines.append("")
lines.append("Note: RLVR rows appear here only after the GPU6 evaluator finishes each checkpoint.")
lines.append("")
readme_text = "\n".join(lines)
(results_dir / "README.md").write_text(readme_text, encoding="utf-8")

root_text = root_readme.read_text(encoding="utf-8", errors="replace") if root_readme.exists() else ""
section_title = "## 진행 중: LFM2.5 ECHO RLVR GPU6 평가"
section = "\n".join(
    [
        section_title,
        "",
        f"2026-06-12 기준 GPU 6번에서 ECHO RLVR LoRA checkpoint TB2-lite 평가를 진행한다. 결과 디렉터리는 `{results_dir}`다.",
        "",
        f"- 현재 비교 최고점: `{best['name'] if best else 'pending'}` Score `{best['score']:.2f}`" if best else "- 현재 비교 최고점: pending",
        f"- RLVR 평가 완료 개수: `{len(rows)}`",
        "- 기준점: SFT 1Epoch Score `52.30`, SFT 2Epoch Score `50.48`, LFM2.5 base Score `36.53`",
        "- RLVR checkpoint가 SFT 1Epoch `52.30`을 넘는지 여부는 full eval 완료 후 판단한다.",
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

run_eval() {
  local adapter="$1"
  local short="$2"
  if [[ ! -d "$adapter" ]]; then
    echo "missing adapter: $adapter" | tee -a "$LOG_FILE"
    return 1
  fi
  echo "[$(date -Is)] eval_start short=$short adapter=$adapter" | tee -a "$LOG_FILE"
  env -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
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
  echo "[$(date -Is)] eval_done short=$short" | tee -a "$LOG_FILE"
}

PARENT_CKPT="${PARENT_CKPT:-$PARENT_OUTPUT_DIR/checkpoint-1880}"
CONT_CKPT="${CONT_CKPT:-$(latest_checkpoint "$CONT_OUTPUT_DIR")}"
CONT_STEP="$(basename "$CONT_CKPT" | awk -F- '{print $NF}')"

write_readme
run_eval "$PARENT_CKPT" "lfm25-echo-rlvr-parent-checkpoint-1880"
run_eval "$CONT_CKPT" "lfm25-echo-rlvr-continue-checkpoint-${CONT_STEP}"
write_readme

echo "[$(date -Is)] all_done results=$RESULTS_DIR" | tee -a "$LOG_FILE"
