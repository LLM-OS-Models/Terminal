#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/work/.projects/LLM-OS-Models/Terminal}"
HRM_ROOT="${HRM_ROOT:-$ROOT/HRM-Text}"
CKPT="${CKPT:-/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch3-from-epoch2-gbs180k-8gpu}"
EXPORT="${EXPORT:-/home/work/.data/hrm_text_exports/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch3}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/work/.data/hrm_text_prepared/kohrm_sft_lfm25_terminal_toolbench_full_v1}"
EVAL_DIR="${EVAL_DIR:-$ROOT/tb2_lite/results/20260606T_kohrm_lfm25_epoch3_eval_sdpa8_b16}"
HF_REPO="${HF_REPO:-LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch3}"
SHORT="${SHORT:-KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch3-sdpa8-b16}"
TRAIN_PID="${TRAIN_PID:-}"
LOG="${LOG:-/home/work/.data/hrm_text_logs/kohrm_lfm25_epoch3_after_training.log}"

mkdir -p "$(dirname "$LOG")" "$EVAL_DIR/logs"
exec >>"$LOG" 2>&1

log() {
  printf '[%s] %s\n' "$(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST')" "$*"
}

load_hf_env() {
  if [[ -f "$HRM_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$HRM_ROOT/.env"
    set +a
  fi
}

wait_for_training() {
  if [[ -n "$TRAIN_PID" ]]; then
    log "waiting for training pid=$TRAIN_PID"
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
      sleep 60
    done
  else
    log "TRAIN_PID not set; waiting for matching epoch3 training process to disappear"
    while pgrep -af "KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch3-from-epoch2-8gpu" | grep -v run_kohrm_lfm25_epoch3_after_training >/dev/null; do
      sleep 60
    done
  fi
}

wait_for_checkpoint() {
  local fsdp="$CKPT/fsdp2_epoch_1"
  log "waiting for complete checkpoint at $fsdp"
  for _ in $(seq 1 120); do
    local shards carry
    shards=$(find "$fsdp" -maxdepth 1 -name '*.distcp' 2>/dev/null | wc -l || true)
    carry=$(find "$CKPT" -maxdepth 1 -name 'carry_epoch_1.*.pt' 2>/dev/null | wc -l || true)
    if [[ -f "$fsdp/.metadata" && "$shards" -ge 8 && "$carry" -ge 8 ]]; then
      sleep 120
      log "checkpoint ready: shards=$shards carry=$carry"
      return
    fi
    sleep 60
  done
  log "checkpoint did not become ready"
  exit 1
}

write_pending_card() {
  mkdir -p "$EXPORT"
  cat >"$EXPORT/README.md" <<EOF
---
language:
- ko
- en
library_name: pytorch
pipeline_tag: text-generation
base_model: LLM-OS-Models/KoHRM-Text-1.4B
base_model_relation: finetune
tags:
- kohrm
- hrm-text
- terminal-agent
- tool-use
- full-sft
- prefixlm
- tb2-lite
---

# KoHRM-Text-1.4B FullSFT LFM25 Terminal ToolBench Epoch3

This is the third full-SFT epoch of the KoHRM-Text 1.4B LFM25/ToolBench terminal checkpoint.

- Base model: \`LLM-OS-Models/KoHRM-Text-1.4B\`
- Parent checkpoint: \`LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch2\`
- Dataset: \`kohrm_sft_lfm25_terminal_toolbench_full_v1\`
- Training: full SFT, 8 x H200, global batch size \`180224\`, learning rate \`2e-5\`
- Status: model uploaded first; TB2-lite full replay evaluation is running or pending.

The final score table will be written back to this card after the 303-step evaluation finishes.
EOF
}

convert_and_upload() {
  log "converting checkpoint to HF export: $EXPORT"
  rm -rf "$EXPORT"
  cd "$HRM_ROOT"
  python conversion/convert_to_hf.py \
    --ckpt_path "$CKPT" \
    --ckpt_epoch 1 \
    --ckpt_use_ema true \
    --out_dir "$EXPORT" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --device cpu

  write_pending_card
  load_hf_env
  log "uploading model export to https://huggingface.co/$HF_REPO"
  python scripts/upload_folder_to_hf.py \
    --folder "$EXPORT" \
    --repo-id "$HF_REPO" \
    --repo-type model \
    --large \
    --num-workers 4
}

run_eval() {
  cd "$ROOT"
  log "starting 8-shard TB2-lite evaluation: $EVAL_DIR"
  local pids=()
  for shard in 0 1 2 3 4 5 6 7; do
    (
      export CUDA_VISIBLE_DEVICES="$shard"
      export PYTHONUNBUFFERED=1
      export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-256}"
      export KOHRM_FORCE_SDPA_KVCACHE=1
      export KOHRM_DISABLE_INFERENCE_COMPILE=1
      python tb2_lite/scripts/replay_eval_hrm_text.py \
        --model "$EXPORT" \
        --model-short "${SHORT}-shard${shard}" \
        --gpu "$shard" \
        --eval-path tb2_lite/data/replay_full.jsonl \
        --output-dir "$EVAL_DIR" \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --max-tokens 1024 \
        --temperature 0.0 \
        --top-p 1.0 \
        --condition synth,cot \
        --head-tokens 1024 \
        --min-tail-tokens 1024 \
        --batch-size 16 \
        --num-shards 8 \
        --shard-index "$shard" \
        --progress-every 5 \
        --save-every 5 \
        --hrm-text-root HRM-Text \
        --local-hrm-export \
        --base-ckpt-path "$CKPT" \
        >"$EVAL_DIR/logs/shard${shard}_gpu${shard}.log" 2>&1
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    log "one or more eval shards failed"
    exit 1
  fi

  log "merging eval shards"
  python tb2_lite/scripts/merge_replay_result_shards.py \
    --output "$EVAL_DIR/${SHORT}-merged.json" \
    "$EVAL_DIR/${SHORT}-shard0.json" \
    "$EVAL_DIR/${SHORT}-shard1.json" \
    "$EVAL_DIR/${SHORT}-shard2.json" \
    "$EVAL_DIR/${SHORT}-shard3.json" \
    "$EVAL_DIR/${SHORT}-shard4.json" \
    "$EVAL_DIR/${SHORT}-shard5.json" \
    "$EVAL_DIR/${SHORT}-shard6.json" \
    "$EVAL_DIR/${SHORT}-shard7.json"
}

write_final_card_and_summary() {
  local merged="$EVAL_DIR/${SHORT}-merged.json"
  local summary="$EVAL_DIR/${SHORT}-summary.md"
  python - "$merged" "$EXPORT/README.md" "$summary" "$CKPT" "$HF_REPO" <<'PY'
import json
import sys
from pathlib import Path

merged = Path(sys.argv[1])
readme = Path(sys.argv[2])
summary = Path(sys.argv[3])
ckpt = sys.argv[4]
repo = sys.argv[5]
data = json.loads(merged.read_text())
agg = data["aggregate"]
score = round(100.0 * float(agg["avg_command_f1"]), 2)
epoch2_score = 45.90
delta = score - epoch2_score
groups = agg.get("by_source_group", {})
strong = sorted(groups.items(), key=lambda kv: kv[1]["avg_command_f1"], reverse=True)[:6]
weak = sorted(groups.items(), key=lambda kv: kv[1]["avg_command_f1"])[:6]
bucket = agg.get("by_bucket", {})

def group_lines(rows):
    return "\n".join(
        f"- `{name}`: `{stats['avg_command_f1']:.4f}` ({stats['steps']} steps)"
        for name, stats in rows
    )

body = f"""---
language:
- ko
- en
library_name: pytorch
pipeline_tag: text-generation
base_model: LLM-OS-Models/KoHRM-Text-1.4B
base_model_relation: finetune
tags:
- kohrm
- hrm-text
- terminal-agent
- tool-use
- full-sft
- prefixlm
- tb2-lite
---

# KoHRM-Text-1.4B FullSFT LFM25 Terminal ToolBench Epoch3

This is the third full-SFT epoch of the KoHRM-Text 1.4B LFM25/ToolBench terminal checkpoint.
It is a fine-tuned version of [`LLM-OS-Models/KoHRM-Text-1.4B`](https://huggingface.co/LLM-OS-Models/KoHRM-Text-1.4B) and continues from the Epoch2 LFM25 terminal full-SFT checkpoint.

## Training

- Base model: `LLM-OS-Models/KoHRM-Text-1.4B`
- Parent checkpoint: `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch2`
- Dataset: `kohrm_sft_lfm25_terminal_toolbench_full_v1`
- Training type: full SFT, not LoRA
- Total SFT epochs on this dataset: `3`
- Epoch3 GPUs: `8 x H200`
- Global batch size: `180224` tokens
- Learning rate: `2e-5`
- Checkpoint: `{ckpt}`

## TB2-lite Evaluation

Result JSON:

`{merged}`

| Checkpoint | Steps | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Avg Pred Cmds | Sec/Step | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `epoch2 full replay` | 303/303 | 45.90 | 0.4590 | 0.5031 | 0.5098 | 44.9% | 68.3% | 25.16 | 10.842 | completed |
| `epoch3 full replay` | {data.get('generated_steps')}/{data.get('total_input_steps')} | {score:.2f} | {agg['avg_command_f1']:.4f} | {agg['avg_command_precision']:.4f} | {agg['avg_command_recall']:.4f} | {agg['first_cmd_exact_pct']:.1f}% | {agg['valid_json_pct']:.1f}% | {agg['avg_pred_cmds']:.2f} | {data.get('avg_sec_per_step')} | completed |

`Score = 100 * avg_command_f1`.

## Result Analysis

Epoch3 changed the score by `{delta:+.2f}` versus Epoch2. Interpret this as an additional-pass SFT result: if it improves, the LFM25/ToolBench distribution was still underfit after Epoch2; if it declines, the third pass likely started to overfit repeated terminal trajectories or reduced robustness on late corrective steps.

Bucket F1:

- early: `{bucket.get('early', {}).get('avg_command_f1', 0.0):.4f}`
- mid: `{bucket.get('mid', {}).get('avg_command_f1', 0.0):.4f}`
- late: `{bucket.get('late', {}).get('avg_command_f1', 0.0):.4f}`

Strong source groups:

{group_lines(strong)}

Weak source groups:

{group_lines(weak)}

## Usage

Use the local HRM-Text PrefixLM evaluator/runtime:

```bash
python tb2_lite/scripts/replay_eval_hrm_text.py \\
  --model /path/to/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch3 \\
  --model-short KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch3 \\
  --eval-path tb2_lite/data/replay_full.jsonl \\
  --output-dir tb2_lite/results/kohrm_lfm25_epoch3_eval \\
  --local-hrm-export \\
  --base-ckpt-path /path/to/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch3-from-epoch2-gbs180k-8gpu \\
  --max-model-len 8192 \\
  --max-tokens 1024 \\
  --condition synth,cot \\
  --batch-size 16
```
"""
readme.write_text(body, encoding="utf-8")
summary.write_text(body, encoding="utf-8")
print(json.dumps({"score": score, "delta_vs_epoch2": round(delta, 2), "summary": str(summary)}, ensure_ascii=False))
PY
}

upload_final_readme() {
  load_hf_env
  log "uploading final scored README.md to $HF_REPO"
  python - "$EXPORT/README.md" "$HF_REPO" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

path = Path(sys.argv[1])
repo_id = sys.argv[2]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
api = HfApi(token=token)
api.upload_file(
    path_or_fileobj=str(path),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Update model card with TB2-lite epoch3 score",
)
print(f"uploaded README.md -> https://huggingface.co/{repo_id}")
PY
}

main() {
  log "postprocess watcher started"
  wait_for_training
  wait_for_checkpoint
  convert_and_upload
  run_eval
  write_final_card_and_summary
  upload_final_readme
  log "epoch3 postprocess complete"
}

main "$@"
