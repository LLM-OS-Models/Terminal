#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prompt_builder import build_prompts, sanitize_name
from replay_metrics import aggregate_scores, parse_prediction, score_commands, step_bucket


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def add_prompt_length_meta(tokenizer: Any, rows: list[dict], prompts: list[str], args: argparse.Namespace, prompt_meta: dict) -> None:
    lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
    too_long = [
        {"idx": idx, "task_id": rows[idx].get("task_id"), "step_idx": rows[idx].get("step_idx"), "prompt_tokens": length}
        for idx, length in enumerate(lengths)
        if length + args.max_tokens > args.max_model_len
    ]
    if too_long:
        raise RuntimeError(
            "prompt context overflow before generation: "
            f"max_model_len={args.max_model_len} max_tokens={args.max_tokens} "
            f"overflow_count={len(too_long)} examples={too_long[:10]}"
        )
    sorted_lengths = sorted(lengths)
    prompt_meta.update(
        {
            "row_count": len(rows),
            "prompt_tokens_min": min(lengths) if lengths else 0,
            "prompt_tokens_max": max(lengths) if lengths else 0,
            "prompt_tokens_p50": sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0,
            "prompt_tokens_p95": sorted_lengths[int(len(sorted_lengths) * 0.95) - 1] if sorted_lengths else 0,
            "prompt_tokens_p99": sorted_lengths[int(len(sorted_lengths) * 0.99) - 1] if sorted_lengths else 0,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--assistant-model", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--model-short", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--thinking-mode", default="off")
    parser.add_argument("--strip-thinking-history", default="on")
    parser.add_argument("--gemma4-empty-thought-channel", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or sanitize_name(args.assistant_model)
    suffix = f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    out_path = output_dir / f"{model_short}{suffix}.json"
    if args.skip_if_exists and out_path.exists():
        print(json.dumps({"output_path": str(out_path), "skipped": True}, ensure_ascii=False))
        return

    tokenizer_path = args.tokenizer_path or args.target_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    mode = str(args.thinking_mode).lower()
    chat_template_kwargs: dict[str, Any] = {}
    if mode in {"on", "true", "1", "yes", "enabled"}:
        chat_template_kwargs["enable_thinking"] = True
    elif mode in {"off", "false", "0", "no", "disabled"}:
        chat_template_kwargs["enable_thinking"] = False
    strip_mode = str(args.strip_thinking_history).lower()
    strip_thinking_history = strip_mode in {"on", "true", "1", "yes", "enabled"}
    channel_mode = str(args.gemma4_empty_thought_channel).lower()
    empty_thought_channel = channel_mode in {"on", "true", "1", "yes", "enabled", "auto"}
    loaded_rows = load_rows(Path(args.eval_path), args.limit)
    rows = [
        {**row, "_eval_row_index": idx}
        for idx, row in enumerate(loaded_rows)
        if idx % args.num_shards == args.shard_index
    ]
    prompts, prompt_meta = build_prompts(
        tokenizer,
        rows,
        model_name=args.target_model,
        chat_template_kwargs=chat_template_kwargs,
        prompt_options={
            "strip_thinking_history": strip_thinking_history,
            "gemma4_empty_thought_channel": empty_thought_channel,
            "use_gemma4_patched_template": True,
        },
    )
    add_prompt_length_meta(tokenizer, rows, prompts, args, prompt_meta)

    load_start = time.time()
    dtype = torch.bfloat16
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    assistant_model = AutoModelForCausalLM.from_pretrained(
        args.assistant_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    target_model.eval()
    assistant_model.eval()
    load_time = round(time.time() - load_start, 1)

    gen_start = time.time()
    per_step: list[dict[str, Any]] = []
    do_sample = args.temperature > 0
    for idx, (row, prompt) in enumerate(zip(rows, prompts), start=1):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = target_model.generate(
                **inputs,
                assistant_model=assistant_model,
                max_new_tokens=args.max_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        pred_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
        pred = parse_prediction(pred_text)
        ref = parse_prediction(row["ref_raw"])
        first_exact, precision, recall, f1 = score_commands(pred["command_units"], ref["command_units"])
        pred_complete_true = bool(pred["task_complete"]) and bool(ref["task_complete"])
        per_step.append(
            {
                "eval_row_index": row.get("_eval_row_index"),
                "task_id": row["task_id"],
                "sample_idx": row["sample_idx"],
                "step_idx": row["step_idx"],
                "bucket": step_bucket(row["step_idx"]),
                "source_group": row["source_group"],
                "valid_json": pred["valid_json"],
                "has_analysis": pred["has_analysis"],
                "has_plan": pred["has_plan"],
                "ref_task_complete": bool(ref["task_complete"]),
                "pred_task_complete": pred["task_complete"],
                "pred_task_complete_true": pred_complete_true,
                "ref_command_units": ref["command_units"],
                "pred_command_units": pred["command_units"],
                "first_cmd_exact": first_exact,
                "command_precision": round(precision, 4),
                "command_recall": round(recall, 4),
                "command_f1": round(f1, 4),
                "pred_preview": pred_text[:1200],
            }
        )
        if idx == 1 or idx % 10 == 0 or idx == len(rows):
            print(f"progress {idx}/{len(rows)}", flush=True)

    gen_time = round(time.time() - gen_start, 1)
    aggregate = aggregate_scores(per_step)
    result = {
        "model": model_short,
        "model_path": args.assistant_model,
        "target_model_path": args.target_model,
        "model_short": model_short,
        "gpu": str(args.gpu),
        "eval_path": str(args.eval_path),
        "timestamp": datetime.utcnow().isoformat(),
        "load_time_sec": load_time,
        "gen_time_sec": gen_time,
        "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "thinking_mode": args.thinking_mode,
            "strip_thinking_history": args.strip_thinking_history,
            "gemma4_empty_thought_channel": args.gemma4_empty_thought_channel,
            "dtype": "bfloat16",
            "backend": "transformers_assisted_generation",
            "max_model_len": args.max_model_len,
        },
        "prompt_template": prompt_meta,
        "shard": {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "loaded_row_count": len(loaded_rows),
            "row_count": len(rows),
        },
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(out_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
