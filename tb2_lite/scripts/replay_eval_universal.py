#!/usr/bin/env python3
"""Universal replay eval — handles both text-only and multimodal models."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_metrics import aggregate_scores, parse_prediction, score_commands, step_bucket
from prompt_builder import build_prompts, sanitize_name


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_multimodal(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    with config_path.open() as f:
        cfg = json.load(f)
    return "vision_config" in cfg or cfg.get("model_type", "").endswith("_vl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-short", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(eval_path)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    mm = is_multimodal(args.model)
    backend_tag = "transformers-mm" if mm else "transformers"

    load_start = time.time()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts, prompt_meta = build_prompts(tokenizer, rows)

    if mm:
        print(f"Detected multimodal model, using AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
        )
    model.eval()
    load_time = round(time.time() - load_start, 1)
    print(f"Model loaded in {load_time}s (backend={backend_tag})", flush=True)

    gen_configs = {
        "max_new_tokens": args.max_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else 1.0,
        "top_p": args.top_p,
    }

    gen_start = time.time()
    predictions: list[str] = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_model_len).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_configs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        predictions.append(pred_text)
        if (i + 1) % 20 == 0 or (i + 1) == len(prompts):
            elapsed = round(time.time() - gen_start, 1)
            rate = round((i + 1) / elapsed, 2) if elapsed > 0 else 0
            print(f"  [{i+1}/{len(prompts)}] {elapsed}s ({rate} samples/s)", flush=True)

    gen_time = round(time.time() - gen_start, 1)

    per_step: list[dict] = []
    for row, pred_text in zip(rows, predictions):
        pred = parse_prediction(pred_text)
        ref = parse_prediction(row["ref_raw"])
        first_exact, precision, recall, f1 = score_commands(
            pred["command_units"], ref["command_units"]
        )
        pred_complete_true = bool(pred["task_complete"]) and bool(ref["task_complete"])
        per_step.append({
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
        })

    aggregate = aggregate_scores(per_step)
    model_short = args.model_short or sanitize_name(args.model)
    result = {
        "model": args.model_short or args.model,
        "model_path": args.model,
        "model_short": model_short,
        "gpu": str(args.gpu),
        "eval_path": str(eval_path),
        "timestamp": datetime.utcnow().isoformat(),
        "load_time_sec": load_time,
        "gen_time_sec": gen_time,
        "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
        "backend": backend_tag,
        "multimodal": mm,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
        },
        "prompt_template": prompt_meta,
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path = output_dir / f"{model_short}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"output_path": str(out_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
