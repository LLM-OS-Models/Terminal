#!/usr/bin/env python3
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

from prompt_builder import build_prompts, sanitize_name
from replay_metrics import aggregate_scores, parse_prediction, score_commands, step_bucket


def append_prompt_suffix(prompt: str, suffix: str) -> str:
    if not suffix:
        return prompt
    marker = "<|role_end|><role>ASSISTANT</role>"
    if prompt.endswith(marker):
        return prompt[: -len(marker)] + "\n\n" + suffix.strip() + marker
    return prompt + "\n\n" + suffix.strip()


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def score_predictions(rows: list[dict], predictions: list[str]) -> list[dict]:
    per_step: list[dict] = []
    for row, pred_text in zip(rows, predictions):
        pred = parse_prediction(pred_text)
        ref = parse_prediction(row["ref_raw"])
        first_exact, precision, recall, f1 = score_commands(
            pred["command_units"], ref["command_units"]
        )
        pred_complete_true = bool(pred["task_complete"]) and bool(ref["task_complete"])
        per_step.append(
            {
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
    return per_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-short", default="")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gen-length", type=int, default=1024)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--editing-threshold", type=float, default=0.0)
    parser.add_argument("--max-post-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--decode-output",
        choices=("full", "new"),
        default="full",
        help="LLaDA generate() returns generated tokens directly; use full for model-card behavior.",
    )
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or sanitize_name(args.model)
    out_path = output_dir / f"{model_short}.json"
    if args.skip_if_exists and out_path.exists():
        print(json.dumps({"output_path": str(out_path), "skipped": True}, ensure_ascii=False))
        return

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    rows = load_rows(Path(args.eval_path), args.limit)
    load_start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts, prompt_meta = build_prompts(tokenizer, rows, model_name=args.model)
    if args.prompt_suffix:
        prompts = [append_prompt_suffix(prompt, args.prompt_suffix) for prompt in prompts]
        prompt_meta = {
            **prompt_meta,
            "llada_prompt_suffix": args.prompt_suffix,
        }
    lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
    too_long = [
        {
            "idx": idx,
            "task_id": rows[idx].get("task_id"),
            "step_idx": rows[idx].get("step_idx"),
            "prompt_tokens": length,
        }
        for idx, length in enumerate(lengths)
        if length + args.gen_length > args.max_model_len
    ]
    if too_long:
        raise RuntimeError(
            "prompt context overflow before generation: "
            f"max_model_len={args.max_model_len} gen_length={args.gen_length} "
            f"overflow_count={len(too_long)} examples={too_long[:10]}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
    )
    model = model.to(torch_dtype)
    model.eval()
    load_time = round(time.time() - load_start, 1)
    print(f"Model loaded in {load_time}s", flush=True)

    gen_start = time.time()
    predictions: list[str] = []
    for idx, prompt in enumerate(prompts):
        input_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                eos_early_stop=True,
                gen_length=args.gen_length,
                block_length=args.block_length,
                threshold=args.threshold,
                editing_threshold=args.editing_threshold,
                max_post_steps=args.max_post_steps,
                temperature=args.temperature,
            )
        if args.decode_output == "new":
            decode_ids = output_ids[0, input_ids.shape[1] :]
        else:
            decode_ids = output_ids[0]
        pred_text = tokenizer.decode(decode_ids, skip_special_tokens=True)
        predictions.append(pred_text)
        if (idx + 1) % args.progress_every == 0 or (idx + 1) == len(prompts):
            elapsed = time.time() - gen_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"[{idx + 1}/{len(prompts)}] elapsed={elapsed:.1f}s rate={rate:.4f}/s",
                flush=True,
            )
        if args.save_every and ((idx + 1) % args.save_every == 0 or (idx + 1) == len(prompts)):
            partial_per_step = score_predictions(rows[: len(predictions)], predictions)
            partial_gen_time = round(time.time() - gen_start, 1)
            partial_aggregate = aggregate_scores(partial_per_step)
            partial_result = {
                "model": args.model_short or args.model,
                "model_path": args.model,
                "model_short": model_short,
                "gpu": "transformers-device_map-auto",
                "eval_path": args.eval_path,
                "timestamp": datetime.utcnow().isoformat(),
                "load_time_sec": load_time,
                "gen_time_sec": partial_gen_time,
                "avg_sec_per_step": round(partial_gen_time / max(len(partial_per_step), 1), 3),
                "backend": "transformers-llada",
                "complete": len(partial_per_step) == len(rows),
                "generated_steps": len(partial_per_step),
                "total_steps": len(rows),
                "sampling": {
                    "dtype": args.dtype,
                    "max_model_len": args.max_model_len,
                    "gen_length": args.gen_length,
                    "block_length": args.block_length,
                    "threshold": args.threshold,
                    "editing_threshold": args.editing_threshold,
                    "max_post_steps": args.max_post_steps,
                    "temperature": args.temperature,
                    "decode_output": args.decode_output,
                },
                "prompt_template": prompt_meta,
                "aggregate": partial_aggregate,
                "per_step": partial_per_step,
            }
            out_path.write_text(json.dumps(partial_result, ensure_ascii=False, indent=2))

    gen_time = round(time.time() - gen_start, 1)
    per_step = score_predictions(rows, predictions)

    aggregate = aggregate_scores(per_step)
    result = {
        "model": args.model_short or args.model,
        "model_path": args.model,
        "model_short": model_short,
        "gpu": "transformers-device_map-auto",
        "eval_path": args.eval_path,
        "timestamp": datetime.utcnow().isoformat(),
        "load_time_sec": load_time,
        "gen_time_sec": gen_time,
        "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
        "backend": "transformers-llada",
        "complete": True,
        "generated_steps": len(per_step),
        "total_steps": len(rows),
        "sampling": {
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "threshold": args.threshold,
            "editing_threshold": args.editing_threshold,
            "max_post_steps": args.max_post_steps,
            "temperature": args.temperature,
            "decode_output": args.decode_output,
        },
        "prompt_template": prompt_meta,
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"output_path": str(out_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
