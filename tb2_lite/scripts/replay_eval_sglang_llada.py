#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

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


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def add_prompt_length_meta(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    prompts: list[str],
    args: argparse.Namespace,
    prompt_meta: dict[str, Any],
) -> None:
    lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
    too_long = [
        {
            "idx": idx,
            "task_id": rows[idx].get("task_id"),
            "step_idx": rows[idx].get("step_idx"),
            "prompt_tokens": length,
        }
        for idx, length in enumerate(lengths)
        if length + args.max_new_tokens > args.max_model_len
    ]
    if too_long:
        raise RuntimeError(
            "prompt context overflow before generation: "
            f"max_model_len={args.max_model_len} max_new_tokens={args.max_new_tokens} "
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
            "max_new_tokens": args.max_new_tokens,
        }
    )


def extract_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        if not data:
            return ""
        return extract_text(data[0])
    if not isinstance(data, dict):
        return str(data)
    if isinstance(data.get("text"), str):
        return data["text"]
    if isinstance(data.get("output"), str):
        return data["output"]
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            if isinstance(first.get("text"), str):
                return first["text"]
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return json.dumps(data, ensure_ascii=False)


def post_generate(args: argparse.Namespace, prompt: str) -> str:
    url = args.base_url.rstrip("/") + "/generate"
    custom_params = {
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "threshold": args.threshold,
        "editing_threshold": args.editing_threshold,
        "max_post_steps": args.max_post_steps,
    }
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "custom_params": custom_params,
        },
    }
    if args.ignore_eos:
        payload["sampling_params"]["ignore_eos"] = True
    last_error: str | None = None
    for attempt in range(args.retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=args.request_timeout)
            if response.status_code == 200:
                return extract_text(response.json())
            last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
        except Exception as exc:
            last_error = repr(exc)
        if attempt < args.retries:
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(last_error or "generate request failed")


def score_one(row: dict[str, Any], pred_text: str) -> dict[str, Any]:
    pred = parse_prediction(pred_text)
    ref = parse_prediction(row["ref_raw"])
    first_exact, precision, recall, f1 = score_commands(
        pred["command_units"], ref["command_units"]
    )
    pred_complete_true = bool(pred["task_complete"]) and bool(ref["task_complete"])
    return {
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


def write_result(
    out_path: Path,
    args: argparse.Namespace,
    model_short: str,
    prompt_meta: dict[str, Any],
    per_step: list[dict[str, Any]],
    gen_start: float,
    complete: bool,
    total_steps: int,
) -> None:
    gen_time = round(time.time() - gen_start, 1)
    aggregate = aggregate_scores(per_step)
    result = {
        "model": args.model_short or args.model_path or model_short,
        "model_path": args.model_path or args.base_url,
        "model_short": model_short,
        "gpu": args.gpu,
        "eval_path": args.eval_path,
        "timestamp": datetime.utcnow().isoformat(),
        "gen_time_sec": gen_time,
        "avg_sec_per_step": round(gen_time / max(len(per_step), 1), 3),
        "backend": "sglang-generate-llada",
        "complete": complete,
        "generated_steps": len(per_step),
        "total_steps": total_steps,
        "sampling": {
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "threshold": args.threshold,
            "editing_threshold": args.editing_threshold,
            "max_post_steps": args.max_post_steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
        },
        "prompt_template": prompt_meta,
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-short", default="")
    parser.add_argument("--gpu", default="sglang-tp8")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--editing-threshold", type=float, default=0.0)
    parser.add_argument("--max-post-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or sanitize_name(args.model_path or args.tokenizer_path)
    out_path = output_dir / f"{model_short}.json"
    if args.skip_if_exists and out_path.exists():
        print(json.dumps({"output_path": str(out_path), "skipped": True}, ensure_ascii=False))
        return

    from transformers import AutoTokenizer

    rows = load_rows(Path(args.eval_path), args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    prompts, prompt_meta = build_prompts(
        tokenizer,
        rows,
        model_name=args.model_path or args.tokenizer_path,
    )
    if args.prompt_suffix:
        prompts = [append_prompt_suffix(prompt, args.prompt_suffix) for prompt in prompts]
        prompt_meta = {**prompt_meta, "llada_prompt_suffix": args.prompt_suffix}
    add_prompt_length_meta(tokenizer, rows, prompts, args, prompt_meta)

    gen_start = time.time()
    per_step: list[dict[str, Any]] = []
    for idx, (row, prompt) in enumerate(zip(rows, prompts), start=1):
        pred_text = post_generate(args, prompt)
        per_step.append(score_one(row, pred_text))
        if idx % args.progress_every == 0 or idx == len(rows):
            elapsed = time.time() - gen_start
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (len(rows) - idx) / rate if rate > 0 else 0.0
            print(
                f"[{idx}/{len(rows)}] elapsed={elapsed:.1f}s rate={rate:.4f}/s eta={eta:.1f}s",
                flush=True,
            )
        if args.save_every and (idx % args.save_every == 0 or idx == len(rows)):
            write_result(
                out_path,
                args,
                model_short,
                prompt_meta,
                per_step,
                gen_start,
                complete=(idx == len(rows)),
                total_steps=len(rows),
            )

    write_result(
        out_path,
        args,
        model_short,
        prompt_meta,
        per_step,
        gen_start,
        complete=True,
        total_steps=len(rows),
    )
    aggregate = aggregate_scores(per_step)
    print(json.dumps({"output_path": str(out_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
