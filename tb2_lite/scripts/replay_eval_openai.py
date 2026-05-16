#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
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

from replay_metrics import (
    aggregate_scores,
    parse_prediction,
    score_commands,
    step_bucket,
)
from prompt_builder import build_prompts, sanitize_name


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
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


def load_tokenizer(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:
        if "Step-3.5-Flash" in tokenizer_path or "Step_hyphen_3_dot_5" in str(exc):
            from huggingface_hub import snapshot_download

            local_path = Path(snapshot_download(tokenizer_path, local_files_only=True))
            return PreTrainedTokenizerFast(
                tokenizer_file=str(local_path / "tokenizer.json"),
                tokenizer_config_file=str(local_path / "tokenizer_config.json"),
            )
        if "deepseek_v4" not in str(exc).lower():
            raise
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def parse_chat_template_kwargs(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    mode = str(args.thinking_mode).lower()
    if mode in {"on", "true", "1", "yes", "enabled"}:
        kwargs["enable_thinking"] = True
    elif mode in {"off", "false", "0", "no", "disabled"}:
        kwargs["enable_thinking"] = False
    else:
        identity = f"{args.model} {args.tokenizer_path or ''} {getattr(tokenizer, 'name_or_path', '')}".lower()
        if "gemma-4" in identity:
            kwargs["enable_thinking"] = False
    return kwargs


def is_gemma4_model(args: argparse.Namespace, tokenizer: Any) -> bool:
    identity = f"{args.model} {args.tokenizer_path or ''} {getattr(tokenizer, 'name_or_path', '')}".lower()
    return "gemma-4" in identity


def gemma4_has_nonthinking_channel(args: argparse.Namespace, tokenizer: Any) -> bool:
    identity = f"{args.model} {args.tokenizer_path or ''} {getattr(tokenizer, 'name_or_path', '')}".lower()
    return "gemma-4-26b" in identity or "gemma-4-31b" in identity


def parse_prompt_options(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    gemma4 = is_gemma4_model(args, tokenizer)

    strip_mode = str(args.strip_thinking_history).lower()
    if strip_mode in {"on", "true", "1", "yes", "enabled"}:
        strip_thinking_history = True
    elif strip_mode in {"off", "false", "0", "no", "disabled"}:
        strip_thinking_history = False
    else:
        strip_thinking_history = gemma4

    channel_mode = str(args.gemma4_empty_thought_channel).lower()
    if channel_mode in {"on", "true", "1", "yes", "enabled"}:
        empty_thought_channel = True
    elif channel_mode in {"off", "false", "0", "no", "disabled"}:
        empty_thought_channel = False
    else:
        empty_thought_channel = gemma4 and gemma4_has_nonthinking_channel(args, tokenizer)

    return {
        "strip_thinking_history": strip_thinking_history,
        "gemma4_empty_thought_channel": empty_thought_channel,
        "use_gemma4_patched_template": gemma4 and (strip_thinking_history or empty_thought_channel),
    }


def add_prompt_length_meta(
    tokenizer: Any,
    rows: list[dict],
    prompts: list[str],
    args: argparse.Namespace,
    prompt_meta: dict,
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
            "messages_rows": sum(isinstance(row.get("messages"), list) and bool(row.get("messages")) for row in rows),
            "prompt_tokens_min": min(lengths) if lengths else 0,
            "prompt_tokens_max": max(lengths) if lengths else 0,
            "prompt_tokens_p50": sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0,
            "prompt_tokens_p95": sorted_lengths[int(len(sorted_lengths) * 0.95) - 1] if sorted_lengths else 0,
            "prompt_tokens_p99": sorted_lengths[int(len(sorted_lengths) * 0.99) - 1] if sorted_lengths else 0,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
        }
    )


def post_completion(
    base_url: str,
    model: str,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    url = base_url.rstrip("/") + "/completions"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.min_p > 0:
        payload["min_p"] = args.min_p
    if args.extra_body:
        payload.update(args.extra_body)

    last_error: str | None = None
    for attempt in range(args.retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=args.request_timeout)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["text"]
            last_error = f"HTTP {response.status_code}: {response.text[:1000]}"
        except Exception as exc:
            last_error = repr(exc)
        if attempt < args.retries:
            time.sleep(min(2**attempt, 15))
    raise RuntimeError(last_error or "completion request failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--model-short", default="")
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument(
        "--extra-body-json",
        default="",
        help="JSON object merged into every OpenAI completion request.",
    )
    parser.add_argument("--thinking-mode", default="auto")
    parser.add_argument("--strip-thinking-history", default="auto")
    parser.add_argument("--gemma4-empty-thought-channel", default="auto")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-raw-fallback", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()
    if args.extra_body_json:
        args.extra_body = json.loads(args.extra_body_json)
        if not isinstance(args.extra_body, dict):
            raise TypeError("--extra-body-json must decode to a JSON object")
    else:
        args.extra_body = {}

    eval_path = Path(args.eval_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or sanitize_name(args.model)
    out_path = output_dir / f"{model_short}.json"
    if args.skip_if_exists and out_path.exists():
        print(json.dumps({"output_path": str(out_path), "skipped": True}, ensure_ascii=False))
        return

    rows = load_rows(eval_path, args.limit)
    tokenizer_path = args.tokenizer_path or args.model
    tokenizer = load_tokenizer(tokenizer_path)
    chat_template_kwargs = parse_chat_template_kwargs(args, tokenizer)
    prompt_options = parse_prompt_options(args, tokenizer)
    prompts, prompt_meta = build_prompts(
        tokenizer,
        rows,
        model_name=args.model,
        chat_template_kwargs=chat_template_kwargs,
        prompt_options=prompt_options,
    )
    if prompt_meta.get("template_status_counts", {}).get("raw_fallback") and not args.allow_raw_fallback:
        raise RuntimeError(f"raw prompt fallback occurred: {prompt_meta}")
    add_prompt_length_meta(tokenizer, rows, prompts, args, prompt_meta)

    gen_start = time.time()
    pred_texts: list[str] = [""] * len(prompts)
    errors: list[dict[str, Any]] = []
    base_urls = [url.rstrip("/") + "/v1" if not url.rstrip("/").endswith("/v1") else url.rstrip("/") for url in args.base_url]

    def run_one(idx: int) -> tuple[int, str]:
        base_url = base_urls[idx % len(base_urls)]
        return idx, post_completion(base_url, args.model, prompts[idx], args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(run_one, idx): idx for idx in range(len(prompts))}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                done_idx, text = future.result()
                pred_texts[done_idx] = text
            except Exception as exc:
                errors.append({"idx": idx, "task_id": rows[idx].get("task_id"), "error": repr(exc)})

    if errors:
        raise RuntimeError(f"completion errors: {errors[:10]} total={len(errors)}")

    gen_time = round(time.time() - gen_start, 1)
    per_step: list[dict] = []
    for row, pred_text in zip(rows, pred_texts):
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

    aggregate = aggregate_scores(per_step)
    result = {
        "model": args.model_short or args.model,
        "model_path": args.model,
        "lora_path": None,
        "model_short": model_short,
        "gpu": "vllm-openai",
        "eval_path": str(eval_path),
        "timestamp": datetime.utcnow().isoformat(),
        "load_time_sec": None,
        "gen_time_sec": gen_time,
        "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "thinking_mode": args.thinking_mode,
            "strip_thinking_history": args.strip_thinking_history,
            "gemma4_empty_thought_channel": args.gemma4_empty_thought_channel,
            "backend": "vllm-openai",
            "max_model_len": args.max_model_len,
            "base_urls": base_urls,
            "concurrency": args.concurrency,
        },
        "prompt_template": prompt_meta,
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"output_path": str(out_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
