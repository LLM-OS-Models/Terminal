#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from llama_cpp import Llama

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prompt_builder import build_prompts, sanitize_name
from replay_metrics import aggregate_scores, parse_prediction, score_commands, step_bucket


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


def load_tokenizer(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as exc:
        if "deepseek_v4" not in str(exc).lower():
            raise
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def build_llama(args: argparse.Namespace) -> Llama:
    common = {
        "repo_id": args.repo_id,
        "filename": args.filename,
        "n_ctx": args.max_model_len,
        "n_batch": args.n_batch,
        "n_ubatch": args.n_ubatch,
        "n_gpu_layers": args.n_gpu_layers,
        "n_threads": args.n_threads,
        "n_threads_batch": args.n_threads_batch,
        "flash_attn": args.flash_attn,
        "offload_kqv": args.offload_kqv,
        "verbose": bool(args.verbose_llama),
    }
    if args.cache_dir:
        common["cache_dir"] = args.cache_dir
    return Llama.from_pretrained(**common)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-short", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--thinking-mode", default="off")
    parser.add_argument("--strip-thinking-history", default="on")
    parser.add_argument("--gemma4-empty-thought-channel", default="auto")
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-batch", type=int, default=1024)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-threads-batch", type=int, default=None)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--no-offload-kqv", dest="offload_kqv", action="store_false")
    parser.set_defaults(offload_kqv=True)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-raw-fallback", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--verbose-llama", action="store_true")
    args = parser.parse_args()
    args.model = args.repo_id

    eval_path = Path(args.eval_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or sanitize_name(f"{args.repo_id}_{args.filename}")
    out_path = output_dir / f"{model_short}.json"
    if args.skip_if_exists and out_path.exists():
        print(json.dumps({"output_path": str(out_path), "skipped": True}, ensure_ascii=False))
        return

    rows = load_rows(eval_path, args.limit)

    tokenizer = load_tokenizer(args.tokenizer_path)
    chat_template_kwargs = parse_chat_template_kwargs(args, tokenizer)
    prompt_options = parse_prompt_options(args, tokenizer)
    prompts, prompt_meta = build_prompts(
        tokenizer,
        rows,
        model_name=args.repo_id,
        chat_template_kwargs=chat_template_kwargs,
        prompt_options=prompt_options,
    )
    if prompt_meta.get("template_status_counts", {}).get("raw_fallback") and not args.allow_raw_fallback:
        raise RuntimeError(f"raw prompt fallback occurred: {prompt_meta}")
    add_prompt_length_meta(tokenizer, rows, prompts, args, prompt_meta)

    load_start = time.time()
    llm = build_llama(args)
    load_time = round(time.time() - load_start, 1)

    gen_start = time.time()
    per_step: list[dict[str, Any]] = []
    for idx, (row, prompt) in enumerate(zip(rows, prompts), start=1):
        generation = llm(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repetition_penalty,
            min_p=args.min_p,
        )
        pred_text = generation["choices"][0].get("text", "") if generation.get("choices") else ""
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
        if idx == 1 or idx % 10 == 0 or idx == len(rows):
            print(f"progress {idx}/{len(rows)}", flush=True)

    gen_time = round(time.time() - gen_start, 1)
    aggregate = aggregate_scores(per_step)
    result = {
        "model": model_short,
        "model_path": f"{args.repo_id}:{args.filename}",
        "lora_path": None,
        "model_short": model_short,
        "gpu": str(args.gpu),
        "eval_path": str(eval_path),
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
            "dtype": "gguf",
            "backend": "llama_cpp",
            "tp": 1,
            "max_model_len": args.max_model_len,
            "repo_id": args.repo_id,
            "filename": args.filename,
            "tokenizer_path": args.tokenizer_path,
            "n_gpu_layers": args.n_gpu_layers,
            "n_batch": args.n_batch,
            "n_ubatch": args.n_ubatch,
            "n_threads": args.n_threads,
            "n_threads_batch": args.n_threads_batch,
            "flash_attn": args.flash_attn,
            "offload_kqv": args.offload_kqv,
        },
        "prompt_template": prompt_meta,
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(out_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
