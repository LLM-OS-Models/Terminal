#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

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


def engine_accepts_kwarg(name: str) -> bool:
    try:
        from vllm.engine.arg_utils import EngineArgs

        return name in inspect.signature(EngineArgs.__init__).parameters
    except Exception:
        return False


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
            # TB2-lite expects the model answer itself to be JSON. The official
            # Gemma 4 template supports thinking, but disabling it keeps the
            # generated JSON in the assistant content instead of a thought channel.
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


def add_prompt_length_meta(tokenizer: Any, rows: list[dict], prompts: list[str], args: argparse.Namespace, prompt_meta: dict) -> None:
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


def build_llm(args: argparse.Namespace) -> tuple[LLM, dict]:
    tokenizer_path = args.tokenizer_path or args.model
    kwargs: dict = {
        "model": args.model,
        "tokenizer": tokenizer_path,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
    }
    if engine_accepts_kwarg("limit_mm_per_prompt") and args.language_model_only:
        kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0, "video": 0}
    if engine_accepts_kwarg("language_model_only") and args.language_model_only:
        kwargs["language_model_only"] = True
    if engine_accepts_kwarg("skip_mm_profiling") and args.language_model_only:
        kwargs["skip_mm_profiling"] = True
    if engine_accepts_kwarg("disable_chunked_mm_input") and args.language_model_only:
        kwargs["disable_chunked_mm_input"] = True
    if engine_accepts_kwarg("enforce_eager") and args.enforce_eager:
        kwargs["enforce_eager"] = True
    return LLM(**kwargs), kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--model-short", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--thinking-mode", default="auto")
    parser.add_argument("--strip-thinking-history", default="auto")
    parser.add_argument("--gemma4-empty-thought-channel", default="auto")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model-impl", default="auto")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-raw-fallback", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or sanitize_name(args.model)
    out_path = output_dir / f"{model_short}.json"
    if args.skip_if_exists and out_path.exists():
        print(json.dumps({"output_path": str(out_path), "skipped": True}, ensure_ascii=False))
        return
    rows = load_rows(eval_path, args.limit)

    from transformers import AutoTokenizer as _AT
    tokenizer_path = args.tokenizer_path or args.model
    tokenizer = _AT.from_pretrained(tokenizer_path, trust_remote_code=True)
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

    load_start = time.time()
    llm, llm_kwargs = build_llm(args)
    load_time = round(time.time() - load_start, 1)

    sampling_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.min_p > 0:
        sampling_kwargs["min_p"] = args.min_p
    sampling = SamplingParams(**sampling_kwargs)

    gen_start = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)
    gen_time = round(time.time() - gen_start, 1)

    per_step: list[dict] = []
    for row, output in zip(rows, outputs):
        pred_text = output.outputs[0].text if output.outputs else ""
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
            "dtype": args.dtype,
            "backend": args.backend,
            "model_impl": args.model_impl,
            "tp": args.tp,
            "cpu_offload_gb": args.cpu_offload_gb,
            "max_model_len": args.max_model_len,
            "language_model_only": args.language_model_only,
            "llm_kwargs": llm_kwargs,
        },
        "prompt_template": prompt_meta,
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"output_path": str(out_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
