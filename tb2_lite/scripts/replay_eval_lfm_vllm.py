#!/usr/bin/env python3
"""vLLM-only replay evaluator for LFM SFT checkpoints.

This evaluator is intentionally separate from the legacy replay evaluator.
It requires model chat templates, prefers cumulative `messages` rows, records
prompt-length preflight metadata, and fails before generation if prompts cannot
fit in the configured context window.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prompt_builder import build_prompt, sanitize_name
from replay_metrics import aggregate_scores, parse_prediction, score_commands, step_bucket


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_llm(args: argparse.Namespace) -> tuple[LLM, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
    }
    sig = inspect.signature(LLM.__init__)
    if "limit_mm_per_prompt" in sig.parameters:
        kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0, "video": 0}
    if "disable_mm_preprocessor_cache" in sig.parameters:
        kwargs["disable_mm_preprocessor_cache"] = True
    if "enforce_eager" in sig.parameters and args.enforce_eager:
        kwargs["enforce_eager"] = True
    return LLM(**kwargs), kwargs


def build_prompts_and_meta(tokenizer: Any, rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    builds = [build_prompt(tokenizer, row) for row in rows]
    status_counts = Counter(build.status for build in builds)
    if status_counts.get("raw_fallback") and not args.allow_raw_fallback:
        errors = sorted({build.error for build in builds if build.error})[:5]
        raise RuntimeError(f"chat_template fallback occurred: counts={dict(status_counts)} errors={errors}")

    prompts = [build.prompt for build in builds]
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=False).input_ids)
        for prompt in prompts
    ]
    too_long = [
        {
            "idx": idx,
            "task_id": rows[idx].get("task_id"),
            "step_idx": rows[idx].get("step_idx"),
            "prompt_tokens": length,
        }
        for idx, length in enumerate(prompt_lengths)
        if length + args.max_tokens > args.max_model_len
    ]
    if too_long:
        preview = too_long[:10]
        raise RuntimeError(
            "prompt context overflow before generation: "
            f"max_model_len={args.max_model_len} max_tokens={args.max_tokens} "
            f"overflow_count={len(too_long)} examples={preview}"
        )

    sorted_lengths = sorted(prompt_lengths)
    prompt_meta = {
        "protocol": "lfm_cumulative_chat_template_v1",
        "template_status_counts": dict(status_counts),
        "template_status": "chat_template" if status_counts == {"chat_template": len(builds)} else "mixed_or_raw",
        "rank_eligible": not status_counts.get("raw_fallback"),
        "row_count": len(rows),
        "messages_rows": sum(isinstance(row.get("messages"), list) and bool(row.get("messages")) for row in rows),
        "prompt_tokens_min": min(prompt_lengths) if prompt_lengths else 0,
        "prompt_tokens_max": max(prompt_lengths) if prompt_lengths else 0,
        "prompt_tokens_p50": sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0,
        "prompt_tokens_p95": sorted_lengths[int(len(sorted_lengths) * 0.95) - 1] if sorted_lengths else 0,
        "prompt_tokens_p99": sorted_lengths[int(len(sorted_lengths) * 0.99) - 1] if sorted_lengths else 0,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
    }
    return prompts, prompt_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-short", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--eval-path", default="tb2_lite/data/replay_full.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-raw-fallback", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or sanitize_name(args.model)
    out_path = output_dir / f"{model_short}.json"
    if args.skip_if_exists and out_path.exists():
        print(json.dumps({"output_path": str(out_path), "skipped": True}, ensure_ascii=False))
        return

    rows = load_rows(Path(args.eval_path), args.limit)
    if not rows:
        raise RuntimeError(f"no eval rows loaded from {args.eval_path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts, prompt_meta = build_prompts_and_meta(tokenizer, rows, args)

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

    per_step: list[dict[str, Any]] = []
    for row, output in zip(rows, outputs, strict=True):
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
        "model_short": model_short,
        "gpu": str(args.gpu),
        "eval_path": args.eval_path,
        "timestamp": datetime.now(UTC).isoformat(),
        "load_time_sec": load_time,
        "gen_time_sec": gen_time,
        "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "dtype": args.dtype,
            "backend": "vllm",
            "tp": args.tp,
            "cpu_offload_gb": args.cpu_offload_gb,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
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
