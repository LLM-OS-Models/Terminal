#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path

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


def sanitize_name(value: str) -> str:
    return value.rstrip("/").split("/")[-1].replace(" ", "-")


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_llm(args: argparse.Namespace) -> tuple[LLM, dict]:
    kwargs: dict = {
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
    if "limit_mm_per_prompt" in sig.parameters and args.language_model_only:
        kwargs["limit_mm_per_prompt"] = {"image": 0, "audio": 0, "video": 0}
    if "disable_mm_preprocessor_cache" in sig.parameters and args.language_model_only:
        kwargs["disable_mm_preprocessor_cache"] = True
    if "enforce_eager" in sig.parameters and args.enforce_eager:
        kwargs["enforce_eager"] = True
    return LLM(**kwargs), kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
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
    parser.add_argument("--thinking-mode", default="auto")
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model-impl", default="auto")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(eval_path)
    prompts = [row["prompt"] for row in rows]

    load_start = time.time()
    llm, llm_kwargs = build_llm(args)
    load_time = round(time.time() - load_start, 1)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

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
    model_short = args.model_short or sanitize_name(args.model)
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
            "dtype": args.dtype,
            "backend": args.backend,
            "model_impl": args.model_impl,
            "tp": args.tp,
            "cpu_offload_gb": args.cpu_offload_gb,
            "max_model_len": args.max_model_len,
            "language_model_only": args.language_model_only,
            "llm_kwargs": llm_kwargs,
        },
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path = output_dir / f"{model_short}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"output_path": str(out_path), "score": aggregate["next_action_score"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
