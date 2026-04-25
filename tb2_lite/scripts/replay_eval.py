#!/usr/bin/env python3
"""Stateful multi-turn replay evaluator powered by vLLM."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams


def build_prompt(tokenizer, model_name: str, prompt: str, thinking_mode: str = "auto") -> str:
    messages = [{"role": "user", "content": prompt}]
    lowered = model_name.lower()
    if "deepseek-v4" in lowered or "deepseek_v4" in lowered:
        if thinking_mode == "thinking":
            return f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>"
        return f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜></think>"
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        if "lfm" in lowered:
            return f"<|user|>\n{prompt}<|end_of_text|>\n<|assistant|>\n"
        if "gemma" in lowered:
            return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        if "qwen" in lowered:
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return f"User: {prompt}\nAssistant: "


def parse_json_blob(text: str) -> dict | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return None


def flatten_keystrokes(commands: list[dict]) -> list[str]:
    units: list[str] = []
    for command in commands:
        keystrokes = str(command.get("keystrokes", "")).replace("\r\n", "\n").replace("\r", "\n")
        pieces = [piece.strip() for piece in keystrokes.split("\n") if piece.strip()]
        if pieces:
            units.extend(pieces)
        else:
            units.append("<WAIT>")
    return units


def fallback_commands(text: str) -> list[str]:
    commands: list[str] = []
    for match in re.findall(r'"keystrokes"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        try:
            decoded = bytes(match, "utf-8").decode("unicode_escape")
        except Exception:
            decoded = match
        pieces = [piece.strip() for piece in decoded.replace("\r\n", "\n").replace("\r", "\n").split("\n") if piece.strip()]
        if pieces:
            commands.extend(pieces)
    return commands


def tokenize_command(command: str) -> list[str]:
    normalized = command.strip().lower()
    if not normalized:
        return []
    if normalized == "<wait>":
        return ["<wait>"]
    try:
        return shlex.split(normalized)
    except Exception:
        return normalized.split()


def token_f1(left: str, right: str) -> float:
    left_tokens = tokenize_command(left)
    right_tokens = tokenize_command(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    left_counter = Counter(left_tokens)
    right_counter = Counter(right_tokens)
    overlap = sum((left_counter & right_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(left_counter.values())
    recall = overlap / sum(right_counter.values())
    return 2 * precision * recall / (precision + recall)


def normalize_units(units: list[str]) -> list[str]:
    normalized: list[str] = []
    for unit in units:
        value = re.sub(r"\s+", " ", unit.strip())
        normalized.append(value if value else "<WAIT>")
    return normalized


def score_commands(pred_units: list[str], ref_units: list[str]) -> tuple[float, float, float, float]:
    pred_units = normalize_units(pred_units)
    ref_units = normalize_units(ref_units)
    if not pred_units and not ref_units:
        return 1.0, 1.0, 1.0, 1.0

    first_exact = float(bool(pred_units and ref_units and pred_units[0].lower() == ref_units[0].lower()))
    if not pred_units:
        return first_exact, 0.0, 0.0, 0.0
    if not ref_units:
        return first_exact, 0.0, 0.0, 0.0

    recall = sum(max(token_f1(ref_unit, pred_unit) for pred_unit in pred_units) for ref_unit in ref_units) / len(ref_units)
    precision = sum(max(token_f1(pred_unit, ref_unit) for ref_unit in ref_units) for pred_unit in pred_units) / len(pred_units)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return first_exact, precision, recall, f1


def step_bucket(step_idx: int) -> str:
    if step_idx <= 1:
        return "early"
    if step_idx <= 4:
        return "mid"
    return "late"


def parse_prediction(text: str) -> dict:
    payload = parse_json_blob(text)
    commands: list[str] = []
    task_complete = None
    has_analysis = False
    has_plan = False
    valid_json = payload is not None

    if payload is not None:
        raw_commands = payload.get("commands", [])
        if isinstance(raw_commands, list):
            commands = flatten_keystrokes([cmd for cmd in raw_commands if isinstance(cmd, dict)])
        task_complete = payload.get("task_complete") if isinstance(payload.get("task_complete"), bool) else None
        has_analysis = bool(payload.get("analysis"))
        has_plan = bool(payload.get("plan"))

    if not commands:
        commands = fallback_commands(text)

    return {
        "valid_json": valid_json,
        "has_analysis": has_analysis,
        "has_plan": has_plan,
        "task_complete": task_complete,
        "command_units": commands,
    }


def aggregate_scores(per_step: list[dict]) -> dict:
    total = len(per_step)
    aggregate: dict[str, object] = {
        "steps": total,
        "tasks": len({row["task_id"] for row in per_step}),
        "avg_ref_cmds": round(sum(len(row["ref_command_units"]) for row in per_step) / total, 2),
        "avg_pred_cmds": round(sum(len(row["pred_command_units"]) for row in per_step) / total, 2),
        "valid_json_pct": round(sum(row["valid_json"] for row in per_step) / total * 100, 1),
        "has_analysis_pct": round(sum(row["has_analysis"] for row in per_step) / total * 100, 1),
        "has_plan_pct": round(sum(row["has_plan"] for row in per_step) / total * 100, 1),
        "first_cmd_exact_pct": round(sum(row["first_cmd_exact"] for row in per_step) / total * 100, 1),
        "avg_command_precision": round(sum(row["command_precision"] for row in per_step) / total, 4),
        "avg_command_recall": round(sum(row["command_recall"] for row in per_step) / total, 4),
        "avg_command_f1": round(sum(row["command_f1"] for row in per_step) / total, 4),
    }

    positive_steps = [row for row in per_step if row["ref_task_complete"]]
    negative_steps = [row for row in per_step if not row["ref_task_complete"]]
    aggregate["complete_true_recall_pct"] = round(
        sum(row["pred_task_complete_true"] for row in positive_steps) / max(len(positive_steps), 1) * 100, 1
    )
    aggregate["premature_complete_rate_pct"] = round(
        sum(row["pred_task_complete_true"] for row in negative_steps) / max(len(negative_steps), 1) * 100, 1
    )

    by_bucket: dict[str, dict[str, float]] = defaultdict(lambda: {"steps": 0, "avg_command_f1": 0.0})
    by_source: dict[str, dict[str, float]] = defaultdict(lambda: {"steps": 0, "avg_command_f1": 0.0, "first_cmd_exact_pct": 0.0})
    for row in per_step:
        bucket_stats = by_bucket[row["bucket"]]
        bucket_stats["steps"] += 1
        bucket_stats["avg_command_f1"] += row["command_f1"]

        source_stats = by_source[row["source_group"]]
        source_stats["steps"] += 1
        source_stats["avg_command_f1"] += row["command_f1"]
        source_stats["first_cmd_exact_pct"] += row["first_cmd_exact"]

    aggregate["by_bucket"] = {
        key: {
            "steps": int(stats["steps"]),
            "avg_command_f1": round(stats["avg_command_f1"] / stats["steps"], 4),
        }
        for key, stats in sorted(by_bucket.items())
    }
    aggregate["by_source_group"] = {
        key: {
            "steps": int(stats["steps"]),
            "avg_command_f1": round(stats["avg_command_f1"] / stats["steps"], 4),
            "first_cmd_exact_pct": round(stats["first_cmd_exact_pct"] / stats["steps"] * 100, 1),
        }
        for key, stats in sorted(by_source.items())
    }
    aggregate["next_action_score"] = round(
        100.0
        * (
            0.7 * aggregate["avg_command_f1"]
            + 0.3 * (aggregate["first_cmd_exact_pct"] / 100.0)
        ),
        2,
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--eval-path", default="tb2_lite/data/replay_full.jsonl")
    parser.add_argument("--output-dir", default="tb2_lite/results")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--thinking-mode", choices=["auto", "chat", "thinking"], default="auto")
    parser.add_argument("--model-impl", choices=["auto", "transformers"], default="auto")
    parser.add_argument("--gdn-triton", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.eval_path) as handle:
        rows = [json.loads(line) for line in handle]

    gpu_label = os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu))
    model_label = args.model_name or args.model
    print(f"[GPU {gpu_label}] Loading {model_label} from {args.model}")
    load_started = time.time()
    llm_kwargs = dict(
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        disable_log_stats=True,
    )
    if args.model_impl != "auto":
        llm_kwargs["model_impl"] = args.model_impl
    if args.gdn_triton:
        llm_kwargs["additional_config"] = {"gdn_prefill_backend": "triton"}
    llm = LLM(model=args.model, **llm_kwargs)
    load_time = time.time() - load_started

    tokenizer = llm.get_tokenizer()
    prompts = [build_prompt(tokenizer, model_label, row["prompt"], args.thinking_mode) for row in rows]

    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"[GPU {gpu_label}] Generating {len(prompts)} replay steps")
    gen_started = time.time()
    outputs = llm.generate(prompts, sampling)
    gen_time = time.time() - gen_started

    per_step: list[dict] = []
    for row, output in zip(rows, outputs):
        prediction = output.outputs[0].text
        parsed = parse_prediction(prediction)
        first_exact, precision, recall, f1 = score_commands(parsed["command_units"], row["ref_command_units"])
        per_step.append({
            "task_id": row["task_id"],
            "sample_idx": row["sample_idx"],
            "step_idx": row["step_idx"],
            "bucket": step_bucket(row["step_idx"]),
            "source_group": row["source_group"],
            "valid_json": parsed["valid_json"],
            "has_analysis": parsed["has_analysis"],
            "has_plan": parsed["has_plan"],
            "ref_task_complete": row["ref_task_complete"],
            "pred_task_complete": parsed["task_complete"],
            "pred_task_complete_true": parsed["task_complete"] is True,
            "ref_command_units": row["ref_command_units"],
            "pred_command_units": normalize_units(parsed["command_units"]),
            "first_cmd_exact": first_exact,
            "command_precision": round(precision, 4),
            "command_recall": round(recall, 4),
            "command_f1": round(f1, 4),
            "pred_preview": prediction[:500],
        })

    aggregate = aggregate_scores(per_step)
    model_short = model_label.split("/")[-1]
    result = {
        "model": model_label,
        "model_path": args.model,
        "model_short": model_short,
        "gpu": gpu_label,
        "eval_path": args.eval_path,
        "timestamp": datetime.now().isoformat(),
        "load_time_sec": round(load_time, 1),
        "gen_time_sec": round(gen_time, 1),
        "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "thinking_mode": args.thinking_mode,
            "dtype": args.dtype,
            "model_impl": args.model_impl,
            "tp": args.tp,
            "cpu_offload_gb": args.cpu_offload_gb,
            "max_model_len": args.max_model_len,
        },
        "aggregate": aggregate,
        "per_step": per_step,
    }

    out_path = output_dir / f"{model_short}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"[GPU {gpu_label}] DONE {model_short}")
    print(f"  Load: {load_time:.1f}s | Gen: {gen_time:.1f}s | {gen_time/max(len(rows),1):.3f}s/step")
    print(
        "  "
        f"Cmd F1: {aggregate['avg_command_f1']} | "
        f"First cmd exact: {aggregate['first_cmd_exact_pct']}% | "
        f"Score: {aggregate['next_action_score']}"
    )
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
