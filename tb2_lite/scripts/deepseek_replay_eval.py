#!/usr/bin/env python3
"""DeepSeek-V4 replay evaluator using the official inference code path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_model
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[2]
DEEPSEEK_DIR = ROOT_DIR / "tb2_lite" / "deepseek_v4"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(DEEPSEEK_DIR / "inference"))
sys.path.insert(0, str(DEEPSEEK_DIR / "encoding"))

from encoding_dsv4 import encode_messages  # type: ignore  # noqa: E402
from generate import generate  # type: ignore  # noqa: E402
from model import ModelArgs, Transformer  # type: ignore  # noqa: E402
from tb2_lite.scripts.replay_metrics import (  # noqa: E402
    aggregate_scores,
    normalize_units,
    parse_prediction,
    score_commands,
    step_bucket,
)


def batched(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--eval-path", default="tb2_lite/data/replay_full.jsonl")
    parser.add_argument("--output-dir", default="tb2_lite/results")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--thinking-mode", choices=["chat", "thinking"], default="thinking")
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--token-progress-every", type=int, default=32)
    args = parser.parse_args()

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    if rank == 0:
        print(f"[DeepSeek] Loading {args.model_name}")
    load_started = time.time()
    with open(args.config_path) as handle:
        model_args = ModelArgs(**json.load(handle))
    model_args.max_batch_size = args.batch_size
    model_args.max_seq_len = args.max_model_len

    with torch.device("cuda"):
        model = Transformer(model_args)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    load_model(model, os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")
    torch.cuda.synchronize()
    load_time = time.time() - load_started

    with open(args.eval_path) as handle:
        all_rows = [json.loads(line) for line in handle]
    rows = [row for idx, row in enumerate(all_rows) if idx % args.shard_count == args.shard_index]
    prepared_rows = []
    for row in rows:
        prompt = encode_messages([{"role": "user", "content": row["prompt"]}], thinking_mode=args.thinking_mode)
        prompt_tokens = tokenizer.encode(prompt)
        row = dict(row)
        row["_prompt"] = prompt
        row["_prompt_tokens"] = prompt_tokens
        row["_prompt_len"] = len(prompt_tokens)
        prepared_rows.append(row)
    rows = sorted(prepared_rows, key=lambda row: row["_prompt_len"])
    model_short = args.model_name.split("/")[-1]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"{model_short}.status.shard{args.shard_index:02d}.json"
    if rank == 0:
        status_path.write_text(
            json.dumps(
                {
                    "model": args.model_name,
                    "model_short": model_short,
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "completed_steps": 0,
                    "total_steps": len(rows),
                    "progress_pct": 0.0,
                    "elapsed_gen_sec": 0.0,
                    "eta_sec": None,
                    "steps_per_sec": None,
                    "batch_size": args.batch_size,
                    "max_model_len": args.max_model_len,
                    "updated_at": datetime.now().isoformat(),
                    "phase": "loaded",
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if rank == 0:
        print(
            f"[DeepSeek] Generating {len(rows)} replay steps "
            f"(shard {args.shard_index + 1}/{args.shard_count}) "
            f"with batch_size={args.batch_size}"
        )
    gen_started = time.time()
    per_step: list[dict] = []
    completed_steps = 0
    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    for batch_index, batch_rows in enumerate(batched(rows, args.batch_size), start=1):
        prompt_tokens = [row["_prompt_tokens"] for row in batch_rows]

        def update_inflight_status(tokens_done: int, tokens_total: int, finished_count: int, batch_size_now: int) -> None:
            if rank != 0:
                return
            if tokens_done != 1 and tokens_done % args.token_progress_every != 0 and tokens_done < tokens_total:
                return
            batch_frac = tokens_done / max(tokens_total, 1)
            approx_completed = completed_steps + batch_frac * len(batch_rows)
            elapsed_gen = max(time.time() - gen_started, 1e-6)
            steps_per_sec = approx_completed / elapsed_gen if approx_completed > 0 else 0.0
            remaining = max(len(rows) - approx_completed, 0.0)
            eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else None
            status = {
                "model": args.model_name,
                "model_short": model_short,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "completed_steps": completed_steps,
                "approx_completed_steps": round(approx_completed, 2),
                "total_steps": len(rows),
                "progress_pct": round(completed_steps / max(len(rows), 1) * 100, 1),
                "approx_progress_pct": round(approx_completed / max(len(rows), 1) * 100, 1),
                "elapsed_gen_sec": round(elapsed_gen, 1),
                "eta_sec": round(eta_sec, 1) if eta_sec is not None else None,
                "steps_per_sec": round(steps_per_sec, 4) if steps_per_sec > 0 else None,
                "batch_size": args.batch_size,
                "max_model_len": args.max_model_len,
                "current_batch_index": batch_index,
                "total_batches": total_batches,
                "current_batch_size": len(batch_rows),
                "current_batch_tokens_done": tokens_done,
                "current_batch_tokens_total": tokens_total,
                "current_batch_finished": finished_count,
                "updated_at": datetime.now().isoformat(),
                "phase": "generating_batch",
            }
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))

        completion_tokens = generate(
            model,
            prompt_tokens,
            args.max_new_tokens,
            tokenizer.eos_token_id,
            args.temperature,
            progress_callback=update_inflight_status,
        )
        if rank != 0:
            continue
        predictions = tokenizer.batch_decode(completion_tokens)
        for row, prediction in zip(batch_rows, predictions):
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
        completed_steps += len(batch_rows)
        if completed_steps == len(rows) or completed_steps == 1 or completed_steps % args.progress_every == 0:
            elapsed_gen = max(time.time() - gen_started, 1e-6)
            steps_per_sec = completed_steps / elapsed_gen
            remaining = max(len(rows) - completed_steps, 0)
            eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else None
            status = {
                "model": args.model_name,
                "model_short": model_short,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "completed_steps": completed_steps,
                "total_steps": len(rows),
                "progress_pct": round(completed_steps / max(len(rows), 1) * 100, 1),
                "elapsed_gen_sec": round(elapsed_gen, 1),
                "eta_sec": round(eta_sec, 1) if eta_sec is not None else None,
                "steps_per_sec": round(steps_per_sec, 4),
                "batch_size": args.batch_size,
                "max_model_len": args.max_model_len,
                "updated_at": datetime.now().isoformat(),
                "phase": "generating",
            }
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
            print(
                f"[DeepSeek][Shard {args.shard_index + 1}/{args.shard_count}] "
                f"{status['completed_steps']}/{status['total_steps']} "
                f"({status['progress_pct']}%) | "
                f"elapsed {status['elapsed_gen_sec']}s | "
                f"eta {status['eta_sec']}s",
                flush=True,
            )
    torch.cuda.synchronize()
    gen_time = time.time() - gen_started

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank != 0:
        return

    aggregate = aggregate_scores(per_step)
    result = {
        "model": args.model_name,
        "model_path": args.ckpt_path,
        "model_short": model_short,
        "gpu": list(range(world_size)),
        "eval_path": args.eval_path,
        "rows_total": len(all_rows),
        "rows_in_shard": len(rows),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "timestamp": datetime.now().isoformat(),
        "load_time_sec": round(load_time, 1),
        "gen_time_sec": round(gen_time, 1),
        "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
        "sampling": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "thinking_mode": args.thinking_mode,
            "tp": world_size,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
        },
        "aggregate": aggregate,
        "per_step": per_step,
    }
    if args.shard_count > 1:
        out_path = output_dir / f"{model_short}.part{args.shard_index:02d}.json"
    else:
        out_path = output_dir / f"{model_short}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    status_path.write_text(
        json.dumps(
            {
                "model": args.model_name,
                "model_short": model_short,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "completed_steps": len(rows),
                "total_steps": len(rows),
                "progress_pct": 100.0,
                "elapsed_gen_sec": round(gen_time, 1),
                "eta_sec": 0.0,
                "steps_per_sec": round(len(rows) / max(gen_time, 1e-6), 4),
                "updated_at": datetime.now().isoformat(),
                "done": True,
                "result_path": str(out_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"[DeepSeek] DONE {model_short}")
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
