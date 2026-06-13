#!/usr/bin/env python3
"""TB2-lite replay evaluation for sapientinc/HRM-Text PrefixLM checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prompt_builder import row_messages, sanitize_name
from replay_metrics import aggregate_scores, parse_prediction, score_commands, step_bucket


CONDITION_TOKENS = {
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
}

TRUNCATION_MARKER = "\n\n[...middle of prompt truncated to fit HRM-Text context...]\n\n"


def load_rows(
    path: Path,
    limit: int | None,
    num_shards: int,
    shard_index: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    total_rows = len(rows)
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]
    return rows, total_rows


def condition_prefix(condition: str) -> str:
    pieces: list[str] = []
    for item in condition.split(","):
        item = item.strip()
        if not item:
            continue
        pieces.append(CONDITION_TOKENS.get(item, item))
    if not pieces:
        raise ValueError("condition resolved to an empty prefix")
    return "".join(pieces)


def flatten_messages(row: dict[str, Any]) -> str:
    if isinstance(row.get("prompt"), str) and row["prompt"].strip():
        return row["prompt"].rstrip()
    parts: list[str] = []
    for message in row_messages(row):
        role = message["role"].upper()
        parts.append(f"{role}:\n{message['content'].rstrip()}")
    return "\n\n".join(parts).rstrip()


def prepare_local_hrm_prompts(
    ckpt: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    tokenizer = ckpt.tokenizer
    max_prompt_tokens = args.max_model_len - args.max_tokens
    if max_prompt_tokens <= 0:
        raise ValueError(
            "--max-model-len must be larger than --max-tokens for local HRM evaluation"
        )

    empty_prompt_tokens = ckpt.tokenize_prompt(args.condition, "")
    available_body = max_prompt_tokens - int(empty_prompt_tokens.size)
    if available_body <= 0:
        raise ValueError(
            "HRM condition/wrapper leaves no room for prompt body: "
            f"max_model_len={args.max_model_len} max_tokens={args.max_tokens}"
        )

    marker_ids = tokenizer(TRUNCATION_MARKER, add_special_tokens=False).input_ids
    marker_len = len(marker_ids)
    prepared: list[str] = []
    body_lengths: list[int] = []
    kept_lengths: list[int] = []
    truncated = 0

    for row in rows:
        body = flatten_messages(row)
        body_ids = tokenizer(body, add_special_tokens=False).input_ids
        body_lengths.append(len(body_ids))
        if len(body_ids) <= available_body:
            prepared.append(body)
            kept_lengths.append(len(body_ids))
            continue

        truncated += 1
        available_without_marker = max(available_body - marker_len, 1)
        if available_without_marker < args.head_tokens + args.min_tail_tokens:
            keep_head = max(available_without_marker // 2, 1)
        else:
            keep_head = min(args.head_tokens, available_without_marker - 1)
        keep_tail = max(available_without_marker - keep_head, 1)
        trimmed_ids = body_ids[:keep_head] + marker_ids + body_ids[-keep_tail:]
        prepared.append(tokenizer.decode(trimmed_ids, skip_special_tokens=False))
        kept_lengths.append(keep_head + keep_tail)

    total = max(len(rows), 1)
    return prepared, {
        "local_prompt_budget_tokens": max_prompt_tokens,
        "local_wrapper_tokens": int(empty_prompt_tokens.size),
        "local_available_body_tokens": available_body,
        "local_truncated_prompts": truncated,
        "local_truncated_pct": round(truncated / total * 100.0, 1),
        "local_max_body_tokens_original": max(body_lengths or [0]),
        "local_avg_body_tokens_original": round(sum(body_lengths) / total, 1),
        "local_avg_body_tokens_kept": round(sum(kept_lengths) / total, 1),
    }


def encode_prompt(
    tokenizer: Any,
    body: str,
    condition: str,
    max_model_len: int,
    max_new_tokens: int,
    head_tokens: int,
    min_tail_tokens: int,
) -> tuple[list[int], dict[str, Any]]:
    prefix = f"<|im_start|>{condition_prefix(condition)}"
    suffix = "\nASSISTANT:\n<|im_end|>"

    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    body_ids = tokenizer(body, add_special_tokens=False).input_ids

    max_prompt_tokens = max_model_len - max_new_tokens
    available_body = max_prompt_tokens - len(prefix_ids) - len(suffix_ids)
    if available_body <= 0:
        raise ValueError(
            "max_model_len/max_new_tokens leaves no room for prompt body: "
            f"max_model_len={max_model_len} max_new_tokens={max_new_tokens}"
        )

    meta = {
        "body_tokens_original": len(body_ids),
        "body_tokens_kept": len(body_ids),
        "truncated": False,
    }
    if len(body_ids) <= available_body:
        return prefix_ids + body_ids + suffix_ids, meta

    marker_ids = tokenizer(TRUNCATION_MARKER, add_special_tokens=False).input_ids
    available_body_without_marker = available_body - len(marker_ids)
    if available_body_without_marker <= 0:
        raise ValueError("truncation marker does not fit in the prompt budget")

    if available_body_without_marker < head_tokens + min_tail_tokens:
        keep_head = max(available_body_without_marker // 2, 1)
    else:
        keep_head = head_tokens
    keep_tail = max(available_body_without_marker - keep_head, 1)
    trimmed_body_ids = body_ids[:keep_head] + marker_ids + body_ids[-keep_tail:]
    meta.update(
        {
            "body_tokens_kept": keep_head + keep_tail,
            "truncated": True,
            "head_tokens": keep_head,
            "tail_tokens": keep_tail,
            "dropped_tokens": len(body_ids) - keep_head - keep_tail,
        }
    )
    return prefix_ids + trimmed_body_ids + suffix_ids, meta


def score_predictions(rows: list[dict[str, Any]], predictions: list[str]) -> list[dict[str, Any]]:
    per_step: list[dict[str, Any]] = []
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


def score_completed_predictions(
    rows: list[dict[str, Any]],
    predictions: list[str],
    completed_indices: set[int],
) -> list[dict[str, Any]]:
    ordered = sorted(completed_indices)
    return score_predictions([rows[idx] for idx in ordered], [predictions[idx] for idx in ordered])


def build_result(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    total_input_steps: int,
    per_step: list[dict[str, Any]],
    load_time: float,
    gen_time: float,
    prompt_meta: dict[str, Any],
    complete: bool,
) -> dict[str, Any]:
    aggregate = aggregate_scores(per_step)
    model_short = args.model_short or sanitize_name(args.model)
    return {
        "model": args.model_short or args.model,
        "model_path": args.model,
        "lora_path": str(args.lora_path) if getattr(args, "lora_path", None) else None,
        "model_short": model_short,
        "gpu": args.gpu,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "eval_path": str(args.eval_path),
        "timestamp": datetime.utcnow().isoformat(),
        "load_time_sec": round(load_time, 1),
        "gen_time_sec": round(gen_time, 1),
        "avg_sec_per_step": round(gen_time / max(len(per_step), 1), 3),
        "backend": "transformers-hrm-text-prefixlm",
        "complete": complete,
        "generated_steps": len(per_step),
        "total_steps": len(rows),
        "total_input_steps": total_input_steps,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "sampling": {
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "condition": args.condition,
            "head_tokens": args.head_tokens,
            "min_tail_tokens": args.min_tail_tokens,
            "batch_size": args.batch_size,
        },
        "prompt_template": prompt_meta,
        "aggregate": aggregate,
        "per_step": per_step,
    }


def resolve_lora_adapter(lora_path: Path, tag: str) -> tuple[Path, Path | None, str]:
    if lora_path.is_file():
        return lora_path, None, lora_path.stem.removeprefix("lora_")
    if not tag:
        latest = lora_path / "latest_lora.txt"
        tag = latest.read_text(encoding="utf-8").strip() if latest.exists() else "epoch_1"
    adapter = lora_path / f"lora_{tag}.pt"
    info = lora_path / f"lora_{tag}_info.json"
    if not adapter.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter}")
    return adapter, info if info.exists() else None, tag


def run_local_lora_eval(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    total_input_steps: int,
    out_path: Path,
) -> None:
    hrm_root = Path(args.hrm_text_root).resolve()
    if str(hrm_root) not in sys.path:
        sys.path.insert(0, str(hrm_root))

    from simple_inference_engine import inference_generate, inference_load_checkpoint
    from models.lora import inject_lora, load_lora_adapter

    adapter_path, info_path, tag = resolve_lora_adapter(Path(args.lora_path), args.lora_tag)
    info: dict[str, Any] = {}
    if info_path is not None:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    lora_cfg = info.get("lora", {})

    load_start = time.time()
    ckpt = inference_load_checkpoint(
        ckpt_path=args.model,
        ckpt_epoch=args.ckpt_epoch,
        ckpt_step=args.ckpt_step,
        ckpt_use_ema=args.ckpt_use_ema,
        device="cuda",
    )
    matched_modules = inject_lora(
        ckpt.model,
        target_suffixes=lora_cfg.get("target_suffixes", ["gqkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"]),
        rank=int(lora_cfg.get("rank", 16)),
        alpha=float(lora_cfg.get("alpha", 32.0)),
        dropout=0.0,
    )
    load_stats = load_lora_adapter(ckpt.model, adapter_path)
    ckpt.model.eval()
    load_time = time.time() - load_start

    prompt_meta = {
        "template_status": "hrm_text_prefixlm_local_lora",
        "rank_eligible": True,
        "condition": args.condition,
        "lora_tag": tag,
        "lora_adapter_path": str(adapter_path),
        "lora_info_path": str(info_path) if info_path else None,
        "lora_load_stats": load_stats,
        "matched_lora_modules": len(matched_modules),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
    }

    gen_start = time.time()
    predictions = [""] * len(rows)
    completed_indices: set[int] = set()
    prompts, trim_meta = prepare_local_hrm_prompts(ckpt, rows, args)
    prompt_meta.update(trim_meta)
    iterator = ((idx, (args.condition, prompt)) for idx, prompt in enumerate(prompts))
    for done, (idx, generated_text) in enumerate(
        inference_generate(
            ckpt,
            iterator,
            max_tokens=args.max_model_len,
            max_generation=args.max_tokens,
            batch_size=args.batch_size,
            temp=args.temperature,
        ),
        start=1,
    ):
        predictions[idx] = generated_text
        completed_indices.add(idx)
        if done % args.progress_every == 0 or done == len(rows):
            elapsed = time.time() - gen_start
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"[{done}/{len(rows)}] elapsed={elapsed:.1f}s rate={rate:.4f}/s", flush=True)
        if args.save_every and (done % args.save_every == 0 or done == len(rows)):
            partial_per_step = score_completed_predictions(rows, predictions, completed_indices)
            partial_result = build_result(
                args=args,
                rows=rows,
                total_input_steps=total_input_steps,
                per_step=partial_per_step,
                load_time=load_time,
                gen_time=time.time() - gen_start,
                prompt_meta=prompt_meta,
                complete=done == len(rows),
            )
            out_path.write_text(json.dumps(partial_result, ensure_ascii=False, indent=2))

    gen_time = time.time() - gen_start
    per_step = score_predictions(rows, predictions)
    result = build_result(
        args=args,
        rows=rows,
        total_input_steps=total_input_steps,
        per_step=per_step,
        load_time=load_time,
        gen_time=gen_time,
        prompt_meta=prompt_meta,
        complete=True,
    )
    result["backend"] = "kohrm-local-lora-prefixlm"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    score = round(100.0 * result["aggregate"]["avg_command_f1"], 2)
    print(json.dumps({"output_path": str(out_path), "score": score}, ensure_ascii=False))


def run_local_hrm_eval(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    total_input_steps: int,
    out_path: Path,
) -> None:
    hrm_root = Path(args.hrm_text_root).resolve()
    if str(hrm_root) not in sys.path:
        sys.path.insert(0, str(hrm_root))

    from simple_inference_engine import inference_generate, inference_load_checkpoint

    load_start = time.time()
    ckpt = inference_load_checkpoint(
        ckpt_path=args.model,
        ckpt_epoch=args.ckpt_epoch,
        ckpt_step=args.ckpt_step,
        ckpt_use_ema=args.ckpt_use_ema,
        device="cuda",
    )
    ckpt.model.eval()
    load_time = time.time() - load_start

    prompt_meta = {
        "template_status": "hrm_text_prefixlm_local_checkpoint",
        "rank_eligible": True,
        "condition": args.condition,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
    }

    gen_start = time.time()
    predictions = [""] * len(rows)
    completed_indices: set[int] = set()
    prompts, trim_meta = prepare_local_hrm_prompts(ckpt, rows, args)
    prompt_meta.update(trim_meta)
    iterator = ((idx, (args.condition, prompt)) for idx, prompt in enumerate(prompts))
    for done, (idx, generated_text) in enumerate(
        inference_generate(
            ckpt,
            iterator,
            max_tokens=args.max_model_len,
            max_generation=args.max_tokens,
            batch_size=args.batch_size,
            temp=args.temperature,
        ),
        start=1,
    ):
        predictions[idx] = generated_text
        completed_indices.add(idx)
        if done % args.progress_every == 0 or done == len(rows):
            elapsed = time.time() - gen_start
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"[{done}/{len(rows)}] elapsed={elapsed:.1f}s rate={rate:.4f}/s", flush=True)
        if args.save_every and (done % args.save_every == 0 or done == len(rows)):
            partial_per_step = score_completed_predictions(rows, predictions, completed_indices)
            partial_result = build_result(
                args=args,
                rows=rows,
                total_input_steps=total_input_steps,
                per_step=partial_per_step,
                load_time=load_time,
                gen_time=time.time() - gen_start,
                prompt_meta=prompt_meta,
                complete=done == len(rows),
            )
            out_path.write_text(json.dumps(partial_result, ensure_ascii=False, indent=2))

    gen_time = time.time() - gen_start
    per_step = score_predictions(rows, predictions)
    result = build_result(
        args=args,
        rows=rows,
        total_input_steps=total_input_steps,
        per_step=per_step,
        load_time=load_time,
        gen_time=gen_time,
        prompt_meta=prompt_meta,
        complete=True,
    )
    result["backend"] = "kohrm-local-prefixlm"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    score = round(100.0 * result["aggregate"]["avg_command_f1"], 2)
    print(json.dumps({"output_path": str(out_path), "score": score}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sapientinc/HRM-Text-1B")
    parser.add_argument("--model-short", default="")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--eval-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--condition",
        default="direct",
        help="Comma-separated HRM condition tags or raw special tokens. Examples: direct, synth,cot.",
    )
    parser.add_argument("--head-tokens", type=int, default=1024)
    parser.add_argument("--min-tail-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--lora-tag", default="")
    parser.add_argument("--hrm-text-root", type=Path, default=Path("HRM-Text"))
    parser.add_argument("--local-hrm-checkpoint", action="store_true")
    parser.add_argument("--local-hrm-export", action="store_true")
    parser.add_argument("--base-ckpt-path", default="")
    parser.add_argument("--ckpt-epoch", type=int, default=None)
    parser.add_argument("--ckpt-step", type=int, default=None)
    parser.add_argument("--ckpt-use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or sanitize_name(args.model)
    out_path = args.output_dir / f"{model_short}.json"
    if args.skip_if_exists and out_path.exists():
        print(json.dumps({"output_path": str(out_path), "skipped": True}, ensure_ascii=False))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rows, total_input_steps = load_rows(
        args.eval_path,
        args.limit,
        args.num_shards,
        args.shard_index,
    )
    if args.lora_path is not None:
        run_local_lora_eval(args, rows, total_input_steps, out_path)
        return
    if args.local_hrm_export:
        hrm_root = Path(args.hrm_text_root).resolve()
        if str(hrm_root) not in sys.path:
            sys.path.insert(0, str(hrm_root))
        from simple_inference_engine import inference_generate, inference_load_hf_export

        if not args.base_ckpt_path:
            raise ValueError("--base-ckpt-path is required with --local-hrm-export")
        load_start = time.time()
        ckpt = inference_load_hf_export(
            export_path=args.model,
            base_ckpt_path=args.base_ckpt_path,
            device="cuda",
        )
        ckpt.model.eval()
        load_time = time.time() - load_start
        print(f"Export model loaded in {load_time:.1f}s from {args.model}", flush=True)

        prompt_meta = {
            "template_status": "hrm_text_prefixlm_local_hf_export",
            "rank_eligible": True,
            "condition": args.condition,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "base_ckpt_path": args.base_ckpt_path,
        }
        gen_start = time.time()
        predictions = [""] * len(rows)
        completed_indices: set[int] = set()
        prompts, trim_meta = prepare_local_hrm_prompts(ckpt, rows, args)
        prompt_meta.update(trim_meta)
        iterator = ((idx, (args.condition, prompt)) for idx, prompt in enumerate(prompts))
        for done, (idx, generated_text) in enumerate(
            inference_generate(
                ckpt,
                iterator,
                max_tokens=args.max_model_len,
                max_generation=args.max_tokens,
                batch_size=args.batch_size,
                temp=args.temperature,
            ),
            start=1,
        ):
            predictions[idx] = generated_text
            completed_indices.add(idx)
            if done % args.progress_every == 0 or done == len(rows):
                elapsed = time.time() - gen_start
                rate = done / elapsed if elapsed > 0 else 0.0
                print(f"[{done}/{len(rows)}] elapsed={elapsed:.1f}s rate={rate:.4f}/s", flush=True)
            if args.save_every and (done % args.save_every == 0 or done == len(rows)):
                partial_per_step = score_completed_predictions(rows, predictions, completed_indices)
                partial_result = build_result(
                    args=args,
                    rows=rows,
                    total_input_steps=total_input_steps,
                    per_step=partial_per_step,
                    load_time=load_time,
                    gen_time=time.time() - gen_start,
                    prompt_meta=prompt_meta,
                    complete=done == len(rows),
                )
                partial_result["backend"] = "kohrm-local-hf-export-prefixlm"
                out_path.write_text(json.dumps(partial_result, ensure_ascii=False, indent=2))

        gen_time = time.time() - gen_start
        per_step = score_predictions(rows, predictions)
        result = build_result(
            args=args,
            rows=rows,
            total_input_steps=total_input_steps,
            per_step=per_step,
            load_time=load_time,
            gen_time=gen_time,
            prompt_meta=prompt_meta,
            complete=True,
        )
        result["backend"] = "kohrm-local-hf-export-prefixlm"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        score = round(100.0 * result["aggregate"]["avg_command_f1"], 2)
        print(json.dumps({"output_path": str(out_path), "score": score}, ensure_ascii=False))
        return
    if args.local_hrm_checkpoint:
        run_local_hrm_eval(args, rows, total_input_steps, out_path)
        return

    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_ids: list[list[int]] = []
    prompt_lengths: list[int] = []
    truncation_meta: list[dict[str, Any]] = []
    for row in rows:
        ids, meta = encode_prompt(
            tokenizer=tokenizer,
            body=flatten_messages(row),
            condition=args.condition,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_tokens,
            head_tokens=args.head_tokens,
            min_tail_tokens=args.min_tail_tokens,
        )
        prompt_ids.append(ids)
        prompt_lengths.append(len(ids))
        truncation_meta.append(
            {
                "task_id": row.get("task_id"),
                "sample_idx": row.get("sample_idx"),
                "step_idx": row.get("step_idx"),
                **meta,
            }
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s on {device}", flush=True)

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else 1.0,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    prompt_meta = {
        "template_status": "hrm_text_prefixlm",
        "rank_eligible": True,
        "condition": args.condition,
        "condition_prefix": condition_prefix(args.condition),
        "max_prompt_tokens": args.max_model_len - args.max_tokens,
        "prompt_tokens_min": min(prompt_lengths) if prompt_lengths else 0,
        "prompt_tokens_max": max(prompt_lengths) if prompt_lengths else 0,
        "prompt_tokens_avg": round(sum(prompt_lengths) / max(len(prompt_lengths), 1), 1),
        "truncated_steps": sum(1 for item in truncation_meta if item["truncated"]),
        "truncation_examples": [item for item in truncation_meta if item["truncated"]][:10],
    }

    gen_start = time.time()
    predictions: list[str] = []
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    for batch_start in range(0, len(prompt_ids), args.batch_size):
        batch_ids = prompt_ids[batch_start : batch_start + args.batch_size]
        max_len = max(len(ids) for ids in batch_ids)
        pad_id = tokenizer.pad_token_id
        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        token_type_rows: list[list[int]] = []
        for ids in batch_ids:
            pad = max_len - len(ids)
            input_rows.append([pad_id] * pad + ids)
            attention_rows.append([0] * pad + [1] * len(ids))
            token_type_rows.append([0] * pad + [1] * len(ids))

        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
        token_type_ids = torch.tensor(token_type_rows, dtype=torch.long, device=device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                **gen_kwargs,
            )

        for row_offset in range(len(batch_ids)):
            new_tokens = output_ids[row_offset, input_ids.shape[1] :]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            predictions.append(pred_text)

        done = len(predictions)
        if done % args.progress_every == 0 or done == len(prompt_ids):
            elapsed = time.time() - gen_start
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"[{done}/{len(prompt_ids)}] elapsed={elapsed:.1f}s rate={rate:.4f}/s", flush=True)

        if args.save_every and (done % args.save_every == 0 or done == len(prompt_ids)):
            partial_per_step = score_predictions(rows[: len(predictions)], predictions)
            partial_gen_time = time.time() - gen_start
            partial_result = build_result(
                args=args,
                rows=rows,
                total_input_steps=total_input_steps,
                per_step=partial_per_step,
                load_time=load_time,
                gen_time=partial_gen_time,
                prompt_meta=prompt_meta,
                complete=len(predictions) == len(rows),
            )
            out_path.write_text(json.dumps(partial_result, ensure_ascii=False, indent=2))

    gen_time = time.time() - gen_start
    per_step = score_predictions(rows, predictions)
    result = build_result(
        args=args,
        rows=rows,
        total_input_steps=total_input_steps,
        per_step=per_step,
        load_time=load_time,
        gen_time=gen_time,
        prompt_meta=prompt_meta,
        complete=True,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    score = round(100.0 * result["aggregate"]["avg_command_f1"], 2)
    print(json.dumps({"output_path": str(out_path), "score": score}, ensure_ascii=False))


if __name__ == "__main__":
    main()
