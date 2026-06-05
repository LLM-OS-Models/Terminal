#!/usr/bin/env python3
"""Text-only Gemma 4 replay eval for checkpoints stored as gemma4_unified."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4 import Gemma4ForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prompt_builder import build_prompts, sanitize_name
from replay_metrics import aggregate_scores, parse_prediction, score_commands, step_bucket


def resolve_model_path(model_id_or_path: str) -> Path:
    path = Path(model_id_or_path)
    if path.exists():
        return path
    return Path(snapshot_download(model_id_or_path))


def load_rows(
    path: Path,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for row_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if shard_count > 1 and row_idx % shard_count != shard_index:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def mapped_key(key: str) -> str | None:
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model.") :]
    if key.startswith("language_model."):
        return "model." + key[len("language_model.") :]
    if key.startswith("model.") or key == "lm_head.weight":
        return key
    return None


def load_gemma4_text_only_model(model_path: Path, torch_dtype: torch.dtype) -> Gemma4ForCausalLM:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        text_config = config

    model = Gemma4ForCausalLM(text_config)
    index_path = model_path / "model.safetensors.index.json"
    single_path = model_path / "model.safetensors"

    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_to_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for old_key, shard_name in weight_map.items():
            new_key = mapped_key(old_key)
            if new_key is not None:
                shard_to_keys[shard_name].append((old_key, new_key))
        for shard_name, key_pairs in sorted(shard_to_keys.items()):
            shard_state = {}
            with safe_open(str(model_path / shard_name), framework="pt", device="cpu") as handle:
                for old_key, new_key in key_pairs:
                    shard_state[new_key] = handle.get_tensor(old_key).to(dtype=torch_dtype)
            model.load_state_dict(shard_state, strict=False)
            del shard_state
    elif single_path.exists():
        shard_state = {}
        with safe_open(str(single_path), framework="pt", device="cpu") as handle:
            for old_key in handle.keys():
                new_key = mapped_key(old_key)
                if new_key is not None:
                    shard_state[new_key] = handle.get_tensor(old_key).to(dtype=torch_dtype)
        model.load_state_dict(shard_state, strict=False)
        del shard_state
    else:
        raise FileNotFoundError(f"No safetensors weights found in {model_path}")

    model.tie_weights()
    return model.to(dtype=torch_dtype)


def score_completed(rows: list[dict[str, Any]], predictions: list[str]) -> list[dict[str, Any]]:
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


def write_result(
    *,
    out_path: Path,
    args: argparse.Namespace,
    model_path: Path,
    rows: list[dict[str, Any]],
    predictions: list[str],
    load_time: float,
    gen_time: float,
    prompt_meta: dict[str, Any],
    complete: bool,
) -> None:
    per_step = score_completed(rows, predictions)
    aggregate = aggregate_scores(per_step)
    result = {
        "model": args.model,
        "model_path": str(model_path),
        "model_short": args.model_short or sanitize_name(args.model),
        "gpu": str(args.gpu),
        "eval_path": str(args.eval_path),
        "timestamp": datetime.utcnow().isoformat(),
        "load_time_sec": round(load_time, 1),
        "gen_time_sec": round(gen_time, 1),
        "avg_sec_per_step": round(gen_time / max(len(predictions), 1), 3),
        "backend": "transformers-gemma4-text-only",
        "complete": complete,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "batch_size": args.batch_size,
        },
        "prompt_template": prompt_meta,
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-short", default="")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short or sanitize_name(args.model)
    out_path = args.output_dir / f"{model_short}.json"

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    rows = load_rows(args.eval_path, args.limit, args.shard_index, args.shard_count)
    load_start = time.time()
    model_path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts, prompt_meta = build_prompts(
        tokenizer,
        rows,
        model_name=args.model,
        chat_template_kwargs={"enable_thinking": False},
        prompt_options={
            "strip_thinking_history": True,
            "gemma4_empty_thought_channel": False,
            "use_gemma4_patched_template": True,
        },
    )
    prompt_meta.update(
        {
            "template_status": "gemma4_text_only_patched_turn",
            "rank_eligible": True,
            "row_count": len(rows),
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
        }
    )

    model = load_gemma4_text_only_model(model_path, torch_dtype=torch_dtype)
    model.to("cuda")
    model.eval()
    load_time = time.time() - load_start
    print(f"Gemma4 text-only model loaded in {load_time:.1f}s from {model_path}", flush=True)

    gen_kwargs = {
        "max_new_tokens": args.max_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = {key: value for key, value in gen_kwargs.items() if value is not None}

    gen_start = time.time()
    predictions: list[str] = []
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_model_len,
        ).to("cuda")
        input_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
        for output in output_ids:
            pred_text = tokenizer.decode(output[input_width:], skip_special_tokens=True)
            predictions.append(pred_text)
        done = len(predictions)
        elapsed = time.time() - gen_start
        if done % args.progress_every == 0 or done == len(prompts):
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"[{done}/{len(prompts)}] elapsed={elapsed:.1f}s rate={rate:.4f}/s", flush=True)
        if args.save_every and (done % args.save_every == 0 or done == len(prompts)):
            write_result(
                out_path=out_path,
                args=args,
                model_path=model_path,
                rows=rows[: len(predictions)],
                predictions=predictions,
                load_time=load_time,
                gen_time=elapsed,
                prompt_meta=prompt_meta,
                complete=done == len(prompts),
            )

    gen_time = time.time() - gen_start
    write_result(
        out_path=out_path,
        args=args,
        model_path=model_path,
        rows=rows,
        predictions=predictions,
        load_time=load_time,
        gen_time=gen_time,
        prompt_meta=prompt_meta,
        complete=True,
    )
    score = round(100.0 * aggregate_scores(score_completed(rows, predictions))["avg_command_f1"], 2)
    print(json.dumps({"output_path": str(out_path), "score": score}, ensure_ascii=False))


if __name__ == "__main__":
    main()
