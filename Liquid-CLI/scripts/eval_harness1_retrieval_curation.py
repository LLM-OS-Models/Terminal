#!/usr/bin/env python
"""Evaluate Harness-1 style retrieval curation adapters.

The evaluator runs the model on RLVR prompts, extracts `curated_doc_ids` from
strict JSON output, and reports retrieval metrics against gold document IDs.
It is intentionally standalone so it can run immediately after RL training
without importing TRL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument(
        "--dataset-path",
        default="/home/work/.data/liquid_cli_sft/datasets/lfm25_harness1_retrieval_rlvr_browsecomp_v1",
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--split-limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        pieces = re.split(r"[\s,;]+", value.strip())
    elif isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("doc_id") or item.get("docid") or item.get("id")
            pieces.append(str(item))
    else:
        pieces = [str(value)]

    out: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        doc = str(piece).strip().strip("\"'`[](){}")
        if not doc or doc in seen:
            continue
        seen.add(doc)
        out.append(doc)
    return out


def parse_curated_ids(text: str) -> tuple[list[str], bool]:
    parsed = extract_json_object(text)
    if parsed is None:
        return [], False
    for key in ("curated_doc_ids", "doc_ids", "selected_doc_ids", "evidence_doc_ids"):
        if key in parsed:
            return normalize_ids(parsed.get(key)), True
    return [], False


def f1_score(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return 2.0 * precision * recall / denom


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    denom = beta * beta * precision + recall
    if denom <= 0:
        return 0.0
    return (1 + beta * beta) * precision * recall / denom


def row_metrics(predicted: list[str], gold_ids: list[str], answer_ids: list[str], candidate_ids: list[str], valid: bool) -> dict[str, Any]:
    gold = set(str(x) for x in gold_ids)
    answer = set(str(x) for x in answer_ids) or gold
    candidates = set(str(x) for x in candidate_ids)
    selected = [doc for doc in predicted if doc in candidates]
    selected_set = set(selected)
    invalid = [doc for doc in predicted if doc not in candidates]

    hits = len(selected_set & gold)
    answer_hits = len(selected_set & answer)
    precision = hits / max(len(selected_set), 1)
    recall = hits / max(len(gold), 1)
    answer_recall = answer_hits / max(len(answer), 1)
    return {
        "valid_json": valid,
        "selected_count": len(selected_set),
        "invalid_count": len(invalid),
        "gold_count": len(gold),
        "hit_count": hits,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "f2": f_beta(precision, recall, beta=2.0),
        "answer_recall": answer_recall,
        "exact_gold_set": selected_set == gold,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[idx]


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    dataset = load_from_disk(args.dataset_path)
    if args.split_limit:
        dataset = dataset.select(range(min(args.split_limit, len(dataset))))
    indices = [idx for idx in range(len(dataset)) if idx % args.num_shards == args.shard_index]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path = Path(args.metrics_json)
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    do_sample = args.temperature > 0
    started = time.time()
    rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as out:
        for offset in range(0, len(indices), args.batch_size):
            batch_indices = indices[offset : offset + args.batch_size]
            batch = [dataset[int(idx)] for idx in batch_indices]
            prompts = [
                tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_length,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_len = encoded["input_ids"].shape[1]
            completions = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            for row_index, row, completion in zip(batch_indices, batch, completions):
                predicted, valid = parse_curated_ids(completion)
                metrics = row_metrics(
                    predicted,
                    row["gold_doc_ids"],
                    row["answer_doc_ids"],
                    row["candidate_doc_ids"],
                    valid,
                )
                record = {
                    "index": int(row_index),
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "answer": row["answer"],
                    "predicted_doc_ids": predicted,
                    "gold_doc_ids": row["gold_doc_ids"],
                    "answer_doc_ids": row["answer_doc_ids"],
                    "candidate_doc_ids": row["candidate_doc_ids"],
                    "completion": completion,
                    "metrics": metrics,
                }
                rows.append(record)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

    summary = {
        "task": "harness1_retrieval_curation_eval",
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "dataset_path": args.dataset_path,
        "output_jsonl": str(output_path),
        "rows": len(rows),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "runtime_seconds": round(time.time() - started, 3),
        "valid_json_rate": mean([float(row["metrics"]["valid_json"]) for row in rows]),
        "exact_gold_set_rate": mean([float(row["metrics"]["exact_gold_set"]) for row in rows]),
        "recall_mean": mean([row["metrics"]["recall"] for row in rows]),
        "recall_p50": percentile([row["metrics"]["recall"] for row in rows], 0.50),
        "recall_p90": percentile([row["metrics"]["recall"] for row in rows], 0.90),
        "answer_recall_mean": mean([row["metrics"]["answer_recall"] for row in rows]),
        "precision_mean": mean([row["metrics"]["precision"] for row in rows]),
        "f1_mean": mean([row["metrics"]["f1"] for row in rows]),
        "f2_mean": mean([row["metrics"]["f2"] for row in rows]),
        "selected_count_mean": mean([row["metrics"]["selected_count"] for row in rows]),
        "invalid_count_mean": mean([row["metrics"]["invalid_count"] for row in rows]),
    }
    metric_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
