#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

NON_CODING_PREFIXES = {
    "file_operations",
    "data_processing",
    "data_querying",
    "dependency_management",
    "security",
}

EXCLUDED_PREFIXES = {
    "data_science",
    "scientific_computing",
    "debugging",
    "software_engineering",
}

CODING_KEYWORDS = [
    "write python",
    "write code",
    "write a script",
    "write a program",
    "solution.py",
    "def ",
    "class ",
    "import torch",
    "import pandas",
    "import numpy",
    "fix the bug",
    "debug the code",
    "pull request",
    "git diff",
]


def task_prefix(task_name: str) -> str:
    if "_task_" in task_name:
        return task_name.split("_task_", 1)[0]
    return task_name


def has_coding_keywords(example: dict) -> bool:
    text = " ".join(
        str(message.get("content", ""))
        for message in example.get("conversations", [])
        if isinstance(message, dict)
    ).lower()
    return any(keyword in text for keyword in CODING_KEYWORDS)


def keep_example(example: dict) -> bool:
    prefix = task_prefix(str(example.get("task", "")))
    if prefix in EXCLUDED_PREFIXES:
        return False
    if prefix not in NON_CODING_PREFIXES:
        return False
    if has_coding_keywords(example):
        return False
    return True


def conversation_to_text(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    bos = tokenizer.bos_token or ""
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/home/work/.data/liquid_cli_sft/datasets",
    )
    parser.add_argument(
        "--model-path",
        default="LiquidAI/LFM2-8B-A1B",
    )
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    args = parser.parse_args()

    root = Path(args.dataset_root)
    raw_path = root / "raw_skill_based"
    filtered_path = root / "sft_data"
    text_path = root / "sft_text"
    tokenized_path = root / "sft_tokenized"
    meta_path = root / "prepare_meta.json"
    root.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        merged = load_from_disk(str(raw_path))
    else:
        splits = []
        for split_name in ["skill_based_easy", "skill_based_medium", "skill_based_mixed"]:
            ds = load_dataset("nvidia/Nemotron-Terminal-Corpus", split_name, split="train")
            splits.append(ds)
        merged = concatenate_datasets(splits)
        merged.save_to_disk(str(raw_path))

    if filtered_path.exists():
        filtered = load_from_disk(str(filtered_path))
    else:
        filtered = merged.filter(keep_example, num_proc=args.num_proc, desc="filter_non_coding")
        filtered.save_to_disk(str(filtered_path))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if text_path.exists():
        text_ds = load_from_disk(str(text_path))
    else:
        text_ds = filtered.map(
            lambda example: conversation_to_text(example, tokenizer),
            num_proc=args.num_proc,
            desc="format_chat_text",
        )
        text_ds.save_to_disk(str(text_path))

    if tokenized_path.exists():
        tokenized = load_from_disk(str(tokenized_path))
    else:
        tokenized = text_ds.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
            ),
            batched=True,
            num_proc=args.num_proc,
            remove_columns=text_ds.column_names,
            desc="tokenize",
        )
        tokenized.save_to_disk(str(tokenized_path))

    meta = {
        "raw_samples": len(merged),
        "filtered_samples": len(filtered),
        "text_samples": len(text_ds),
        "tokenized_samples": len(tokenized),
        "raw_path": str(raw_path),
        "filtered_path": str(filtered_path),
        "text_path": str(text_path),
        "tokenized_path": str(tokenized_path),
        "model_path": args.model_path,
        "max_seq_length": args.max_seq_length,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
