from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def tokenizer_metadata(tokenizer: Any, model_path: str) -> dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    payload = {
        "name_or_path": getattr(tokenizer, "name_or_path", model_path),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
    }
    return {
        "template_model_id": model_path,
        "tokenizer_hash": stable_hash(json.dumps(payload, sort_keys=True, default=str)),
        "chat_template_hash": stable_hash(chat_template),
        "has_chat_template": bool(chat_template),
    }


def row_holdout_keys(row: dict[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    source = row.get("_source_file")
    original_index = row.get("_original_index")
    if source is not None and original_index is not None:
        keys.add(("source_index", f"{source}::{original_index}"))

    task = row.get("task")
    episode = row.get("episode")
    if task is not None and episode is not None:
        keys.add(("task_episode", f"{task}::{episode}"))

    return keys


def load_holdout_keys(path: str | None) -> set[tuple[str, str]]:
    if not path:
        return set()
    holdout_path = Path(path)
    if not holdout_path.exists():
        return set()

    keys: set[tuple[str, str]] = set()
    with holdout_path.open() as handle:
        if holdout_path.suffix == ".json":
            rows = json.load(handle)
            if isinstance(rows, dict):
                rows = rows.get("rows", rows.get("samples", []))
            for row in rows:
                if isinstance(row, dict):
                    keys.update(row_holdout_keys(row))
            return keys

        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.update(row_holdout_keys(row))
    return keys


def is_holdout_row(row: dict[str, Any], holdout_keys: set[tuple[str, str]]) -> bool:
    if not holdout_keys:
        return False
    return bool(row_holdout_keys(row) & holdout_keys)


def build_turn_pair_examples(raw_dataset: Any, holdout_keys: set[tuple[str, str]] | None = None) -> tuple[list[dict[str, str]], int]:
    examples: list[dict[str, str]] = []
    skipped_holdout = 0
    holdout_keys = holdout_keys or set()

    for row in raw_dataset:
        if is_holdout_row(row, holdout_keys):
            skipped_holdout += 1
            continue

        conversations = row.get("conversations", [])
        for idx in range(1, len(conversations)):
            prev_msg = conversations[idx - 1]
            cur_msg = conversations[idx]
            if prev_msg.get("role") != "user" or cur_msg.get("role") != "assistant":
                continue

            assistant_text = cur_msg.get("content", "")
            has_commands = '"keystrokes"' in assistant_text or '"commands"' in assistant_text
            empty_commands = '"commands": []' in assistant_text or '"commands":[]' in assistant_text
            if not has_commands or empty_commands:
                continue

            examples.append(
                {
                    "user": prev_msg.get("content", ""),
                    "assistant": assistant_text,
                }
            )
    return examples, skipped_holdout


def encode_assistant_only(
    *,
    tokenizer: Any,
    user: str,
    assistant: str,
    max_seq_length: int,
) -> dict[str, Any] | None:
    prompt_messages = [{"role": "user", "content": user}]
    full_messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )
    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

    input_ids = full_enc["input_ids"]
    prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))
    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len
    assistant_token_count = sum(label != -100 for label in labels)
    if assistant_token_count <= 0:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
        "assistant_token_count": assistant_token_count,
    }


def assert_assistant_label_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No train rows remain after assistant-only masking.")
    counts = [int(row.get("assistant_token_count", 0)) for row in rows]
    zero_count = sum(count <= 0 for count in counts)
    if zero_count:
        raise RuntimeError(f"{zero_count} rows have zero assistant label tokens.")
    return {
        "assistant_label_min": min(counts),
        "assistant_label_max": max(counts),
        "assistant_label_avg": round(sum(counts) / len(counts), 2),
    }
