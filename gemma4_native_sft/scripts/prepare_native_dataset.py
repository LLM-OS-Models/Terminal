#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

from terminal_sft_utils import (
    is_holdout_row,
    load_holdout_keys,
    stable_hash,
    tokenizer_metadata,
)


SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }
  ],
  "task_complete": false
}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete
"""


THINK_BLOCK_RE = re.compile(r"<think>\s*.*?</think>\s*", re.DOTALL)
THOUGHT_CHANNEL_RE = re.compile(r"<\|channel\>thought\n.*?<channel\|>\s*", re.DOTALL)


BASE_TO_TEMPLATE_MODEL = {
    "google/gemma-4-E2B": "google/gemma-4-E2B-it",
    "google/gemma-4-E4B": "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B": "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B": "google/gemma-4-31B-it",
}


def strip_thinking_blocks(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    text = THOUGHT_CHANNEL_RE.sub("", text)
    return text.lstrip()


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def normalize_role(role: str) -> str | None:
    if role == "model":
        return "assistant"
    if role == "tool":
        return "user"
    if role in {"system", "user", "assistant"}:
        return role
    return None


def normalize_conversation(messages: Any, *, strip_history_thinking: bool) -> list[dict[str, str]] | None:
    if not isinstance(messages, list) or not messages:
        return None

    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(str(message.get("role", "")))
        if role is None:
            continue
        text = content_to_text(message.get("content", ""))
        if role == "assistant" and strip_history_thinking:
            text = strip_thinking_blocks(text)
        normalized.append({"role": role, "content": text})

    if not normalized:
        return None
    if normalized[0]["role"] != "system":
        normalized.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return normalized


def extract_first_json_object(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def normalize_assistant_target(
    text: str,
    *,
    target_json_only: bool,
    keep_empty_commands: bool,
    pretty_json: bool,
) -> str | None:
    text = strip_thinking_blocks(text).strip()
    if not target_json_only:
        return text or None

    obj = extract_first_json_object(text)
    if not isinstance(obj, dict):
        return None
    commands = obj.get("commands")
    if not isinstance(commands, list):
        return None
    if not keep_empty_commands and not commands:
        return None
    if pretty_json:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def encode_example(
    *,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    assistant_target: str,
    max_seq_length: int,
) -> tuple[dict[str, Any] | None, str | None]:
    prompt_text = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    # Build the supervised sequence from the actual generation prompt. This matters
    # for Gemma 4 26B/31B, where non-thinking generation prompts include an empty
    # thought channel that is not inserted when rendering a completed assistant turn.
    full_text = prompt_text + assistant_target + "<turn|>\n"

    prompt_enc = tokenizer(prompt_text, add_special_tokens=False, truncation=False)
    full_enc = tokenizer(full_text, add_special_tokens=False, truncation=False)
    input_ids = full_enc["input_ids"]
    prompt_len = len(prompt_enc["input_ids"])
    if len(input_ids) > max_seq_length:
        return None, "too_long"
    if prompt_len >= len(input_ids):
        return None, "empty_assistant"

    labels = input_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len
    assistant_token_count = sum(label != -100 for label in labels)
    if assistant_token_count <= 0:
        return None, "empty_assistant"

    return (
        {
            "input_ids": input_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels": labels,
            "prompt_token_count": prompt_len,
            "assistant_token_count": assistant_token_count,
        },
        None,
    )


def load_source_dataset(args: argparse.Namespace) -> Any:
    if args.source_path:
        return load_from_disk(args.source_path)
    configs = [name.strip() for name in args.hf_configs.split(",") if name.strip()]
    if not configs:
        raise ValueError("--hf-configs must not be empty")
    datasets = [
        load_dataset(args.hf_dataset_name, config, split=args.hf_split)
        for config in configs
    ]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def infer_template_model_id(model_id: str, template_model_id: str | None) -> str:
    if template_model_id:
        return template_model_id
    return BASE_TO_TEMPLATE_MODEL.get(model_id, model_id)


def load_gemma_tokenizer(model_id: str, template_model_id: str) -> tuple[Any, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_has_template = bool(getattr(tokenizer, "chat_template", None))
    template_was_injected = False
    if template_model_id != model_id or not original_has_template:
        template_tokenizer = AutoTokenizer.from_pretrained(template_model_id, trust_remote_code=False)
        if not getattr(template_tokenizer, "chat_template", None):
            raise RuntimeError(f"Template tokenizer has no chat_template: {template_model_id}")
        tokenizer.chat_template = template_tokenizer.chat_template
        template_was_injected = True

    probe_messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    probe = apply_chat_template(tokenizer, probe_messages, add_generation_prompt=True)
    return tokenizer, {
        "model_id": model_id,
        "template_model_id": template_model_id,
        "model_tokenizer_had_template": original_has_template,
        "template_was_injected": template_was_injected,
        "nonthinking_has_empty_thought_channel": "<|channel>thought\n<channel|>" in probe,
        "nonthinking_probe_tail": probe[-120:],
    }


def percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", default="/home/work/.data/liquid_cli_sft/datasets/sft_data")
    parser.add_argument("--hf-dataset-name", default="nvidia/Nemotron-Terminal-Corpus")
    parser.add_argument("--hf-configs", default="dataset_adapters,skill_based_easy,skill_based_medium,skill_based_mixed")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--template-model-id", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--holdout-path", default="eval/eval_dataset.jsonl")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--source-limit", type=int, default=None)
    parser.add_argument("--strip-thinking-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-json-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-empty-commands", action="store_true")
    parser.add_argument("--compact-json", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    template_model_id = infer_template_model_id(args.model_id, args.template_model_id)
    tokenizer, tokenizer_meta = load_gemma_tokenizer(args.model_id, template_model_id)
    holdout_keys = load_holdout_keys(args.holdout_path)
    dataset = load_source_dataset(args)
    raw_rows = len(dataset)
    limit = args.source_limit if args.source_limit is not None else raw_rows

    rows: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    assistant_lengths: list[int] = []
    stats = {
        "raw_rows": raw_rows,
        "visited_rows": 0,
        "skipped_holdout_rows": 0,
        "skipped_no_conversation": 0,
        "candidate_assistant_turns": 0,
        "skipped_non_json_or_no_commands": 0,
        "skipped_too_long": 0,
        "skipped_empty_assistant": 0,
        "processed_rows": 0,
    }

    for row_index, row in enumerate(dataset):
        if row_index >= limit:
            break
        stats["visited_rows"] += 1
        if holdout_keys and is_holdout_row(row, holdout_keys):
            stats["skipped_holdout_rows"] += 1
            continue
        messages = normalize_conversation(
            row.get("conversations"),
            strip_history_thinking=args.strip_thinking_history,
        )
        if not messages:
            stats["skipped_no_conversation"] += 1
            continue

        for turn_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            if turn_index == 0:
                continue
            stats["candidate_assistant_turns"] += 1
            target = normalize_assistant_target(
                message["content"],
                target_json_only=args.target_json_only,
                keep_empty_commands=args.keep_empty_commands,
                pretty_json=not args.compact_json,
            )
            if target is None:
                stats["skipped_non_json_or_no_commands"] += 1
                continue

            prompt_messages = messages[:turn_index]
            encoded, reason = encode_example(
                tokenizer=tokenizer,
                prompt_messages=prompt_messages,
                assistant_target=target,
                max_seq_length=args.max_seq_length,
            )
            if encoded is None:
                if reason == "too_long":
                    stats["skipped_too_long"] += 1
                else:
                    stats["skipped_empty_assistant"] += 1
                continue

            prompt_lengths.append(int(encoded.pop("prompt_token_count")))
            assistant_lengths.append(int(encoded.pop("assistant_token_count")))
            rows.append(encoded)

        if args.log_every and stats["visited_rows"] % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "visited_rows": stats["visited_rows"],
                        "processed_rows": len(rows),
                        "skipped_too_long": stats["skipped_too_long"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    stats["processed_rows"] = len(rows)
    if not rows:
        raise RuntimeError("No rows were produced. Check source data and template settings.")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_out = Dataset.from_list(rows)
    dataset_out.save_to_disk(str(output_path))

    meta = {
        "source_mode": "gemma4_native_conversation_prefix_masked",
        "source_path": args.source_path,
        "hf_dataset_name": None if args.source_path else args.hf_dataset_name,
        "hf_configs": None if args.source_path else args.hf_configs,
        "hf_split": None if args.source_path else args.hf_split,
        "holdout_path": args.holdout_path,
        "holdout_hash": stable_hash(json.dumps(sorted(holdout_keys))),
        "max_seq_length": args.max_seq_length,
        "strip_thinking_history": args.strip_thinking_history,
        "target_json_only": args.target_json_only,
        "keep_empty_commands": args.keep_empty_commands,
        "compact_json": args.compact_json,
        "target_appended_to_generation_prompt": True,
        **tokenizer_metadata(tokenizer, args.model_id),
        **tokenizer_meta,
        **stats,
        "prompt_token_min": min(prompt_lengths),
        "prompt_token_p50": percentile(prompt_lengths, 0.50),
        "prompt_token_p95": percentile(prompt_lengths, 0.95),
        "prompt_token_max": max(prompt_lengths),
        "assistant_token_min": min(assistant_lengths),
        "assistant_token_p50": percentile(assistant_lengths, 0.50),
        "assistant_token_p95": percentile(assistant_lengths, 0.95),
        "assistant_token_max": max(assistant_lengths),
    }
    (output_path / "prepare_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
