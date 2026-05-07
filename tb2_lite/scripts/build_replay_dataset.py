#!/usr/bin/env python3
"""Build flattened multi-turn replay datasets from eval/eval_dataset.jsonl."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def extract_json_blob(text: str) -> dict | None:
    text = text.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
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


def source_group(source: str) -> str:
    if not source:
        return "unknown"
    return source.split("/")[-1]


def normalize_message(message: dict) -> dict[str, str] | None:
    role = message.get("role")
    content = message.get("content")
    if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
        return None
    return {"role": role, "content": content}


def build_records(
    input_path: Path,
    *,
    include_invalid_refs: bool = False,
    include_empty_command_refs: bool = False,
) -> tuple[list[dict], list[dict]]:
    task_records: list[dict] = []
    step_records: list[dict] = []

    with input_path.open() as handle:
        for sample_idx, line in enumerate(handle):
            row = json.loads(line)
            conversations = row["conversations"]
            assistants = [msg for msg in conversations if msg.get("role") == "assistant"]
            task_id = f"task-{sample_idx:03d}"

            task_records.append({
                "task_id": task_id,
                "sample_idx": sample_idx,
                "source": row.get("_source_file", ""),
                "source_group": source_group(row.get("_source_file", "")),
                "total_steps": len(assistants),
            })

            context_messages: list[dict[str, str]] = []
            assistant_idx = 0
            for message in conversations:
                normalized = normalize_message(message)
                if normalized is None:
                    continue
                role = message.get("role")
                if role != "assistant":
                    context_messages.append(normalized)
                    continue
                if not context_messages:
                    continue

                payload = extract_json_blob(message.get("content", "")) or {}
                ref_parse_ok = bool(payload)
                if not ref_parse_ok and not include_invalid_refs:
                    context_messages.append(normalized)
                    assistant_idx += 1
                    continue
                commands = payload.get("commands", [])
                if not isinstance(commands, list):
                    commands = []
                command_units = flatten_keystrokes(commands)
                if not command_units and not include_empty_command_refs:
                    context_messages.append(normalized)
                    assistant_idx += 1
                    continue

                step_records.append({
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "step_idx": assistant_idx,
                    "total_steps": len(assistants),
                    "is_final_step": assistant_idx == len(assistants) - 1,
                    "source": row.get("_source_file", ""),
                    "source_group": source_group(row.get("_source_file", "")),
                    "prompt": "\n\n".join(
                        f"{msg['role'].upper()}:\n{msg['content']}" for msg in context_messages
                    ),
                    "messages": list(context_messages),
                    "ref_raw": message.get("content", ""),
                    "ref_parse_ok": ref_parse_ok,
                    "ref_analysis": payload.get("analysis", ""),
                    "ref_plan": payload.get("plan", ""),
                    "ref_task_complete": bool(payload.get("task_complete", False)),
                    "ref_commands": commands,
                    "ref_command_units": command_units,
                })
                context_messages.append(normalized)
                assistant_idx += 1

    return task_records, step_records


def select_dev_tasks(task_records: list[dict], dev_tasks: int, seed: int) -> set[str]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in task_records:
        grouped[record["source_group"]].append(record)

    for records in grouped.values():
        rng.shuffle(records)

    chosen: list[dict] = []
    while len(chosen) < min(dev_tasks, len(task_records)):
        progressed = False
        for group in sorted(grouped):
            if not grouped[group]:
                continue
            chosen.append(grouped[group].pop())
            progressed = True
            if len(chosen) >= dev_tasks:
                break
        if not progressed:
            break

    return {record["task_id"] for record in chosen}


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="eval/eval_dataset.jsonl")
    parser.add_argument("--output-dir", default="tb2_lite/data")
    parser.add_argument("--dev-tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-invalid-refs", action="store_true")
    parser.add_argument("--include-empty-command-refs", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_records, step_records = build_records(
        input_path,
        include_invalid_refs=args.include_invalid_refs,
        include_empty_command_refs=args.include_empty_command_refs,
    )
    dev_task_ids = select_dev_tasks(task_records, args.dev_tasks, args.seed)

    full_path = output_dir / "replay_full.jsonl"
    dev_path = output_dir / f"replay_dev_{args.dev_tasks}.jsonl"
    manifest_path = output_dir / "task_manifest.json"

    write_jsonl(full_path, step_records)
    write_jsonl(dev_path, [record for record in step_records if record["task_id"] in dev_task_ids])
    manifest_path.write_text(json.dumps({
        "input_path": str(input_path),
        "num_tasks": len(task_records),
        "num_steps": len(step_records),
        "prompt_mode": "cumulative_messages",
        "include_invalid_refs": args.include_invalid_refs,
        "include_empty_command_refs": args.include_empty_command_refs,
        "dev_tasks": args.dev_tasks,
        "dev_task_ids": sorted(dev_task_ids),
        "source_groups": sorted({record["source_group"] for record in task_records}),
    }, ensure_ascii=False, indent=2))

    print(
        json.dumps(
            {
                "tasks": len(task_records),
                "steps": len(step_records),
                "full_path": str(full_path),
                "dev_path": str(dev_path),
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
