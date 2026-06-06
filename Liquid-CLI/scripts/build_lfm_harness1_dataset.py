#!/usr/bin/env python
"""Build an LFM chat dataset from Harness-1 SFT trajectories.

The upstream Harness-1 training code emits trajectory JSON records for a
Tinker/Harmony stack. This converter keeps the Harness search semantics but
emits the local LFM conversation schema already consumed by
train_unsloth_processed.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, Features, Value


MESSAGE_FEATURES = Features(
    {
        "conversations": [
            {
                "role": Value("string"),
                "content": Value("string"),
                "tool_calls": Value("string"),
            }
        ],
        "source": Value("string"),
    }
)

DEFAULT_SYSTEM_PROMPT = """You are a stateful retrieval search subagent.

You operate a Harness-1 style search environment. The environment maintains candidate documents, curated evidence, evidence links, verification records, and a context budget. Your job is to choose the next semantic action: search broadly, grep exactly, read or review documents, curate evidence, verify claims, or end the search when the curated evidence is sufficient.

Use tool calls instead of free-form final answers. Each assistant turn should contain concise reasoning and exactly one structured tool call unless the trajectory says otherwise. Prefer evidence-seeking actions over guessing from memory. Keep the search state useful: curate plausible documents, remove weak evidence, verify multi-constraint claims, and stop only when the evidence is sufficient."""

TOOL_SUMMARY = """Available Harness tools:
- fan_out_search({"queries": string[]}): run diverse searches in parallel.
- search_corpus({"query": string}): search the corpus for relevant documents.
- grep_corpus({"pattern": string}): run exact or regex-style corpus search.
- read_document({"doc_id": string}): read one document by id.
- review_docs({"doc_ids": string[]}): re-render already seen documents.
- curate({"add_ids": string[], "remove_ids": string[], "importance": object?}): update final curated evidence.
- verify({"doc_ids": string[], "claim": string}): check whether documents support a claim.
- end_search({"answer": string?, "doc_ids": string[]?}): finish when evidence is sufficient."""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dataset_paths(path: Path) -> tuple[Path, Path, Path]:
    return path, path.with_name(path.name + ".tmp"), path.with_name(path.name + ".ready")


def save_atomic(dataset: Dataset, output_path: Path) -> None:
    final_path, tmp_path, ready_path = dataset_paths(output_path)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if ready_path.exists():
        ready_path.unlink()
    if final_path.exists():
        shutil.rmtree(final_path)
    dataset.save_to_disk(str(tmp_path))
    tmp_path.rename(final_path)
    ready_path.write_text("ok\n", encoding="utf-8")


def iter_input_files(paths: list[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.json"))
            yield from sorted(path.rglob("*.jsonl"))
        elif path.is_file():
            yield path


def iter_json_records(paths: list[Path]) -> Iterable[tuple[dict[str, Any], str]]:
    for path in iter_input_files(paths):
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {path}:{idx}: {exc}") from exc
                    if isinstance(obj, dict):
                        yield obj, f"{path}:{idx}"
            continue

        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if isinstance(obj, list):
            for idx, item in enumerate(obj):
                if isinstance(item, dict):
                    yield item, f"{path}[{idx}]"
        elif isinstance(obj, dict):
            for key in ("trajectories", "records", "data", "examples"):
                items = obj.get(key)
                if isinstance(items, list):
                    for idx, item in enumerate(items):
                        if isinstance(item, dict):
                            yield item, f"{path}:{key}[{idx}]"
                    break
            else:
                yield obj, str(path)


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", []):
            return value
    return None


def extract_query(record: dict[str, Any]) -> str:
    value = first_present(
        record,
        (
            "query_text",
            "query",
            "question",
            "task",
            "prompt",
            "user_query",
            "input",
        ),
    )
    if isinstance(value, dict):
        value = first_present(value, ("text", "query", "question", "content"))
    query = clean_text(value)
    if query:
        return query

    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return clean_text(message.get("content"))
    return ""


def extract_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("turn_history", "turns", "trajectory", "steps", "actions"):
        value = record.get(key)
        if isinstance(value, list):
            return [turn for turn in value if isinstance(turn, dict)]
    return []


def extract_tool_name(turn: dict[str, Any]) -> str:
    for key in ("tool_name", "tool", "name", "function_name"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    action = turn.get("action")
    if isinstance(action, dict):
        for key in ("tool_name", "tool", "name"):
            value = action.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        tools = action.get("tools")
        if isinstance(tools, list) and tools:
            first = tools[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                schema = first.get("tool_schema") or first.get("function") or first
                value = schema.get("name") if isinstance(schema, dict) else None
                if value:
                    return str(value)
    return ""


def extract_params(turn: dict[str, Any]) -> dict[str, Any]:
    value = first_present(turn, ("params", "parameters", "arguments", "args"))
    if value is None and isinstance(turn.get("action"), dict):
        action = turn["action"]
        value = first_present(action, ("params", "parameters", "arguments", "args"))
        if value is None and isinstance(action.get("params"), list) and action["params"]:
            value = action["params"][0]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"input": value}
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"input": value}


def extract_reasoning(turn: dict[str, Any]) -> str:
    value = first_present(turn, ("reasoning", "rationale", "analysis", "thought", "content"))
    if value is None and isinstance(turn.get("action"), dict):
        value = first_present(turn["action"], ("reasoning", "rationale", "analysis", "thought"))
    return clean_text(value)


def extract_observation(turn: dict[str, Any]) -> str:
    value = first_present(turn, ("observation", "observations", "result", "output", "tool_result"))
    if isinstance(value, dict):
        for key in ("text", "content", "observation"):
            if key in value:
                return clean_text(value[key])
    if isinstance(value, list):
        return "\n".join(clean_text(item) for item in value if clean_text(item))
    return clean_text(value)


def make_tool_calls(tool_name: str, params: dict[str, Any]) -> str:
    if not tool_name:
        return ""
    return compact_json(
        [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": params if isinstance(params, dict) else {"input": params},
                },
            }
        ]
    )


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...[truncated {len(text) - max_chars} chars]...\n{text[-tail:]}"


def render_history(
    turns: list[dict[str, Any]],
    *,
    recent_turns: int,
    max_observation_chars: int,
    max_older_summary_chars: int,
) -> str:
    if not turns:
        return "No previous turns."

    older = turns[:-recent_turns] if recent_turns > 0 else turns
    recent = turns[-recent_turns:] if recent_turns > 0 else []
    parts: list[str] = []

    if older:
        lines = []
        for idx, turn in enumerate(older, start=1):
            tool_name = extract_tool_name(turn) or "unknown_tool"
            params = truncate_text(compact_json(extract_params(turn)), 360)
            lines.append(f"{idx}. {tool_name}({params})")
        summary = "\n".join(lines)
        parts.append(
            "# Older Turn Summary\n"
            + truncate_text(summary, max_older_summary_chars)
        )

    if recent:
        rendered = ["# Recent Turns"]
        start_idx = len(older) + 1
        for offset, turn in enumerate(recent, start=start_idx):
            tool_name = extract_tool_name(turn) or "unknown_tool"
            params = compact_json(extract_params(turn))
            reasoning = extract_reasoning(turn)
            observation = truncate_text(extract_observation(turn), max_observation_chars)
            rendered.append(f"## Turn {offset}")
            if reasoning:
                rendered.append(f"Assistant reasoning:\n{reasoning}")
            rendered.append(f"Tool call:\n{tool_name}({params})")
            if observation:
                rendered.append(f"Observation:\n{observation}")
        parts.append("\n\n".join(rendered))

    return "\n\n".join(parts)


def build_user_context(
    *,
    query: str,
    previous_turns: list[dict[str, Any]],
    recent_turns: int,
    max_observation_chars: int,
    max_older_summary_chars: int,
) -> str:
    return "\n\n".join(
        [
            "# Query",
            query,
            "# Tool Interface",
            TOOL_SUMMARY,
            "# Search State And Prior History",
            render_history(
                previous_turns,
                recent_turns=recent_turns,
                max_observation_chars=max_observation_chars,
                max_older_summary_chars=max_older_summary_chars,
            ),
            "# Next Action",
            "Choose the next Harness action. Return concise reasoning and the matching structured tool call.",
        ]
    )


def messages_to_rows(record: dict[str, Any], source: str) -> Iterable[dict[str, Any]]:
    messages = record.get("messages") or record.get("conversations")
    if not isinstance(messages, list):
        return
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = clean_text(message.get("role")).lower()
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        normalized.append(
            {
                "role": role,
                "content": clean_text(message.get("content")),
                "tool_calls": clean_text(message.get("tool_calls")),
            }
        )
    if normalized:
        yield {"conversations": normalized, "source": source}


def trajectory_to_turn_rows(
    record: dict[str, Any],
    *,
    source: str,
    min_final_recall: float,
    keep_unscored: bool,
    recent_turns: int,
    max_observation_chars: int,
    max_older_summary_chars: int,
    max_turns: int,
    system_prompt: str,
) -> Iterable[dict[str, Any]]:
    final_recall = record.get("final_recall")
    if final_recall is not None:
        try:
            if float(final_recall) < min_final_recall:
                return
        except (TypeError, ValueError):
            pass
    elif not keep_unscored:
        return

    query = extract_query(record)
    turns = extract_turns(record)
    if not query or not turns:
        return

    if max_turns > 0:
        turns = turns[:max_turns]

    for idx, turn in enumerate(turns):
        tool_name = extract_tool_name(turn)
        if not tool_name:
            continue
        params = extract_params(turn)
        reasoning = extract_reasoning(turn)
        conversations = [
            {"role": "system", "content": system_prompt, "tool_calls": ""},
            {
                "role": "user",
                "content": build_user_context(
                    query=query,
                    previous_turns=turns[:idx],
                    recent_turns=recent_turns,
                    max_observation_chars=max_observation_chars,
                    max_older_summary_chars=max_older_summary_chars,
                ),
                "tool_calls": "",
            },
            {
                "role": "assistant",
                "content": reasoning,
                "tool_calls": make_tool_calls(tool_name, params),
            },
        ]
        yield {"conversations": conversations, "source": source}


def build_dataset(args: argparse.Namespace) -> tuple[Dataset, dict[str, Any]]:
    input_paths = [Path(path).expanduser().resolve() for path in args.harness_input]
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()

    for record, source in iter_json_records(input_paths):
        stats["input_records"] += 1
        if args.max_trajectories and stats["kept_trajectories"] >= args.max_trajectories:
            break

        produced = list(
            trajectory_to_turn_rows(
                record,
                source=source,
                min_final_recall=args.min_final_recall,
                keep_unscored=args.keep_unscored,
                recent_turns=args.recent_turns,
                max_observation_chars=args.max_observation_chars,
                max_older_summary_chars=args.max_older_summary_chars,
                max_turns=args.max_turns,
                system_prompt=args.system_prompt,
            )
        )
        if not produced and args.pass_through_conversations:
            produced = list(messages_to_rows(record, source))

        if not produced:
            stats["skipped_records"] += 1
            continue

        stats["kept_trajectories"] += 1
        for row in produced:
            if args.max_samples and len(rows) >= args.max_samples:
                break
            rows.append(row)
            stats["samples"] += 1
            assistant = row["conversations"][-1]
            if assistant.get("tool_calls"):
                try:
                    call = json.loads(assistant["tool_calls"])[0]
                    tool_counts[call["function"]["name"]] += 1
                except Exception:
                    tool_counts["unparsed_tool_call"] += 1
        if args.max_samples and len(rows) >= args.max_samples:
            break

    if not rows:
        raise RuntimeError(
            "No Harness rows were produced. Check --harness-input, min recall, and trajectory schema."
        )

    dataset = Dataset.from_list(rows, features=MESSAGE_FEATURES)
    meta = {
        "format": "LFM conversations from Harness-1 SFT turn trajectories",
        "sample_mode": "turn_context",
        "input_paths": [str(path) for path in input_paths],
        "min_final_recall": args.min_final_recall,
        "keep_unscored": args.keep_unscored,
        "recent_turns": args.recent_turns,
        "max_observation_chars": args.max_observation_chars,
        "max_older_summary_chars": args.max_older_summary_chars,
        "max_turns": args.max_turns,
        "stats": dict(stats),
        "tool_counts": dict(tool_counts.most_common()),
    }
    return dataset, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-input", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--min-final-recall", type=float, default=0.10)
    parser.add_argument("--keep-unscored", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recent-turns", type=int, default=5)
    parser.add_argument("--max-observation-chars", type=int, default=12000)
    parser.add_argument("--max-older-summary-chars", type=int, default=2000)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-trajectories", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--pass-through-conversations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_path).expanduser().resolve()
    final_path, _, ready_path = dataset_paths(output_path)
    if final_path.exists() and ready_path.exists() and not args.overwrite:
        print(f"dataset already exists: {final_path}")
        return

    dataset, meta = build_dataset(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(dataset, output_path)
    (output_path / "build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
