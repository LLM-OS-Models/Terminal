#!/usr/bin/env python
"""Build high-recall SFT rows from failed or partial retrieval evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-f2", type=float, default=0.999)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--include-repair-rows", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def clean_ids(values: Any, candidate_ids: set[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = str(value).strip()
        if item and item in candidate_ids and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def f2_of(row: dict[str, Any]) -> float:
    if "f2" in row:
        return float(row.get("f2") or 0.0)
    metrics = row.get("metrics") or {}
    return float(metrics.get("f2") or 0.0)


def row_index(row: dict[str, Any]) -> int | None:
    value = row.get("row_idx")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_target(gold_ids: list[str], answer: str) -> dict[str, Any]:
    reason = "Selected every supporting evidence document, including intermediate clue pages and final-answer evidence."
    if answer:
        reason = f"Selected every supporting evidence document needed to verify the answer: {answer[:220]}"
    return {"curated_doc_ids": gold_ids, "reasoning": reason}


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)
    predictions = [row for row in read_jsonl(Path(args.predictions_jsonl)) if f2_of(row) < args.min_f2]
    if args.max_rows > 0:
        predictions = predictions[: args.max_rows]

    records: list[dict[str, Any]] = []
    for pred in predictions:
        idx = row_index(pred)
        if idx is None or idx < 0 or idx >= len(dataset):
            continue
        row = dict(dataset[idx])
        candidate_ids = {str(item) for item in row.get("candidate_doc_ids", [])}
        gold_ids = clean_ids(row.get("gold_doc_ids", []), candidate_ids)
        if not gold_ids:
            continue
        answer = str(row.get("answer") or "").strip()
        target = build_target(gold_ids, answer)

        base_messages = list(row["prompt"])
        direct_record = {
            "messages": base_messages + [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}],
            "query_id": row.get("query_id"),
            "source": row.get("source"),
            "gold_doc_ids": gold_ids,
            "task_type": "hardcase_direct_high_recall",
            "prior_f2": f2_of(pred),
        }
        for _ in range(max(args.repeat, 1)):
            records.append(direct_record)

        if args.include_repair_rows:
            wrong = str(pred.get("completion") or pred.get("raw") or "")
            repair_messages = (
                base_messages
                + [{"role": "assistant", "content": wrong[:1200]}]
                + [
                    {
                        "role": "user",
                        "content": (
                            "The previous curation missed or corrupted required evidence IDs. "
                            "Return corrected strict JSON now. Include all supporting gold evidence IDs "
                            "from the candidate pool and do not invent IDs."
                        ),
                    },
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ]
            )
            records.append(
                {
                    "messages": repair_messages,
                    "query_id": row.get("query_id"),
                    "source": row.get("source"),
                    "gold_doc_ids": gold_ids,
                    "task_type": "hardcase_repair_high_recall",
                    "prior_f2": f2_of(pred),
                }
            )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    Dataset.from_list(records).save_to_disk(str(output_path.with_suffix("")))
    print(
        json.dumps(
            {
                "prediction_rows": len(predictions),
                "sft_rows": len(records),
                "jsonl": str(output_path),
                "dataset": str(output_path.with_suffix("")),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
