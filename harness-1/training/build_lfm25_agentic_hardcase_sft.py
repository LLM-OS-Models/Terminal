#!/usr/bin/env python
"""Build agentic prompt hardcase SFT rows from local vLLM agent evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from eval_scripts.eval_lfm25_agentic_vllm import SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--min-f2", type=float, default=0.999)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--include-repair-rows", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_f2(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    return float(row.get("f2") or metrics.get("f2") or 0.0)


def prediction_index(row: dict[str, Any], fallback: int) -> int:
    for key in ("row_idx", "dataset_idx", "idx"):
        value = row.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return fallback


def unique_valid_ids(values: Any, candidate_ids: set[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        doc_id = str(value).strip()
        if doc_id and doc_id in candidate_ids and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def build_user_content(row: dict[str, Any]) -> str:
    source_messages = list(row["prompt"])
    source_user = next((m.get("content", "") for m in source_messages if m.get("role") == "user"), "")
    if not source_user:
        source_user = f"# Query\n{clean_text(row.get('query'))}"
    return (
        source_user.rstrip()
        + "\n\n# Agent Instructions\n"
        + "Ignore any earlier natural-language final-answer format in this prompt. "
        + "Select all supporting evidence from the Candidate Document Pool above and output the final curation action now. "
        + "Prefer exactly this JSON shape: {\"curated_doc_ids\":[\"doc_id\"],\"reasoning\":\"brief reason\"}. "
        + "Most hard queries need multiple IDs; do not stop after only the final-answer page if intermediate evidence is relevant. "
        + "Only use search/review tools if the pool is insufficient."
    )


def build_target(row: dict[str, Any], gold_ids: list[str]) -> dict[str, Any]:
    answer = clean_text(row.get("answer"))
    reason = "Selected every supporting evidence document from the candidate pool."
    if answer:
        reason = f"Selected every supporting evidence document needed to verify the answer: {answer[:220]}"
    return {"curated_doc_ids": gold_ids, "reasoning": reason}


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)
    predictions = [row for row in read_jsonl(Path(args.predictions_jsonl)) if metric_f2(row) < args.min_f2]
    if args.max_rows > 0:
        predictions = predictions[: args.max_rows]

    records: list[dict[str, Any]] = []
    for fallback_idx, pred in enumerate(predictions):
        idx = prediction_index(pred, fallback_idx)
        if idx < 0 or idx >= len(dataset):
            continue
        row = dict(dataset[idx])
        candidate_ids = {str(item) for item in row.get("candidate_doc_ids", [])}
        gold_ids = unique_valid_ids(row.get("gold_doc_ids", []), candidate_ids)
        if not gold_ids:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(row)},
        ]
        target = build_target(row, gold_ids)
        target_text = json.dumps(target, ensure_ascii=False)
        base_record = {
            "messages": messages + [{"role": "assistant", "content": target_text}],
            "query_id": row.get("query_id"),
            "source": row.get("source"),
            "gold_doc_ids": gold_ids,
            "task_type": "agentic_direct_hardcase_high_recall",
            "prior_f2": metric_f2(pred),
        }
        for _ in range(max(args.repeat, 1)):
            records.append(base_record)

        if args.include_repair_rows:
            trace = pred.get("trace") or []
            wrong = ""
            if trace and isinstance(trace, list):
                wrong = str(trace[0].get("raw") or trace[0].get("action") or "")
            if not wrong:
                wrong = json.dumps({"curated_doc_ids": pred.get("curated_doc_ids", [])}, ensure_ascii=False)
            repair_messages = (
                messages
                + [{"role": "assistant", "content": wrong[:1200]}]
                + [
                    {
                        "role": "user",
                        "content": (
                            "The previous agent action missed required supporting evidence. "
                            "Return the corrected final curation JSON now. Include all gold evidence IDs "
                            "that appear in the candidate pool and do not invent IDs."
                        ),
                    },
                    {"role": "assistant", "content": target_text},
                ]
            )
            records.append(
                {
                    "messages": repair_messages,
                    "query_id": row.get("query_id"),
                    "source": row.get("source"),
                    "gold_doc_ids": gold_ids,
                    "task_type": "agentic_repair_hardcase_high_recall",
                    "prior_f2": metric_f2(pred),
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
