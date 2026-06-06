#!/usr/bin/env python
"""Convert Context-1 SEC generation outputs into Harness-style RLVR JSONL.

The Context-1 SEC pipeline writes one JSON file per ticker. Each file contains
tasks with a question, truth, supporting item IDs, and `items_and_contents`.
This converter materializes those tasks into the local JSONL schema consumed by
`build_harness1_retrieval_rlvr_dataset.py`:

  query, answer, gold_docs, evidence_docs, negative_docs

Gold documents are supporting chunks for the task. Negatives are sampled from
other SEC chunks found in the same output directory, so the resulting curation
task stays SEC-only without needing the private Harness-1 Chroma collection.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def iter_json_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def load_objects(input_dir: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for path in iter_json_files(input_dir):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict):
            obj["_source_path"] = str(path)
            objects.append(obj)
    return objects


def as_doc(docid: str, text: str, *, ticker: str, company: str, source_path: str) -> dict[str, str]:
    title_bits = [bit for bit in (ticker, company, docid) if bit]
    return {
        "docid": clean_text(docid),
        "title": " | ".join(title_bits),
        "url": "",
        "text": clean_text(text),
        "source_path": source_path,
    }


def supporting_ids(task: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in task.get("supporting_items", []) or []:
        if not isinstance(item, dict):
            continue
        candidates = [
            item.get("id"),
            item.get("item_id"),
            item.get("chunk_id"),
            item.get("docid"),
        ]
        for candidate in candidates:
            cid = clean_text(candidate)
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
    bridging = task.get("bridging_item")
    if isinstance(bridging, dict):
        for key in ("id", "item_id", "chunk_id", "docid"):
            cid = clean_text(bridging.get(key))
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
    return ids


def additional_docs(task: dict[str, Any], contents: dict[str, Any], *, ticker: str, company: str, source_path: str) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in task.get("supporting_items", []) or []:
        if not isinstance(item, dict):
            continue
        for extra in item.get("additional_chunks", []) or []:
            if isinstance(extra, dict):
                cid = clean_text(extra.get("id") or extra.get("chunk_id") or extra.get("docid"))
                text = clean_text(extra.get("text") or extra.get("content") or contents.get(cid))
            else:
                cid = clean_text(extra)
                text = clean_text(contents.get(cid))
            if cid and text and cid not in seen:
                seen.add(cid)
                docs.append(as_doc(cid, text, ticker=ticker, company=company, source_path=source_path))
    return docs


def collect_global_docs(objects: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    docs: dict[str, dict[str, str]] = {}
    for obj in objects:
        ticker = clean_text(obj.get("ticker"))
        company = clean_text(obj.get("company_name"))
        source_path = clean_text(obj.get("_source_path"))
        for task in obj.get("tasks", []) or []:
            if not isinstance(task, dict):
                continue
            contents = task.get("items_and_contents") or {}
            if not isinstance(contents, dict):
                continue
            for docid, text in contents.items():
                did = clean_text(docid)
                body = clean_text(text)
                if did and body and did not in docs:
                    docs[did] = as_doc(did, body, ticker=ticker, company=company, source_path=source_path)
    return docs


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_dir = Path(args.input_dir).expanduser().resolve()
    objects = load_objects(input_dir)
    global_docs = collect_global_docs(objects)
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for obj in objects:
        ticker = clean_text(obj.get("ticker"))
        company = clean_text(obj.get("company_name"))
        source_path = clean_text(obj.get("_source_path"))
        tasks = obj.get("tasks", []) or []
        for task_index, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            question = clean_text(task.get("question") or task.get("clues"))
            answer = clean_text(task.get("truth") or task.get("answer"))
            contents = task.get("items_and_contents") or {}
            if not question or not isinstance(contents, dict):
                stats["skipped_no_question_or_contents"] += 1
                continue

            gold_docs: list[dict[str, str]] = []
            for sid in supporting_ids(task):
                text = clean_text(contents.get(sid) or global_docs.get(sid, {}).get("text"))
                if text:
                    gold_docs.append(as_doc(sid, text, ticker=ticker, company=company, source_path=source_path))

            if not gold_docs:
                stats["skipped_no_gold_docs"] += 1
                continue

            evidence_docs = additional_docs(
                task,
                contents,
                ticker=ticker,
                company=company,
                source_path=source_path,
            )
            gold_ids = {doc["docid"] for doc in gold_docs}
            evidence_ids = {doc["docid"] for doc in evidence_docs}
            negative_pool = [
                doc for did, doc in global_docs.items()
                if did not in gold_ids and did not in evidence_ids
            ]
            rng.shuffle(negative_pool)
            negative_docs = negative_pool[: args.max_negative_docs]

            query_id = clean_text(task.get("query_id")) or f"{ticker}_{task_index}_{clean_text(task.get('level')) or 0}"
            rows.append(
                {
                    "query_id": query_id,
                    "query": question,
                    "answer": answer,
                    "gold_docs": gold_docs[: args.max_gold_docs],
                    "evidence_docs": evidence_docs[: args.max_evidence_docs],
                    "negative_docs": negative_docs,
                    "source_dataset": "context1_sec",
                    "source_path": source_path,
                }
            )
            stats["rows"] += 1
            stats["gold_docs"] += min(len(gold_docs), args.max_gold_docs)
            stats["evidence_docs"] += min(len(evidence_docs), args.max_evidence_docs)
            stats["negative_docs"] += len(negative_docs)
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    meta = {
        "format": "context1_sec_rlvr_jsonl_v1",
        "input_dir": str(input_dir),
        "rows": len(rows),
        "global_docs": len(global_docs),
        "seed": args.seed,
        "max_gold_docs": args.max_gold_docs,
        "max_evidence_docs": args.max_evidence_docs,
        "max_negative_docs": args.max_negative_docs,
        "stats": dict(stats),
    }
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-gold-docs", type=int, default=24)
    parser.add_argument("--max-evidence-docs", type=int, default=24)
    parser.add_argument("--max-negative-docs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_jsonl).expanduser().resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() and not args.overwrite:
        raise RuntimeError(f"output exists: {output_path} (pass --overwrite)")

    rows, meta = build_rows(args)
    if not rows:
        raise RuntimeError(f"no rows produced from {args.input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
