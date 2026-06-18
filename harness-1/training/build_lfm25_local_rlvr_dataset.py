#!/usr/bin/env python
"""Build local GRPO data for LFM2.5 Harness-style retrieval RLVR.

This is the local-HF path for LFM2.5 experiments. It does not replace the
original Tinker/Chroma Harness-1 RL pipeline in ``training/train_rl.py``.

The paper-aligned SEC path requires both:
- the SEC query dataset, e.g. ``kellyhongg/1_18_sec_train``;
- a real retrieval candidate pool from the same SEC corpus/index.

If SEC records are available only as gold facts/chunk IDs, this builder fails by
default so we do not silently train on gold-only candidates and call it paper RL.
Use ``--allow-gold-only-candidates`` only for wiring checks.

For local reproduction, ``--dataset-kind sec_jsonl`` accepts materialized rows
with query/gold docs plus retrieved candidate docs from the SEC index.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Sequence, Value, load_dataset


FEATURES = Features(
    {
        "prompt": [{"role": Value("string"), "content": Value("string")}],
        "query_id": Value("string"),
        "query": Value("string"),
        "answer": Value("string"),
        "gold_doc_ids": Sequence(Value("string")),
        "answer_doc_ids": Sequence(Value("string")),
        "candidate_doc_ids": Sequence(Value("string")),
        "source": Value("string"),
    }
)

SYSTEM_PROMPT = """You are training for a Harness-1 retrieval environment.

Given a query and a candidate document pool, curate all document IDs that are
needed to answer or verify the query. Prefer recall over precision, but avoid
irrelevant distractors.

Output only strict JSON:
{"curated_doc_ids":["doc_id_1","doc_id_2"],"reasoning":"brief evidence-based reason"}

Rules:
- Use only IDs from the candidate pool.
- Include all documents supporting constraints, intermediate clues, or final answer.
- Keep reasoning short. No markdown."""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except Exception:
        pass
    try:
        return ast.literal_eval(stripped)
    except Exception:
        return value


def compact(text: str, limit: int) -> str:
    text = clean_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(0, limit // 2)
    tail = max(0, limit - head)
    return f"{text[:head]} ...[truncated {len(text) - limit} chars]... {text[-tail:]}"


def doc_id(doc: dict[str, Any]) -> str:
    return clean_text(doc.get("docid") or doc.get("doc_id") or doc.get("id"))


def doc_title(doc: dict[str, Any]) -> str:
    text = str(doc.get("text") or "")
    match = re.search(r"^title:\s*(.+)$", text, flags=re.MULTILINE)
    if match:
        return clean_text(match.group(1))
    return clean_text(doc.get("title")) or doc_id(doc)


def unique_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for doc in docs:
        did = doc_id(doc)
        if not did or did in seen:
            continue
        seen.add(did)
        out.append(doc)
    return out


def load_browsecomp_records(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("query") and row.get("gold_docs"):
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def docs_from_document_ids(value: Any) -> list[dict[str, Any]]:
    document_ids = parse_jsonish(value)
    docs: list[dict[str, Any]] = []
    if not isinstance(document_ids, list):
        return docs
    for item in document_ids:
        if isinstance(item, dict):
            fact = clean_text(item.get("fact"))
            chunk_ids = parse_jsonish(item.get("chunk_ids"))
            if not isinstance(chunk_ids, list):
                continue
            for cid in chunk_ids:
                cid_s = clean_text(cid)
                if cid_s:
                    docs.append(
                        {
                            "docid": cid_s,
                            "title": "SEC filing evidence chunk",
                            "text": fact or cid_s,
                            "is_final_answer": bool(item.get("is_final_answer", False)),
                        }
                    )
        else:
            cid_s = clean_text(item)
            if cid_s:
                docs.append({"docid": cid_s, "title": "SEC chunk", "text": cid_s})
    return docs


def normalize_doc_list(value: Any, *, default_title: str) -> list[dict[str, Any]]:
    value = parse_jsonish(value)
    if not value:
        return []
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return []
    docs: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            did = doc_id(item)
            if did:
                docs.append(item)
            elif item.get("chunk_ids"):
                docs.extend(docs_from_document_ids([item]))
        else:
            did = clean_text(item)
            if did:
                docs.append({"docid": did, "title": default_title, "text": did})
    return docs


def load_sec_jsonl_records(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            query = clean_text(row.get("query") or row.get("question"))
            if not query:
                continue
            gold_docs = normalize_doc_list(row.get("gold_docs"), default_title="SEC gold chunk")
            if not gold_docs and row.get("document_ids"):
                gold_docs = docs_from_document_ids(row.get("document_ids"))
            candidate_docs = normalize_doc_list(row.get("candidate_docs"), default_title="SEC candidate chunk")
            if not candidate_docs and row.get("candidates"):
                candidate_docs = normalize_doc_list(row.get("candidates"), default_title="SEC candidate chunk")
            evidence_docs = normalize_doc_list(row.get("evidence_docs"), default_title="SEC evidence chunk")
            negative_docs = normalize_doc_list(row.get("negative_docs"), default_title="SEC retrieved distractor")
            if not gold_docs:
                continue
            rows.append(
                {
                    "query_id": clean_text(row.get("query_id") or row.get("id") or idx),
                    "query": query,
                    "answer": clean_text(row.get("answer")),
                    "gold_docs": gold_docs,
                    "candidate_docs": candidate_docs,
                    "evidence_docs": evidence_docs,
                    "negative_docs": negative_docs,
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def load_harness_hf_records(
    *,
    hf_path: str,
    split: str,
    split_ids_path: Path | None,
    split_id_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        dataset = load_dataset(hf_path, split=split, token=token)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load HF dataset {hf_path!r} split {split!r}. "
            "For paper-aligned SEC RL, provide access to the Harness-1 SEC HF "
            "data or materialize an equivalent local JSONL with retrieved "
            "candidate documents."
        ) from exc

    allowed_ids: set[str] | None = None
    if split_ids_path:
        split_obj = json.loads(split_ids_path.read_text(encoding="utf-8"))
        ids = split_obj.get(split_id_key)
        if not isinstance(ids, list):
            raise RuntimeError(f"{split_ids_path} has no list key {split_id_key!r}")
        allowed_ids = {str(item) for item in ids}

    rows: list[dict[str, Any]] = []
    for row in dataset:
        query_id = clean_text(row.get("query_id"))
        if allowed_ids is not None and query_id not in allowed_ids:
            continue
        query = clean_text(row.get("query"))
        answer = clean_text(row.get("answer"))
        docs = docs_from_document_ids(row.get("document_ids"))
        if not query or not docs:
            continue
        if docs:
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "answer": answer,
                    "gold_docs": docs,
                    "evidence_docs": [],
                    "negative_docs": [],
                }
            )
        if limit and len(rows) >= limit:
            break
    return rows


def render_candidates(docs: list[dict[str, Any]], max_doc_chars: int) -> str:
    parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        parts.append(
            "\n".join(
                [
                    f"[{idx}] doc_id: {doc_id(doc)}",
                    f"title: {doc_title(doc)}",
                    f"snippet: {compact(str(doc.get('text') or ''), max_doc_chars)}",
                ]
            )
        )
    return "\n\n".join(parts)


def make_candidates(
    record: dict[str, Any],
    *,
    max_gold_docs: int,
    max_evidence_docs: int,
    max_negative_docs: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    gold_docs = [doc for doc in record.get("gold_docs", []) if isinstance(doc, dict)]
    candidate_docs = [doc for doc in record.get("candidate_docs", []) if isinstance(doc, dict)]
    evidence_docs = [doc for doc in record.get("evidence_docs", []) if isinstance(doc, dict)]
    negative_docs = [doc for doc in record.get("negative_docs", []) if isinstance(doc, dict)]
    gold_docs = gold_docs[:max_gold_docs]
    if candidate_docs:
        candidates = unique_docs(candidate_docs + gold_docs)
        rng.shuffle(candidates)
        return candidates
    evidence_docs = evidence_docs[:max_evidence_docs]
    if max_negative_docs > 0 and len(negative_docs) > max_negative_docs:
        negative_docs = rng.sample(negative_docs, max_negative_docs)
    candidates = unique_docs(gold_docs + evidence_docs + negative_docs)
    rng.shuffle(candidates)
    return candidates


def answer_doc_ids(record: dict[str, Any], gold_ids: list[str]) -> list[str]:
    gold_set = set(gold_ids)
    out: list[str] = []
    for doc in record.get("gold_docs", []):
        if not isinstance(doc, dict):
            continue
        did = doc_id(doc)
        if did in gold_set and bool(doc.get("is_final_answer", False)):
            out.append(did)
    return out or gold_ids


def build_prompt(record: dict[str, Any], candidates: list[dict[str, Any]], max_doc_chars: int) -> list[dict[str, str]]:
    user = "\n\n".join(
        [
            "# Query",
            clean_text(record.get("query")),
            "# Candidate Document Pool",
            render_candidates(candidates, max_doc_chars),
            "# Task",
            "Return strict JSON with schema {\"curated_doc_ids\":[\"...\"],\"reasoning\":\"...\"}.",
        ]
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_dataset(args: argparse.Namespace) -> tuple[Dataset, dict[str, Any]]:
    rng = random.Random(args.seed)
    if args.dataset_kind == "browsecomp":
        if not args.input_jsonl:
            raise RuntimeError("--input-jsonl is required for browsecomp")
        source_path = Path(args.input_jsonl).expanduser().resolve()
        records = load_browsecomp_records(source_path, args.limit)
        source = str(source_path)
    elif args.dataset_kind == "sec_hf":
        split_ids_path = Path(args.split_ids_path).expanduser().resolve() if args.split_ids_path else None
        records = load_harness_hf_records(
            hf_path=args.hf_dataset,
            split=args.hf_split,
            split_ids_path=split_ids_path,
            split_id_key=args.split_id_key,
            limit=args.limit,
        )
        source = f"{args.hf_dataset}:{args.hf_split}:{args.split_id_key}"
    elif args.dataset_kind == "sec_jsonl":
        if not args.input_jsonl:
            raise RuntimeError("--input-jsonl is required for sec_jsonl")
        source_path = Path(args.input_jsonl).expanduser().resolve()
        records = load_sec_jsonl_records(source_path, args.limit)
        source = str(source_path)
    else:
        raise RuntimeError(f"Unsupported dataset kind: {args.dataset_kind}")

    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for record in records:
        candidates = make_candidates(
            record,
            max_gold_docs=args.max_gold_docs,
            max_evidence_docs=args.max_evidence_docs,
            max_negative_docs=args.max_negative_docs,
            rng=rng,
        )
        gold_ids = [doc_id(doc) for doc in record.get("gold_docs", []) if isinstance(doc, dict) and doc_id(doc)]
        candidate_ids = [doc_id(doc) for doc in candidates if doc_id(doc)]
        if (
            args.dataset_kind in {"sec_hf", "sec_jsonl"}
            and not args.allow_gold_only_candidates
            and not (set(candidate_ids) - set(gold_ids))
        ):
            stats["skipped_gold_only_candidates"] += 1
            continue
        if not gold_ids or not set(gold_ids).intersection(candidate_ids):
            stats["skipped_no_gold_in_candidates"] += 1
            continue
        if len(candidate_ids) < args.min_candidates:
            stats["skipped_too_few_candidates"] += 1
            continue
        rows.append(
            {
                "prompt": build_prompt(record, candidates, args.max_doc_chars),
                "query_id": clean_text(record.get("query_id")),
                "query": clean_text(record.get("query")),
                "answer": clean_text(record.get("answer")),
                "gold_doc_ids": gold_ids,
                "answer_doc_ids": answer_doc_ids(record, gold_ids),
                "candidate_doc_ids": candidate_ids,
                "source": source,
            }
        )
        stats["rows"] += 1
        stats["candidate_docs"] += len(candidate_ids)
        stats["gold_docs"] += len(gold_ids)

    if not rows:
        if stats.get("skipped_gold_only_candidates"):
            raise RuntimeError(
                "No RLVR rows were produced because SEC candidates were gold-only. "
                "For paper-aligned SEC RL, materialize retrieved candidate_docs or "
                "negative_docs from sec_1_4. Use --allow-gold-only-candidates only "
                "for smoke tests."
            )
        raise RuntimeError("No RLVR rows were produced.")
    dataset = Dataset.from_list(rows, features=FEATURES)
    meta = {
        "format": "harness1_lfm25_local_rlvr_v1",
        "dataset_kind": args.dataset_kind,
        "source": source,
        "rows": len(rows),
        "stats": dict(stats),
        "paper_aligned": args.dataset_kind in {"sec_hf", "sec_jsonl"} and not args.allow_gold_only_candidates,
        "note": "For paper SEC RL, use SEC rows with real retrieval candidates from sec_1_4.",
    }
    return dataset, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-kind", choices=("browsecomp", "sec_hf", "sec_jsonl"), default="sec_hf")
    parser.add_argument("--input-jsonl", default="")
    parser.add_argument("--hf-dataset", default="kellyhongg/1_18_sec_train")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--split-ids-path", default="datagen/splits/sec_splits.json")
    parser.add_argument("--split-id-key", default="rl_query_ids")
    parser.add_argument("--allow-gold-only-candidates", action="store_true")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-doc-chars", type=int, default=900)
    parser.add_argument("--max-gold-docs", type=int, default=12)
    parser.add_argument("--max-evidence-docs", type=int, default=12)
    parser.add_argument("--max-negative-docs", type=int, default=18)
    parser.add_argument("--min-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_path).expanduser().resolve()
    ready_path = output_path.with_name(output_path.name + ".ready")
    if output_path.exists() and ready_path.exists() and not args.overwrite:
        print(f"dataset already exists: {output_path}")
        return
    if output_path.exists() and args.overwrite:
        shutil.rmtree(output_path)
    if ready_path.exists():
        ready_path.unlink()

    dataset, meta = build_dataset(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    dataset.save_to_disk(str(tmp_path))
    tmp_path.rename(output_path)
    (output_path / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ready_path.write_text("ok\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
