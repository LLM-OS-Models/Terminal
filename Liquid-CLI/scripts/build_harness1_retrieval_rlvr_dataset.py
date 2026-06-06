#!/usr/bin/env python
"""Build a local retrieval-RLVR dataset for Harness-1 style curation.

The local public path uses BrowseComp+ because it ships with query, gold,
evidence, and negative documents. The paper-aligned path is SEC-only RL: when
the Harness-1 SEC HF datasets and retrieval backend are accessible, this script
can build prompts from their fact-level gold chunk IDs. If no retrieved candidate
text is available for SEC, it fails instead of silently training on the wrong
domain.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Sequence, Value, load_dataset


FEATURES = Features(
    {
        "prompt": [
            {
                "role": Value("string"),
                "content": Value("string"),
            }
        ],
        "query_id": Value("string"),
        "query": Value("string"),
        "answer": Value("string"),
        "gold_doc_ids": Sequence(Value("string")),
        "answer_doc_ids": Sequence(Value("string")),
        "candidate_doc_ids": Sequence(Value("string")),
        "source": Value("string"),
    }
)


SYSTEM_PROMPT = """You are training for a Harness-1 style retrieval environment.

Your job is high-recall evidence curation. Given a query and a candidate document
pool, select every document that is plausibly needed to answer or verify the
query. Do not answer from memory. Prefer recall over precision, but do not select
irrelevant distractors.

Output only strict JSON:
{"curated_doc_ids":["doc_id_1","doc_id_2"],"reasoning":"brief evidence-based reason"}

Rules:
- Use only doc IDs from the candidate pool.
- Include all documents that support required constraints, intermediate clues,
  or the final answer.
- If several documents are needed together, curate all of them.
- Keep reasoning short. No markdown."""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compact(text: str, limit: int) -> str:
    text = clean_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(0, limit // 2)
    tail = max(0, limit - head)
    return f"{text[:head]} ...[truncated {len(text) - limit} chars]... {text[-tail:]}"


def doc_title(doc: dict[str, Any]) -> str:
    text = str(doc.get("text") or "")
    match = re.search(r"^title:\s*(.+)$", text, flags=re.MULTILINE)
    if match:
        return clean_text(match.group(1))
    return clean_text(doc.get("title")) or clean_text(doc.get("docid"))


def doc_id(doc: dict[str, Any]) -> str:
    return clean_text(doc.get("docid"))


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


def load_browsecomp_records(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("query") and row.get("gold_docs"):
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        for parser in (json.loads,):
            try:
                return parser(stripped)
            except Exception:
                pass
        try:
            import ast

            return ast.literal_eval(stripped)
        except Exception:
            return value
    return value


def load_harness_hf_records(
    *,
    hf_path: str,
    split: str,
    split_ids_path: Path | None,
    split_id_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        dataset = load_dataset(hf_path, split=split, token=hf_token)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load HF dataset {hf_path!r} split {split!r}. "
            "For paper-aligned SEC RL, provide access to kellyhongg SEC data "
            "or first materialize an equivalent local JSONL with retrieved candidates."
        ) from exc

    allowed_ids: set[str] | None = None
    if split_ids_path:
        split_obj = json.loads(split_ids_path.read_text(encoding="utf-8"))
        ids = split_obj.get(split_id_key)
        if not isinstance(ids, list):
            raise RuntimeError(f"{split_ids_path} does not contain list key {split_id_key!r}")
        allowed_ids = {str(item) for item in ids}

    rows: list[dict[str, Any]] = []
    for row in dataset:
        query_id = clean_text(row.get("query_id"))
        if allowed_ids is not None and query_id not in allowed_ids:
            continue
        query = clean_text(row.get("query"))
        document_ids = parse_jsonish(row.get("document_ids"))
        answer = clean_text(row.get("answer"))
        if not query or not document_ids:
            continue

        docs: list[dict[str, Any]] = []
        if isinstance(document_ids, list):
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
                                    "url": "",
                                    "text": fact or cid_s,
                                    "is_final_answer": bool(item.get("is_final_answer", False)),
                                }
                            )
                else:
                    cid_s = clean_text(item)
                    if cid_s:
                        docs.append(
                            {
                                "docid": cid_s,
                                "title": "SEC filing evidence chunk",
                                "url": "",
                                "text": cid_s,
                                "is_final_answer": True,
                            }
                        )
        if not docs:
            continue
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


def answer_doc_ids(record: dict[str, Any], gold_ids: list[str]) -> list[str]:
    evidence_ids = [doc_id(doc) for doc in record.get("evidence_docs", []) if isinstance(doc, dict)]
    out = [did for did in evidence_ids if did in set(gold_ids)]
    return out or gold_ids


def render_candidates(docs: list[dict[str, Any]], *, max_doc_chars: int) -> str:
    lines: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        did = doc_id(doc)
        title = doc_title(doc)
        url = clean_text(doc.get("url"))
        text = compact(str(doc.get("text") or ""), max_doc_chars)
        lines.append(
            "\n".join(
                [
                    f"[{idx}] doc_id: {did}",
                    f"title: {title}",
                    f"url: {url}",
                    f"snippet: {text}",
                ]
            )
        )
    return "\n\n".join(lines)


def build_prompt(record: dict[str, Any], candidates: list[dict[str, Any]], *, max_doc_chars: int) -> list[dict[str, str]]:
    query = clean_text(record.get("query"))
    user = "\n\n".join(
        [
            "# Query",
            query,
            "# Candidate Document Pool",
            render_candidates(candidates, max_doc_chars=max_doc_chars),
            "# Task",
            (
                "Return strict JSON selecting the document IDs that should be in the final "
                "curated evidence set. Use the schema exactly: "
                "{\"curated_doc_ids\":[\"...\"],\"reasoning\":\"...\"}."
            ),
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def make_candidates(
    record: dict[str, Any],
    *,
    max_gold_docs: int,
    max_evidence_docs: int,
    max_negative_docs: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    gold_docs = [doc for doc in record.get("gold_docs", []) if isinstance(doc, dict)]
    evidence_docs = [doc for doc in record.get("evidence_docs", []) if isinstance(doc, dict)]
    negative_docs = [doc for doc in record.get("negative_docs", []) if isinstance(doc, dict)]

    gold_docs = gold_docs[:max_gold_docs]
    evidence_docs = evidence_docs[:max_evidence_docs]
    if max_negative_docs > 0 and len(negative_docs) > max_negative_docs:
        negative_docs = rng.sample(negative_docs, max_negative_docs)

    candidates = unique_docs(gold_docs + evidence_docs + negative_docs)
    rng.shuffle(candidates)
    return candidates


def build_dataset(args: argparse.Namespace) -> tuple[Dataset, dict[str, Any]]:
    rng = random.Random(args.seed)
    input_path = Path(args.input_jsonl).expanduser().resolve() if args.input_jsonl else None
    if args.dataset_kind in {"browsecomp", "local_jsonl"}:
        if input_path is None:
            raise RuntimeError("--input-jsonl is required for local JSONL dataset kinds")
        records = load_browsecomp_records(input_path, limit=args.limit)
        source = str(input_path)
    elif args.dataset_kind == "harness_hf":
        split_ids_path = (
            Path(args.split_ids_path).expanduser().resolve()
            if args.split_ids_path
            else None
        )
        records = load_harness_hf_records(
            hf_path=args.hf_dataset,
            split=args.hf_split,
            split_ids_path=split_ids_path,
            split_id_key=args.split_id_key,
            limit=args.limit,
        )
        source = f"{args.hf_dataset}:{args.hf_split}:{args.split_id_key}"
    else:
        raise RuntimeError(f"unsupported dataset kind: {args.dataset_kind}")

    if args.dataset_kind == "harness_hf" and not args.allow_gold_only_candidates:
        raise RuntimeError(
            "Harness HF records contain gold IDs/facts but no retrieval candidate pool. "
            "Set --allow-gold-only-candidates only for wiring checks, or provide a "
            "materialized SEC JSONL with retrieved negatives."
        )

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
        if not gold_ids or not set(gold_ids).intersection(candidate_ids):
            stats["skipped_no_gold_in_candidates"] += 1
            continue
        if len(candidate_ids) < args.min_candidates:
            stats["skipped_too_few_candidates"] += 1
            continue

        rows.append(
            {
                "prompt": build_prompt(record, candidates, max_doc_chars=args.max_doc_chars),
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
        raise RuntimeError(f"No RLVR rows were produced from {input_path}")

    dataset = Dataset.from_list(rows, features=FEATURES)
    meta = {
        "format": "harness1_retrieval_rlvr_v1",
        "dataset_kind": args.dataset_kind,
        "source": source,
        "rows": len(rows),
        "seed": args.seed,
        "max_doc_chars": args.max_doc_chars,
        "max_gold_docs": args.max_gold_docs,
        "max_evidence_docs": args.max_evidence_docs,
        "max_negative_docs": args.max_negative_docs,
        "stats": dict(stats),
        "note": (
            "Local RLVR candidate-curation dataset. BrowseComp+ is a public proxy; "
            "paper-aligned RL should use SEC-only records with a real retrieval "
            "candidate pool and held-out evaluation."
        ),
    }
    return dataset, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-kind",
        choices=("browsecomp", "local_jsonl", "harness_hf"),
        default="local_jsonl",
    )
    parser.add_argument(
        "--input-jsonl",
        default="/home/work/.data/harness1/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl",
    )
    parser.add_argument("--hf-dataset", default="kellyhongg/1_18_sec_train")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--split-ids-path", default="")
    parser.add_argument("--split-id-key", default="rl_query_ids")
    parser.add_argument(
        "--allow-gold-only-candidates",
        action="store_true",
        help="Allow HF gold-only candidate pools. Use only as a wiring check, not final RL.",
    )
    parser.add_argument(
        "--output-path",
        default="/home/work/.data/liquid_cli_sft/datasets/lfm25_harness1_retrieval_rlvr_browsecomp_v1",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-doc-chars", type=int, default=1400)
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
        import shutil

        shutil.rmtree(output_path)
    if ready_path.exists():
        ready_path.unlink()

    dataset, meta = build_dataset(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    if tmp_path.exists():
        import shutil

        shutil.rmtree(tmp_path)
    dataset.save_to_disk(str(tmp_path))
    tmp_path.rename(output_path)
    (output_path / "build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ready_path.write_text("ok\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
