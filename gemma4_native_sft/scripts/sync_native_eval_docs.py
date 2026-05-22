#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/work/.data/gemma4_native_sft")
STATE_PATH = DATA_ROOT / "monitor_state.json"
LOG_PATH = DATA_ROOT / "logs/post_eval_doc_watcher.log"
DOC_SYNC_STATE_PATH = DATA_ROOT / "doc_sync_state.json"
REPORT_PATH = ROOT_DIR / "README.md"
README_PATH = ROOT_DIR / "PROJECT_OVERVIEW_2026-05-02.md"

START = "<!-- GEMMA4_NATIVE_AUTO_RESULTS_START -->"
END = "<!-- GEMMA4_NATIVE_AUTO_RESULTS_END -->"


@dataclass(frozen=True)
class EvalResult:
    repo_id: str
    checkpoint: str
    score: float
    f1: float
    precision: float
    recall: float
    first_cmd: float
    valid_json: float
    sec_per_step: float
    load_time: float
    result_path: str


def kst_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"published": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def load_doc_sync_state() -> dict:
    if not DOC_SYNC_STATE_PATH.exists():
        return {}
    try:
        return json.loads(DOC_SYNC_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_doc_sync_state(data: dict) -> None:
    DOC_SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = DOC_SYNC_STATE_PATH.with_suffix(DOC_SYNC_STATE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(DOC_SYNC_STATE_PATH)


def read_eval_result(item: dict) -> EvalResult | None:
    repo_id = item.get("repo_id")
    checkpoint = item.get("checkpoint")
    tb2_eval = item.get("tb2_eval")
    if not repo_id or not checkpoint or not isinstance(tb2_eval, dict):
        return None
    result_path = tb2_eval.get("result")
    if not result_path:
        return None
    path = Path(result_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    agg = data.get("aggregate", {})
    f1 = float(agg.get("avg_command_f1", 0.0))
    return EvalResult(
        repo_id=repo_id,
        checkpoint=checkpoint,
        score=round(100.0 * f1, 2),
        f1=f1,
        precision=float(agg.get("avg_command_precision", 0.0)),
        recall=float(agg.get("avg_command_recall", 0.0)),
        first_cmd=float(agg.get("first_cmd_exact_pct", 0.0)),
        valid_json=float(agg.get("valid_json_pct", 0.0)),
        sec_per_step=float(data.get("avg_sec_per_step", 0.0)),
        load_time=float(data.get("load_time_sec", 0.0)),
        result_path=str(path),
    )


def pending_items(state: dict) -> list[str]:
    repos: list[str] = []
    for item in state.get("published", {}).values():
        repo_id = item.get("repo_id")
        if not repo_id:
            continue
        if item.get("tb2_eval") in (None, "pending"):
            repos.append(repo_id)
    return sorted(set(repos))


def load_results() -> tuple[list[EvalResult], list[str]]:
    state = load_state()
    results: list[EvalResult] = []
    for item in state.get("published", {}).values():
        result = read_eval_result(item)
        if result:
            results.append(result)
    results.sort(key=lambda item: (-item.score, item.repo_id))
    return results, pending_items(state)


def result_signature(results: list[EvalResult], pending: list[str]) -> str:
    payload = {
        "results": [item.__dict__ for item in results],
        "pending": pending,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def epoch_label(repo_id: str) -> str:
    match = re.search(r"-(1|2)Epoch$", repo_id)
    return f"{match.group(1)}epoch" if match else "-"


def checkpoint_name(path: str) -> str:
    return Path(path).name


def generated_section(results: list[EvalResult], pending: list[str]) -> str:
    lines = [
        START,
        "## Gemma 4 Native 자동 평가 현황",
        "",
        f"업데이트: `{kst_now().strftime('%Y-%m-%d %H:%M:%S KST')}`",
        "",
        "점수 기준: `Score = 100 * avg_command_f1`. 이 섹션은 `monitor_state.json`과 TB2-lite 결과 JSON에서 자동 생성된다.",
        "",
        "| Native 순위 | HF repo | Epoch | Checkpoint | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Sec/Step |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(results, start=1):
        lines.append(
            "| "
            f"{rank} | `{item.repo_id}` | {epoch_label(item.repo_id)} | `{checkpoint_name(item.checkpoint)}` | "
            f"{item.score:.2f} | {item.f1:.4f} | {item.precision:.4f} | {item.recall:.4f} | "
            f"{item.first_cmd:.1f}% | {item.valid_json:.1f}% | {item.sec_per_step:.3f} |"
        )
    if pending:
        lines += [
            "",
            "평가 대기:",
            "",
        ]
        for repo_id in pending:
            lines.append(f"- `{repo_id}`")
    else:
        lines += ["", "평가 대기: 없음"]
    lines.append(END)
    return "\n".join(lines) + "\n"


def replace_or_append_section(text: str, section: str) -> str:
    if START in text and END in text:
        pattern = re.compile(re.escape(START) + r".*?" + re.escape(END) + r"\n?", re.DOTALL)
        return pattern.sub(section, text)
    return text.rstrip() + "\n\n" + section


def parse_number(raw: str) -> float:
    raw = raw.strip().rstrip("%")
    if raw in {"", "-"}:
        return 0.0
    return float(raw)


def merge_overall_ranking(text: str, results: list[EvalResult]) -> str:
    header = "## 전체 순위"
    start = text.find(header)
    if start < 0:
        return text
    table_start = text.find("| 순위 |", start)
    if table_start < 0:
        return text
    table_end = text.find("\n\n", table_start)
    if table_end < 0:
        return text

    table = text[table_start:table_end]
    rows = []
    row_re = re.compile(
        r"^\|\s*\d+\s*\|\s*(?P<model>.+?)\s*\|\s*(?P<score>-?\d+(?:\.\d+)?)\s*\|"
        r"\s*(?P<f1>-?\d+(?:\.\d+)?)\s*\|\s*(?P<precision>-?\d+(?:\.\d+)?)\s*\|"
        r"\s*(?P<recall>-?\d+(?:\.\d+)?)\s*\|\s*(?P<first>-?\d+(?:\.\d+)?)%\s*\|"
        r"\s*(?P<valid>-?\d+(?:\.\d+)?)%\s*\|\s*(?P<template>[^|]+?)\s*\|"
        r"\s*(?P<sec>-?\d+(?:\.\d+)?)\s*\|\s*(?P<load>-?\d+(?:\.\d+)?)\s*\|"
    )
    for line in table.splitlines():
        match = row_re.match(line.strip())
        if not match:
            continue
        model_cell = match.group("model").strip()
        repo_id = model_cell.strip("`")
        rows.append(
            {
                "repo_id": repo_id,
                "score": parse_number(match.group("score")),
                "f1": parse_number(match.group("f1")),
                "precision": parse_number(match.group("precision")),
                "recall": parse_number(match.group("recall")),
                "first": parse_number(match.group("first")),
                "valid": parse_number(match.group("valid")),
                "template": match.group("template").strip(),
                "sec": parse_number(match.group("sec")),
                "load": parse_number(match.group("load")),
            }
        )

    by_repo = {row["repo_id"]: row for row in rows}
    for item in results:
        by_repo[item.repo_id] = {
            "repo_id": item.repo_id,
            "score": item.score,
            "f1": item.f1,
            "precision": item.precision,
            "recall": item.recall,
            "first": item.first_cmd,
            "valid": item.valid_json,
            "template": "gemma4_native",
            "sec": item.sec_per_step,
            "load": item.load_time,
        }

    merged = sorted(by_repo.values(), key=lambda row: (-float(row["score"]), str(row["repo_id"])))
    new_lines = [
        "| 순위 | 모델(HF 저장소명) | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Template | Sec/Step | Load(s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for rank, row in enumerate(merged, start=1):
        new_lines.append(
            "| "
            f"{rank} | `{row['repo_id']}` | {float(row['score']):.2f} | {float(row['f1']):.4f} | "
            f"{float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['first']):.1f}% | "
            f"{float(row['valid']):.1f}% | {row['template']} | {float(row['sec']):.3f} | {float(row['load']):.1f} |"
        )
    return text[:table_start] + "\n".join(new_lines) + text[table_end:]


def sync_docs(force: bool = False) -> bool:
    results, pending = load_results()
    signature = result_signature(results, pending)
    sync_state = load_doc_sync_state()
    if not force and sync_state.get("signature") == signature:
        return False

    section = generated_section(results, pending)
    changed = False
    for path in (REPORT_PATH, README_PATH):
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated = replace_or_append_section(original, section)
        if path == REPORT_PATH:
            updated = merge_overall_ranking(updated, results)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed = True
            log(f"DOC_UPDATED path={path}")
    if changed:
        print("\a", end="", flush=True)
        log(f"ALERT docs synced evaluated={len(results)} pending={len(pending)}")
    save_doc_sync_state(
        {
            "signature": signature,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "evaluated": len(results),
            "pending": len(pending),
        }
    )
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    while True:
        try:
            sync_docs(force=args.force)
        except Exception as exc:
            log(f"ERROR {type(exc).__name__}: {exc}")
        if not args.watch:
            return 0
        args.force = False
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
