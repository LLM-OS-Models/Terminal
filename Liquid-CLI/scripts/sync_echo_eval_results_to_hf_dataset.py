#!/usr/bin/env python3
"""Sync ECHO RLVR TB2-lite evaluation snapshots to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


def load_dotenv_token(env_file: Path) -> str | None:
    if not env_file.exists():
        return None
    for raw_line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() not in {"HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            return value
    return None


def get_token(env_file: Path) -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or load_dotenv_token(env_file)
    )
    if not token:
        raise RuntimeError(f"HF token not found in environment or {env_file}")
    return token


def read_score(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    aggregate = data.get("aggregate") or {}
    return {
        "file": path.name,
        "model_short": data.get("model_short") or path.stem,
        "score": (
            100.0 * aggregate["avg_command_f1"]
            if isinstance(aggregate.get("avg_command_f1"), int | float)
            else None
        ),
        "next_action_score": aggregate.get("next_action_score"),
        "avg_command_f1": aggregate.get("avg_command_f1"),
        "first_cmd_exact_pct": aggregate.get("first_cmd_exact_pct"),
        "valid_json_pct": aggregate.get("valid_json_pct"),
        "gen_time_sec": data.get("gen_time_sec"),
        "lora_path": data.get("lora_path"),
        "timestamp": data.get("timestamp"),
    }


def stage_results(results_dir: Path, stage_root: Path, repo_id: str, path_in_repo: str) -> tuple[Path, dict]:
    stage_dir = stage_root / results_dir.name
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    scores: list[dict] = []
    for src in sorted(results_dir.glob("*.json")):
        dst = stage_dir / src.name
        shutil.copy2(src, dst)
        copied.append(src.name)
        try:
            scores.append(read_score(src))
        except Exception as exc:  # noqa: BLE001
            scores.append({"file": src.name, "error": str(exc)})

    readme = results_dir / "README.md"
    if readme.exists():
        shutil.copy2(readme, stage_dir / "SOURCE_README.md")
        copied.append("SOURCE_README.md")

    best = None
    numeric_scores = [row for row in scores if isinstance(row.get("score"), int | float)]
    if numeric_scores:
        best = max(numeric_scores, key=lambda row: row["score"])

    manifest = {
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "results_dir": str(results_dir),
        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": copied,
        "result_count": len([name for name in copied if name.endswith(".json")]),
        "best": best,
        "scores": sorted(
            scores,
            key=lambda row: row.get("score") if isinstance(row.get("score"), int | float) else -1,
            reverse=True,
        ),
        "logs_excluded": True,
    }
    (stage_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (stage_dir / "README.md").write_text(
        f"""---
license: apache-2.0
task_categories:
- text-generation
tags:
- terminal-agent
- rlvr
- echo
- evaluation
- tb2-lite
pretty_name: ECHO Terminal RLVR Evaluation Results
---

# ECHO Terminal RLVR Evaluation Results

This snapshot stores TB2-lite evaluation JSON files for ECHO-style terminal
RLVR checkpoints.

- Repository: `{repo_id}`
- Path in repo: `{path_in_repo}`
- Source directory: `{results_dir}`
- Synced at UTC: `{manifest["synced_at_utc"]}`
- Result count: `{manifest["result_count"]}`
- Current best: `{manifest["best"]}`

The TB2-lite score is a fast proxy based on corrected 303-step replay metrics,
not the official Docker-isolated TerminalBench score. It is intended for
checkpoint selection and regression tracking during RLVR.
""",
        encoding="utf-8",
    )
    return stage_dir, manifest


def sync_once(args: argparse.Namespace, api: HfApi) -> dict:
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(results_dir)
    stage_dir, manifest = stage_results(
        results_dir=results_dir,
        stage_root=Path(args.stage_root).resolve(),
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
    )
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(stage_dir),
        path_in_repo=args.path_in_repo,
        commit_message=(
            f"sync eval {results_dir.name}: {manifest['result_count']} results"
        ),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts")
    parser.add_argument("--results-dir", default="tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612")
    parser.add_argument("--path-in-repo", default="eval/tb2_lite_gpu6/lfm25_echo_rlvr_gpu6_eval_20260612")
    parser.add_argument("--stage-root", default="/home/work/.data/hf_upload_stage/echo_eval_results")
    parser.add_argument("--env-file", default="/home/work/.projects/LLM-OS-Models/Terminal/.env")
    parser.add_argument("--interval-sec", type=int, default=900)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = get_token(Path(args.env_file))
    api = HfApi(token=token)

    while True:
        try:
            manifest = sync_once(args, api)
            print(
                json.dumps(
                    {
                        "event": "hf_eval_sync_done",
                        "repo_id": args.repo_id,
                        "path_in_repo": args.path_in_repo,
                        "result_count": manifest["result_count"],
                        "best": manifest["best"],
                        "synced_at_utc": manifest["synced_at_utc"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "event": "hf_eval_sync_error",
                        "repo_id": args.repo_id,
                        "error": str(exc),
                        "time_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not args.loop:
                raise
        if not args.loop:
            return
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
