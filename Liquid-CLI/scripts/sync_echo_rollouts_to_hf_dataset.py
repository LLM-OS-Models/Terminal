#!/usr/bin/env python3
"""Sync live ECHO terminal RLVR rollout traces to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


SECRET_KEY_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)


def load_dotenv_token(env_file: Path) -> str | None:
    if not env_file.exists():
        return None
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
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


def iter_json_events(log_path: Path, event: str) -> Iterable[dict]:
    if not log_path.exists():
        return
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("event") == event:
                yield obj


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def redact_env_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    lines: list[str] = []
    for raw_line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw_line:
            lines.append(raw_line)
            continue
        key, _ = raw_line.split("=", 1)
        clean_key = key.replace("export ", "").strip()
        if SECRET_KEY_RE.search(clean_key):
            lines.append(f"{key}=<redacted>")
        else:
            lines.append(raw_line)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if SECRET_KEY_RE.search(key):
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def latest_checkpoints(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(
        [p.name for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda name: int(name.split("-")[-1]) if name.split("-")[-1].isdigit() else -1,
    )


def write_dataset_card(stage_dir: Path, *, repo_id: str, run_id: str, manifest: dict) -> None:
    card = f"""---
license: apache-2.0
task_categories:
- text-generation
tags:
- terminal-agent
- rlvr
- echo
- vllm
- liquid-ai
pretty_name: ECHO Terminal RLVR Rollouts
---

# ECHO Terminal RLVR Rollouts

This dataset snapshot stores live terminal-agent RLVR rollouts generated during
training for `{run_id}`.

The raw trace JSONL files include model generations served through vLLM,
executed terminal commands, terminal observations, verifier results, and reward
fields. The training run uses an ECHO-style objective: verifier RL plus
cross-entropy on terminal observation tokens.

## Why This Data Matters

These rollouts are intended to be reusable training data, not just debug logs.

- Future RLVR: successful and failed command trajectories can be replayed or
  reweighted for later verifier-reward training.
- Future SFT: high-quality successful trajectories can be filtered into
  terminal-agent demonstrations.
- World-model learning: terminal observations, including stderr, stack traces,
  directory listings, and test output, provide supervised targets for learning
  how the shell environment responds to actions.
- Error analysis: timeout, unsafe-command, verifier-failure, and partial-credit
  examples can be mined to improve prompts, rewards, sandboxing, and curricula.

In short, each row preserves the action, the real terminal feedback, and the
learning signal needed to build stronger terminal agents later.

## Current Snapshot

- Repository: `{repo_id}`
- Run ID: `{run_id}`
- Synced at UTC: `{manifest["synced_at_utc"]}`
- Trace rows: `{manifest["trace_rows_total"]}`
- Train steps logged: `{manifest["train_steps_logged"]}`
- Checkpoints: `{", ".join(manifest["checkpoints"]) if manifest["checkpoints"] else "none"}`
- No Docker: `{manifest["no_docker"]}`
- vLLM GPUs: `{manifest["vllm_gpus"]}`
- Training GPUs: `{manifest["train_gpus"]}`

## Files

- `traces/*.jsonl`: raw rollout traces.
- `train_steps.jsonl`: parsed training metrics from `train.log`.
- `manifest.json`: sync metadata and counts.
- `run.env.redacted`: run configuration with secret-like values redacted.
- `checkpoint_eval_candidates.*`: optional checkpoint summary for later eval sweeps.
"""
    (stage_dir / "README.md").write_text(card, encoding="utf-8")


def stage_snapshot(run_dir: Path, output_dir: Path, stage_root: Path, repo_id: str) -> tuple[Path, dict]:
    run_id = run_dir.name
    stage_dir = stage_root / run_id
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    (stage_dir / "traces").mkdir(parents=True, exist_ok=True)

    trace_rows_total = 0
    trace_files: dict[str, int] = {}
    for src in sorted((run_dir / "traces").glob("*.jsonl")):
        dst = stage_dir / "traces" / src.name
        shutil.copy2(src, dst)
        rows = count_jsonl(dst)
        trace_files[src.name] = rows
        trace_rows_total += rows

    train_log = first_existing(
        [
            run_dir / "train.log",
            run_dir / "logs" / "train.log",
            run_dir / "logs" / "train_online.log",
        ]
    )
    train_steps = list(iter_json_events(train_log, "train_step"))
    with (stage_dir / "train_steps.jsonl").open("w", encoding="utf-8") as fh:
        for obj in train_steps:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    env_path = first_existing([run_dir / "run.env", run_dir / "run_env.sh"])
    redact_env_file(env_path, stage_dir / "run.env.redacted")
    for aux_name in ("checkpoint_eval_candidates.md", "checkpoint_eval_candidates.json"):
        aux_src = run_dir / aux_name
        if aux_src.exists():
            shutil.copy2(aux_src, stage_dir / aux_name)

    run_env = parse_env_file(env_path)

    manifest = {
        "repo_id": repo_id,
        "run_id": run_id,
        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "trace_files": trace_files,
        "trace_rows_total": trace_rows_total,
        "train_steps_logged": len(train_steps),
        "latest_train_step": train_steps[-1] if train_steps else None,
        "train_log": str(train_log),
        "checkpoints": latest_checkpoints(output_dir),
        "no_docker": True,
        "vllm_gpus": run_env.get("VLLM_GPUS", "0,1,2,3"),
        "train_gpus": run_env.get("TRAIN_GPUS", "4,5"),
        "vllm_base_url": run_env.get("VLLM_BASE_URL", ""),
    }
    (stage_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_dataset_card(stage_dir, repo_id=repo_id, run_id=run_id, manifest=manifest)
    return stage_dir, manifest


def sync_once(args: argparse.Namespace, api: HfApi) -> dict:
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path("")
    stage_root = Path(args.stage_root).resolve()
    stage_dir, manifest = stage_snapshot(run_dir, output_dir, stage_root, args.repo_id)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )
    path_in_repo = args.path_in_repo or f"runs/{run_dir.name}"
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(stage_dir),
        path_in_repo=path_in_repo,
        commit_message=(
            f"sync {run_dir.name}: {manifest['trace_rows_total']} traces, "
            f"{manifest['train_steps_logged']} steps"
        ),
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--path-in-repo", default="")
    parser.add_argument("--stage-root", default="/home/work/.data/hf_upload_stage/echo_rlvr_rollouts")
    parser.add_argument("--env-file", default="/home/work/.projects/LLM-OS-Models/Terminal/.env")
    parser.add_argument("--interval-sec", type=int, default=600)
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
                        "event": "hf_dataset_sync_done",
                        "repo_id": args.repo_id,
                        "run_id": manifest["run_id"],
                        "trace_rows_total": manifest["trace_rows_total"],
                        "train_steps_logged": manifest["train_steps_logged"],
                        "checkpoints": manifest["checkpoints"],
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
                        "event": "hf_dataset_sync_error",
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
