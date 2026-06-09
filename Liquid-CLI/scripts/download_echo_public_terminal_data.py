#!/usr/bin/env python
"""Download public ECHO-related terminal task sources with retry.

This downloader is intentionally conservative for Hugging Face resolver limits:
it resumes existing local snapshots and sleeps/retries on 429 errors.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="/home/work/.data/echo_terminal_data/raw")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--retry-seconds", type=int, default=180)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--skip-openthoughts-if-present", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-endless", action="store_true")
    return parser.parse_args()


def load_env_token(env_file: str) -> None:
    path = Path(env_file)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.split(None, 1)[1]
        if key in {"HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            os.environ["HF_TOKEN"] = value.strip().strip('"').strip("'")
            return


def retry_after_seconds(message: str, default: int) -> int:
    match = re.search(r"Retry after\s+(\d+)\s+seconds", message, re.IGNORECASE)
    if match:
        return max(default, int(match.group(1)) + 5)
    return default


def run_with_retry(name: str, fn, max_attempts: int, retry_seconds: int) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[{name}] attempt {attempt}/{max_attempts}", flush=True)
            result = fn()
            print(f"[{name}] done: {result}", flush=True)
            return
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            wait = retry_after_seconds(message, retry_seconds)
            print(f"[{name}] failed: {message}", flush=True)
            print(f"[{name}] sleeping {wait}s before retry", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{name} failed after {max_attempts} attempts")


def main() -> None:
    args = parse_args()
    load_env_token(args.env_file)
    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    openthoughts_dir = raw_root / "openthoughts_agent_v1_rl"
    openthoughts_file = openthoughts_dir / "tasks.parquet"
    if not (args.skip_openthoughts_if_present and openthoughts_file.exists()):
        run_with_retry(
            "openthoughts",
            lambda: hf_hub_download(
                "open-thoughts/OpenThoughts-Agent-v1-RL",
                "tasks.parquet",
                repo_type="dataset",
                local_dir=openthoughts_dir,
                token=token,
            ),
            args.max_attempts,
            args.retry_seconds,
        )
    else:
        print(f"[openthoughts] already present: {openthoughts_file}", flush=True)

    if not args.skip_endless:
        run_with_retry(
            "endless",
            lambda: snapshot_download(
                "obiwan96/endless-terminals",
                repo_type="dataset",
                local_dir=raw_root / "endless_terminals",
                token=token,
                allow_patterns=[
                    "README.md",
                    "task_*/instruction.md",
                    "task_*/task.toml",
                    "task_*/environment/*",
                    "task_*/tests/*",
                    "task_*/solution/solve.sh",
                ],
                ignore_patterns=["**/*summary.json", "**/.git*"],
            ),
            args.max_attempts,
            args.retry_seconds,
        )


if __name__ == "__main__":
    main()
