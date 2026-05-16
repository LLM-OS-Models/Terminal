#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from huggingface_hub import snapshot_download


def stamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def download(repo_id: str, cache_dir: str, revision: str | None, retries: int, sleep_sec: int) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        print(f"[{stamp()}] START repo={repo_id} attempt={attempt}/{retries}", flush=True)
        try:
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=False,
                force_download=False,
            )
            print(f"[{stamp()}] DONE repo={repo_id} path={path}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - log full failure and retry.
            last_error = exc
            print(f"[{stamp()}] ERROR repo={repo_id} attempt={attempt}: {exc!r}", flush=True)
            if attempt < retries:
                time.sleep(sleep_sec)
    raise RuntimeError(f"download failed for {repo_id}: {last_error!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", nargs="+")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE", "/home/work/.data/huggingface/hub"))
    parser.add_argument("--revision", default=None)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sleep-sec", type=int, default=60)
    args = parser.parse_args()

    for repo_id in args.repo_id:
        download(repo_id, args.cache_dir, args.revision, args.retries, args.sleep_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
