#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--commit-message", default="Upload Gemma 4 native SFT model")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing token in environment variable: {args.token_env}")

    staging_dir = Path(args.staging_dir)
    if not staging_dir.exists():
        raise FileNotFoundError(f"Staging directory not found: {staging_dir}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "staging_dir": str(staging_dir),
                "commit_message": args.commit_message,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
