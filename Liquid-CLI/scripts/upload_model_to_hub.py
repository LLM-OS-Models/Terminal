#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hub-model-id", required=True)
    parser.add_argument("--upload-subdir", default="final")
    parser.add_argument("--token-env-var", default="HF_TOKEN")
    parser.add_argument("--commit-message", default=None)
    return parser.parse_args()


def require_final_artifacts(upload_dir: Path) -> None:
    required = [
        upload_dir / "config.json",
        upload_dir / "tokenizer_config.json",
    ]
    weight_candidates = [
        upload_dir / "model.safetensors",
        upload_dir / "model-00001-of-00002.safetensors",
        upload_dir / "pytorch_model.bin",
    ]
    missing = [path for path in required if not path.exists()]
    if not any(path.exists() for path in weight_candidates):
        missing.append(upload_dir / "<model weights>")
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"upload artifacts are incomplete: {missing_text}")


def main() -> None:
    args = parse_args()

    token = os.environ.get(args.token_env_var)
    if not token:
        raise RuntimeError(f"{args.token_env_var} is required for upload.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    upload_dir = (output_dir / args.upload_subdir).resolve()
    if not upload_dir.exists():
        raise FileNotFoundError(f"upload directory does not exist: {upload_dir}")

    require_final_artifacts(upload_dir)

    api = HfApi(token=token)
    api.create_repo(repo_id=args.hub_model_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=args.hub_model_id,
        repo_type="model",
        commit_message=args.commit_message or f"Upload {upload_dir.name} from {output_dir.name}",
    )
    print(f"uploaded {upload_dir} -> {args.hub_model_id}", flush=True)


if __name__ == "__main__":
    main()
