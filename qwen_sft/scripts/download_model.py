#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--output-dir",
        default="/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=args.model_name,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    meta = {
        "model_name": args.model_name,
        "output_dir": str(out_dir),
        "snapshot_path": snapshot_path,
    }
    (out_dir / "download_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
