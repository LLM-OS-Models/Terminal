#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def rewrite_model(src_model: Path, dst_model: Path) -> dict[str, int]:
    tensors: dict[str, torch.Tensor] = {}
    renamed = 0
    untouched = 0

    with safe_open(str(src_model), framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            if key.startswith("model.language_model.language_model."):
                new_key = "model." + key[len("model.language_model.language_model."):]
            elif key.startswith("model.language_model."):
                new_key = "model." + key[len("model.language_model."):]
            else:
                new_key = key
            if new_key != key:
                renamed += 1
            else:
                untouched += 1
            tensors[new_key] = f.get_tensor(key)

    dst_model.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst_model), metadata=metadata)
    return {"renamed": renamed, "untouched": untouched, "total": renamed + untouched}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--dst-dir", required=True)
    parser.add_argument(
        "--copy-all-files",
        action="store_true",
        help="Copy every non-model file. Default copies only inference-relevant files.",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    keep_names = {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "training_args.bin",
    }

    stats = rewrite_model(src_dir / "model.safetensors", dst_dir / "model.safetensors")

    for path in src_dir.iterdir():
        if path.name == "model.safetensors":
            continue
        if path.is_dir():
            continue
        if args.copy_all_files or path.name in keep_names:
            shutil.copy2(path, dst_dir / path.name)

    print(
        json.dumps(
            {
                "src_dir": str(src_dir),
                "dst_dir": str(dst_dir),
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
