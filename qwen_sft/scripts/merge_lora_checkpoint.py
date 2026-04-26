#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from unsloth import FastLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--save-method", default="merged_16bit")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        backup_dir = output_dir.with_name(output_dir.name + "_broken_backup")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        output_dir.rename(backup_dir)
        print(json.dumps({"moved_existing_output_to": str(backup_dir)}), flush=True)

    print(
        json.dumps(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "output_dir": str(output_dir),
                "device": args.device,
                "max_seq_length": args.max_seq_length,
                "save_method": args.save_method,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint_dir),
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        device_map={"": args.device},
    )
    model.save_pretrained_merged(str(output_dir), tokenizer, save_method=args.save_method)
    print(json.dumps({"status": "ok", "output_dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    main()
