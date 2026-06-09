#!/usr/bin/env python3
"""Upload model-family prepared/tokenized terminal datasets to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi


DEFAULT_STAGE_ROOT = Path("/home/work/.data/hf_upload_stage/family_tokenized_datasets")

FAMILY_SPECS = {
    "lfm": {
        "repo_id": "LLM-OS-Models/LFM25-Terminal-ToolBench-Full-Tokenized",
        "title": "LFM2.5 Terminal ToolBench Full Tokenized Dataset",
        "summary": "LFM2.5-8B-A1B train-ready token IDs for the Terminal + ToolBench full SFT run.",
        "source_dirs": [
            "/home/work/.data/liquid_cli_sft/datasets/lfm25_8b_a1b_terminal_full_toolbench_full_train_ready_v1",
        ],
        "notes": [
            "This dataset stores token IDs and labels, not raw conversations.",
            "It was used by the LFM2.5-8B-A1B Terminal ToolBench full SFT config.",
            "Features: input_ids, seq_lengths, labels.",
        ],
    },
    "qwen": {
        "repo_id": "LLM-OS-Models/Qwen-Terminal-ToolBench-Processed-Tokenized",
        "title": "Qwen Terminal ToolBench Processed Datasets",
        "summary": "Qwen-family processed/template-applied and selected tokenized terminal datasets.",
        "source_dirs": [
            "/home/work/.data/qwen_sft/datasets/qwen35_2b_full_terminal_toolcall_processed_v1",
            "/home/work/.data/qwen_sft/datasets/qwen35_2b_terminal_toolcall_fullconv_from_samecount_v1",
            "/home/work/.data/qwen_sft/datasets/qwen35_27b_hf_fsdp_1024",
            "/home/work/.data/qwen_sft/datasets/qwen35_27b_processed_2bdata",
            "/home/work/.data/qwen_sft/datasets/qwen36_35b_a3b_hf_fsdp_1024",
            "/home/work/.data/qwen_sft/datasets/qwen35_2b_fast_continue_fullconv_2048_v1",
            "/home/work/.data/qwen_sft/datasets/qwen35_2b_fast_continue_samecount_2048_v1",
            "/home/work/.data/qwen_sft/datasets/qwen35_4b_processed_2bdata",
            "/home/work/.data/qwen_sft/datasets/qwen35_9b_processed_2bdata",
        ],
        "notes": [
            "Large Qwen3.5-2B full datasets are template-applied text processed datasets.",
            "HF/FSDP datasets include input_ids, attention_mask, and labels where present.",
            "This repo intentionally preserves the local training-ready folder layout.",
        ],
    },
    "gemma": {
        "repo_id": "LLM-OS-Models/Gemma4-Terminal-ToolBench-Tokenized",
        "title": "Gemma4 Terminal ToolBench Tokenized Datasets",
        "summary": "Gemma4-family tokenized terminal/tool datasets for HF/FSDP experiments.",
        "source_dirs": [
            "/home/work/.data/qwen_sft/datasets/gemma4_e2b_hf_fsdp_1024",
            "/home/work/.data/qwen_sft/datasets/gemma4_e2b_hf_fsdp_1024_4gpu",
            "/home/work/.data/qwen_sft/datasets/gemma4_e4b_hf_fsdp_1024",
            "/home/work/.data/qwen_sft/datasets/gemma4_26b_a4b_hf_fsdp_1024",
            "/home/work/.data/qwen_sft/datasets/gemma4_31b_hf_fsdp_1024",
        ],
        "notes": [
            "These datasets store input_ids, attention_mask, and labels.",
            "The smoke duplicate is excluded; only regular training/eval preparation folders are staged.",
            "All folders are preserved as separate subdirectories because tokenizer/model variants differ.",
        ],
    },
}


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


def dataset_summary(path: Path) -> dict:
    ds = load_from_disk(str(path))
    size_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return {
        "name": path.name,
        "source_path": str(path),
        "rows": len(ds),
        "features": {key: str(value) for key, value in ds.features.items()},
        "size_bytes": size_bytes,
    }


def hardlink_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, copy_function=os.link)


def write_readme(stage_dir: Path, spec: dict, manifest: dict) -> None:
    lines = [
        "---",
        "license: apache-2.0",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- terminal-agent",
        "- sft",
        "- tokenized",
        "- toolbench",
        "- liquid-cli",
        "---",
        "",
        f"# {spec['title']}",
        "",
        spec["summary"],
        "",
        "## Contents",
        "",
    ]
    for item in manifest["datasets"]:
        gib = item["size_bytes"] / 1024**3
        features = ", ".join(item["features"].keys())
        lines.append(f"- `{item['name']}`: {item['rows']} rows, {gib:.2f} GiB, features: {features}")
    lines.extend(["", "## Notes", ""])
    for note in spec["notes"]:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Loading",
            "",
            "Download a subdirectory and use `datasets.load_from_disk`:",
            "",
            "```python",
            "from huggingface_hub import snapshot_download",
            "from datasets import load_from_disk",
            "",
            f"repo_id = \"{spec['repo_id']}\"",
            "local_dir = snapshot_download(repo_id, repo_type=\"dataset\")",
            f"dataset = load_from_disk(f\"{{local_dir}}/{manifest['datasets'][0]['name']}\")",
            "print(dataset)",
            "```",
            "",
            "## Manifest",
            "",
            "See `manifest.json` for exact local source paths, row counts, features, and byte sizes.",
            "",
        ]
    )
    stage_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def stage_family(family: str, spec: dict, stage_root: Path) -> tuple[Path, dict]:
    stage_dir = stage_root / family
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for raw_path in spec["source_dirs"]:
        src = Path(raw_path)
        if not src.exists():
            raise FileNotFoundError(src)
        summary = dataset_summary(src)
        hardlink_tree(src, stage_dir / src.name)
        summaries.append(summary)

    manifest = {
        "family": family,
        "repo_id": spec["repo_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": summaries,
        "total_size_bytes": sum(item["size_bytes"] for item in summaries),
        "total_rows": sum(item["rows"] for item in summaries),
    }
    stage_dir.joinpath("manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(stage_dir, spec, manifest)
    return stage_dir, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["lfm", "qwen", "gemma"])
    parser.add_argument("--stage-root", default=str(DEFAULT_STAGE_ROOT))
    parser.add_argument("--env-file", default="/home/work/.projects/LLM-OS-Models/Terminal/.env")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()

    stage_root = Path(args.stage_root)
    token = get_token(Path(args.env_file))
    api = HfApi(token=token)

    for family in args.families:
        if family not in FAMILY_SPECS:
            raise KeyError(f"Unknown family: {family}")
        spec = FAMILY_SPECS[family]
        stage_dir, manifest = stage_family(family, spec, stage_root)
        print(
            json.dumps(
                {
                    "event": "staged_family",
                    "family": family,
                    "repo_id": spec["repo_id"],
                    "stage_dir": str(stage_dir),
                    "total_rows": manifest["total_rows"],
                    "total_size_bytes": manifest["total_size_bytes"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.stage_only:
            continue
        api.create_repo(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            exist_ok=True,
            private=args.private,
        )
        api.upload_large_folder(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            folder_path=str(stage_dir),
            private=args.private,
            num_workers=args.num_workers,
        )
        print(
            json.dumps(
                {
                    "event": "uploaded_family",
                    "family": family,
                    "url": f"https://huggingface.co/datasets/{spec['repo_id']}",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
