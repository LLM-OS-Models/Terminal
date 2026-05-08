#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def build_readme(
    *,
    repo_id: str,
    base_model: str,
    source_dataset: str,
    tb2_result: str | None,
    notes: str | None,
) -> str:
    lines = [
        "---",
        "license: apache-2.0",
        "library_name: transformers",
        f"base_model: {base_model}",
        "tags:",
        "- gemma-4",
        "- terminal-agent",
        "- full-finetuning",
        "- tb2-lite",
        "- gemma4-native-template",
        "---",
        "",
        f"# {repo_id}",
        "",
        "## Summary",
        "",
        f"- Base model: `{base_model}`",
        f"- Source dataset/cache: `{source_dataset}`",
        "- Training format: Gemma 4 native chat template",
        "- Labels: assistant JSON command response only",
        "- Prompt/history labels are masked with `-100`",
        "- Previous assistant thinking blocks are stripped from history",
    ]
    if tb2_result:
        lines.extend(["", "## TB2-lite", "", f"- Result: `{tb2_result}`"])
    if notes:
        lines.extend(["", "## Notes", "", notes.strip()])
    lines.extend(
        [
            "",
            "## Loading",
            "",
            "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")',
            f'model = AutoModelForCausalLM.from_pretrained("{repo_id}", torch_dtype="auto")',
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--tb2-result", default=None)
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    staging_dir = Path(args.staging_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    copy_tree_contents(src_dir, staging_dir)
    (staging_dir / "README.md").write_text(
        build_readme(
            repo_id=args.repo_id,
            base_model=args.base_model,
            source_dataset=args.source_dataset,
            tb2_result=args.tb2_result,
            notes=args.notes,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "src_dir": str(src_dir),
                "staging_dir": str(staging_dir),
                "repo_id": args.repo_id,
                "base_model": args.base_model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
