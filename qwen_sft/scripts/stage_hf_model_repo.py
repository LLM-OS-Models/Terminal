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
    src_output_root: Path,
    root_subdir: str,
    checkpoints: list[str],
    tb2_results: list[str],
    notes: str | None,
) -> str:
    lines = [
        "---",
        "license: apache-2.0",
        "library_name: transformers",
        f"base_model: {base_model}",
        "tags:",
        "- terminal-agent",
        "- qwen3.5",
        "- full-finetuning",
        "- unsloth",
        "- tb2-lite",
        "---",
        "",
        f"# {repo_id}",
        "",
        "## Summary",
        "",
        f"- Base model: `{base_model}`",
        f"- Source output root: `{src_output_root}`",
        f"- Root export copied from: `{root_subdir}`",
        f"- Included checkpoints: `{', '.join(checkpoints) if checkpoints else 'none'}`",
        "",
        "## Layout",
        "",
        "- Repository root contains the eval/inference-ready final export.",
        "- Trainer checkpoints are included as subdirectories for recovery and inspection.",
    ]

    if tb2_results:
        lines.extend(
            [
                "",
                "## TB2-lite Results",
                "",
            ]
        )
        for path in tb2_results:
            lines.append(f"- `{path}`")

    if notes:
        lines.extend(
            [
                "",
                "## Notes",
                "",
                notes.strip(),
            ]
        )

    lines.extend(
        [
            "",
            "## Loading",
            "",
            "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}", trust_remote_code=True)',
            f'model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)',
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-output-root", required=True)
    parser.add_argument("--root-subdir", default="final")
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--tb2-result", action="append", default=[])
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    src_output_root = Path(args.src_output_root)
    root_src = src_output_root / args.root_subdir
    staging_dir = Path(args.staging_dir)

    if not root_src.exists():
        raise FileNotFoundError(f"Root export directory not found: {root_src}")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    copy_tree_contents(root_src, staging_dir)

    copied_checkpoints: list[str] = []
    for checkpoint_name in args.checkpoint:
        checkpoint_src = src_output_root / checkpoint_name
        checkpoint_dst = staging_dir / checkpoint_name
        if not checkpoint_src.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_src}")
        shutil.copytree(checkpoint_src, checkpoint_dst, dirs_exist_ok=True)
        copied_checkpoints.append(checkpoint_name)

    readme = build_readme(
        repo_id=args.repo_id,
        base_model=args.base_model,
        src_output_root=src_output_root,
        root_subdir=args.root_subdir,
        checkpoints=copied_checkpoints,
        tb2_results=args.tb2_result,
        notes=args.notes,
    )
    (staging_dir / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "src_output_root": str(src_output_root),
                "root_subdir": args.root_subdir,
                "staging_dir": str(staging_dir),
                "repo_id": args.repo_id,
                "checkpoints": copied_checkpoints,
                "tb2_results": args.tb2_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
