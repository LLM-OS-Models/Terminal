#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_ROOT = Path("/home/work/.data/hf_upload_stage/lfm_epoch_checkpoints_20260508")


@dataclass(frozen=True)
class UploadTarget:
    name: str
    source_dir: Path
    repo_id: str
    base_model: str
    epoch: int
    checkpoint: str
    score: str
    notes: str


TARGETS: tuple[UploadTarget, ...] = (
    UploadTarget(
        name="lfm2_8b_a1b_epoch1",
        source_dir=Path("/home/work/.data/liquid_cli_sft/models/LFM2-8B-A1B__terminal_sft_h200_7gpu_processed_template_holdout/checkpoint-830"),
        repo_id="LLM-OS-Models/LFM2-8B-A1B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout",
        base_model="LiquidAI/LFM2-8B-A1B",
        epoch=1,
        checkpoint="checkpoint-830",
        score="32.41",
        notes="Best checkpoint for the A1B run under corrected TB2-lite avg_command_f1 scoring.",
    ),
    UploadTarget(
        name="lfm2_8b_a1b_epoch2",
        source_dir=Path("/home/work/.data/liquid_cli_sft/models/LFM2-8B-A1B__terminal_sft_h200_7gpu_processed_template_holdout/checkpoint-1660"),
        repo_id="LLM-OS-Models/LFM2-8B-A1B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout",
        base_model="LiquidAI/LFM2-8B-A1B",
        epoch=2,
        checkpoint="checkpoint-1660",
        score="31.02",
        notes="Final epoch checkpoint for the A1B run.",
    ),
    UploadTarget(
        name="lfm25_1p2b_epoch1",
        source_dir=Path("/home/work/.data/liquid_cli_sft/models/LFM2.5-1.2B-Base__terminal_sft_h200_8gpu_processed_template_holdout/checkpoint-545"),
        repo_id="LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout",
        base_model="LiquidAI/LFM2.5-1.2B-Base",
        epoch=1,
        checkpoint="checkpoint-545",
        score="28.10",
        notes="Epoch 1 checkpoint for the 1.2B run.",
    ),
    UploadTarget(
        name="lfm25_1p2b_epoch2",
        source_dir=Path("/home/work/.data/liquid_cli_sft/models/LFM2.5-1.2B-Base__terminal_sft_h200_8gpu_processed_template_holdout/checkpoint-1090"),
        repo_id="LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout",
        base_model="LiquidAI/LFM2.5-1.2B-Base",
        epoch=2,
        checkpoint="checkpoint-1090",
        score="28.64",
        notes="Final epoch checkpoint for the 1.2B run.",
    ),
    UploadTarget(
        name="lfm2_2p6b_epoch1",
        source_dir=Path("/home/work/.data/liquid_cli_sft/models/LFM2-2.6B__terminal_sft_h200_8gpu_processed_template_holdout/checkpoint-545"),
        repo_id="LLM-OS-Models/LFM2-2.6B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout",
        base_model="LiquidAI/LFM2-2.6B",
        epoch=1,
        checkpoint="checkpoint-545",
        score="31.86",
        notes="Epoch 1 checkpoint for the 2.6B run.",
    ),
    UploadTarget(
        name="lfm2_2p6b_epoch2",
        source_dir=Path("/home/work/.data/liquid_cli_sft/models/LFM2-2.6B__terminal_sft_h200_8gpu_processed_template_holdout/checkpoint-1090"),
        repo_id="LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout",
        base_model="LiquidAI/LFM2-2.6B",
        epoch=2,
        checkpoint="checkpoint-1090",
        score="32.85",
        notes="Final epoch checkpoint for the 2.6B run.",
    ),
    UploadTarget(
        name="lfm2_24b_a2b_epoch1",
        source_dir=Path("/home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp_template_masked/checkpoint-730"),
        repo_id="LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-TemplateMasked",
        base_model="LiquidAI/LFM2-24B-A2B",
        epoch=1,
        checkpoint="checkpoint-730",
        score="33.46",
        notes="Best checkpoint in the current LFM sweep under corrected TB2-lite avg_command_f1 scoring.",
    ),
    UploadTarget(
        name="lfm2_24b_a2b_epoch2",
        source_dir=Path("/home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp_template_masked/checkpoint-1460"),
        repo_id="LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-TemplateMasked",
        base_model="LiquidAI/LFM2-24B-A2B",
        epoch=2,
        checkpoint="checkpoint-1460",
        score="33.35",
        notes="Final epoch checkpoint and best final LFM model in the current sweep.",
    ),
)


KEEP_PATTERNS = (
    re.compile(r"^config\.json$"),
    re.compile(r"^generation_config\.json$"),
    re.compile(r"^tokenizer.*"),
    re.compile(r"^special_tokens_map\.json$"),
    re.compile(r"^added_tokens\.json$"),
    re.compile(r"^chat_template\.jinja$"),
    re.compile(r"^model.*\.safetensors$"),
    re.compile(r"^model\.safetensors\.index\.json$"),
    re.compile(r"^training_args\.bin$"),
    re.compile(r"^trainer_state\.json$"),
)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    for env_path in (ROOT_DIR / ".env", Path.home() / ".env"):
        values = parse_env_file(env_path)
        token = values.get("HF_TOKEN") or values.get("HUGGINGFACE_TOKEN")
        if token:
            return token
    raise RuntimeError("HF_TOKEN is not available. Put it in environment or .env.")


def should_keep(path: Path) -> bool:
    return any(pattern.match(path.name) for pattern in KEEP_PATTERNS)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def require_artifacts(source_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {source_dir}")
    required = [source_dir / "config.json", source_dir / "tokenizer_config.json"]
    missing = [str(path) for path in required if not path.exists()]
    weights = list(source_dir.glob("model*.safetensors"))
    if not weights:
        missing.append("model*.safetensors")
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint {source_dir}: {', '.join(missing)}")


def build_readme(target: UploadTarget) -> str:
    return f"""---
license: other
base_model: {target.base_model}
tags:
- lfm
- terminal
- sft
- tb2-lite
---

# {target.repo_id.split('/', 1)[1]}

Terminal SFT checkpoint uploaded from the LFM retraining sweep.

## Training

- Base model: `{target.base_model}`
- Epoch: `{target.epoch}`
- Source checkpoint: `{target.checkpoint}`
- Recipe: Terminal SFT with model chat template and holdout-aware preprocessing
- Evaluation protocol: corrected TB2-lite replay, 303 steps / 50 tasks, vLLM, tokenizer chat template
- Recomputed score: `{target.score}` (`100 * avg_command_f1`)

## Notes

{target.notes}

This upload intentionally excludes optimizer, scheduler, and RNG state files.
It contains model weights, tokenizer/config files, chat template, and lightweight training metadata only.
"""


def stage_target(target: UploadTarget, stage_root: Path) -> Path:
    require_artifacts(target.source_dir)
    stage_dir = stage_root / target.name
    stage_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(target.source_dir.iterdir()):
        if src.is_file() and should_keep(src):
            link_or_copy(src, stage_dir / src.name)
    (stage_dir / "README.md").write_text(build_readme(target), encoding="utf-8")
    return stage_dir


def upload_target(api: HfApi, target: UploadTarget, stage_dir: Path, *, private: bool, num_workers: int) -> None:
    api.create_repo(repo_id=target.repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_large_folder(
        repo_id=target.repo_id,
        repo_type="model",
        folder_path=stage_dir,
        private=private,
        num_workers=num_workers,
        print_report=True,
        print_report_every=120,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", default=str(DEFAULT_STAGE_ROOT))
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = [target for target in TARGETS if not args.only or target.name in set(args.only)]
    if not selected:
        raise RuntimeError("No upload targets selected.")

    stage_root = Path(args.stage_root)
    print(f"staging_root={stage_root}", flush=True)
    for target in selected:
        stage_dir = stage_target(target, stage_root)
        size_gb = sum(path.stat().st_size for path in stage_dir.iterdir() if path.is_file()) / 1e9
        print(f"staged {target.name}: {size_gb:.2f} GB -> {target.repo_id}", flush=True)

    if args.dry_run:
        print("dry_run_complete", flush=True)
        return

    token = get_token()
    api = HfApi(token=token)
    for index, target in enumerate(selected, start=1):
        stage_dir = stage_root / target.name
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"[{started}] upload {index}/{len(selected)} start: {target.name} -> {target.repo_id}", flush=True)
        upload_target(api, target, stage_dir, private=args.private, num_workers=args.num_workers)
        finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        marker = stage_dir / ".upload_complete"
        marker.write_text(f"{finished}\n{target.repo_id}\n", encoding="utf-8")
        print(f"[{finished}] upload {index}/{len(selected)} complete: {target.repo_id}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
