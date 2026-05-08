#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_STATE = Path("/home/work/.data/gemma4_native_sft/monitor_state.json")
DEFAULT_STAGING_ROOT = Path("/home/work/.data/gemma4_native_sft/staging")


@dataclass(frozen=True)
class RunSpec:
    model_id: str
    output_dir: Path
    dataset_path: Path
    repo_1epoch: str
    repo_2epoch: str


RUNS = [
    RunSpec(
        model_id="google/gemma-4-E2B-it",
        output_dir=Path("/home/work/.data/gemma4_native_sft/models/google__gemma-4-E2B-it__terminal_sft_native_liquid_2epoch"),
        dataset_path=Path("/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E2B-it__liquid_raw_json_masked_8192"),
        repo_1epoch="LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-1Epoch",
        repo_2epoch="LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-2Epoch",
    ),
    RunSpec(
        model_id="google/gemma-4-E4B-it",
        output_dir=Path("/home/work/.data/gemma4_native_sft/models/google__gemma-4-E4B-it__terminal_sft_native_liquid_2epoch"),
        dataset_path=Path("/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E4B-it__liquid_raw_json_masked_8192"),
        repo_1epoch="LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-1Epoch",
        repo_2epoch="LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-2Epoch",
    ),
    RunSpec(
        model_id="google/gemma-4-E2B",
        output_dir=Path("/home/work/.data/gemma4_native_sft/models/google__gemma-4-E2B__terminal_sft_native_liquid_2epoch"),
        dataset_path=Path("/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E2B__liquid_raw_json_masked_8192"),
        repo_1epoch="LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-1Epoch",
        repo_2epoch="LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-2Epoch",
    ),
    RunSpec(
        model_id="google/gemma-4-E4B",
        output_dir=Path("/home/work/.data/gemma4_native_sft/models/google__gemma-4-E4B__terminal_sft_native_liquid_2epoch"),
        dataset_path=Path("/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E4B__liquid_raw_json_masked_8192"),
        repo_1epoch="LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-1Epoch",
        repo_2epoch="LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-2Epoch",
    ),
    RunSpec(
        model_id="google/gemma-4-26B-A4B-it",
        output_dir=Path("/home/work/.data/gemma4_native_sft/models/google__gemma-4-26B-A4B-it__terminal_sft_native_liquid_2epoch"),
        dataset_path=Path("/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-26B-A4B-it__liquid_raw_json_masked_8192"),
        repo_1epoch="LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-1Epoch",
        repo_2epoch="LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch",
    ),
    RunSpec(
        model_id="google/gemma-4-26B-A4B",
        output_dir=Path("/home/work/.data/gemma4_native_sft/models/google__gemma-4-26B-A4B__terminal_sft_native_liquid_2epoch"),
        dataset_path=Path("/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-26B-A4B__liquid_raw_json_masked_8192"),
        repo_1epoch="LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-1Epoch",
        repo_2epoch="LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch",
    ),
    RunSpec(
        model_id="google/gemma-4-31B-it",
        output_dir=Path("/home/work/.data/gemma4_native_sft/models/google__gemma-4-31B-it__terminal_sft_native_liquid_2epoch"),
        dataset_path=Path("/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-31B-it__liquid_raw_json_masked_8192"),
        repo_1epoch="LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-1Epoch",
        repo_2epoch="LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch",
    ),
    RunSpec(
        model_id="google/gemma-4-31B",
        output_dir=Path("/home/work/.data/gemma4_native_sft/models/google__gemma-4-31B__terminal_sft_native_liquid_2epoch"),
        dataset_path=Path("/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-31B__liquid_raw_json_masked_8192"),
        repo_1epoch="LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-1Epoch",
        repo_2epoch="LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"published": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except Exception:
        return -1


def checkpoint_complete(path: Path, settle_seconds: int) -> bool:
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", "trainer_state.json"]
    if any(not (path / name).exists() for name in required):
        return False
    if not list(path.glob("*.safetensors")) and not list(path.glob("*.bin")):
        return False
    newest_mtime = max(item.stat().st_mtime for item in path.iterdir())
    return (time.time() - newest_mtime) >= settle_seconds


def trainer_epoch(path: Path) -> float:
    state_path = path / "trainer_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return float(state.get("epoch") or 0.0)
    except Exception:
        return 0.0


def select_repo(spec: RunSpec, checkpoint: Path) -> tuple[str, str]:
    epoch = trainer_epoch(checkpoint)
    if epoch >= 1.5:
        return spec.repo_2epoch, "2Epoch"
    return spec.repo_1epoch, "1Epoch"


def run_checked(cmd: list[str]) -> None:
    log("RUN " + " ".join(cmd[:3]) + " ...")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def publish_checkpoint(
    *,
    spec: RunSpec,
    checkpoint: Path,
    staging_root: Path,
    env_file: Path,
    private: bool,
) -> tuple[str, Path]:
    repo_id, epoch_label = select_repo(spec, checkpoint)
    staging_dir = staging_root / repo_id.replace("/", "__")
    epoch = trainer_epoch(checkpoint)
    step = checkpoint_step(checkpoint)
    notes = (
        f"- Source checkpoint: `{checkpoint}`\n"
        f"- Checkpoint step: `{step}`\n"
        f"- Trainer epoch: `{epoch:.4f}`\n"
        "- TB2-lite score: pending GPU evaluation\n"
        "- Upload policy: checkpoint uploaded immediately after save; score card updates after evaluation."
    )
    run_checked(
        [
            sys.executable,
            "gemma4_native_sft/scripts/stage_model_repo.py",
            "--src-dir",
            str(checkpoint),
            "--staging-dir",
            str(staging_dir),
            "--repo-id",
            repo_id,
            "--base-model",
            spec.model_id,
            "--source-dataset",
            str(spec.dataset_path),
            "--tb2-result",
            "pending",
            "--notes",
            notes,
        ]
    )
    upload_cmd = [
        sys.executable,
        "gemma4_native_sft/scripts/upload_model_repo.py",
        "--staging-dir",
        str(staging_dir),
        "--repo-id",
        repo_id,
        "--env-file",
        str(env_file),
        "--commit-message",
        f"Upload Gemma 4 native SFT {epoch_label} checkpoint",
    ]
    if private:
        upload_cmd.append("--private")
    run_checked(upload_cmd)
    return repo_id, staging_dir


def scan_once(args: argparse.Namespace, state: dict) -> bool:
    changed = False
    published = state.setdefault("published", {})
    for spec in RUNS:
        if not spec.output_dir.exists():
            continue
        checkpoints = sorted(spec.output_dir.glob("checkpoint-*"), key=checkpoint_step)
        for checkpoint in checkpoints:
            key = str(checkpoint)
            if key in published:
                continue
            if not checkpoint_complete(checkpoint, args.settle_seconds):
                continue
            log(f"PUBLISH_READY model={spec.model_id} checkpoint={checkpoint}")
            repo_id, staging_dir = publish_checkpoint(
                spec=spec,
                checkpoint=checkpoint,
                staging_root=Path(args.staging_root),
                env_file=Path(args.env_file),
                private=args.private,
            )
            published[key] = {
                "model_id": spec.model_id,
                "checkpoint": key,
                "repo_id": repo_id,
                "staging_dir": str(staging_dir),
                "published_at": utc_now(),
                "tb2_eval": "pending",
            }
            changed = True
            save_state(Path(args.state_file), state)
            log(f"PUBLISHED repo={repo_id} checkpoint={checkpoint}")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", default=str(DEFAULT_STATE))
    parser.add_argument("--staging-root", default=str(DEFAULT_STAGING_ROOT))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--settle-seconds", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    state_path = Path(args.state_file)
    state = load_state(state_path)
    while True:
        try:
            scan_once(args, state)
            save_state(state_path, state)
        except Exception as exc:
            log(f"ERROR {type(exc).__name__}: {exc}")
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
