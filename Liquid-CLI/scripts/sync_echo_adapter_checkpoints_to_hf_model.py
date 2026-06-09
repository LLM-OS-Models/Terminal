#!/usr/bin/env python3
"""Sync ECHO RLVR LoRA adapter checkpoints to a Hugging Face model repo."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


SECRET_KEY_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_token(env_file: Path) -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if token:
        return token
    env = parse_env_file(env_file)
    token = env.get("HF_TOKEN") or env.get("HUGGINGFACE_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(f"HF token not found in environment or {env_file}")
    return token


def redact_env(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    lines: list[str] = []
    for raw_line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw_line:
            lines.append(raw_line)
            continue
        key, _ = raw_line.split("=", 1)
        clean_key = key.replace("export ", "").strip()
        if SECRET_KEY_RE.search(clean_key):
            lines.append(f"{key}=<redacted>")
        else:
            lines.append(raw_line)
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def checkpoints(output_dir: Path) -> list[Path]:
    items = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    return sorted(items, key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)


def require_adapter_artifacts(path: Path) -> None:
    required = [path / "adapter_config.json", path / "tokenizer_config.json"]
    missing = [str(p) for p in required if not p.exists()]
    if not ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()):
        missing.append("adapter_model.safetensors or adapter_model.bin")
    if missing:
        raise FileNotFoundError(f"incomplete adapter checkpoint {path}: {', '.join(missing)}")


def read_state(path: Path) -> dict:
    if not path.exists():
        return {"uploaded_checkpoints": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"uploaded_checkpoints": []}


def write_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_readme(repo_id: str, run_env: dict[str, str], manifest: dict) -> str:
    model_name = repo_id.split("/", 1)[-1]
    uploaded = manifest["uploaded_checkpoints"]
    latest = uploaded[-1] if uploaded else "none"
    return f"""---
license: apache-2.0
base_model: {run_env.get("MODEL_PATH", "LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch")}
tags:
- liquid-ai
- lfm2.5
- terminal-agent
- rlvr
- echo
- grpo
- lora
- adapter
---

# {model_name}

LoRA adapter checkpoints from an ECHO-style terminal RLVR run.

## Training Setup

- Base model: `{run_env.get("MODEL_PATH", "")}`
- Resume adapter: `{run_env.get("RESUME_ADAPTER", "")}`
- Run ID: `{run_env.get("RUN_ID", "")}`
- vLLM GPUs: `{run_env.get("VLLM_GPUS", "0,1,2,3")}`
- Training GPUs: `{run_env.get("TRAIN_GPUS", "4,5")}`
- No-Docker sandbox: `{run_env.get("NO_DOCKER", "true")}`
- World-model coefficient: `{run_env.get("WORLD_MODEL_COEFF", "")}`
- Learning rate: `{run_env.get("LEARNING_RATE", "")}`
- Save interval: every `{run_env.get("SAVE_STEPS", "")}` train steps

## Latest Snapshot

- Synced at UTC: `{manifest["synced_at_utc"]}`
- Latest uploaded checkpoint: `{latest}`
- Uploaded checkpoints: `{", ".join(uploaded) if uploaded else "none"}`

## Layout

- `checkpoints/checkpoint-*`: PEFT/LoRA adapter checkpoint directories.
- `manifest.json`: sync metadata.
- `run.env.redacted`: training configuration with secret-like values redacted.

The paired rollout dataset is:

`LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts`

That dataset stores the terminal actions, observations, verifier outcomes, and
train-step metrics that produced these adapter checkpoints.
"""


def stage_metadata(stage_dir: Path, repo_id: str, run_dir: Path, output_dir: Path, uploaded: list[str]) -> Path:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    run_env = parse_env_file(run_dir / "run.env")
    manifest = {
        "repo_id": repo_id,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        "uploaded_checkpoints": uploaded,
    }
    (stage_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    redact_env(run_dir / "run.env", stage_dir / "run.env.redacted")
    (stage_dir / "README.md").write_text(build_readme(repo_id, run_env, manifest), encoding="utf-8")
    return stage_dir


def sync_once(args: argparse.Namespace, api: HfApi) -> dict:
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    state_path = Path(args.state_file).resolve() if args.state_file else run_dir / "hf_model_sync_state.json"
    state = read_state(state_path)
    uploaded: list[str] = list(state.get("uploaded_checkpoints", []))
    uploaded_set = set(uploaded)

    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    new_uploads: list[str] = []
    for ckpt in checkpoints(output_dir):
        if ckpt.name in uploaded_set:
            continue
        require_adapter_artifacts(ckpt)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(ckpt),
            path_in_repo=f"checkpoints/{ckpt.name}",
            commit_message=f"Upload {ckpt.name} from {run_dir.name}",
        )
        uploaded.append(ckpt.name)
        uploaded_set.add(ckpt.name)
        new_uploads.append(ckpt.name)
        state["uploaded_checkpoints"] = uploaded
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_state(state_path, state)

    stage_dir = stage_metadata(
        Path(args.stage_root).resolve() / run_dir.name,
        args.repo_id,
        run_dir,
        output_dir,
        uploaded,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(stage_dir),
        path_in_repo=".",
        commit_message=f"Update metadata for {run_dir.name}",
    )

    return {
        "event": "hf_model_sync_done",
        "repo_id": args.repo_id,
        "run_id": run_dir.name,
        "new_uploads": new_uploads,
        "uploaded_checkpoints": uploaded,
        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-root", default="/home/work/.data/hf_upload_stage/echo_rlvr_adapter_checkpoints")
    parser.add_argument("--state-file", default="")
    parser.add_argument("--env-file", default="/home/work/.projects/LLM-OS-Models/Terminal/.env")
    parser.add_argument("--interval-sec", type=int, default=900)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi(token=get_token(Path(args.env_file)))

    while True:
        try:
            result = sync_once(args, api)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "hf_model_sync_error",
                        "repo_id": args.repo_id,
                        "run_dir": args.run_dir,
                        "error": repr(exc),
                        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if not args.loop:
                raise
        if not args.loop:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
