#!/usr/bin/env python
"""Live no-Docker terminal RLVR with GRPO.

This is the real terminal-feedback path for the local workspace:

  prompt -> model emits command(s) -> subprocess executes them -> verifier runs
  -> stdout/stderr/exit code + verifier result become the RL reward.

It intentionally does not use Docker. Harbor task archives are unpacked into an
isolated workspace under /home/work/.data, and hard-coded /workspace, /output,
and /logs paths are rewritten to that sandbox before command/test execution.

This is GRPO with live terminal verification. The original ECHO paper also adds
an auxiliary CE loss on terminal observation tokens via SkyRL hooks; that exact
auxiliary loss requires SkyRL's custom batch tensors. The trace logs written by
this script preserve command outputs for adding that loss later.
"""

from __future__ import annotations

import argparse
import base64
import builtins
import importlib.machinery
import io
import json
import os
import random
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = ROOT_DIR.parent
ECHO_RL_DIR = REPO_DIR / "echo-rl"
TB2_SCRIPT_DIR = REPO_DIR / "tb2_lite" / "scripts"
for path in (ROOT_DIR, REPO_DIR, ECHO_RL_DIR, TB2_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if not hasattr(builtins, "PreTrainedConfig"):
    if hasattr(transformers, "PreTrainedConfig"):
        builtins.PreTrainedConfig = transformers.PreTrainedConfig
    elif hasattr(transformers, "PretrainedConfig"):
        builtins.PreTrainedConfig = transformers.PretrainedConfig

# TRL imports optional integrations when GRPO is imported. They are not used
# here and are brittle in this workspace, so expose no-op modules.
if "llm_blender" not in sys.modules:
    llm_blender_stub = types.ModuleType("llm_blender")
    llm_blender_stub.__spec__ = importlib.machinery.ModuleSpec("llm_blender", loader=None)

    class _UnusedBlender:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("llm_blender is not used by live terminal RLVR.")

    llm_blender_stub.Blender = _UnusedBlender
    sys.modules["llm_blender"] = llm_blender_stub

if "weave" not in sys.modules:
    weave_stub = types.ModuleType("weave")
    weave_stub.__spec__ = importlib.machinery.ModuleSpec("weave", loader=None)

    class _UnusedEvaluationLogger:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("weave logging is not used by live terminal RLVR.")

    weave_stub.EvaluationLogger = _UnusedEvaluationLogger
    weave_stub.init = lambda *args, **kwargs: None
    trace_stub = types.ModuleType("weave.trace")
    trace_stub.__spec__ = importlib.machinery.ModuleSpec("weave.trace", loader=None)
    context_stub = types.ModuleType("weave.trace.context")
    context_stub.__spec__ = importlib.machinery.ModuleSpec("weave.trace.context", loader=None)

    class _UnusedWeaveClientContext:
        @staticmethod
        def get_weave_client() -> None:
            return None

    context_stub.weave_client_context = _UnusedWeaveClientContext
    sys.modules["weave"] = weave_stub
    sys.modules["weave.trace"] = trace_stub
    sys.modules["weave.trace.context"] = context_stub

from echo_rl.terminal_agent.parsers import HFHermesParser, Qwen35XMLParser, XMLParser
from huggingface_hub import snapshot_download
from trl import GRPOConfig, GRPOTrainer


SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You can use bash commands to inspect files, transform data, write outputs, and verify your work.

Return JSON only, with this structure:

{
  "analysis": "Briefly describe the current state and your plan.",
  "commands": [
    {"keystrokes": "ONE_NON_INTERACTIVE_BASH_COMMAND\\n", "duration": 0.1}
  ],
  "task_complete": false
}

Rules:
- Use non-interactive commands only.
- Prefer one robust shell command or a short command batch.
- Commands execute from /workspace. Required outputs usually belong in /output.
- stdout, stderr, exit codes, files, and verifier results are used as training feedback.
- Do not touch files outside /workspace, /output, and /logs.
"""

DESTRUCTIVE_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\brm\s+-rf\s+\$?\{?(HOME|PWD)\}?",
    r"\bdd\s+.*\bof=/dev/",
    r"\bmkfs\b",
    r"\bmount\b",
    r"\bumount\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bsystemctl\b",
    r"\bservice\b",
    r"\bsudo\b",
    r"\bsu\s",
    r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",
    r"\bchmod\s+-R\s+777\s+/",
    r"\bchown\s+-R\s+.*\s+/",
    r"\b(find|du|grep|rg|ls|tree)\s+/(?:\s|$)",
    r"\b(find|du|grep|rg|ls|tree)\s+/(etc|home|root|proc|sys|dev|var|usr|bin|sbin|lib|lib64|opt|mnt|media|run|tmp)\b",
    r"\b(cat|head|tail|sed|awk|python|python3|perl|ruby|node)\s+/(etc|home|root|proc|sys|dev|var|usr|bin|sbin|lib|lib64|opt|mnt|media|run|tmp)\b",
]

GPU_TASK_PATTERNS = [
    r"\bnvidia-smi\b",
    r"\bcuda_visible_devices\b",
    r"\bnvidia_visible_devices\b",
    r"\bhip_visible_devices\b",
    r"\brocr_visible_devices\b",
    r"\btorch\.cuda\b",
    r"\bcuda:\d+\b",
    r"--device\s+cuda(?::\d+)?\b",
    r"/dev/nvidia",
]

GLOBAL_CONFIG: dict[str, Any] = {}


@dataclass
class SandboxResult:
    reward: float
    parse_ok: bool
    verifier_reward: float
    executed: int
    blocked: bool
    last_exit_code: int | None
    trace: dict[str, Any]


def parse_target_modules(value: str) -> str | list[str]:
    value = value.strip()
    if value == "all-linear":
        return value
    if value.startswith("regex:"):
        return value.removeprefix("regex:")
    modules = [module.strip() for module in value.split(",") if module.strip()]
    return modules if len(modules) > 1 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch",
    )
    parser.add_argument("--sft-adapter-path", default=None)
    parser.add_argument(
        "--output-dir",
        default="/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__live_terminal_rlvr_grpo_r32",
    )
    parser.add_argument("--sandbox-root", default="/home/work/.data/liquid_cli_sft/live_terminal_rlvr/sandboxes")
    parser.add_argument("--trace-dir", default="/home/work/.data/liquid_cli_sft/live_terminal_rlvr/traces")
    parser.add_argument("--hf-cache-dir", default="/home/work/.data/liquid_cli_sft/live_terminal_rlvr/hf")
    parser.add_argument("--hf-rl-dataset", default="open-thoughts/OpenThoughts-Agent-v1-RL")
    parser.add_argument(
        "--hf-rl-parquet",
        default=None,
        help="Optional local OpenThoughts-Agent-v1-RL parquet path. If omitted, cached HF snapshots are searched first.",
    )
    parser.add_argument("--include-openthoughts-rl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-endless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--endless-repo", default="obiwan96/endless-terminals")
    parser.add_argument("--include-tb-dev", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tb-dev-repo", default="open-thoughts/OpenThoughts-TB-dev")
    parser.add_argument("--include-tblite-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tblite-repo", default="open-thoughts/OpenThoughts-TBLite")
    parser.add_argument("--local-task-dir", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--steps-per-generation", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--random-state", type=int, default=3407)
    parser.add_argument("--command-timeout", type=float, default=25.0)
    parser.add_argument("--verifier-timeout", type=float, default=45.0)
    parser.add_argument("--max-commands-per-completion", type=int, default=4)
    parser.add_argument("--max-command-chars", type=int, default=6000)
    parser.add_argument("--max-terminal-output-chars", type=int, default=12000)
    parser.add_argument("--keep-sandboxes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test-reward", action="store_true")
    return parser.parse_args()


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def text_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        return len(token_ids)
    except Exception:
        text = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
        return len(tokenizer(text, add_special_tokens=False).input_ids)


def safe_extract_tar_bytes(payload: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        root = dest.resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"Unsafe tar path: {member.name}")
        tar.extractall(dest)


def safe_extract_tar_file(path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="r:*") as tar:
        root = dest.resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"Unsafe tar path: {member.name}")
        tar.extractall(dest)


def find_task_root(root: Path) -> Path:
    if (root / "instruction.md").exists() and (root / "task.toml").exists():
        return root
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "instruction.md").exists()]
    if len(candidates) == 1:
        return candidates[0]
    for candidate in candidates:
        if (candidate / "task.toml").exists():
            return candidate
    return root


def has_gpu_dependency_text(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in GPU_TASK_PATTERNS)


def path_tree_has_gpu_dependency(root: Path, max_bytes: int = 1_000_000) -> bool:
    if not root.exists():
        return False
    paths = [root] if root.is_file() else list(root.rglob("*"))
    for path in paths:
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if has_gpu_dependency_text(text):
            return True
    return False


def task_archive_has_gpu_dependency(payload: bytes, max_bytes: int = 1_000_000) -> bool:
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or member.size > max_bytes:
                    continue
                file_obj = tar.extractfile(member)
                if not file_obj:
                    continue
                text = file_obj.read().decode("utf-8", "replace")
                if has_gpu_dependency_text(text):
                    return True
    except Exception:
        return False
    return False


def extract_instruction_from_task_binary(payload: bytes) -> str:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        names = tar.getnames()
        instruction_names = [n for n in names if n.endswith("instruction.md")]
        if not instruction_names:
            return ""
        member = tar.getmember(instruction_names[0])
        file_obj = tar.extractfile(member)
        if not file_obj:
            return ""
        return file_obj.read().decode("utf-8", "replace").strip()


def make_prompt(instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task:\n{instruction.strip()}"},
    ]


def b64_encode_bytes(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def b64_decode_bytes(payload: str) -> bytes:
    return base64.b64decode(payload.encode("ascii"))


def iter_task_dirs(root: Path) -> list[Path]:
    if (root / "instruction.md").exists():
        return [root]
    task_dirs = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "instruction.md").exists() and ((path / "tests").exists() or (path / "task.toml").exists()):
            task_dirs.append(path)
    return task_dirs


def load_hf_task_repo(repo_id: str, cache_root: Path) -> Path:
    safe_name = repo_id.replace("/", "__")
    local_dir = cache_root / "repos" / safe_name
    if not local_dir.exists() or not any(local_dir.iterdir()):
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir), local_dir_use_symlinks=False)
    return local_dir


def find_cached_hf_parquet(repo_id: str, filename: str = "tasks.parquet") -> Path | None:
    repo_name = repo_id.replace("/", "--")
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{repo_name}" / "snapshots",
        Path("/home/work/.cache/huggingface/hub") / f"datasets--{repo_name}" / "snapshots",
    ]
    for snapshots in candidates:
        if not snapshots.exists():
            continue
        matches = sorted(snapshots.glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def load_openthoughts_rl_dataset(args: argparse.Namespace) -> Dataset:
    parquet_path = Path(args.hf_rl_parquet).expanduser().resolve() if args.hf_rl_parquet else None
    if parquet_path is None:
        parquet_path = find_cached_hf_parquet(args.hf_rl_dataset)
    if parquet_path is not None and parquet_path.exists():
        print(
            json.dumps({"event": "using_cached_hf_rl_parquet", "path": str(parquet_path)}, ensure_ascii=False),
            flush=True,
        )
        return load_dataset("parquet", data_files=str(parquet_path), split="train")
    return load_dataset(args.hf_rl_dataset, split="train")


def record_from_task_dir(tokenizer: Any, task_dir: Path, source: str, args: argparse.Namespace) -> dict[str, Any] | None:
    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        return None
    if path_tree_has_gpu_dependency(task_dir):
        return None
    instruction = instruction_path.read_text(encoding="utf-8", errors="replace").strip()
    if not instruction:
        return None
    messages = make_prompt(instruction)
    prompt_tokens = text_token_count(tokenizer, messages)
    if prompt_tokens > args.max_prompt_length:
        return None
    return {
        "prompt": messages,
        "task_id": task_dir.name,
        "source": source,
        "task_dir": str(task_dir.resolve()),
        "task_binary_b64": "",
        "prompt_tokens": prompt_tokens,
    }


def build_dataset(tokenizer: Any, args: argparse.Namespace) -> Dataset:
    cache_root = Path(args.hf_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    skipped_long = 0
    skipped_gpu_dependency = 0

    def can_add(source: str) -> bool:
        if args.max_rows is not None and len(records) >= args.max_rows:
            return False
        if args.max_rows_per_source is not None and source_counts.get(source, 0) >= args.max_rows_per_source:
            return False
        return True

    def add_record(record: dict[str, Any]) -> None:
        records.append(record)
        source_counts[record["source"]] = source_counts.get(record["source"], 0) + 1

    if args.include_openthoughts_rl:
        source = "openthoughts_rl"
        ds = load_openthoughts_rl_dataset(args)
        for row in ds:
            if not can_add(source):
                break
            payload = bytes(row["task_binary"])
            if task_archive_has_gpu_dependency(payload):
                skipped_gpu_dependency += 1
                continue
            instruction = extract_instruction_from_task_binary(payload)
            if not instruction:
                continue
            messages = make_prompt(instruction)
            prompt_tokens = text_token_count(tokenizer, messages)
            if prompt_tokens > args.max_prompt_length:
                skipped_long += 1
                continue
            add_record(
                {
                    "prompt": messages,
                    "task_id": str(row["path"]),
                    "source": source,
                    "task_dir": "",
                    "task_binary_b64": b64_encode_bytes(payload),
                    "prompt_tokens": prompt_tokens,
                }
            )

    if args.include_endless:
        source = "endless_terminals"
        endless_root = load_hf_task_repo(args.endless_repo, cache_root)
        for task_dir in iter_task_dirs(endless_root):
            if not can_add(source):
                break
            record = record_from_task_dir(tokenizer, task_dir, source, args)
            if record is None:
                if path_tree_has_gpu_dependency(task_dir):
                    skipped_gpu_dependency += 1
                else:
                    skipped_long += 1
                continue
            add_record(record)

    if args.include_tb_dev:
        source = "openthoughts_tb_dev"
        tb_dev_root = load_hf_task_repo(args.tb_dev_repo, cache_root)
        for task_dir in iter_task_dirs(tb_dev_root):
            if not can_add(source):
                break
            record = record_from_task_dir(tokenizer, task_dir, source, args)
            if record is None:
                if path_tree_has_gpu_dependency(task_dir):
                    skipped_gpu_dependency += 1
                else:
                    skipped_long += 1
                continue
            add_record(record)

    if args.include_tblite_train:
        source = "openthoughts_tblite"
        tblite_root = load_hf_task_repo(args.tblite_repo, cache_root)
        for task_dir in iter_task_dirs(tblite_root):
            if not can_add(source):
                break
            record = record_from_task_dir(tokenizer, task_dir, source, args)
            if record is None:
                if path_tree_has_gpu_dependency(task_dir):
                    skipped_gpu_dependency += 1
                else:
                    skipped_long += 1
                continue
            add_record(record)

    for local in args.local_task_dir:
        root = Path(local).expanduser().resolve()
        source = f"local:{root.name}"
        for task_dir in iter_task_dirs(root):
            if not can_add(source):
                break
            record = record_from_task_dir(tokenizer, task_dir, source, args)
            if record is None:
                if path_tree_has_gpu_dependency(task_dir):
                    skipped_gpu_dependency += 1
                else:
                    skipped_long += 1
                continue
            add_record(record)

    if not records:
        raise RuntimeError("No live terminal tasks were loaded.")
    rng = random.Random(args.random_state)
    rng.shuffle(records)
    dataset = Dataset.from_list(records)
    print(
        json.dumps(
            {
                "event": "live_dataset_ready",
                "train_rows": len(dataset),
                "source_counts": source_counts,
                "skipped_long_or_invalid": skipped_long,
                "skipped_gpu_dependency": skipped_gpu_dependency,
                "prompt_tokens_min": min(r["prompt_tokens"] for r in records),
                "prompt_tokens_max": max(r["prompt_tokens"] for r in records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return dataset


def parse_json_blob(text: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def split_keystrokes(value: str) -> list[str]:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return [piece.strip() for piece in value.split("\n") if piece.strip()]


def parse_commands(text: str) -> tuple[list[str], bool, bool, list[str]]:
    commands: list[str] = []
    warnings: list[str] = []
    is_done = False
    valid = False

    payload = parse_json_blob(text)
    if payload is not None:
        valid = True
        if isinstance(payload.get("task_complete"), bool):
            is_done = bool(payload["task_complete"])
        raw_commands = payload.get("commands")
        if isinstance(raw_commands, list):
            for item in raw_commands:
                if not isinstance(item, dict):
                    continue
                keystrokes = item.get("keystrokes") or item.get("command") or item.get("cmd")
                if isinstance(keystrokes, str):
                    commands.extend(split_keystrokes(keystrokes))
        for key in ("command", "cmd", "keystrokes"):
            value = payload.get(key)
            if isinstance(value, str):
                commands.extend(split_keystrokes(value))

    for parser in (XMLParser(), HFHermesParser(), Qwen35XMLParser()):
        try:
            result = parser.parse_response(text)
        except TypeError:
            result = parser.parse_response(text, extract_answer_tags_for_done=False)
        except Exception as exc:
            warnings.append(f"parser_error:{type(exc).__name__}")
            continue
        if result.is_done:
            is_done = True
            valid = True
        if result.commands:
            valid = True
        for command in result.commands:
            if command.error:
                warnings.append(command.error)
                continue
            if command.name not in {"bash", "shell"}:
                warnings.append(f"unsupported_tool:{command.name}")
                continue
            value = command.arguments.get("command") or command.arguments.get("cmd") or command.arguments.get("keystrokes")
            if isinstance(value, str):
                commands.extend(split_keystrokes(value))

    # Fallback for common raw tags/fences.
    for match in re.findall(r"<command>(.*?)</command>", text, flags=re.DOTALL):
        commands.extend(split_keystrokes(match))
    fenced_shell = re.findall(r"```(?:bash|sh|shell)?\s*(.*?)```", text, flags=re.DOTALL)
    for block in fenced_shell:
        if "{" not in block[:20]:
            commands.extend(split_keystrokes(block))

    deduped: list[str] = []
    seen = set()
    for command in commands:
        if command and command not in seen:
            deduped.append(command)
            seen.add(command)
    return deduped, is_done, valid, warnings


def is_unsafe_command(command: str) -> bool:
    lowered = command.lower()
    gpu_forbidden_patterns = [
        r"\bcuda_visible_devices\s*=",
        r"\bnvidia_visible_devices\s*=",
        r"\bhip_visible_devices\s*=",
        r"\brocr_visible_devices\s*=",
        r"\bnvidia-smi\b",
        r"--device\s+cuda(?::\d+)?\b",
        r"\bcuda:\d+\b",
        r"\btorch\.cuda\b",
        r"\.to\(\s*['\"]cuda",
    ]
    for pattern in gpu_forbidden_patterns:
        if re.search(pattern, lowered):
            return True
    allowed_abs = {
        "/workspace",
        "/output",
        "/logs",
        "/dev/null",
        "$workspace",
        "$output",
        "$logs",
        "${workspace}",
        "${output}",
        "${logs}",
    }
    for pattern in DESTRUCTIVE_PATTERNS:
        if re.search(pattern, lowered):
            return True
    for match in re.finditer(r"(?<![\w.-])/(?:[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)?", lowered):
        path = match.group(0).rstrip("/") or "/"
        if path in {"/workspace", "/output", "/logs", "/dev/null"}:
            continue
        if any(path.startswith(prefix + "/") for prefix in allowed_abs if prefix.startswith("/")):
            continue
        return True
    return False


def ensure_no_gpu_wrappers(sandbox: Path) -> Path:
    bin_dir = sandbox / "bin_no_gpu"
    bin_dir.mkdir(parents=True, exist_ok=True)
    path_env = os.environ.get("PATH", "")
    wrappers = {
        "python": shutil.which("python", path=path_env) or sys.executable,
        "python3": shutil.which("python3", path=path_env) or sys.executable,
        "pytest": shutil.which("pytest", path=path_env) or "",
        "pip": shutil.which("pip", path=path_env) or "",
        "pip3": shutil.which("pip3", path=path_env) or "",
    }
    for name, real_path in wrappers.items():
        if not real_path:
            continue
        wrapper = bin_dir / name
        wrapper.write_text(
            "#!/usr/bin/env bash\n"
            "unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES CUDA_DEVICE_ORDER "
            "HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES\n"
            "export CUDA_VISIBLE_DEVICES=\n"
            "export NVIDIA_VISIBLE_DEVICES=none\n"
            f"exec {shlex.quote(real_path)} \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'nvidia-smi is disabled inside no-Docker RLVR sandboxes' >&2\n"
        "exit 127\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(nvidia_smi.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def sandbox_subprocess_env(cwd: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    ]:
        env.pop(key, None)
    no_gpu_bin = ensure_no_gpu_wrappers(cwd.parent)
    env.update(
        {
            "WORKSPACE": str(cwd),
            "OUTPUT": str(cwd.parent / "output"),
            "LOGS": str(cwd.parent / "logs"),
            "HOME": str(cwd),
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "PATH": f"{no_gpu_bin}:{env.get('PATH', '')}",
        }
    )
    return env


def rewrite_paths(command: str, sandbox: Path) -> str:
    replacements = {
        "/workspace": str(sandbox / "workspace"),
        "/output": str(sandbox / "output"),
        "/logs": str(sandbox / "logs"),
    }
    rewritten = command
    for src, dst in replacements.items():
        rewritten = rewritten.replace(src, dst)
    return rewritten


def terminate_process_group(proc: subprocess.Popen[str] | None, grace_sec: float = 1.0) -> None:
    if proc is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        except Exception:
            return
        if sig == signal.SIGTERM:
            time.sleep(grace_sec)


def copytree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target, symlinks=False)
        else:
            shutil.copy2(child, target)


def setup_sandbox(task_binary_b64: str, task_dir_value: str, task_id: str) -> tuple[Path, Path]:
    root = Path(GLOBAL_CONFIG["sandbox_root"]) / f"rank{os.environ.get('LOCAL_RANK', '0')}" / f"{task_id}_{uuid.uuid4().hex[:10]}"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    task_extract = root / "task"
    workspace = root / "workspace"
    output = root / "output"
    logs = root / "logs"
    for path in (task_extract, workspace, output, logs / "verifier"):
        path.mkdir(parents=True, exist_ok=True)

    if task_binary_b64:
        safe_extract_tar_bytes(b64_decode_bytes(task_binary_b64), task_extract)
        task_root = find_task_root(task_extract)
    else:
        task_root = Path(task_dir_value).resolve()
        if not task_root.exists():
            raise RuntimeError(f"task_dir does not exist: {task_root}")

    env_dir = task_root / "environment"
    if env_dir.exists():
        seeds_dir = env_dir / "seeds"
        if seeds_dir.exists():
            copytree_contents(seeds_dir, workspace)
        for child in env_dir.iterdir():
            if child.name in {"Dockerfile", "docker-compose.yaml", "container.def", "task.json", "seeds"}:
                continue
            target = workspace / child.name
            if child.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(child, target, symlinks=False)
            else:
                shutil.copy2(child, target)

    tests_src = task_root / "tests"
    if tests_src.exists():
        tests_dst = root / "tests"
        shutil.copytree(tests_src, tests_dst, symlinks=False)
        rewrite_text_tree(tests_dst, root)

    return root, task_root


def rewrite_text_tree(root: Path, sandbox: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text = text.replace("/workspace", str(sandbox / "workspace"))
        text = text.replace("/output", str(sandbox / "output"))
        text = text.replace("/logs", str(sandbox / "logs"))
        path.write_text(text, encoding="utf-8")


def run_subprocess(command: str, cwd: Path, timeout: float) -> dict[str, Any]:
    start = time.monotonic()
    proc: subprocess.Popen[str] | None = None
    try:
        if has_gpu_dependency_text(command):
            return {
                "command": command,
                "exit_code": 127,
                "stdout": "",
                "stderr": "GPU/CUDA commands are disabled in no-Docker RLVR sandboxes.",
                "duration_sec": round(time.monotonic() - start, 4),
                "timeout": False,
                "blocked": True,
                "reason": "gpu_dependency",
            }
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=sandbox_subprocess_env(cwd),
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        # The shell can exit while leaving background children in its session.
        # Clean the whole process group so verifier scripts cannot leak loops.
        terminate_process_group(proc, grace_sec=0.1)
        return {
            "command": command,
            "exit_code": proc.returncode,
            "stdout": stdout[-int(GLOBAL_CONFIG["max_terminal_output_chars"]) :],
            "stderr": stderr[-int(GLOBAL_CONFIG["max_terminal_output_chars"]) :],
            "duration_sec": round(time.monotonic() - start, 4),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        if proc is not None:
            terminate_process_group(proc, grace_sec=0.2)
            stdout, stderr = proc.communicate()
        else:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "command": command,
            "exit_code": -124,
            "stdout": (stdout or "")[-int(GLOBAL_CONFIG["max_terminal_output_chars"]) :],
            "stderr": (stderr or "")[-int(GLOBAL_CONFIG["max_terminal_output_chars"]) :],
            "duration_sec": round(time.monotonic() - start, 4),
            "timeout": True,
        }


def run_verifier(sandbox: Path) -> tuple[float, dict[str, Any]]:
    tests_dir = sandbox / "tests"
    test_sh = tests_dir / "test.sh"
    if not test_sh.exists():
        return 0.0, {"verifier_missing": True, "exit_code": None, "stdout": "", "stderr": "missing tests/test.sh"}
    if path_tree_has_gpu_dependency(tests_dir):
        return 0.0, {
            "verifier_skipped": True,
            "reason": "gpu_dependency",
            "exit_code": 127,
            "stdout": "",
            "stderr": "GPU/CUDA verifier tasks are skipped in no-Docker RLVR sandboxes.",
        }
    test_sh.chmod(test_sh.stat().st_mode | 0o111)
    result = run_subprocess(str(test_sh), sandbox / "workspace", float(GLOBAL_CONFIG["verifier_timeout"]))
    reward_path = sandbox / "logs" / "verifier" / "reward.txt"
    reward = 1.0 if result["exit_code"] == 0 else 0.0
    if reward_path.exists():
        try:
            reward = float(reward_path.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    return max(0.0, min(1.0, reward)), result


def execute_live_task(
    completion: Any,
    task_id: str,
    source: str,
    task_binary_b64: str,
    task_dir: str,
) -> SandboxResult:
    text = completion_text(completion)
    commands, is_done, parse_ok, warnings = parse_commands(text)
    trace: dict[str, Any] = {
        "task_id": task_id,
        "source": source,
        "parse_ok": parse_ok,
        "is_done": is_done,
        "warnings": warnings,
        "raw_completion": text[:12000],
        "commands": commands[: int(GLOBAL_CONFIG["max_commands_per_completion"])],
        "events": [],
    }
    if not parse_ok:
        return SandboxResult(-0.75, False, 0.0, 0, False, None, trace)
    if not commands and is_done:
        return SandboxResult(-0.25, True, 0.0, 0, False, None, trace)
    if not commands:
        return SandboxResult(-0.65, True, 0.0, 0, False, None, trace)

    sandbox: Path | None = None
    last_exit_code: int | None = None
    blocked = False
    try:
        sandbox, _ = setup_sandbox(task_binary_b64, task_dir, re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:80])
        for raw_command in commands[: int(GLOBAL_CONFIG["max_commands_per_completion"])]:
            if len(raw_command) > int(GLOBAL_CONFIG["max_command_chars"]):
                trace["events"].append({"blocked": True, "reason": "command_too_long", "raw": raw_command[:500]})
                blocked = True
                break
            if is_unsafe_command(raw_command):
                trace["events"].append({"blocked": True, "reason": "unsafe_pattern", "raw": raw_command[:500]})
                blocked = True
                break
            command = rewrite_paths(raw_command, sandbox)
            event = run_subprocess(command, sandbox / "workspace", float(GLOBAL_CONFIG["command_timeout"]))
            event["raw_command"] = raw_command
            trace["events"].append(event)
            last_exit_code = int(event["exit_code"])
            if event["timeout"]:
                break
        verifier_reward, verifier_event = run_verifier(sandbox)
        trace["verifier"] = verifier_event
        trace["verifier_reward"] = verifier_reward
    except Exception as exc:
        trace["exception"] = f"{type(exc).__name__}: {exc}"
        verifier_reward = 0.0
    finally:
        if sandbox is not None and not bool(GLOBAL_CONFIG["keep_sandboxes"]):
            shutil.rmtree(sandbox, ignore_errors=True)

    executed = len([event for event in trace["events"] if not event.get("blocked")])
    score = 5.0 * verifier_reward
    score += 0.25 if parse_ok else -0.5
    score += min(executed, 2) * 0.1
    if last_exit_code == 0:
        score += 0.15
    elif last_exit_code is not None:
        score -= 0.1
    if blocked:
        score -= 1.0
    if is_done and verifier_reward < 0.5:
        score -= 0.25
    return SandboxResult(float(max(-1.0, min(score, 6.0))), parse_ok, verifier_reward, executed, blocked, last_exit_code, trace)


def append_trace(trace: dict[str, Any]) -> None:
    trace_dir = Path(GLOBAL_CONFIG["trace_dir"])
    trace_dir.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("LOCAL_RANK", "0")
    path = trace_dir / f"live_rollouts_rank{rank}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")


def reward_live_terminal_execution(completions, **kwargs) -> list[float]:
    task_ids = kwargs.get("task_id", [])
    sources = kwargs.get("source", [])
    task_binaries = kwargs.get("task_binary_b64", [])
    task_dirs = kwargs.get("task_dir", [])
    rewards: list[float] = []
    for completion, task_id, source, task_binary_b64, task_dir in zip(
        completions,
        task_ids,
        sources,
        task_binaries,
        task_dirs,
    ):
        result = execute_live_task(completion, str(task_id), str(source), str(task_binary_b64), str(task_dir))
        trace = result.trace
        trace.update(
            {
                "reward": result.reward,
                "executed": result.executed,
                "blocked": result.blocked,
                "last_exit_code": result.last_exit_code,
            }
        )
        append_trace(trace)
        rewards.append(result.reward)
    return rewards


def smoke_test_reward(dataset: Dataset) -> None:
    row = dataset[0]
    completion = {
        "content": json.dumps(
            {
                "analysis": "Inspect files first.",
                "commands": [{"keystrokes": "ls -la && find . -maxdepth 2 -type f | sort\n", "duration": 0.1}],
                "task_complete": False,
            }
        )
    }
    result = execute_live_task(
        completion,
        str(row["task_id"]),
        str(row["source"]),
        str(row["task_binary_b64"]),
        str(row["task_dir"]),
    )
    print(json.dumps({"event": "smoke_reward", "reward": result.reward, "trace": result.trace}, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    GLOBAL_CONFIG.clear()
    GLOBAL_CONFIG.update(
        {
            "sandbox_root": args.sandbox_root,
            "trace_dir": args.trace_dir,
            "command_timeout": args.command_timeout,
            "verifier_timeout": args.verifier_timeout,
            "max_commands_per_completion": args.max_commands_per_completion,
            "max_command_chars": args.max_command_chars,
            "max_terminal_output_chars": args.max_terminal_output_chars,
            "keep_sandboxes": args.keep_sandboxes,
        }
    )

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --push-to-hub is enabled.")
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = build_dataset(tokenizer, args)

    if args.smoke_test_reward:
        smoke_test_reward(dataset)
    if args.dry_run:
        print(json.dumps({"event": "dry_run_ok", "rows": len(dataset)}, ensure_ascii=False), flush=True)
        return

    target_modules = parse_target_modules(args.target_modules)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        model.to(torch.device("cuda", local_rank))

    if args.sft_adapter_path:
        model = PeftModel.from_pretrained(model, args.sft_adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    for candidate in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        if candidate is not None and not hasattr(candidate, "warnings_issued"):
            candidate.warnings_issued = {}

    global_train_batch = world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    generation_batch_size = args.generation_batch_size or global_train_batch
    if local_rank == 0:
        print(
            json.dumps(
                {
                    "task": "live_terminal_rlvr_grpo_no_docker",
                    "paper_reference": "ECHO uses live terminal observations; this run uses live terminal verifier reward and logs observations.",
                    "model_path": args.model_path,
                    "dataset_rows": len(dataset),
                    "output_dir": args.output_dir,
                    "world_size": world_size,
                    "effective_batch_size": global_train_batch,
                    "generation_batch_size": generation_batch_size,
                    "num_generations": args.num_generations,
                    "max_steps": args.max_steps,
                    "learning_rate": args.learning_rate,
                    "beta": args.beta,
                    "target_modules": target_modules,
                    "use_vllm": args.use_vllm,
                    "sandbox_root": args.sandbox_root,
                    "trace_dir": args.trace_dir,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    train_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="constant_with_warmup",
        optim="adamw_8bit",
        bf16=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        generation_batch_size=generation_batch_size,
        steps_per_generation=args.steps_per_generation,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=1.03,
        use_vllm=args.use_vllm,
        remove_unused_columns=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_strategy="every_save" if args.push_to_hub else "checkpoint",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_live_terminal_execution],
        args=train_args,
        train_dataset=dataset,
    )
    trainer.train()

    final_dir = Path(args.output_dir) / "final_lora"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if local_rank == 0:
        (final_dir / "LIVE_RLVR_NOTES.md").write_text(
            "\n".join(
                [
                    "# Live Terminal RLVR Adapter",
                    "",
                    "This adapter was trained with live no-Docker terminal execution.",
                    "The reward function parsed model commands, executed them in a local",
                    "sandbox, captured stdout/stderr/exit codes, and ran task verifiers.",
                    "",
                    "Training sources are controlled by the run args. By default this",
                    "uses open-thoughts/OpenThoughts-Agent-v1-RL, obiwan96/endless-terminals,",
                    "and open-thoughts/OpenThoughts-TB-dev when available.",
                    "",
                    "TB2/TBLite should remain final evaluation unless explicitly enabled.",
                    "",
                    "Trace JSONL files contain terminal observations for later ECHO-style",
                    "environment-token CE integration.",
                    "",
                ]
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
