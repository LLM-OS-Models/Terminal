#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/work/.data")
GEMMA_MODEL_ROOT = DATA_ROOT / "gemma4_native_sft/models"
GEMMA_OUTPUT_DIR = DATA_ROOT / "tb2_lite_eval/gemma4_native_sft_20260509"
REQUESTED_OUTPUT_DIR = DATA_ROOT / "tb2_lite_eval/requested_models_20260514"
DEFAULT_EVAL_PATH = ROOT_DIR / "tb2_lite/data/replay_full.jsonl"
DEFAULT_LOG_DIR = DATA_ROOT / "tb2_lite_eval/priority_queue_20260514/logs"
ZAYA_READY_MARKER = ROOT_DIR / ".vllm-zaya/.zaya_ready"


@dataclass(frozen=True)
class EvalJob:
    name: str
    backend: str
    priority: int
    gpu_count: int
    model_short: str
    output_dir: Path
    model: str = ""
    tokenizer: str = ""
    checkpoint: Path | None = None
    gguf_repo: str = ""
    gguf_filename: str = ""
    gguf_quant: str = ""
    dtype: str = "bfloat16"
    tp_size: int | None = None
    data_parallel_size: int | None = None
    max_model_len: int = 49152
    max_tokens: int = 1024
    gpu_memory_utilization: float = 0.96
    cpu_offload_gb: float = 0.0
    thinking_mode: str = "off"
    strip_thinking_history: str = "on"
    gemma4_empty_thought_channel: str = "auto"
    language_model_only: bool = True
    allow_raw_fallback: bool = False
    enforce_eager: bool = False
    llamacpp_n_batch: int = 2048
    llamacpp_n_ubatch: int = 512
    llamacpp_n_threads: int = 32
    llamacpp_n_threads_batch: int = 64
    llamacpp_flash_attn: bool = True
    skip_if_exists: bool = True
    wait_training: bool = True
    settle_seconds: int = 180
    env_var: str = "VLLM_PYTHON"
    extra_args: tuple[str, ...] = ()
    retry_gpu_counts: tuple[int, ...] = ()
    retry_of: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str, log_path: Path | None = None) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def process_rows() -> list[tuple[int, int, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid,pgid,cmd"],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=True,
    )
    rows: list[tuple[int, int, str]] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def training_running() -> bool:
    own_pid = os.getpid()
    return any(
        pid != own_pid and "gemma4_native_sft/scripts/train_native_hf_fsdp.py" in cmd
        for pid, _pgid, cmd in process_rows()
    )


def newest_mtime(path: Path) -> float:
    newest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except FileNotFoundError:
            pass
    return newest


def checkpoint_ready(path: Path, settle_seconds: int) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    required = ["config.json", "trainer_state.json"]
    missing = [name for name in required if not (path / name).exists()]
    has_weights = (path / "model.safetensors.index.json").exists() or any(path.glob("*.safetensors"))
    if missing:
        return False, f"missing_files={','.join(missing)}"
    if not has_weights:
        return False, "missing_weights"
    age = time.time() - newest_mtime(path)
    if age < settle_seconds:
        return False, f"settling={age:.0f}s/{settle_seconds}s"
    return True, "ready"


def gpu_memory_used() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=True,
    )
    memory: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        idx, used = [part.strip() for part in line.split(",", 1)]
        memory[int(idx)] = int(used)
    return memory


def result_path(job: EvalJob) -> Path:
    return job.output_dir / f"{job.model_short}.json"


def job_record(job: EvalJob) -> dict[str, Any]:
    data = asdict(job)
    data["output_dir"] = str(job.output_dir)
    data["checkpoint"] = str(job.checkpoint) if job.checkpoint is not None else None
    return data


def score_from_result(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return round(100.0 * float(data["aggregate"]["avg_command_f1"]), 2)
    except Exception:
        return None


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:180]


def vllm_python_for(job: EvalJob, args: argparse.Namespace) -> Path:
    if job.env_var and os.environ.get(job.env_var):
        return Path(os.environ[job.env_var]).expanduser()
    if job.env_var == "ZAYA_VLLM_PYTHON":
        return ROOT_DIR / ".vllm-zaya/bin/python"
    if args.vllm_python:
        return args.vllm_python
    candidates = [
        ROOT_DIR / ".vllm-eval-cu129/bin/python",
        ROOT_DIR / ".vllm-020/bin/python",
        ROOT_DIR / ".vllm-nightly/bin/python",
        ROOT_DIR / ".vllm-0_19_1/bin/python",
        ROOT_DIR / ".vllm-work/bin/python",
        ROOT_DIR / ".vllm-env/bin/python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def llamacpp_python_for(args: argparse.Namespace) -> Path:
    if os.environ.get("LLAMACPP_PYTHON"):
        return Path(os.environ["LLAMACPP_PYTHON"]).expanduser()
    if args.llamacpp_python:
        return args.llamacpp_python
    return ROOT_DIR / ".llamacpp-cu/bin/python"


def extend_vllm_env(env: dict[str, str], python_bin: Path) -> None:
    venv_root = python_bin.expanduser().parent.parent
    lib_candidates = [
        venv_root / "lib/python3.12/site-packages/nvidia/cu13/lib",
        venv_root / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
        venv_root / "lib/python3.12/site-packages/torch/lib",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    paths = [str(path) for path in lib_candidates if path.exists()]
    env["LD_LIBRARY_PATH"] = ":".join(paths + ([existing] if existing else []))
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = ""


def gguf_model_id(job: EvalJob) -> str:
    return f"{job.gguf_repo}:{job.gguf_quant or job.gguf_filename}"


def launch_job(
    job: EvalJob,
    gpus: list[int],
    args: argparse.Namespace,
    records: dict[str, dict[str, Any]],
    state_path: Path,
    main_log: Path,
) -> subprocess.Popen:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    gpu_arg = ",".join(str(gpu) for gpu in gpus)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_arg
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.hf_home:
        env["HF_HOME"] = str(args.hf_home)
        env["HF_HUB_CACHE"] = str(args.hf_home / "hub")
        env["TRANSFORMERS_CACHE"] = str(args.hf_home / "transformers")
        env["HF_DATASETS_CACHE"] = str(args.hf_home / "datasets")

    if job.backend == "vllm":
        python_bin = vllm_python_for(job, args)
        extend_vllm_env(env, python_bin)
        env.setdefault("VLLM_USE_DEEP_GEMM", "0")
        env.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
        env.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")
        env.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
        model = str(job.checkpoint) if job.checkpoint is not None else (gguf_model_id(job) if job.gguf_repo else job.model)
        tokenizer = str(job.checkpoint) if job.checkpoint is not None else job.tokenizer
        tp = job.tp_size or len(gpus)
        cmd = [
            str(python_bin),
            "tb2_lite/scripts/replay_eval.py",
            "--model",
            model,
            "--tokenizer-path",
            tokenizer,
            "--model-short",
            job.model_short,
            "--gpu",
            gpu_arg,
            "--tp",
            str(tp),
            "--eval-path",
            str(args.eval_path),
            "--output-dir",
            str(job.output_dir),
            "--dtype",
            job.dtype,
            "--max-model-len",
            str(job.max_model_len),
            "--max-tokens",
            str(job.max_tokens),
            "--gpu-memory-utilization",
            str(job.gpu_memory_utilization),
            "--cpu-offload-gb",
            str(job.cpu_offload_gb),
            "--thinking-mode",
            job.thinking_mode,
            "--strip-thinking-history",
            job.strip_thinking_history,
            "--gemma4-empty-thought-channel",
            job.gemma4_empty_thought_channel,
            "--skip-if-exists",
        ]
        if job.language_model_only:
            cmd.append("--language-model-only")
        if job.allow_raw_fallback:
            cmd.append("--allow-raw-fallback")
        if job.enforce_eager:
            cmd.append("--enforce-eager")
        cmd.append("--disable-custom-all-reduce")
        if job.data_parallel_size is not None:
            cmd.extend(["--data-parallel-size", str(job.data_parallel_size)])
        cmd.extend(job.extra_args)
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
    elif job.backend == "llamacpp":
        python_bin = llamacpp_python_for(args)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = ""
        env.setdefault("NUMEXPR_MAX_THREADS", "256")
        env.setdefault("OMP_NUM_THREADS", str(job.llamacpp_n_threads))
        env.setdefault("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "0")
        cmd = [
            str(python_bin),
            "tb2_lite/scripts/replay_eval_llamacpp.py",
            "--repo-id",
            job.gguf_repo,
            "--filename",
            job.gguf_filename,
            "--tokenizer-path",
            job.tokenizer,
            "--model-short",
            job.model_short,
            "--gpu",
            gpu_arg,
            "--eval-path",
            str(args.eval_path),
            "--output-dir",
            str(job.output_dir),
            "--max-model-len",
            str(job.max_model_len),
            "--max-tokens",
            str(job.max_tokens),
            "--n-batch",
            str(job.llamacpp_n_batch),
            "--n-ubatch",
            str(job.llamacpp_n_ubatch),
            "--n-threads",
            str(job.llamacpp_n_threads),
            "--n-threads-batch",
            str(job.llamacpp_n_threads_batch),
            "--thinking-mode",
            job.thinking_mode,
            "--strip-thinking-history",
            job.strip_thinking_history,
            "--gemma4-empty-thought-channel",
            job.gemma4_empty_thought_channel,
            "--skip-if-exists",
        ]
        if job.llamacpp_flash_attn:
            cmd.append("--flash-attn")
        if job.allow_raw_fallback:
            cmd.append("--allow-raw-fallback")
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
    else:
        raise ValueError(f"unknown backend: {job.backend}")

    log_path = args.log_dir / f"{safe_name(job.name)}_{job.backend}.log"
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"{utc_now()} COMMAND {' '.join(cmd)}\n")
    handle.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT_DIR,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    setattr(proc, "_log_handle", handle)
    records[job.name] = {
        **job_record(job),
        "checkpoint": str(job.checkpoint) if job.checkpoint is not None else None,
        "output_dir": str(job.output_dir),
        "status": "running",
        "pid": proc.pid,
        "gpu": gpu_arg,
        "started_at": utc_now(),
        "log_path": str(log_path),
        "result": str(result_path(job)),
    }
    save_json(state_path, build_state(records, args))
    log(f"EVAL_START name={job.name} backend={job.backend} gpu={gpu_arg} pid={proc.pid}", main_log)
    return proc


def build_state(records: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "updated_at": utc_now(),
        "eval_path": str(args.eval_path),
        "log_dir": str(args.log_dir),
        "records": records,
    }


def base_gemma_jobs() -> list[EvalJob]:
    specs = [
        (
            "gemma4_26b_a4b_it_native_e1",
            "LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-1Epoch",
            GEMMA_MODEL_ROOT / "google__gemma-4-26B-A4B-it__terminal_sft_native_liquid_2epoch/checkpoint-1020",
            (2,),
        ),
        (
            "gemma4_26b_a4b_it_native_e2",
            "LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch",
            GEMMA_MODEL_ROOT / "google__gemma-4-26B-A4B-it__terminal_sft_native_liquid_2epoch/checkpoint-2040",
            (2,),
        ),
        (
            "gemma4_26b_a4b_base_native_e1",
            "LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-1Epoch",
            GEMMA_MODEL_ROOT / "google__gemma-4-26B-A4B__terminal_sft_native_liquid_2epoch/checkpoint-510",
            (2,),
        ),
        (
            "gemma4_26b_a4b_base_native_e2",
            "LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch",
            GEMMA_MODEL_ROOT / "google__gemma-4-26B-A4B__terminal_sft_native_liquid_2epoch/checkpoint-1020",
            (2,),
        ),
        (
            "gemma4_31b_it_native_e1",
            "LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-1Epoch",
            GEMMA_MODEL_ROOT / "google__gemma-4-31B-it__terminal_sft_native_liquid_2epoch/checkpoint-510",
            (4,),
        ),
        (
            "gemma4_31b_it_native_e2",
            "LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch",
            GEMMA_MODEL_ROOT / "google__gemma-4-31B-it__terminal_sft_native_liquid_2epoch/checkpoint-1020",
            (4,),
        ),
        (
            "gemma4_31b_base_native_e1",
            "LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-1Epoch",
            GEMMA_MODEL_ROOT / "google__gemma-4-31B__terminal_sft_native_liquid_2epoch/checkpoint-510",
            (4,),
        ),
        (
            "gemma4_31b_base_native_e2",
            "LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch",
            GEMMA_MODEL_ROOT / "google__gemma-4-31B__terminal_sft_native_liquid_2epoch/checkpoint-1020",
            (4,),
        ),
    ]
    jobs: list[EvalJob] = []
    for short, repo_id, checkpoint, retries in specs:
        jobs.append(
            EvalJob(
                name=short,
                backend="vllm",
                priority=0,
                gpu_count=1,
                model_short=short,
                model=repo_id,
                tokenizer=str(checkpoint),
                checkpoint=checkpoint,
                output_dir=GEMMA_OUTPUT_DIR,
                gpu_memory_utilization=0.97,
                retry_gpu_counts=retries,
                env_var="VLLM_PYTHON",
            )
        )
    return jobs


def requested_jobs() -> list[EvalJob]:
    out = REQUESTED_OUTPUT_DIR
    return [
        EvalJob(
            name="zaya1_8b",
            backend="vllm",
            priority=10,
            gpu_count=1,
            model_short="zaya1_8b",
            model="Zyphra/ZAYA1-8B",
            tokenizer="Zyphra/ZAYA1-8B",
            output_dir=out,
            env_var="ZAYA_VLLM_PYTHON",
            extra_args=("--mamba-cache-dtype", "float32"),
        ),
        EvalJob(
            name="qwopus36_35b_a3b_v1_gguf_q4km",
            backend="llamacpp",
            priority=11,
            gpu_count=1,
            model_short="qwopus36_35b_a3b_v1_gguf_q4km",
            gguf_repo="Jackrong/Qwopus3.6-35B-A3B-v1-GGUF",
            gguf_filename="Qwopus3.6-35B-A3B-v1-Q4_K_M.gguf",
            gguf_quant="Q4_K_M",
            tokenizer="Qwen/Qwen3.6-35B-A3B",
            output_dir=out,
        ),
        EvalJob(
            name="gemma4_31b_it_assistant",
            backend="vllm",
            priority=12,
            gpu_count=2,
            tp_size=2,
            model_short="gemma4_31b_it_assistant",
            model="google/gemma-4-31B-it",
            tokenizer="google/gemma-4-31B-it",
            output_dir=out,
            gpu_memory_utilization=0.92,
            extra_args=(
                "--speculative-config-json",
                '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}',
                "--max-num-seqs",
                "8",
                "--max-num-batched-tokens",
                "16384",
            ),
            retry_gpu_counts=(4,),
        ),
        EvalJob(
            name="qwen36_27b_mtp_gguf_q4km",
            backend="llamacpp",
            priority=13,
            gpu_count=1,
            model_short="qwen36_27b_mtp_gguf_q4km",
            gguf_repo="unsloth/Qwen3.6-27B-MTP-GGUF",
            gguf_filename="Qwen3.6-27B-Q4_K_M.gguf",
            gguf_quant="Q4_K_M",
            tokenizer="Qwen/Qwen3.6-27B",
            output_dir=out,
        ),
        EvalJob(
            name="supergemma4_26b_uncensored_gguf_v2_q4km",
            backend="llamacpp",
            priority=14,
            gpu_count=1,
            model_short="supergemma4_26b_uncensored_gguf_v2_q4km",
            gguf_repo="Jiunsong/supergemma4-26b-uncensored-gguf-v2",
            gguf_filename="supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf",
            gguf_quant="Q4_K_M",
            tokenizer="google/gemma-4-26B-A4B-it",
            output_dir=out,
        ),
        EvalJob(
            name="qwen36_27b_heretic_gguf_q4km",
            backend="llamacpp",
            priority=15,
            gpu_count=1,
            model_short="qwen36_27b_heretic_gguf_q4km",
            gguf_repo="DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF",
            gguf_filename="Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf",
            gguf_quant="Q4_K_M",
            tokenizer="Qwen/Qwen3.6-27B",
            output_dir=out,
        ),
        EvalJob(
            name="qwen36_40b_deckard_gguf_q4km",
            backend="llamacpp",
            priority=16,
            gpu_count=1,
            model_short="qwen36_40b_deckard_gguf_q4km",
            gguf_repo="DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF",
            gguf_filename="Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf",
            gguf_quant="Q4_K_M",
            tokenizer="Qwen/Qwen3.6-27B",
            output_dir=out,
        ),
        EvalJob(
            name="zaya1_74b_preview",
            backend="vllm",
            priority=30,
            gpu_count=8,
            tp_size=1,
            data_parallel_size=8,
            model_short="zaya1_74b_preview",
            model="Zyphra/ZAYA1-74B-preview",
            tokenizer="Zyphra/ZAYA1-74B-preview",
            output_dir=out,
            env_var="ZAYA_VLLM_PYTHON",
            allow_raw_fallback=True,
            extra_args=("--mamba-cache-dtype", "float32", "--enable-expert-parallel"),
        ),
        EvalJob(
            name="deepseek_v4_flash",
            backend="vllm",
            priority=31,
            gpu_count=8,
            tp_size=8,
            model_short="deepseek_v4_flash",
            model="deepseek-ai/DeepSeek-V4-Flash",
            tokenizer="deepseek-ai/DeepSeek-V4-Flash",
            output_dir=out,
            env_var="DEEPSEEK_VLLM_PYTHON",
            gpu_memory_utilization=0.95,
            enforce_eager=True,
            extra_args=(
                "--tokenizer-mode",
                "deepseek_v4",
                "--kv-cache-dtype",
                "fp8",
                "--block-size",
                "256",
                "--max-num-batched-tokens",
                "16384",
                "--max-num-seqs",
                "8",
                "--enable-expert-parallel",
            ),
        ),
    ]


def all_jobs() -> list[EvalJob]:
    return base_gemma_jobs() + requested_jobs()


def retry_job(job: EvalJob, gpu_count: int, remaining_gpu_counts: tuple[int, ...] = ()) -> EvalJob:
    return replace(
        job,
        name=f"{job.name}_retry_{gpu_count}gpu",
        priority=max(job.priority - 1, 0),
        gpu_count=gpu_count,
        tp_size=None if job.data_parallel_size is None else job.tp_size,
        data_parallel_size=gpu_count if job.data_parallel_size is not None else None,
        retry_gpu_counts=remaining_gpu_counts,
        retry_of=job.name,
        skip_if_exists=False,
    )


def llamacpp_fallback_job(job: EvalJob) -> EvalJob:
    return replace(
        job,
        name=f"{job.name}_llamacpp_fallback",
        backend="llamacpp",
        priority=job.priority + 1,
        gpu_count=1,
        tp_size=1,
        data_parallel_size=None,
        retry_gpu_counts=(),
        retry_of=job.name,
        skip_if_exists=False,
        extra_args=(),
    )


def ready_reason(job: EvalJob, training_done: bool) -> tuple[bool, str]:
    if job.wait_training and not training_done:
        return False, "training_running"
    if job.checkpoint is not None:
        return checkpoint_ready(job.checkpoint, job.settle_seconds)
    return True, "ready"


def allocated_gpus(running: dict[str, tuple[EvalJob, subprocess.Popen, list[int]]]) -> set[int]:
    allocated: set[int] = set()
    for _job, _proc, gpus in running.values():
        allocated.update(gpus)
    return allocated


def free_gpus(args: argparse.Namespace, running: dict[str, tuple[EvalJob, subprocess.Popen, list[int]]]) -> list[int]:
    memory = gpu_memory_used()
    allocated = allocated_gpus(running)
    return [
        gpu
        for gpu in sorted(memory)
        if gpu not in allocated and memory[gpu] <= args.free_gpu_used_mb
    ]


def stop_prior_queue_processes(main_log: Path) -> None:
    own_pid = os.getpid()
    seen_pgids: set[int] = set()
    patterns = [
        "gemma4_native_sft/scripts/run_large_native_eval_8gpu.py",
        "tb2_lite/scripts/run_priority_eval_queue.py",
    ]
    for pid, pgid, cmd in process_rows():
        if pid == own_pid:
            continue
        if pgid == os.getpgrp():
            continue
        if any(pattern in cmd for pattern in patterns):
            if pgid in seen_pgids:
                continue
            seen_pgids.add(pgid)
            try:
                os.killpg(pgid, signal.SIGTERM)
                log(f"STOP_OLD_QUEUE pid={pid} pgid={pgid} cmd={cmd[:160]}", main_log)
            except ProcessLookupError:
                pass
    if seen_pgids:
        time.sleep(5)


def run(args: argparse.Namespace) -> int:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    main_log = args.log_dir / "priority_eval_queue.log"
    state_path = args.log_dir / "priority_eval_queue_state.json"
    if args.stop_prior_queues:
        stop_prior_queue_processes(main_log)

    pending = requested_jobs() if args.requested_only else all_jobs()
    running: dict[str, tuple[EvalJob, subprocess.Popen, list[int]]] = {}
    records: dict[str, dict[str, Any]] = {}
    training_done = not training_running()
    log(f"PRIORITY_QUEUE_START training_done={training_done} pending={len(pending)}", main_log)
    save_json(state_path, build_state(records, args))

    while pending or running:
        if not training_done and not training_running():
            training_done = True
            log("TRAINING_DONE scheduler can launch ready evals", main_log)

        for name, (job, proc, gpus) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            handle = getattr(proc, "_log_handle", None)
            if handle is not None:
                handle.close()
            path = result_path(job)
            ok = rc == 0 and path.exists()
            score = score_from_result(path) if path.exists() else None
            records[name].update(
                {
                    "status": "completed" if ok else "failed",
                    "returncode": rc,
                    "finished_at": utc_now(),
                    "score": score,
                    "result": str(path) if path.exists() else None,
                }
            )
            running.pop(name)
            log(f"EVAL_FINISH name={name} returncode={rc} ok={ok} score={score}", main_log)
            if not ok and job.gguf_repo and job.backend == "vllm":
                fallback = llamacpp_fallback_job(job)
                pending.append(fallback)
                records[fallback.name] = {**job_record(fallback), "status": "pending_fallback", "retry_of": job.name}
                log(f"FALLBACK_ENQUEUE name={fallback.name} backend=llamacpp retry_of={job.name}", main_log)
            elif not ok and job.retry_gpu_counts:
                next_count = job.retry_gpu_counts[0]
                retry = retry_job(job, next_count, job.retry_gpu_counts[1:])
                pending.append(retry)
                records[retry.name] = {**job_record(retry), "status": "pending_retry", "retry_of": job.name}
                log(f"RETRY_ENQUEUE name={retry.name} gpu_count={next_count} retry_of={job.name}", main_log)
            save_json(state_path, build_state(records, args))

        launched_any = False
        pending.sort(key=lambda item: (item.priority, item.gpu_count, item.name))
        available = free_gpus(args, running)
        for job in list(pending):
            path = result_path(job)
            if job.skip_if_exists and path.exists():
                score = score_from_result(path)
                records[job.name] = {
                    **job_record(job),
                    "checkpoint": str(job.checkpoint) if job.checkpoint is not None else None,
                    "output_dir": str(job.output_dir),
                    "status": "skipped_existing",
                    "result": str(path),
                    "score": score,
                }
                pending.remove(job)
                log(f"EVAL_SKIP_EXISTING name={job.name} score={score}", main_log)
                save_json(state_path, build_state(records, args))
                continue
            ready, reason = ready_reason(job, training_done)
            if not ready:
                records.setdefault(job.name, {**job_record(job), "status": "waiting", "wait_reason": reason})
                records[job.name].update({"status": "waiting", "wait_reason": reason, "updated_at": utc_now()})
                continue
            if job.backend == "vllm":
                python_bin = vllm_python_for(job, args)
                if not python_bin.exists():
                    records.setdefault(job.name, {**job_record(job), "status": "waiting_env"})
                    records[job.name].update(
                        {"status": "waiting_env", "wait_reason": f"missing {python_bin}", "updated_at": utc_now()}
                    )
                    continue
                if job.env_var == "ZAYA_VLLM_PYTHON" and not ZAYA_READY_MARKER.exists():
                    records.setdefault(job.name, {**job_record(job), "status": "waiting_env"})
                    records[job.name].update(
                        {
                            "status": "waiting_env",
                            "wait_reason": f"waiting for {ZAYA_READY_MARKER}",
                            "updated_at": utc_now(),
                        }
                    )
                    continue
            elif job.backend == "llamacpp":
                python_bin = llamacpp_python_for(args)
                if not python_bin.exists():
                    records.setdefault(job.name, {**job_record(job), "status": "waiting_env"})
                    records[job.name].update(
                        {"status": "waiting_env", "wait_reason": f"missing {python_bin}", "updated_at": utc_now()}
                    )
                    continue
            if len(available) < job.gpu_count:
                records.setdefault(job.name, {**job_record(job), "status": "waiting_gpu"})
                records[job.name].update(
                    {
                        "status": "waiting_gpu",
                        "wait_reason": f"need={job.gpu_count} free={available}",
                        "updated_at": utc_now(),
                    }
                )
                continue
            gpus = available[: job.gpu_count]
            available = available[job.gpu_count :]
            proc = launch_job(job, gpus, args, records, state_path, main_log)
            running[job.name] = (job, proc, gpus)
            pending.remove(job)
            launched_any = True
            if args.launch_stagger_seconds > 0:
                time.sleep(args.launch_stagger_seconds)

        if pending:
            waiting_summary: dict[str, int] = {}
            for job in pending:
                status = records.get(job.name, {}).get("status", "pending")
                waiting_summary[status] = waiting_summary.get(status, 0) + 1
            log(f"QUEUE_MONITOR running={sorted(running)} pending={len(pending)} waiting={waiting_summary}", main_log)
            save_json(state_path, build_state(records, args))
        if pending or running:
            time.sleep(args.poll_seconds if not launched_any else max(args.short_poll_seconds, 1))

    failed = [name for name, record in records.items() if record.get("status") == "failed"]
    log(f"PRIORITY_QUEUE_DONE failed={failed}", main_log)
    save_json(state_path, build_state(records, args))
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--vllm-python", type=Path, default=None)
    parser.add_argument("--llamacpp-python", type=Path, default=None)
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--free-gpu-used-mb", type=int, default=12000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--short-poll-seconds", type=int, default=10)
    parser.add_argument("--launch-stagger-seconds", type=int, default=20)
    parser.add_argument("--stop-prior-queues", action="store_true")
    parser.add_argument("--requested-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
