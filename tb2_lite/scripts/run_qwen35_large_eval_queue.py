#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/work/.data")
HF_HOME = DATA_ROOT / "huggingface"
OUT_DIR = DATA_ROOT / "tb2_lite_eval/requested_models_20260514"
LOG_DIR = DATA_ROOT / "tb2_lite_eval/priority_queue_20260514/logs"
STATE_PATH = LOG_DIR / "qwen35_large_eval_queue_state.json"
STATUS_PATH = LOG_DIR / "qwen35_large_eval_queue_status.md"
STRICT_ENV = ROOT_DIR / ".vllm-eval-cu129-strict"
STRICT_PYTHON = STRICT_ENV / "bin/python"
STRICT_SITE = STRICT_ENV / "lib/python3.12/site-packages"
EVAL_SCRIPT = ROOT_DIR / "tb2_lite/scripts/replay_eval.py"
EVAL_PATH = ROOT_DIR / "tb2_lite/data/replay_full.jsonl"
HF_BIN = Path("/home/work/.local/bin/hf")


@dataclass(frozen=True)
class QwenJob:
    name: str
    model: str
    model_short: str
    log_name: str
    priority: int
    fallback_of: str = ""
    quantization: str = ""
    max_model_len: int = 32768
    max_tokens: int = 1024
    gpu_memory_utilization: float = 0.94
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 8
    max_cudagraph_capture_size: int | None = 16
    data_parallel_size: int = 1
    tensor_parallel_size: int = 8
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    disable_chunked_prefill: bool = False
    disable_async_scheduling: bool = False

    @property
    def output_path(self) -> Path:
        return OUT_DIR / f"{self.model_short}.json"

    @property
    def log_path(self) -> Path:
        return LOG_DIR / self.log_name


JOBS = [
    QwenJob(
        name="qwen35_122b_a10b_fp8",
        model="Qwen/Qwen3.5-122B-A10B-FP8",
        model_short="qwen35_122b_a10b_fp8",
        log_name="qwen35_122b_a10b_fp8_vllm_dp8_ep.log",
        priority=10,
        gpu_memory_utilization=0.92,
    ),
    QwenJob(
        name="qwen35_122b_a10b_fp8_safe",
        model="Qwen/Qwen3.5-122B-A10B-FP8",
        model_short="qwen35_122b_a10b_fp8_safe",
        log_name="qwen35_122b_a10b_fp8_safe_vllm_tp8_ep.log",
        priority=10,
        fallback_of="qwen35_122b_a10b_fp8",
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=4096,
        max_num_seqs=2,
        max_cudagraph_capture_size=8,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_async_scheduling=True,
    ),
    QwenJob(
        name="qwen35_122b_a10b_gptq_int4",
        model="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        model_short="qwen35_122b_a10b_gptq_int4",
        log_name="qwen35_122b_a10b_gptq_int4_vllm_dp8_ep.log",
        priority=11,
        fallback_of="qwen35_122b_a10b_fp8",
        quantization="moe_wna16",
        gpu_memory_utilization=0.92,
    ),
    QwenJob(
        name="qwen35_122b_a10b_bf16",
        model="Qwen/Qwen3.5-122B-A10B",
        model_short="qwen35_122b_a10b_bf16",
        log_name="qwen35_122b_a10b_bf16_vllm_tp8_ep.log",
        priority=12,
        fallback_of="qwen35_122b_a10b_gptq_int4",
        data_parallel_size=1,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    ),
    QwenJob(
        name="qwen35_397b_a17b_fp8",
        model="Qwen/Qwen3.5-397B-A17B-FP8",
        model_short="qwen35_397b_a17b_fp8",
        log_name="qwen35_397b_a17b_fp8_vllm_dp8_ep.log",
        priority=20,
        gpu_memory_utilization=0.95,
    ),
    QwenJob(
        name="qwen35_397b_a17b_fp8_safe",
        model="Qwen/Qwen3.5-397B-A17B-FP8",
        model_short="qwen35_397b_a17b_fp8_safe",
        log_name="qwen35_397b_a17b_fp8_safe_vllm_tp8_ep.log",
        priority=20,
        fallback_of="qwen35_397b_a17b_fp8",
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=4096,
        max_num_seqs=2,
        max_cudagraph_capture_size=8,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_async_scheduling=True,
    ),
    QwenJob(
        name="qwen35_397b_a17b_gptq_int4",
        model="Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
        model_short="qwen35_397b_a17b_gptq_int4",
        log_name="qwen35_397b_a17b_gptq_int4_vllm_dp8_ep.log",
        priority=21,
        fallback_of="qwen35_397b_a17b_fp8",
        quantization="moe_wna16",
        gpu_memory_utilization=0.95,
    ),
    QwenJob(
        name="qwen35_397b_a17b_bf16",
        model="Qwen/Qwen3.5-397B-A17B",
        model_short="qwen35_397b_a17b_bf16",
        log_name="qwen35_397b_a17b_bf16_vllm_tp8_ep.log",
        priority=22,
        fallback_of="qwen35_397b_a17b_gptq_int4",
        data_parallel_size=1,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    ),
]


FATAL_PATTERNS = (
    "EngineDeadError",
    "EngineCore encountered a fatal error",
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "RuntimeError: cancelled",
    "ValueError:",
    "ImportError:",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_text(cmd: list[str], timeout: int = 30) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def save_state(state: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_group(pid: int) -> None:
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return
    time.sleep(10)
    if pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


def strict_ld_library_path() -> str:
    libs = sorted(str(path) for path in STRICT_SITE.glob("nvidia/**/lib") if path.is_dir())
    base = ":".join(libs + ["/usr/local/cuda-12.9/lib64"])
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{base}:{existing}" if existing else base


def model_cached(model: str) -> bool:
    code = (
        "from huggingface_hub import snapshot_download\n"
        "import sys\n"
        "snapshot_download(sys.argv[1], local_files_only=True)\n"
    )
    env = os.environ.copy()
    env.pop("PYTHONNOUSERSITE", None)
    env.pop("PYTHONPATH", None)
    env["HF_HOME"] = str(HF_HOME)
    env["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    result = subprocess.run(
        [str(STRICT_PYTHON), "-c", code, model],
        cwd=ROOT_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def start_download(model: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe = model.replace("/", "__").replace(":", "_")
    log_path = LOG_DIR / f"{safe}_download.log"
    env = os.environ.copy()
    env.pop("PYTHONNOUSERSITE", None)
    env.pop("PYTHONPATH", None)
    env["HF_HOME"] = str(HF_HOME)
    env["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    code = (
        "from huggingface_hub import snapshot_download\n"
        "import sys\n"
        "path = snapshot_download(sys.argv[1], cache_dir=sys.argv[2], resume_download=True)\n"
        "print(path, flush=True)\n"
    )
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            [str(STRICT_PYTHON), "-u", "-c", code, model, str(HF_HOME / "hub")],
            cwd=ROOT_DIR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def download_running(model: str) -> int | None:
    output = run_text(["pgrep", "-af", model], timeout=10)
    for line in output.splitlines():
        parts = line.split(maxsplit=1)
        if parts and parts[0].isdigit() and model in line:
            return int(parts[0])
    return None


def running_eval_job() -> tuple[QwenJob, int] | None:
    output = run_text(["pgrep", "-af", "tb2_lite/scripts/replay_eval.py"], timeout=10)
    for line in output.splitlines():
        if "Qwen/Qwen3.5" not in line:
            continue
        parts = line.split(maxsplit=1)
        if not parts or not parts[0].isdigit():
            continue
        for job in JOBS:
            if f"--model-short {job.model_short}" in line:
                return job, int(parts[0])
    return None


def gpu_snapshot() -> dict[str, Any]:
    raw = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    rows = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        rows.append(
            {
                "gpu": int(parts[0]),
                "mem_used_mib": int(parts[1]) if parts[1].isdigit() else None,
                "mem_total_mib": int(parts[2]) if parts[2].isdigit() else None,
                "util_pct": int(parts[3]) if parts[3].isdigit() else None,
                "pstate": parts[4],
            }
        )
    return {"raw": raw.strip(), "rows": rows}


def gpus_free(threshold_mib: int = 2_000) -> bool:
    rows = gpu_snapshot()["rows"]
    return bool(rows) and all((row["mem_used_mib"] or 0) < threshold_mib for row in rows)


def read_tail(path: Path, max_bytes: int = 400_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        return handle.read().decode("utf-8", errors="replace")


def fatal_marker(path: Path) -> str:
    text = read_tail(path)
    for pattern in FATAL_PATTERNS:
        if pattern in text:
            return pattern
    return ""


def score_from_result(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return round(100.0 * float(data["aggregate"]["avg_command_f1"]), 2)
    except Exception:
        return None


def progress_signature(path: Path) -> str:
    text = read_tail(path)
    matches = re.findall(r"Processed prompts:.*?\|\s*(\d+)/(\d+)", text)
    if matches:
        done, total = matches[-1]
        return f"processed:{done}/{total}"
    if "GPU KV cache size:" in text:
        return "kv_cache_ready"
    if "Graph capturing finished" in text:
        return "graph_captured"
    if "torch.compile took" in text:
        return "torch_compile_done"
    if "Loading weights took" in text:
        return "weights_loaded"
    matches = re.findall(r"Loading safetensors checkpoint shards:\s+\d+% Completed \| (\d+)/(\d+)", text)
    if matches:
        done, total = matches[-1]
        return f"loading:{done}/{total}"
    return "starting"


def start_eval(job: QwenJob) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HOME"] = str(HF_HOME)
    env["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    env["TRANSFORMERS_CACHE"] = str(HF_HOME / "hub")
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["CUDA_PATH"] = "/usr/local/cuda-12.9"
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    env["LD_LIBRARY_PATH"] = strict_ld_library_path()
    cmd = [
        str(STRICT_PYTHON),
        "-u",
        str(EVAL_SCRIPT),
        "--model",
        job.model,
        "--tokenizer-path",
        job.model,
        "--model-short",
        job.model_short,
        "--gpu",
        "0,1,2,3,4,5,6,7",
        "--tp",
        str(job.tensor_parallel_size),
        "--data-parallel-size",
        str(job.data_parallel_size),
        "--eval-path",
        str(EVAL_PATH),
        "--output-dir",
        str(OUT_DIR),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(job.max_model_len),
        "--max-tokens",
        str(job.max_tokens),
        "--gpu-memory-utilization",
        str(job.gpu_memory_utilization),
        "--thinking-mode",
        "off",
        "--strip-thinking-history",
        "on",
        "--gemma4-empty-thought-channel",
        "auto",
        "--language-model-only",
        "--keep-chunked-mm-input",
        "--enable-expert-parallel",
        "--reasoning-parser",
        "qwen3",
        "--max-num-batched-tokens",
        str(job.max_num_batched_tokens),
        "--max-num-seqs",
        str(job.max_num_seqs),
        "--allow-raw-fallback",
        "--skip-if-exists",
    ]
    if job.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if job.disable_chunked_prefill:
        cmd.append("--disable-chunked-prefill")
    if job.disable_async_scheduling:
        cmd.append("--disable-async-scheduling")
    if job.max_cudagraph_capture_size is not None:
        cmd.extend(["--max-cudagraph-capture-size", str(job.max_cudagraph_capture_size)])
    if job.quantization:
        cmd.extend(["--quantization", job.quantization])
    if job.enforce_eager:
        cmd.append("--enforce-eager")
    if job.tensor_parallel_size > 1:
        cmd.append("--disable-custom-all-reduce")
    with job.log_path.open("ab") as log:
        log.write((utc_now() + " COMMAND " + " ".join(cmd) + "\n").encode())
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def write_status(state: dict[str, Any], note: str) -> None:
    gpu = gpu_snapshot()
    lines = [
        "# Qwen3.5 Large Eval Queue Status",
        "",
        f"- updated_at: `{utc_now()}`",
        f"- note: `{note}`",
        f"- active_job: `{state.get('active_job')}`",
        f"- active_pid: `{state.get('active_pid')}`",
        f"- phase: `{state.get('phase')}`",
        f"- completed: `{state.get('completed', [])}`",
        f"- failed: `{state.get('failed', [])}`",
        "",
        "## Downloads",
        "",
    ]
    for model, info in state.get("downloads", {}).items():
        lines.append(f"- `{model}`: `{info}`")
    lines.extend(["", "## GPU", ""])
    if gpu["rows"]:
        for row in gpu["rows"]:
            lines.append(
                f"- GPU {row['gpu']}: `{row['mem_used_mib']}/{row['mem_total_mib']} MiB`, "
                f"util `{row['util_pct']}%`, `{row['pstate']}`"
            )
    else:
        lines.append(f"- raw: `{gpu['raw']}`")
    lines.extend(["", "## Results", ""])
    for job in JOBS:
        if job.output_path.exists():
            lines.append(f"- `{job.name}`: `{score_from_result(job.output_path)}` -> `{job.output_path}`")
    active = next((job for job in JOBS if job.name == state.get("active_job")), None)
    if active:
        lines.extend(["", "## Active Log Tail", "", "```text", "\n".join(read_tail(active.log_path).splitlines()[-25:]), "```"])
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-seconds", type=int, default=120)
    parser.add_argument("--stall-seconds", type=int, default=1800)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "phase": "starting",
        "active_job": "",
        "active_pid": None,
        "completed": [],
        "failed": [],
        "downloads": {},
        "last_progress_signature": "",
        "last_progress_at": time.time(),
        "started_at": utc_now(),
    }

    primary_order = ["qwen35_122b_a10b_fp8", "qwen35_397b_a17b_fp8"]
    fallback_chain = {
        "qwen35_122b_a10b_fp8": [
            "qwen35_122b_a10b_fp8_safe",
            "qwen35_122b_a10b_gptq_int4",
            "qwen35_122b_a10b_bf16",
        ],
        "qwen35_397b_a17b_fp8": [
            "qwen35_397b_a17b_fp8_safe",
            "qwen35_397b_a17b_gptq_int4",
            "qwen35_397b_a17b_bf16",
        ],
    }
    jobs_by_name = {job.name: job for job in JOBS}
    pending_primary = list(primary_order)
    active_job: QwenJob | None = None

    while True:
        if active_job is None and not state.get("active_job"):
            running = running_eval_job()
            if running is not None:
                active_job, pid = running
                state["active_job"] = active_job.name
                state["active_pid"] = pid
                state["phase"] = "running_eval"
                state["last_progress_signature"] = progress_signature(active_job.log_path)
                state["last_progress_at"] = time.time()

        for model in ["Qwen/Qwen3.5-122B-A10B-FP8", "Qwen/Qwen3.5-397B-A17B-FP8"]:
            if model_cached(model):
                state["downloads"][model] = {"status": "cached"}
            elif pid := download_running(model):
                state["downloads"][model] = {"status": "downloading", "pid": pid}
            elif state["downloads"].get(model, {}).get("status") != "downloading" or not pid_alive(
                state["downloads"].get(model, {}).get("pid")
            ):
                if model.endswith("122B-A10B-FP8") or state["downloads"].get("Qwen/Qwen3.5-122B-A10B-FP8", {}).get("status") == "cached":
                    state["downloads"][model] = {"status": "downloading", "pid": start_download(model)}

        if active_job and pid_alive(state.get("active_pid")):
            signature = progress_signature(active_job.log_path)
            if signature != state.get("last_progress_signature"):
                state["last_progress_signature"] = signature
                state["last_progress_at"] = time.time()
            elif time.time() - float(state.get("last_progress_at", time.time())) > args.stall_seconds:
                kill_group(int(state["active_pid"]))
                state["failed"].append({"job": active_job.name, "reason": f"progress_stalled:{signature}"})
                failed_name = active_job.name
                state["phase"] = "failed_job"
                state["active_job"] = ""
                state["active_pid"] = None
                active_job = None
                for primary, fallbacks in fallback_chain.items():
                    if failed_name == primary or failed_name in fallbacks:
                        next_jobs = [
                            name
                            for name in fallbacks
                            if name not in state["completed"]
                            and all(item.get("job") != name for item in state["failed"])
                        ]
                        if next_jobs:
                            pending_primary.insert(0, next_jobs[0])
                        break

        if active_job and active_job.output_path.exists():
            state["completed"].append(active_job.name)
            state["phase"] = "completed_job"
            state["active_job"] = ""
            state["active_pid"] = None
            active_job = None
        elif active_job and not pid_alive(state.get("active_pid")):
            marker = fatal_marker(active_job.log_path) or "process_exited_without_result"
            state["failed"].append({"job": active_job.name, "reason": marker})
            state["phase"] = "failed_job"
            state["active_job"] = ""
            state["active_pid"] = None
            failed_name = active_job.name
            active_job = None
            for primary, fallbacks in fallback_chain.items():
                if failed_name == primary or failed_name in fallbacks:
                    next_jobs = [name for name in fallbacks if name not in state["completed"] and all(item.get("job") != name for item in state["failed"])]
                    if next_jobs:
                        pending_primary.insert(0, next_jobs[0])
                    break

        if not active_job and pending_primary:
            next_name = pending_primary[0]
            job = jobs_by_name[next_name]
            if job.output_path.exists():
                state["completed"].append(job.name)
                pending_primary.pop(0)
            elif model_cached(job.model) and gpus_free():
                pending_primary.pop(0)
                active_job = job
                state["active_job"] = job.name
                state["active_pid"] = start_eval(job)
                state["phase"] = "running_eval"
            else:
                state["phase"] = "waiting_download_or_gpu"

        if not active_job and not pending_primary:
            state["phase"] = "done"
            write_status(state, "done")
            return

        write_status(state, "loop")
        time.sleep(args.check_seconds)


if __name__ == "__main__":
    main()
