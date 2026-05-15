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


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/work/.data")
LOG_DIR = DATA_ROOT / "tb2_lite_eval/priority_queue_20260514/logs"
OUT_DIR = DATA_ROOT / "tb2_lite_eval/requested_models_20260514"
STATUS_PATH = LOG_DIR / "deepseek_v4_flash_fp8_watchdog_status.md"
STATE_PATH = LOG_DIR / "deepseek_v4_flash_fp8_watchdog_state.json"
STRICT_ENV = REPO_ROOT / ".vllm-eval-cu129-strict"
STRICT_PYTHON = STRICT_ENV / "bin/python"
STRICT_SITE = STRICT_ENV / "lib/python3.12/site-packages"
EVAL_SCRIPT = REPO_ROOT / "tb2_lite/scripts/replay_eval.py"
EVAL_DATA = REPO_ROOT / "tb2_lite/data/replay_full.jsonl"


@dataclass(frozen=True)
class RunConfig:
    name: str
    model_short: str
    log_name: str
    max_model_len: int = 32768
    max_tokens: int = 1024
    gpu_memory_utilization: float = 0.94
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 8
    enforce_eager: bool = False

    @property
    def log_path(self) -> Path:
        return LOG_DIR / self.log_name

    @property
    def output_path(self) -> Path:
        return OUT_DIR / f"{self.model_short}.json"


RUN_CONFIGS = [
    RunConfig(
        name="nomtp_ctx32768_seq8",
        model_short="deepseek_v4_flash_fp8_nomtp",
        log_name="deepseek_v4_flash_fp8_vllm_strict_cu129_tp8_nomtp.log",
    ),
    RunConfig(
        name="nomtp_ctx32768_seq4",
        model_short="deepseek_v4_flash_fp8_nomtp_seq4",
        log_name="deepseek_v4_flash_fp8_vllm_strict_cu129_tp8_nomtp_seq4.log",
        gpu_memory_utilization=0.92,
        max_num_batched_tokens=8192,
        max_num_seqs=4,
    ),
    RunConfig(
        name="nomtp_ctx32768_seq2_eager",
        model_short="deepseek_v4_flash_fp8_nomtp_seq2_eager",
        log_name="deepseek_v4_flash_fp8_vllm_strict_cu129_tp8_nomtp_seq2_eager.log",
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=4096,
        max_num_seqs=2,
        enforce_eager=True,
    ),
    RunConfig(
        name="nomtp_ctx32768_seq1_eager",
        model_short="deepseek_v4_flash_fp8_nomtp_seq1_eager",
        log_name="deepseek_v4_flash_fp8_vllm_strict_cu129_tp8_nomtp_seq1_eager.log",
        gpu_memory_utilization=0.88,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        enforce_eager=True,
    ),
]


FATAL_PATTERNS = (
    "EngineDeadError",
    "EngineCore encountered a fatal error",
    "Traceback (most recent call last)",
    "RuntimeError: cancelled",
    "CUDA out of memory",
    "Child process unexpectedly failed",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def run_text(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


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


def find_running_pid() -> int | None:
    output = run_text(["pgrep", "-af", "tb2_lite/scripts/replay_eval.py"], timeout=10)
    for line in output.splitlines():
        if "sgl-project/DeepSeek-V4-Flash-FP8" not in line:
            continue
        if "--model-short" not in line:
            continue
        parts = line.split(maxsplit=1)
        if parts and parts[0].isdigit():
            return int(parts[0])
    return None


def kill_process_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
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


def start_run(config: RunConfig) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HOME"] = str(DATA_ROOT / "huggingface")
    env["TRANSFORMERS_CACHE"] = str(DATA_ROOT / "huggingface/hub")
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["CUDA_PATH"] = "/usr/local/cuda-12.9"
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    env["LD_LIBRARY_PATH"] = strict_ld_library_path()
    cmd = [
        str(STRICT_PYTHON),
        "-u",
        str(EVAL_SCRIPT),
        "--model",
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "--tokenizer-path",
        "deepseek-ai/DeepSeek-V4-Flash",
        "--model-short",
        config.model_short,
        "--gpu",
        "0,1,2,3,4,5,6,7",
        "--tp",
        "8",
        "--eval-path",
        str(EVAL_DATA),
        "--output-dir",
        str(OUT_DIR),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(config.max_model_len),
        "--max-tokens",
        str(config.max_tokens),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--thinking-mode",
        "off",
        "--strip-thinking-history",
        "on",
        "--gemma4-empty-thought-channel",
        "auto",
        "--language-model-only",
        "--disable-custom-all-reduce",
        "--tokenizer-mode",
        "deepseek_v4",
        "--kv-cache-dtype",
        "fp8",
        "--block-size",
        "256",
        "--max-num-batched-tokens",
        str(config.max_num_batched_tokens),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--enable-expert-parallel",
        "--hf-overrides-json",
        '{"expert_dtype":"fp8"}',
        "--allow-raw-fallback",
    ]
    if config.enforce_eager:
        cmd.append("--enforce-eager")
    with config.log_path.open("ab") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
    return proc.pid


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
        cols = [part.strip() for part in line.split(",")]
        if len(cols) != 5 or not cols[0].isdigit():
            continue
        rows.append(
            {
                "gpu": int(cols[0]),
                "mem_used_mib": int(cols[1]) if cols[1].isdigit() else None,
                "mem_total_mib": int(cols[2]) if cols[2].isdigit() else None,
                "util_pct": int(cols[3]) if cols[3].isdigit() else None,
                "pstate": cols[4],
            }
        )
    return {"raw": raw.strip(), "rows": rows}


def read_tail(path: Path, max_bytes: int = 2_000_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        return handle.read().decode("utf-8", errors="replace")


def detect_fatal(log_text: str) -> str:
    for pattern in FATAL_PATTERNS:
        if pattern in log_text:
            return pattern
    return ""


def parse_progress(log_text: str) -> dict[str, Any]:
    matches = re.findall(r"Processed prompts:.*?\|\s*(\d+)/(\d+)\s*\[", log_text)
    if matches:
        done, total = matches[-1]
        return {"done": int(done), "total": int(total)}
    matches = re.findall(r"Processed prompts:.*?\|\s*(\d+)/(\d+)", log_text)
    if matches:
        done, total = matches[-1]
        return {"done": int(done), "total": int(total)}
    if "DeepGEMM warmup:" in log_text:
        warmups = re.findall(r"DeepGEMM warmup:\s+(\d+)%", log_text)
        return {"warmup_pct": int(warmups[-1]) if warmups else None}
    if "Loading safetensors checkpoint shards:" in log_text:
        shards = re.findall(r"Loading safetensors checkpoint shards:\s+(\d+)%", log_text)
        return {"loading_pct": int(shards[-1]) if shards else None}
    return {}


def result_status() -> dict[str, Any] | None:
    for config in RUN_CONFIGS:
        if not config.output_path.exists():
            continue
        try:
            data = json.loads(config.output_path.read_text())
            score = data.get("aggregate", {}).get("next_action_score")
            return {"config": config.name, "path": str(config.output_path), "score": score}
        except Exception as exc:
            return {"config": config.name, "path": str(config.output_path), "error": str(exc)}
    return None


def write_status(state: dict[str, Any], config: RunConfig, gpu: dict[str, Any], note: str) -> None:
    result = result_status()
    log_text = read_tail(config.log_path, max_bytes=300_000)
    progress = parse_progress(log_text)
    fatal = detect_fatal(log_text)
    lines = [
        "# DeepSeek V4 Flash FP8 Watchdog Status",
        "",
        f"- updated_at: `{utc_now()}`",
        f"- state: `{state.get('status', 'running')}`",
        f"- note: `{note}`",
        f"- current_config: `{config.name}`",
        f"- current_pid: `{state.get('pid')}`",
        f"- restart_count: `{state.get('restart_count', 0)}`",
        f"- progress: `{progress}`",
        f"- fatal_marker: `{fatal}`",
        f"- output: `{result}`",
        f"- log: `{config.log_path}`",
        "",
        "## GPU",
        "",
    ]
    if gpu["rows"]:
        for row in gpu["rows"]:
            lines.append(
                f"- GPU {row['gpu']}: `{row['mem_used_mib']}/{row['mem_total_mib']} MiB`, "
                f"util `{row['util_pct']}%`, `{row['pstate']}`"
            )
    else:
        lines.append(f"- raw: `{gpu['raw']}`")
    lines.extend(["", "## Last Log", "", "```text", "\n".join(log_text.splitlines()[-25:]), "```", ""])
    STATUS_PATH.write_text("\n".join(lines), encoding="utf-8")
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def config_index_for_pid(default_idx: int) -> int:
    output = run_text(["pgrep", "-af", "tb2_lite/scripts/replay_eval.py"], timeout=10)
    for line in output.splitlines():
        for idx, config in enumerate(RUN_CONFIGS):
            if config.model_short in line:
                return idx
    return default_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-pid", type=int, default=0)
    parser.add_argument("--check-seconds", type=int, default=120)
    parser.add_argument("--report-seconds", type=int, default=7200)
    parser.add_argument("--stall-seconds", type=int, default=3600)
    args = parser.parse_args()

    state: dict[str, Any] = {
        "status": "starting",
        "pid": args.current_pid or find_running_pid(),
        "config_idx": 0,
        "restart_count": 0,
        "last_progress": {},
        "last_progress_at": time.time(),
        "last_report_at": 0,
        "started_at": utc_now(),
    }
    state["config_idx"] = config_index_for_pid(0)
    if not pid_alive(state["pid"]):
        state["pid"] = start_run(RUN_CONFIGS[state["config_idx"]])
        state["restart_count"] += 1

    while True:
        config = RUN_CONFIGS[int(state["config_idx"])]
        gpu = gpu_snapshot()
        result = result_status()
        if result and not result.get("error"):
            state["status"] = "completed"
            write_status(state, config, gpu, "completed")
            return

        log_text = read_tail(config.log_path)
        fatal = detect_fatal(log_text)
        pid = int(state.get("pid") or 0)
        alive = pid_alive(pid)
        progress = parse_progress(log_text)
        now = time.time()

        if progress and progress != state.get("last_progress"):
            state["last_progress"] = progress
            state["last_progress_at"] = now

        low_gpu_after_start = False
        if alive and ("Processed prompts:" in log_text or "GPU KV cache size:" in log_text):
            used = [row["mem_used_mib"] or 0 for row in gpu["rows"]]
            low_gpu_after_start = bool(used) and max(used) < 20_000

        stale = alive and now - float(state.get("last_progress_at", now)) > args.stall_seconds

        if fatal or not alive or low_gpu_after_start or stale:
            reason = fatal or ("process_dead" if not alive else "gpu_memory_dropped" if low_gpu_after_start else "progress_stalled")
            if alive:
                kill_process_group(pid)
            next_idx = int(state["config_idx"]) + 1
            if next_idx >= len(RUN_CONFIGS):
                state["status"] = "failed_no_more_fallbacks"
                write_status(state, config, gpu, reason)
                return
            state["config_idx"] = next_idx
            config = RUN_CONFIGS[next_idx]
            state["pid"] = start_run(config)
            state["restart_count"] = int(state.get("restart_count", 0)) + 1
            state["last_progress"] = {}
            state["last_progress_at"] = time.time()
            state["status"] = "restarted"
            write_status(state, config, gpu, f"restart_after_{reason}")
        elif now - float(state.get("last_report_at", 0)) >= args.report_seconds:
            state["status"] = "running"
            state["last_report_at"] = now
            write_status(state, config, gpu, "periodic_report")
        else:
            state["status"] = "running"
            write_status(state, config, gpu, "check")

        time.sleep(args.check_seconds)


if __name__ == "__main__":
    main()
