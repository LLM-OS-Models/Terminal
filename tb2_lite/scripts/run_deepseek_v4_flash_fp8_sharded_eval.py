#!/usr/bin/env python3
from __future__ import annotations

import json
import os
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
STRICT_ENV = REPO_ROOT / ".vllm-eval-cu129-strict"
STRICT_PYTHON = STRICT_ENV / "bin/python"
STRICT_SITE = STRICT_ENV / "lib/python3.12/site-packages"
EVAL_SCRIPT = REPO_ROOT / "tb2_lite/scripts/replay_eval.py"
EVAL_DATA = REPO_ROOT / "tb2_lite/data/replay_full.jsonl"
FINAL_SHORT = "deepseek_v4_flash_fp8_deepgemm_tp4x2_t256_stopbrace"
STATUS_PATH = LOG_DIR / f"{FINAL_SHORT}_status.md"


@dataclass(frozen=True)
class ShardConfig:
    index: int
    gpus: str

    @property
    def model_short(self) -> str:
        return f"{FINAL_SHORT}_shard{self.index}"

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"{self.model_short}.log"

    @property
    def output_path(self) -> Path:
        return OUT_DIR / f"{self.model_short}.json"


SHARDS = [
    ShardConfig(index=0, gpus="0,1,2,3"),
    ShardConfig(index=1, gpus="4,5,6,7"),
]


FATAL_PATTERNS = (
    "EngineDeadError",
    "EngineCore encountered a fatal error",
    "Traceback (most recent call last)",
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


def strict_ld_library_path() -> str:
    libs = sorted(str(path) for path in STRICT_SITE.glob("nvidia/**/lib") if path.is_dir())
    base = ":".join(libs + ["/usr/local/cuda-12.9/lib64"])
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return f"{base}:{existing}" if existing else base


def base_env(gpus: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["HF_HOME"] = str(DATA_ROOT / "huggingface")
    env["TRANSFORMERS_CACHE"] = str(DATA_ROOT / "huggingface/hub")
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["CUDA_PATH"] = "/usr/local/cuda-12.9"
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["LD_LIBRARY_PATH"] = strict_ld_library_path()
    return env


def cleanup_stale() -> None:
    output = run_text(["ps", "-eo", "pid,pgid,cmd"], timeout=20)
    targets: set[tuple[int, int]] = set()
    for line in output.splitlines()[1:]:
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_s, pgid_s, cmd = parts
        if not pid_s.isdigit() or not pgid_s.isdigit():
            continue
        pid = int(pid_s)
        pgid = int(pgid_s)
        if "DeepSeek-V4-Flash-FP8" in cmd or "VLLM::" in cmd:
            targets.add((pid, pgid))
    for pid, pgid in targets:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    if targets:
        time.sleep(5)
    for pid, pgid in targets:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass


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


def read_tail(path: Path, max_bytes: int = 400_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        return handle.read().decode("utf-8", errors="replace")


def active_log(path: Path) -> str:
    text = read_tail(path)
    pos = text.rfind("[sharded-runner] START ")
    return text[pos:] if pos >= 0 else text


def fatal_marker(text: str) -> str:
    for pattern in FATAL_PATTERNS:
        if pattern in text:
            return pattern
    return ""


def start_shard(shard: ShardConfig) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(STRICT_PYTHON),
        "-u",
        str(EVAL_SCRIPT),
        "--model",
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "--tokenizer-path",
        "deepseek-ai/DeepSeek-V4-Flash",
        "--model-short",
        shard.model_short,
        "--gpu",
        shard.gpus,
        "--tp",
        "4",
        "--eval-path",
        str(EVAL_DATA),
        "--output-dir",
        str(OUT_DIR),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--max-tokens",
        "256",
        "--gpu-memory-utilization",
        "0.90",
        "--temperature",
        "0.0",
        "--top-p",
        "1.0",
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
        "16384",
        "--max-num-seqs",
        "8",
        "--enable-expert-parallel",
        "--hf-overrides-json",
        '{"expert_dtype":"fp8"}',
        "--allow-raw-fallback",
        "--moe-backend",
        "deep_gemm",
        "--reasoning-parser",
        "deepseek_v4",
        "--stop-string",
        "\n}",
        "--include-stop-str-in-output",
        "--disable-async-scheduling",
        "--shard-index",
        str(shard.index),
        "--shard-count",
        str(len(SHARDS)),
    ]
    with shard.log_path.open("ab") as log:
        log.write(f"\n\n[sharded-runner] START {utc_now()} shard={shard.index} gpus={shard.gpus}\n".encode())
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=base_env(shard.gpus),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
    return proc.pid


def gpu_snapshot() -> str:
    return run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    ).strip()


def merge_outputs() -> Path:
    from replay_metrics import aggregate_scores

    loaded = [json.loads(shard.output_path.read_text()) for shard in SHARDS]
    per_step = []
    for item in loaded:
        per_step.extend(item["per_step"])
    per_step.sort(key=lambda row: (row["task_id"], row["sample_idx"], row["step_idx"]))
    aggregate = aggregate_scores(per_step)
    result = {
        "model": "DeepSeek-V4-Flash-FP8",
        "model_path": "sgl-project/DeepSeek-V4-Flash-FP8",
        "model_short": FINAL_SHORT,
        "eval_path": str(EVAL_DATA),
        "timestamp": datetime.utcnow().isoformat(),
        "shards": [
            {
                "index": shard.index,
                "gpus": shard.gpus,
                "path": str(shard.output_path),
            }
            for shard in SHARDS
        ],
        "load_time_sec": max(item.get("load_time_sec", 0) for item in loaded),
        "gen_time_sec": max(item.get("gen_time_sec", 0) for item in loaded),
        "avg_sec_per_step": round(max(item.get("gen_time_sec", 0) for item in loaded) / max(len(per_step), 1), 3),
        "sampling": {
            **loaded[0].get("sampling", {}),
            "sharded_runner": "2x tp4",
            "note": "DeepSeek FP8 did not emit EOS reliably at max_tokens=1024; this run caps max_tokens at 256 and stops on final JSON brace when present.",
        },
        "prompt_template": loaded[0].get("prompt_template", {}),
        "aggregate": aggregate,
        "per_step": per_step,
    }
    out_path = OUT_DIR / f"{FINAL_SHORT}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return out_path


def write_status(pids: dict[int, int], restarts: dict[int, int], note: str) -> None:
    lines = [
        "# DeepSeek V4 Flash FP8 Sharded Eval Status",
        "",
        f"- updated_at: `{utc_now()}`",
        f"- note: `{note}`",
        f"- final_output: `{OUT_DIR / (FINAL_SHORT + '.json')}`",
        "",
        "## Shards",
        "",
    ]
    for shard in SHARDS:
        log_text = active_log(shard.log_path)
        lines.append(f"- shard `{shard.index}` GPUs `{shard.gpus}` pid `{pids.get(shard.index)}` restarts `{restarts.get(shard.index, 0)}`")
        lines.append(f"  - output_exists: `{shard.output_path.exists()}`")
        lines.append(f"  - fatal: `{fatal_marker(log_text)}`")
        lines.append(f"  - log: `{shard.log_path}`")
        tail = "\\n".join(log_text.splitlines()[-8:])
        lines.append("  - tail:")
        lines.append("```text")
        lines.append(tail)
        lines.append("```")
    lines.extend(["", "## GPU", "", "```text", gpu_snapshot(), "```", ""])
    STATUS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cleanup_stale()
    pids = {shard.index: start_shard(shard) for shard in SHARDS}
    restarts = {shard.index: 0 for shard in SHARDS}
    last_progress = {shard.index: shard.log_path.stat().st_size if shard.log_path.exists() else 0 for shard in SHARDS}
    last_progress_at = {shard.index: time.time() for shard in SHARDS}
    write_status(pids, restarts, "started")

    while True:
        all_done = all(shard.output_path.exists() for shard in SHARDS)
        if all_done:
            out_path = merge_outputs()
            write_status(pids, restarts, f"completed {out_path}")
            print(json.dumps({"output_path": str(out_path)}, ensure_ascii=False))
            return

        for shard in SHARDS:
            if shard.output_path.exists():
                continue
            log_text = active_log(shard.log_path)
            fatal = fatal_marker(log_text)
            pid = pids.get(shard.index, 0)
            alive = pid_alive(pid)
            size = shard.log_path.stat().st_size if shard.log_path.exists() else 0
            if size != last_progress.get(shard.index):
                last_progress[shard.index] = size
                last_progress_at[shard.index] = time.time()
            stalled = time.time() - last_progress_at.get(shard.index, time.time()) > 900
            if fatal or not alive or stalled:
                if alive:
                    kill_group(pid)
                if restarts[shard.index] >= 2:
                    write_status(pids, restarts, f"failed shard={shard.index} fatal={fatal} alive={alive} stalled={stalled}")
                    raise RuntimeError(f"shard {shard.index} failed after restarts")
                restarts[shard.index] += 1
                pids[shard.index] = start_shard(shard)
                last_progress[shard.index] = shard.log_path.stat().st_size if shard.log_path.exists() else 0
                last_progress_at[shard.index] = time.time()
                write_status(pids, restarts, f"restarted shard={shard.index}")
        write_status(pids, restarts, "running")
        time.sleep(30)


if __name__ == "__main__":
    main()
