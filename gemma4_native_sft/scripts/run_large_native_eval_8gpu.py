#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/work/.data/gemma4_native_sft")
MODEL_ROOT = DATA_ROOT / "models"
DEFAULT_RESULTS_DIR = Path("/home/work/.data/tb2_lite_eval/gemma4_native_sft_20260509")
DEFAULT_EVAL_PATH = ROOT_DIR / "tb2_lite/data/replay_full.jsonl"
DEFAULT_VLLM_PYTHON = ROOT_DIR / ".vllm-eval-cu129/bin/python"


@dataclass(frozen=True)
class EvalJob:
    gpu: str
    short: str
    repo_id: str
    checkpoint: Path


JOBS = [
    EvalJob(
        gpu="0",
        short="gemma4_26b_a4b_it_native_e1",
        repo_id="LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-1Epoch",
        checkpoint=MODEL_ROOT / "google__gemma-4-26B-A4B-it__terminal_sft_native_liquid_2epoch/checkpoint-1020",
    ),
    EvalJob(
        gpu="1",
        short="gemma4_26b_a4b_it_native_e2",
        repo_id="LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch",
        checkpoint=MODEL_ROOT / "google__gemma-4-26B-A4B-it__terminal_sft_native_liquid_2epoch/checkpoint-2040",
    ),
    EvalJob(
        gpu="2",
        short="gemma4_26b_a4b_base_native_e1",
        repo_id="LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-1Epoch",
        checkpoint=MODEL_ROOT / "google__gemma-4-26B-A4B__terminal_sft_native_liquid_2epoch/checkpoint-510",
    ),
    EvalJob(
        gpu="3",
        short="gemma4_26b_a4b_base_native_e2",
        repo_id="LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch",
        checkpoint=MODEL_ROOT / "google__gemma-4-26B-A4B__terminal_sft_native_liquid_2epoch/checkpoint-1020",
    ),
    EvalJob(
        gpu="4",
        short="gemma4_31b_it_native_e1",
        repo_id="LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-1Epoch",
        checkpoint=MODEL_ROOT / "google__gemma-4-31B-it__terminal_sft_native_liquid_2epoch/checkpoint-510",
    ),
    EvalJob(
        gpu="5",
        short="gemma4_31b_it_native_e2",
        repo_id="LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch",
        checkpoint=MODEL_ROOT / "google__gemma-4-31B-it__terminal_sft_native_liquid_2epoch/checkpoint-1020",
    ),
    EvalJob(
        gpu="6",
        short="gemma4_31b_base_native_e1",
        repo_id="LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-1Epoch",
        checkpoint=MODEL_ROOT / "google__gemma-4-31B__terminal_sft_native_liquid_2epoch/checkpoint-510",
    ),
    EvalJob(
        gpu="7",
        short="gemma4_31b_base_native_e2",
        repo_id="LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch",
        checkpoint=MODEL_ROOT / "google__gemma-4-31B__terminal_sft_native_liquid_2epoch/checkpoint-1020",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str, log_path: Path | None = None) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def save_json(path: Path, data: dict) -> None:
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


def training_processes() -> list[str]:
    lines = []
    for pid, _pgid, cmd in process_rows():
        if pid == os.getpid():
            continue
        if "gemma4_native_sft/scripts/train_native_hf_fsdp.py" in cmd:
            lines.append(f"{pid} {cmd}")
    return lines


def wait_for_training_done(interval: int, log_path: Path) -> None:
    while True:
        lines = training_processes()
        if not lines:
            log("TRAINING_DONE no train_native_hf_fsdp.py processes remain", log_path)
            return
        log(f"TRAINING_WAIT running_processes={len(lines)}", log_path)
        time.sleep(interval)


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


def wait_for_checkpoints(interval: int, settle_seconds: int, log_path: Path) -> None:
    while True:
        pending = []
        for job in JOBS:
            ready, reason = checkpoint_ready(job.checkpoint, settle_seconds)
            if not ready:
                pending.append(f"{job.short}:{reason}")
        if not pending:
            log("CHECKPOINTS_READY all 8 checkpoints are present and settled", log_path)
            return
        log("CHECKPOINT_WAIT " + " ".join(pending), log_path)
        time.sleep(interval)


def gpu_memory_used() -> dict[str, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        cwd=ROOT_DIR,
        text=True,
        capture_output=True,
        check=True,
    )
    memory: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        idx, used = [part.strip() for part in line.split(",", 1)]
        memory[idx] = int(used)
    return memory


def wait_for_gpus_free(max_used_mb: int, interval: int, log_path: Path) -> None:
    while True:
        memory = gpu_memory_used()
        busy = {gpu: memory.get(gpu, 0) for gpu in sorted({job.gpu for job in JOBS}) if memory.get(gpu, 0) > max_used_mb}
        if not busy:
            log(f"GPUS_READY max_used_mb={max_used_mb}", log_path)
            return
        log(f"GPU_WAIT max_used_mb={max_used_mb} busy={busy}", log_path)
        time.sleep(interval)


def terminate_process_group(pid: int, pgid: int, log_path: Path, label: str) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
        log(f"STALE_STOP_TERM label={label} pid={pid} pgid={pgid}", log_path)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        log(f"STALE_STOP_DENIED label={label} pid={pid} pgid={pgid} error={exc}", log_path)


def clean_stale_eval_processes(log_path: Path) -> None:
    patterns = [
        "gemma4_native_sft/scripts/parallel_native_eval_scheduler.py",
        "gemma4_native_sft/scripts/eval_native_checkpoint.sh",
        "tb2_lite/scripts/replay_eval.py",
    ]
    own_pid = os.getpid()
    seen_pgids: set[int] = set()
    for pid, pgid, cmd in process_rows():
        if pid == own_pid:
            continue
        if any(pattern in cmd for pattern in patterns):
            if pgid in seen_pgids:
                continue
            seen_pgids.add(pgid)
            terminate_process_group(pid, pgid, log_path, label=cmd[:120])
    if seen_pgids:
        time.sleep(10)
        for pid, pgid, cmd in process_rows():
            if pgid in seen_pgids:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    log(f"STALE_STOP_KILL pid={pid} pgid={pgid} label={cmd[:120]}", log_path)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    log(f"STALE_KILL_DENIED pid={pid} pgid={pgid} error={exc}", log_path)
    else:
        log("STALE_STOP none", log_path)


def score_from_result(path: Path) -> float | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return round(100.0 * float(data["aggregate"]["avg_command_f1"]), 2)
    except Exception:
        return None


def fallback_plan(job: EvalJob) -> tuple[str, int] | None:
    name = f"{job.short} {job.repo_id}".lower()
    if "31b" in name:
        return "0,1,2,3", 4
    if "26b" in name:
        return "0,1", 2
    return None


def launch_job(
    job: EvalJob,
    args: argparse.Namespace,
    log_dir: Path,
    gpu_arg: str | None = None,
    tp: int = 1,
    log_suffix: str = "1gpu",
) -> subprocess.Popen:
    gpu_arg = gpu_arg or job.gpu
    env = os.environ.copy()
    env.update(
        {
            "VLLM_PYTHON": str(args.vllm_python),
            "MAX_MODEL_LEN": str(args.max_model_len),
            "MAX_TOKENS": str(args.max_tokens),
            "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
            "CPU_OFFLOAD_GB": str(args.cpu_offload_gb),
            "MAX_NUM_SEQS": str(args.max_num_seqs) if args.max_num_seqs is not None else "",
            "MAX_NUM_BATCHED_TOKENS": (
                str(args.max_num_batched_tokens)
                if args.max_num_batched_tokens is not None
                else ""
            ),
            "DISABLE_CUSTOM_ALL_REDUCE": "1" if args.disable_custom_all_reduce else "0",
            "THINKING_MODE": args.thinking_mode,
            "STRIP_THINKING_HISTORY": args.strip_thinking_history,
            "GEMMA4_EMPTY_THOUGHT_CHANNEL": args.gemma4_empty_thought_channel,
            "VLLM_USE_DEEP_GEMM": "0",
            "VLLM_MOE_USE_DEEP_GEMM": "0",
            "VLLM_USE_DEEP_GEMM_E8M0": "0",
            "VLLM_DEEP_GEMM_WARMUP": "skip",
        }
    )
    if args.limit is not None:
        env["LIMIT"] = str(args.limit)
    cmd = [
        "bash",
        "gemma4_native_sft/scripts/eval_native_checkpoint.sh",
        "--model-path",
        str(job.checkpoint),
        "--model-short",
        job.short,
        "--gpu",
        gpu_arg,
        "--tp",
        str(tp),
        "--eval-path",
        str(args.eval_path),
        "--output-dir",
        str(args.output_dir),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.max_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])
    if args.no_skip_if_exists:
        cmd.append("--no-skip-if-exists")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.disable_custom_all_reduce:
        cmd.append("--disable-custom-all-reduce")
    cmd.extend(["--thinking-mode", args.thinking_mode])
    cmd.extend(["--strip-thinking-history", args.strip_thinking_history])
    cmd.extend(["--gemma4-empty-thought-channel", args.gemma4_empty_thought_channel])
    log_path = log_dir / f"{job.short}_{log_suffix}_eval.log"
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
    return proc


def build_state(records: dict[str, dict], args: argparse.Namespace) -> dict:
    return {
        "updated_at": utc_now(),
        "output_dir": str(args.output_dir),
        "eval_path": str(args.eval_path),
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "jobs": records,
    }


def run(args: argparse.Namespace) -> int:
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log = log_dir / "large_native_eval_8gpu.log"
    state_path = log_dir / "large_native_eval_8gpu_state.json"

    log("LARGE_NATIVE_EVAL_8GPU_START", main_log)
    if args.wait_training:
        wait_for_training_done(args.poll_seconds, main_log)
    if args.wait_checkpoints:
        wait_for_checkpoints(args.poll_seconds, args.settle_seconds, main_log)
    if args.clean_stale:
        clean_stale_eval_processes(main_log)
    wait_for_gpus_free(args.max_gpu_used_mb_before_launch, args.poll_seconds, main_log)

    records: dict[str, dict] = {}
    processes: dict[str, subprocess.Popen] = {}
    for job in JOBS:
        result_path = args.output_dir / f"{job.short}.json"
        if result_path.exists() and not args.no_skip_if_exists:
            score = score_from_result(result_path)
            records[job.short] = {
                **asdict(job),
                "checkpoint": str(job.checkpoint),
                "status": "skipped_existing",
                "result": str(result_path),
                "score": score,
            }
            log(f"EVAL_SKIP_EXISTING short={job.short} score={score}", main_log)
            continue
        proc = launch_job(job, args, log_dir)
        processes[job.short] = proc
        records[job.short] = {
            **asdict(job),
            "checkpoint": str(job.checkpoint),
            "status": "running",
            "pid": proc.pid,
            "started_at": utc_now(),
            "log_path": str(log_dir / f"{job.short}_1gpu_eval.log"),
            "attempts": [
                {
                    "kind": "1gpu",
                    "gpu": job.gpu,
                    "tp": 1,
                    "pid": proc.pid,
                    "started_at": utc_now(),
                    "log_path": str(log_dir / f"{job.short}_1gpu_eval.log"),
                }
            ],
        }
        log(f"EVAL_START short={job.short} gpu={job.gpu} pid={proc.pid}", main_log)
        save_json(state_path, build_state(records, args))
        if args.launch_stagger_seconds > 0:
            time.sleep(args.launch_stagger_seconds)

    if args.launch_only:
        save_json(state_path, build_state(records, args))
        log("LARGE_NATIVE_EVAL_8GPU_LAUNCH_ONLY", main_log)
        return 0

    while processes:
        for short, proc in list(processes.items()):
            rc = proc.poll()
            if rc is None:
                continue
            handle = getattr(proc, "_log_handle", None)
            if handle is not None:
                handle.close()
            result_path = args.output_dir / f"{short}.json"
            score = score_from_result(result_path) if result_path.exists() else None
            records[short].update(
                {
                    "status": "completed" if rc == 0 and result_path.exists() else "failed",
                    "returncode": rc,
                    "finished_at": utc_now(),
                    "result": str(result_path) if result_path.exists() else None,
                    "score": score,
                }
            )
            if records[short].get("attempts"):
                records[short]["attempts"][-1].update(
                    {
                        "returncode": rc,
                        "finished_at": utc_now(),
                        "result": str(result_path) if result_path.exists() else None,
                        "score": score,
                    }
                )
            log(f"EVAL_FINISH short={short} returncode={rc} result_exists={result_path.exists()} score={score}", main_log)
            processes.pop(short)
            save_json(state_path, build_state(records, args))
        if processes:
            log(f"EVAL_MONITOR running={sorted(processes)}", main_log)
            time.sleep(args.monitor_seconds)

    if args.retry_failed_with_tp:
        for job in JOBS:
            record = records.get(job.short, {})
            if record.get("status") != "failed":
                continue
            plan = fallback_plan(job)
            if plan is None:
                continue
            gpu_arg, tp = plan
            result_path = args.output_dir / f"{job.short}.json"
            log(f"FALLBACK_START short={job.short} gpu={gpu_arg} tp={tp}", main_log)
            wait_for_gpus_free(args.max_gpu_used_mb_before_launch, args.poll_seconds, main_log)
            proc = launch_job(job, args, log_dir, gpu_arg=gpu_arg, tp=tp, log_suffix=f"tp{tp}_retry")
            record.update(
                {
                    "status": "retry_running",
                    "retry_pid": proc.pid,
                    "retry_started_at": utc_now(),
                    "retry_log_path": str(log_dir / f"{job.short}_tp{tp}_retry_eval.log"),
                }
            )
            record.setdefault("attempts", []).append(
                {
                    "kind": f"tp{tp}_retry",
                    "gpu": gpu_arg,
                    "tp": tp,
                    "pid": proc.pid,
                    "started_at": utc_now(),
                    "log_path": str(log_dir / f"{job.short}_tp{tp}_retry_eval.log"),
                }
            )
            save_json(state_path, build_state(records, args))
            while True:
                rc = proc.poll()
                if rc is not None:
                    handle = getattr(proc, "_log_handle", None)
                    if handle is not None:
                        handle.close()
                    score = score_from_result(result_path) if result_path.exists() else None
                    record.update(
                        {
                            "status": "completed_after_retry" if rc == 0 and result_path.exists() else "failed_after_retry",
                            "retry_returncode": rc,
                            "retry_finished_at": utc_now(),
                            "result": str(result_path) if result_path.exists() else None,
                            "score": score,
                        }
                    )
                    record["attempts"][-1].update(
                        {
                            "returncode": rc,
                            "finished_at": utc_now(),
                            "result": str(result_path) if result_path.exists() else None,
                            "score": score,
                        }
                    )
                    log(
                        f"FALLBACK_FINISH short={job.short} returncode={rc} result_exists={result_path.exists()} score={score}",
                        main_log,
                    )
                    save_json(state_path, build_state(records, args))
                    break
                log(f"FALLBACK_MONITOR short={job.short} gpu={gpu_arg} tp={tp}", main_log)
                time.sleep(args.monitor_seconds)

    failed = [
        short
        for short, record in records.items()
        if record.get("status") in {"failed", "failed_after_retry"}
    ]
    log(f"LARGE_NATIVE_EVAL_8GPU_DONE failed={failed}", main_log)
    save_json(state_path, build_state(records, args))
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--vllm-python", type=Path, default=DEFAULT_VLLM_PYTHON)
    parser.add_argument("--max-model-len", type=int, default=49152)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--monitor-seconds", type=int, default=300)
    parser.add_argument("--settle-seconds", type=int, default=180)
    parser.add_argument("--max-gpu-used-mb-before-launch", type=int, default=12000)
    parser.add_argument("--launch-stagger-seconds", type=int, default=30)
    parser.add_argument("--no-wait-training", dest="wait_training", action="store_false")
    parser.add_argument("--no-wait-checkpoints", dest="wait_checkpoints", action="store_false")
    parser.add_argument("--no-clean-stale", dest="clean_stale", action="store_false")
    parser.add_argument("--no-skip-if-exists", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true", default=True)
    parser.add_argument("--thinking-mode", choices=["auto", "on", "off"], default="off")
    parser.add_argument("--strip-thinking-history", choices=["auto", "on", "off"], default="on")
    parser.add_argument("--gemma4-empty-thought-channel", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--launch-only", action="store_true")
    parser.add_argument("--no-retry-failed-with-tp", dest="retry_failed_with_tp", action="store_false")
    parser.set_defaults(wait_training=True, wait_checkpoints=True, clean_stale=True, retry_failed_with_tp=True)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
