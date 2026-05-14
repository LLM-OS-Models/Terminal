#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
STATE_PATH = Path("/home/work/.data/gemma4_native_sft/monitor_state.json")
SCHEDULER_STATE_PATH = Path("/home/work/.data/gemma4_native_sft/parallel_eval_scheduler_state.json")
RESULTS_DIR = Path("/home/work/.data/tb2_lite_eval/gemma4_native_sft_20260509")
LOG_DIR = RESULTS_DIR / "logs"
SCHEDULER_LOG = Path("/home/work/.data/gemma4_native_sft/logs/parallel_native_eval_scheduler.log")
DEFAULT_STAGING_ROOT = Path("/home/work/.data/gemma4_native_sft/staging")
DEFAULT_ENV_FILE = ROOT_DIR / ".env"


MODEL_SHORT_MAP = {
    "gemma-4-E2B-it-Terminal-SFT-Native-Liquid-1Epoch": "gemma4_e2b_it_native_e1",
    "gemma-4-E2B-it-Terminal-SFT-Native-Liquid-2Epoch": "gemma4_e2b_it_native_e2",
    "gemma-4-E2B-Terminal-SFT-Native-Liquid-1Epoch": "gemma4_e2b_base_native_e1",
    "gemma-4-E2B-Terminal-SFT-Native-Liquid-2Epoch": "gemma4_e2b_base_native_e2",
    "gemma-4-E4B-it-Terminal-SFT-Native-Liquid-1Epoch": "gemma4_e4b_it_native_e1",
    "gemma-4-E4B-it-Terminal-SFT-Native-Liquid-2Epoch": "gemma4_e4b_it_native_e2",
    "gemma-4-E4B-Terminal-SFT-Native-Liquid-1Epoch": "gemma4_e4b_base_native_e1",
    "gemma-4-E4B-Terminal-SFT-Native-Liquid-2Epoch": "gemma4_e4b_base_native_e2",
    "gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-1Epoch": "gemma4_26b_a4b_it_native_e1",
    "gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch": "gemma4_26b_a4b_it_native_e2",
    "gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-1Epoch": "gemma4_26b_a4b_base_native_e1",
    "gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch": "gemma4_26b_a4b_base_native_e2",
    "gemma-4-31B-it-Terminal-SFT-Native-Liquid-1Epoch": "gemma4_31b_it_native_e1",
    "gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch": "gemma4_31b_it_native_e2",
    "gemma-4-31B-Terminal-SFT-Native-Liquid-1Epoch": "gemma4_31b_base_native_e1",
    "gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch": "gemma4_31b_base_native_e2",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    SCHEDULER_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    with SCHEDULER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def gpu_memory() -> dict[str, int]:
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


def free_gpus(max_used_mb: int) -> list[str]:
    return [idx for idx, used in sorted(gpu_memory().items(), key=lambda item: int(item[0])) if used <= max_used_mb]


def repo_short(repo_id: str) -> str | None:
    name = repo_id.split("/", 1)[-1]
    return MODEL_SHORT_MAP.get(name)


def result_path(short: str) -> Path:
    return RESULTS_DIR / f"{short}.json"


def score_from_result(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return 100.0 * float(data["aggregate"]["avg_command_f1"])


def upload_after_eval(item: dict, score: float, result: Path) -> tuple[bool, str | None]:
    if item.get("upload_state") not in {"deferred_until_vllm_eval", "upload_failed"}:
        return False, item.get("staging_dir")

    repo_id = item["repo_id"]
    checkpoint = item["checkpoint"]
    staging_dir = DEFAULT_STAGING_ROOT / repo_id.replace("/", "__")
    epoch_label = item.get("epoch_label") or ("2Epoch" if repo_id.endswith("2Epoch") else "1Epoch")
    notes = (
        f"- Source checkpoint: `{checkpoint}`\n"
        f"- TB2-lite result: `{result}`\n"
        f"- TB2-lite score: `{score:.2f}`\n"
        "- Upload policy: uploaded after vLLM evaluation successfully loaded and completed."
    )
    stage_cmd = [
        ".liquid-sft-env/bin/python",
        "gemma4_native_sft/scripts/stage_model_repo.py",
        "--src-dir",
        checkpoint,
        "--staging-dir",
        str(staging_dir),
        "--repo-id",
        repo_id,
        "--base-model",
        item.get("model_id", "unknown"),
        "--source-dataset",
        item.get("dataset_path", "unknown"),
        "--tb2-result",
        str(result),
        "--notes",
        notes,
    ]
    upload_cmd = [
        ".liquid-sft-env/bin/python",
        "gemma4_native_sft/scripts/upload_model_repo.py",
        "--staging-dir",
        str(staging_dir),
        "--repo-id",
        repo_id,
        "--env-file",
        str(DEFAULT_ENV_FILE),
        "--commit-message",
        f"Upload Gemma 4 native SFT {epoch_label} checkpoint after vLLM eval",
    ]
    subprocess.run(stage_cmd, cwd=ROOT_DIR, check=True)
    subprocess.run(upload_cmd, cwd=ROOT_DIR, check=True)
    log(f"HF_UPLOAD_DONE repo={repo_id} staging={staging_dir}")
    return True, str(staging_dir)


def mark_result(repo_id: str, checkpoint: str, short: str) -> bool:
    path = result_path(short)
    if not path.exists():
        return False
    score = round(score_from_result(path), 2)
    monitor_state = load_json(STATE_PATH, {"published": {}})
    changed = False
    matched_item: dict | None = None
    for item in monitor_state.get("published", {}).values():
        if item.get("repo_id") == repo_id and item.get("checkpoint") == checkpoint:
            matched_item = item
            desired = {"score": score, "result": str(path)}
            if item.get("tb2_eval") != desired:
                item["tb2_eval"] = desired
                changed = True
            if item.get("upload_state") in {"deferred_until_vllm_eval", "upload_failed"}:
                try:
                    _, staging_dir = upload_after_eval(item, score, path)
                    item["upload_state"] = "uploaded"
                    item["staging_dir"] = staging_dir
                    item["published_at"] = utc_now()
                    changed = True
                except Exception as exc:
                    item["upload_state"] = "upload_failed"
                    item["upload_error"] = f"{type(exc).__name__}: {exc}"
                    item["upload_failed_at"] = utc_now()
                    changed = True
                    save_json(STATE_PATH, monitor_state)
                    log(f"HF_UPLOAD_FAILED repo={repo_id} error={type(exc).__name__}: {exc}")
                    return False
            break
    if matched_item is None:
        log(f"EVAL_RESULT_ORPHAN repo={repo_id} checkpoint={checkpoint} result={path}")
    if changed:
        save_json(STATE_PATH, monitor_state)
    update_card(repo_id)
    log(f"EVAL_DONE repo={repo_id} score={score:.2f} result={path}")
    return True


def update_card(repo_id: str) -> None:
    subprocess.run(
        [
            ".liquid-sft-env/bin/python",
            "tb2_lite/scripts/update_hf_model_cards.py",
            "--upload",
            "--force",
            "--results-dir",
            str(RESULTS_DIR),
            "--repo",
            repo_id,
        ],
        cwd=ROOT_DIR,
        check=True,
    )


def tp_for_repo(repo_id: str, free_count: int) -> int:
    name = repo_id.lower()
    if "31b" in name:
        return 4 if free_count >= 4 else 0
    if "26b" in name:
        return 2 if free_count >= 2 else 0
    return 1 if free_count >= 1 else 0


def pending_items(monitor_state: dict, scheduler_state: dict) -> list[dict]:
    running_keys = set(scheduler_state.get("running", {}).keys())
    items: list[dict] = []
    for checkpoint, item in sorted(monitor_state.get("published", {}).items()):
        repo_id = item.get("repo_id")
        if not repo_id:
            continue
        short = repo_short(repo_id)
        if not short:
            continue
        if result_path(short).exists():
            mark_result(repo_id, checkpoint, short)
            continue
        if item.get("tb2_eval") not in (None, "pending"):
            continue
        key = f"{repo_id}::{checkpoint}"
        if key in running_keys:
            continue
        items.append({"key": key, "repo_id": repo_id, "checkpoint": checkpoint, "short": short})
    return items


def launch_eval(item: dict, gpu_ids: list[str]) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    gpu_arg = ",".join(gpu_ids)
    tp = len(gpu_ids)
    log_path = LOG_DIR / f"{item['short']}_parallel_eval.log"
    env = os.environ.copy()
    env.update(
        {
            "VLLM_PYTHON": ".vllm-0_19_1/bin/python",
            "OUTPUT_DIR": str(RESULTS_DIR),
            "MAX_MODEL_LEN": "49152",
            "GPU_MEMORY_UTILIZATION": "0.90",
        }
    )
    cmd = [
        "bash",
        "gemma4_native_sft/scripts/eval_native_checkpoint.sh",
        "--model-path",
        item["checkpoint"],
        "--model-short",
        item["short"],
        "--gpu",
        gpu_arg,
        "--tp",
        str(tp),
        "--output-dir",
        str(RESULTS_DIR),
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    log(f"EVAL_START repo={item['repo_id']} short={item['short']} gpu={gpu_arg} tp={tp} pid={proc.pid}")
    return {
        "pid": proc.pid,
        "repo_id": item["repo_id"],
        "checkpoint": item["checkpoint"],
        "short": item["short"],
        "gpus": gpu_ids,
        "tp": tp,
        "log_path": str(log_path),
        "started_at": utc_now(),
        "load_confirmed": False,
    }


def reconcile_running(scheduler_state: dict, load_confirm_mb: int) -> bool:
    changed = False
    running = scheduler_state.setdefault("running", {})
    memory = gpu_memory()
    for key, record in list(running.items()):
        pid = int(record.get("pid") or 0)
        if pid_alive(pid):
            if not record.get("load_confirmed"):
                used = {gpu: memory.get(str(gpu), 0) for gpu in record.get("gpus", [])}
                if used and all(value >= load_confirm_mb for value in used.values()):
                    record["load_confirmed"] = True
                    record["load_confirmed_at"] = utc_now()
                    changed = True
                    log(f"VLLM_LOAD_CONFIRMED repo={record['repo_id']} gpu_used_mb={used}")
            continue
        if mark_result(record["repo_id"], record["checkpoint"], record["short"]):
            scheduler_state.setdefault("completed", {})[key] = {
                **record,
                "completed_at": utc_now(),
            }
            running.pop(key, None)
            changed = True
        else:
            scheduler_state.setdefault("failed", {})[key] = {
                **record,
                "ended_at": utc_now(),
                "reason": "process exited without result json",
            }
            running.pop(key, None)
            changed = True
            log(f"EVAL_FAILED repo={record['repo_id']} log={record.get('log_path')}")
    return changed


def scan_once(args: argparse.Namespace) -> bool:
    monitor_state = load_json(args.monitor_state, {"published": {}})
    scheduler_state = load_json(args.scheduler_state, {"running": {}, "completed": {}, "failed": {}})
    changed = reconcile_running(scheduler_state, args.load_confirm_mb)

    free = free_gpus(args.max_used_mb)
    for item in pending_items(monitor_state, scheduler_state):
        tp = tp_for_repo(item["repo_id"], len(free))
        if tp <= 0 or len(free) < tp:
            continue
        gpu_ids = free[:tp]
        free = free[tp:]
        scheduler_state.setdefault("running", {})[item["key"]] = launch_eval(item, gpu_ids)
        changed = True

    if changed:
        save_json(args.scheduler_state, scheduler_state)
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor-state", type=Path, default=STATE_PATH)
    parser.add_argument("--scheduler-state", type=Path, default=SCHEDULER_STATE_PATH)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--max-used-mb", type=int, default=8000)
    parser.add_argument("--load-confirm-mb", type=int, default=30000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            changed = scan_once(args)
        except Exception as exc:
            log(f"ERROR {type(exc).__name__}: {exc}")
            changed = False
        if not args.watch:
            return 0
        if changed:
            continue
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
