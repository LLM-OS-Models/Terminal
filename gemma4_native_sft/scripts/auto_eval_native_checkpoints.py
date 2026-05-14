#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
STATE_PATH = Path("/home/work/.data/gemma4_native_sft/monitor_state.json")
RESULTS_DIR = Path("/home/work/.data/tb2_lite_eval/gemma4_native_sft_20260509")
LOG_DIR = RESULTS_DIR / "logs"


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"published": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def model_short(repo_id: str) -> str | None:
    name = repo_id.split("/", 1)[-1]
    match = re.match(r"gemma-4-(?P<body>.+?)-Terminal-SFT-Native-Liquid-(?P<epoch>[12])Epoch$", name)
    if not match:
        return None
    body = match.group("body")
    epoch = match.group("epoch")
    mapping = {
        "E2B-it": "gemma4_e2b_it_native",
        "E2B": "gemma4_e2b_base_native",
        "E4B-it": "gemma4_e4b_it_native",
        "E4B": "gemma4_e4b_base_native",
        "26B-A4B-it": "gemma4_26b_a4b_it_native",
        "26B-A4B": "gemma4_26b_a4b_base_native",
        "31B-it": "gemma4_31b_it_native",
        "31B": "gemma4_31b_base_native",
    }
    prefix = mapping.get(body)
    if not prefix:
        return None
    return f"{prefix}_e{epoch}"


def tp_for_repo(repo_id: str, free_count: int) -> int:
    name = repo_id.lower()
    if any(marker in name for marker in ("31b", "26b")):
        return min(4, free_count)
    return 1


def free_gpus(max_used_mb: int, excluded: set[str]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    free: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, used = [part.strip() for part in line.split(",", 1)]
        if index in excluded:
            continue
        try:
            if int(used) <= max_used_mb:
                free.append(index)
        except ValueError:
            continue
    return free


def score_from_result(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    return 100.0 * float(data["aggregate"]["avg_command_f1"])


def update_card(repo_id: str, results_dir: Path) -> None:
    subprocess.run(
        [
            ".liquid-sft-env/bin/python",
            "tb2_lite/scripts/update_hf_model_cards.py",
            "--upload",
            "--force",
            "--results-dir",
            str(results_dir),
            "--repo",
            repo_id,
        ],
        cwd=ROOT_DIR,
        check=True,
    )


def run_eval(item: dict, short: str, gpus: list[str], results_dir: Path) -> Path:
    checkpoint = item["checkpoint"]
    repo_id = item["repo_id"]
    gpu_arg = ",".join(gpus)
    tp = len(gpus)
    log_path = LOG_DIR / f"{short}_auto_eval.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "VLLM_PYTHON": ".vllm-0_19_1/bin/python",
            "OUTPUT_DIR": str(results_dir),
            "MAX_MODEL_LEN": "49152",
            "GPU_MEMORY_UTILIZATION": "0.90",
        }
    )
    cmd = [
        "bash",
        "gemma4_native_sft/scripts/eval_native_checkpoint.sh",
        "--model-path",
        checkpoint,
        "--model-short",
        short,
        "--gpu",
        gpu_arg,
        "--tp",
        str(tp),
        "--output-dir",
        str(results_dir),
    ]
    log(f"EVAL_START repo={repo_id} short={short} gpu={gpu_arg} tp={tp}")
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return results_dir / f"{short}.json"


def scan_once(args: argparse.Namespace) -> bool:
    state = load_state(args.state_path)
    changed = False
    published = state.get("published", {})
    for key, item in sorted(published.items()):
        repo_id = item.get("repo_id")
        checkpoint = item.get("checkpoint")
        if not repo_id or not checkpoint:
            continue
        short = model_short(repo_id)
        if not short:
            continue
        result_path = args.results_dir / f"{short}.json"
        if result_path.exists():
            score = score_from_result(result_path)
            desired = {"score": round(score, 2), "result": str(result_path)}
            if item.get("tb2_eval") != desired:
                item["tb2_eval"] = desired
                changed = True
                log(f"EVAL_RECORDED repo={repo_id} score={score:.2f}")
                update_card(repo_id, args.results_dir)
            continue
        if item.get("tb2_eval") not in (None, "pending"):
            continue
        free = free_gpus(args.max_used_mb, set(args.exclude_gpu))
        tp = tp_for_repo(repo_id, len(free))
        if tp <= 0 or len(free) < tp:
            continue
        result_path = run_eval(item, short, free[:tp], args.results_dir)
        score = score_from_result(result_path)
        item["tb2_eval"] = {"score": round(score, 2), "result": str(result_path)}
        changed = True
        save_state(args.state_path, state)
        update_card(repo_id, args.results_dir)
        log(f"EVAL_DONE repo={repo_id} score={score:.2f}")
        return True
    if changed:
        save_state(args.state_path, state)
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max-used-mb", type=int, default=8000)
    parser.add_argument("--exclude-gpu", action="append", default=[])
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
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
