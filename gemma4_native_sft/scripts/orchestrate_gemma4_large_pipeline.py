#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/home/work/.data/gemma4_native_sft")
LOG_ROOT = DATA_ROOT / "logs"
STATE_PATH = DATA_ROOT / "large_pipeline_orchestrator_state.json"
MONITOR_STATE_PATH = DATA_ROOT / "monitor_state.json"
STATUS_PATH = LOG_ROOT / "large_pipeline_orchestrator_latest.md"

FULL_GPU_IDS = "0,1,2,3,4,5,6,7"
FULL_NPROC = 8
OPPORTUNISTIC_GPU_COUNT = 4
ENABLE_OPPORTUNISTIC_4GPU = False
FSDP_CONFIG = ROOT_DIR / "qwen_sft/configs/fsdp_qwen_full_shard_sizewrap.json"


@dataclass(frozen=True)
class Stage:
    name: str
    model_id: str
    template_model_id: str | None
    dataset_path: Path
    output_dir: Path
    repo_2epoch: str
    depends_on_repos: tuple[str, ...]
    master_port: int


STAGES = [
    Stage(
        name="31b_it",
        model_id="google/gemma-4-31B-it",
        template_model_id=None,
        dataset_path=DATA_ROOT / "datasets/google__gemma-4-31B-it__liquid_raw_json_masked_8192",
        output_dir=DATA_ROOT / "models/google__gemma-4-31B-it__terminal_sft_native_liquid_2epoch",
        repo_2epoch="LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch",
        depends_on_repos=(
            "LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch",
            "LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch",
        ),
        master_port=29631,
    ),
    Stage(
        name="31b_base",
        model_id="google/gemma-4-31B",
        template_model_id="google/gemma-4-31B-it",
        dataset_path=DATA_ROOT / "datasets/google__gemma-4-31B__liquid_raw_json_masked_8192",
        output_dir=DATA_ROOT / "models/google__gemma-4-31B__terminal_sft_native_liquid_2epoch",
        repo_2epoch="LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch",
        depends_on_repos=(
            "LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch",
        ),
        master_port=29632,
    ),
]

BATCH_CANDIDATES = (8, 4, 2, 1)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def kst_now() -> datetime:
    return utc_now().astimezone(timezone(timedelta(hours=9)))


def stamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def log(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with (LOG_ROOT / "large_pipeline_orchestrator.log").open("a", encoding="utf-8") as handle:
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


def shell_output(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=ROOT_DIR, text=True, capture_output=True, check=False)
    return result.stdout


def pgrep(pattern: str) -> list[str]:
    out = shell_output(["bash", "-lc", f"pgrep -af {json.dumps(pattern)} || true"])
    return [line for line in out.splitlines() if line.strip()]


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


def active_training_lines() -> list[str]:
    lines = pgrep("gemma4_native_sft/scripts/run_native_hf_fsdp.sh")
    lines += pgrep("torchrun .*gemma4_native_sft/scripts/train_native_hf_fsdp.py")
    return sorted(set(lines))


def active_eval_lines() -> list[str]:
    patterns = (
        "gemma4_native_sft/scripts/eval_native_checkpoint.sh",
        "tb2_lite/scripts/replay_eval.py",
        "vllm",
    )
    lines: list[str] = []
    for pattern in patterns:
        lines += pgrep(pattern)
    return sorted(set(lines))


def gpu_memory() -> list[tuple[int, int, int]]:
    out = shell_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"]
    )
    rows: list[tuple[int, int, int]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            try:
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                pass
    return rows


def free_gpu_ids(max_used_mb: int = 8000) -> list[str]:
    return [str(idx) for idx, used, _ in gpu_memory() if used <= max_used_mb]


def selected_gpus_free(gpu_ids: list[str], max_used_mb: int = 8000) -> bool:
    memory_by_id = {str(idx): used for idx, used, _ in gpu_memory()}
    return all(memory_by_id.get(gpu_id, max_used_mb + 1) <= max_used_mb for gpu_id in gpu_ids)


def all_gpus_free(max_used_mb: int = 8000) -> bool:
    rows = gpu_memory()
    return len(rows) == 8 and all(used <= max_used_mb for _, used, _ in rows)


def repo_eval_done(repo_id: str) -> bool:
    state = load_json(MONITOR_STATE_PATH, {"published": {}})
    for item in state.get("published", {}).values():
        if item.get("repo_id") != repo_id:
            continue
        tb2_eval = item.get("tb2_eval")
        return isinstance(tb2_eval, dict) and isinstance(tb2_eval.get("score"), (int, float))
    return False


def pending_eval_repos() -> list[str]:
    state = load_json(MONITOR_STATE_PATH, {"published": {}})
    pending: list[str] = []
    for item in state.get("published", {}).values():
        repo_id = item.get("repo_id")
        if not repo_id:
            continue
        tb2_eval = item.get("tb2_eval")
        if tb2_eval in (None, "pending"):
            pending.append(repo_id)
    return sorted(set(pending))


def deps_done(stage: Stage) -> bool:
    return all(repo_eval_done(repo) for repo in stage.depends_on_repos)


def stage_done(stage: Stage) -> bool:
    return repo_eval_done(stage.repo_2epoch)


def acc_for_batch(batch: int) -> int:
    return 2 if batch == 1 else 1


def write_config(stage: Stage, batch: int, gpu_ids: list[str], nproc: int) -> Path:
    acc = acc_for_batch(batch)
    run_name = f"gemma4_{stage.name}_native_{nproc}gpu_bsz{batch}_acc{acc}_gc_qwenfsdp"
    path = ROOT_DIR / f"gemma4_native_sft/configs/auto_{run_name}.env"
    lines = [
        f"MODEL_ID={stage.model_id}",
    ]
    if stage.template_model_id:
        lines.append(f"TEMPLATE_MODEL_ID={stage.template_model_id}")
    lines += [
        f"DATASET_PATH={stage.dataset_path}",
        f"OUTPUT_DIR={stage.output_dir}",
        f"CUDA_VISIBLE_DEVICES_OVERRIDE={','.join(gpu_ids)}",
        f"NPROC_PER_NODE={nproc}",
        f"MASTER_PORT={stage.master_port}",
        f"RUN_NAME={run_name}",
        "OMP_NUM_THREADS=8",
        f"PER_DEVICE_TRAIN_BATCH_SIZE={batch}",
        f"GRADIENT_ACCUMULATION_STEPS={acc}",
        "GRADIENT_CHECKPOINTING=1",
        "LEARNING_RATE=1e-5",
        "NUM_TRAIN_EPOCHS=2",
        "SAVE_STRATEGY=epoch",
        "LOGGING_STEPS=1",
        "WARMUP_RATIO=0.03",
        "OPTIM=adafactor",
        "BF16=0",
        'FSDP="full_shard auto_wrap"',
        f"FSDP_CONFIG={FSDP_CONFIG}",
        "SAVE_ONLY_MODEL=1",
        "SAVE_TOTAL_LIMIT=2",
        "SKIP_FINAL_SAVE=1",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def launch_stage(stage: Stage, batch: int, state: dict, gpu_ids: list[str], mode: str) -> None:
    nproc = len(gpu_ids)
    config_path = write_config(stage, batch, gpu_ids, nproc)
    launch_log = LOG_ROOT / f"launcher_{stage.name}_bsz{batch}_acc{acc_for_batch(batch)}_{stamp()}.log"
    cmd = [
        "bash",
        "gemma4_native_sft/scripts/run_native_hf_fsdp.sh",
        "--config",
        str(config_path.relative_to(ROOT_DIR)),
    ]
    with launch_log.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    stage_state = state.setdefault("stages", {}).setdefault(stage.name, {})
    stage_state.update(
        {
            "status": "running",
            "pid": proc.pid,
            "mode": mode,
            "gpu_ids": gpu_ids,
            "nproc": nproc,
            "batch": batch,
            "accumulation": acc_for_batch(batch),
            "config": str(config_path),
            "launcher_log": str(launch_log),
            "started_at": utc_now().isoformat(timespec="seconds"),
        }
    )
    save_json(STATE_PATH, state)
    log(
        f"LAUNCHED stage={stage.name} mode={mode} gpu={','.join(gpu_ids)} "
        f"batch={batch} acc={acc_for_batch(batch)} pid={proc.pid}"
    )


def attempt_failed_for_memory(stage_state: dict) -> bool:
    log_paths = [stage_state.get("launcher_log")]
    config_path = stage_state.get("config")
    if config_path:
        config_text = Path(config_path).read_text(encoding="utf-8") if Path(config_path).exists() else ""
        run_name = ""
        for line in config_text.splitlines():
            if line.startswith("RUN_NAME="):
                run_name = line.split("=", 1)[1].strip()
                break
        if run_name:
            log_paths += [str(path) for path in LOG_ROOT.glob(f"{run_name}_*.log")]
    text = ""
    for raw in log_paths:
        if not raw:
            continue
        path = Path(raw)
        if path.exists():
            text += path.read_text(encoding="utf-8", errors="ignore")[-50000:]
    markers = ("CUDA out of memory", "OutOfMemoryError", "Tried to allocate", "torch.OutOfMemoryError")
    return any(marker in text for marker in markers)


def next_batch_after(batch: int) -> int | None:
    try:
        index = BATCH_CANDIDATES.index(batch)
    except ValueError:
        return BATCH_CANDIDATES[0]
    if index + 1 >= len(BATCH_CANDIDATES):
        return None
    return BATCH_CANDIDATES[index + 1]


def write_status(state: dict, message: str) -> None:
    rows = gpu_memory()
    gpu_lines = [f"- GPU {idx}: `{used / 1024:.1f} GiB / {total / 1024:.1f} GiB`" for idx, used, total in rows]
    stage_lines = []
    for stage in STAGES:
        stage_state = state.get("stages", {}).get(stage.name, {})
        stage_lines.append(
            f"- `{stage.name}`: status=`{stage_state.get('status', 'pending')}`, "
            f"mode=`{stage_state.get('mode', '-')}`, "
            f"gpu=`{','.join(stage_state.get('gpu_ids', [])) if stage_state.get('gpu_ids') else '-'}`, "
            f"batch=`{stage_state.get('batch', '-')}`, pid=`{stage_state.get('pid', '-')}`, "
            f"repo2_eval_done=`{stage_done(stage)}`"
        )
    content = "\n".join(
        [
            f"# Gemma 4 Large Pipeline Orchestrator",
            "",
            f"- KST: `{kst_now().strftime('%Y-%m-%d %H:%M:%S')}`",
            f"- Status: {message}",
            "",
            "## Stages",
            *stage_lines,
            "",
            "## GPU",
            *gpu_lines,
            "",
        ]
    )
    STATUS_PATH.write_text(content, encoding="utf-8")


def step() -> None:
    state = load_json(STATE_PATH, {"stages": {}})
    message = "watching"

    for stage in STAGES:
        if stage_done(stage):
            state.setdefault("stages", {}).setdefault(stage.name, {})["status"] = "done"
            continue

        stage_state = state.setdefault("stages", {}).setdefault(stage.name, {})
        if stage_state.get("status") == "running":
            pid = int(stage_state.get("pid") or 0)
            if pid_alive(pid):
                message = f"stage {stage.name} running pid={pid}"
                break
            batch = int(stage_state.get("batch") or BATCH_CANDIDATES[0])
            if attempt_failed_for_memory(stage_state):
                stage_state.setdefault("failed_batches", []).append(batch)
                next_batch = next_batch_after(batch)
                stage_state["status"] = "oom" if next_batch else "failed"
                stage_state["ended_at"] = utc_now().isoformat(timespec="seconds")
                save_json(STATE_PATH, state)
                log(f"OOM stage={stage.name} batch={batch} next={next_batch}")
                gpu_ids = [str(gpu_id) for gpu_id in stage_state.get("gpu_ids", FULL_GPU_IDS.split(","))]
                if next_batch is not None and selected_gpus_free(gpu_ids):
                    launch_stage(stage, next_batch, state, gpu_ids, stage_state.get("mode", "full_8gpu"))
                message = f"stage {stage.name} oom batch={batch}"
                break
            stage_state["status"] = "exited_waiting_publish_or_eval"
            stage_state["ended_at"] = utc_now().isoformat(timespec="seconds")
            message = f"stage {stage.name} exited; waiting publish/eval"
            break

        if not deps_done(stage):
            pending_eval = pending_eval_repos()
            free = free_gpu_ids()
            if (
                ENABLE_OPPORTUNISTIC_4GPU
                and stage.name == "31b_it"
                and not pending_eval
                and len(free) >= OPPORTUNISTIC_GPU_COUNT
            ):
                failed = set(stage_state.get("failed_batches", []))
                batch = next((candidate for candidate in BATCH_CANDIDATES if candidate not in failed), None)
                if batch is None:
                    stage_state["status"] = "failed_all_batches"
                    message = f"all opportunistic batches failed for {stage.name}"
                    break
                gpu_ids = free[:OPPORTUNISTIC_GPU_COUNT]
                launch_stage(stage, batch, state, gpu_ids, "opportunistic_4gpu")
                message = f"launched opportunistic {stage.name} gpu={','.join(gpu_ids)} batch={batch}"
                break
            missing = [repo for repo in stage.depends_on_repos if not repo_eval_done(repo)]
            if pending_eval:
                message = f"waiting pending eval before {stage.name}: {', '.join(pending_eval[:4])}"
            else:
                message = f"waiting deps for {stage.name}: {', '.join(missing)}"
            break

        if active_training_lines():
            message = f"waiting active training before {stage.name}"
            break

        if active_eval_lines():
            message = f"waiting active eval before {stage.name}"
            break

        if not all_gpus_free():
            message = f"waiting all GPUs free before {stage.name}"
            break

        failed = set(stage_state.get("failed_batches", []))
        batch = next((candidate for candidate in BATCH_CANDIDATES if candidate not in failed), None)
        if batch is None:
            stage_state["status"] = "failed_all_batches"
            message = f"all batches failed for {stage.name}"
            break
        launch_stage(stage, batch, state, FULL_GPU_IDS.split(","), "full_8gpu")
        message = f"launched {stage.name} batch={batch}"
        break

    save_json(STATE_PATH, state)
    write_status(state, message)


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log("START large pipeline orchestrator")
    while True:
        try:
            step()
        except Exception as exc:
            log(f"ERROR {type(exc).__name__}: {exc}")
        time.sleep(180)


if __name__ == "__main__":
    raise SystemExit(main())
