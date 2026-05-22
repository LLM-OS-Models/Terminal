#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("OUT_DIR", "/home/work/.data/tb2_lite_eval/corrected_readme_models_vllm"))
LOG_DIR = Path(os.environ.get("LOG_DIR", str(OUT_DIR / "logs")))
EVAL_PATH = os.environ.get("EVAL_PATH", "tb2_lite/data/replay_full.jsonl")
REPORT_PATH = Path(os.environ.get("REPORT_PATH", "README.md"))
VLLM_PY = os.environ.get("VLLM_PY", str(ROOT / ".vllm-0_19_1/bin/python"))
VLLM_SITE = os.environ.get("VLLM_SITE", str(ROOT / ".vllm-0_19_1/lib/python3.12/site-packages"))
GPU_COUNT = int(os.environ.get("GPU_COUNT", "8"))
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "32768")
MAX_TOKENS = os.environ.get("MAX_TOKENS", "1024")
GPU_MEMORY_UTILIZATION = os.environ.get("GPU_MEMORY_UTILIZATION", "0.94")
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "10"))


@dataclass(frozen=True)
class Job:
    short: str
    model: str
    extra: tuple[str, ...] = ()


JOBS: list[Job] = [
    Job("gyung_lfm2_8b_terminal_sft_unsloth", "gyung/LFM2-8B-Terminal-SFT-Unsloth"),
    Job("nemotron_terminal_8b", "nvidia/Nemotron-Terminal-8B"),
    Job("qwen35_2b_base", "Qwen/Qwen3.5-2B"),
    Job("qwen35_4b_base", "Qwen/Qwen3.5-4B"),
    Job("qwen35_9b_base", "Qwen/Qwen3.5-9B"),
    Job("liquid_lfm25_1p2b_instruct_base", "LiquidAI/LFM2.5-1.2B-Instruct"),
    Job("liquid_lfm2_2p6b_base", "LiquidAI/LFM2-2.6B"),
    Job("liquid_lfm2_8b_a1b_base", "LiquidAI/LFM2-8B-A1B"),
    Job("qwen35_2b_sft_samecount_e2", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/checkpoint-110-fixed"),
    Job("qwen35_2b_sft_samecount_e1", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/checkpoint-55-vllmfix3"),
    Job("qwen35_4b_sft_2bdata_e1", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata/checkpoint-960-vllmfix4"),
    Job("qwen35_4b_sft_2bdata_e2", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata/checkpoint-1920-fixed"),
    Job("qwen35_9b_sft_2bdata_e2", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata/checkpoint-4386-vllmfix9"),
    Job("qwen35_9b_sft_2bdata_e1", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata/checkpoint-2193-fixed"),
    Job("lfm25_1p2b_sft_unsloth_e2", "LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth"),
    Job("lfm2_8b_sft_unsloth_e2", "LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth"),
    Job("lfm2_2p6b_sft_unsloth_e2", "LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth"),
    Job("qwen35_2b_sft_unsloth_e2", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_unsloth_lora/final"),
    Job("nemotron_terminal_14b", "nvidia/Nemotron-Terminal-14B"),
    Job("nemotron_terminal_32b", "nvidia/Nemotron-Terminal-32B"),
    Job("qwen35_27b_base", "Qwen/Qwen3.5-27B"),
    Job("qwen36_27b_base", "Qwen/Qwen3.6-27B"),
    Job("qwen36_35b_a3b_fp8_base", "Qwen/Qwen3.6-35B-A3B-FP8"),
    Job("gemma4_26b_a4b_it_base", "google/gemma-4-26B-A4B-it"),
    Job("gemma4_31b_it_base", "google/gemma-4-31B-it"),
    Job("liquid_lfm2_24b_a2b_base", "LiquidAI/LFM2-24B-A2B"),
    Job("jackrong_qwen35_27b_claude_opus_distill", "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"),
    Job("qwen36_27b_sft_hf_fsdp_e2", "LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData"),
    Job("qwen36_27b_sft_hf_fsdp_e1", "LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData"),
    Job("qwen35_27b_sft_hf_fsdp_e2", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-3834"),
    Job("qwen35_27b_sft_hf_fsdp_e1", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-1917"),
    Job("qwen35_35b_a3b_sft_hf_fsdp_e2", "LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData"),
    Job("qwen35_35b_a3b_sft_hf_fsdp_e1", "LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData"),
    Job("qwen36_35b_a3b_sft_hf_fsdp_e2", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.6-35B-A3B__terminal_sft_2epoch_hf_fsdp/checkpoint-5868"),
    Job("qwen36_35b_a3b_sft_hf_fsdp_e1", "/home/work/.data/qwen_sft/models/Qwen__Qwen3.6-35B-A3B__terminal_sft_2epoch_hf_fsdp/checkpoint-2934"),
    Job("gemma4_26b_a4b_sft_hf_fsdp_e2", "/home/work/.data/qwen_sft/models/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-1468"),
    Job("gemma4_26b_a4b_sft_hf_fsdp_e1", "/home/work/.data/qwen_sft/models/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-734"),
    Job("lfm2_24b_a2b_sft_hf_fsdp_e2", "/home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-1468"),
    Job("lfm2_24b_a2b_sft_hf_fsdp_e1", "/home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-734"),
    Job("gemma4_e4b_it_base", "google/gemma-4-E4B-it"),
    Job("gemma4_e2b_it_base", "google/gemma-4-E2B-it"),
    Job("gemma4_e2b_sft_ddp_e2", "/home/work/.data/qwen_sft/models/google__gemma-4-E2B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-734"),
    Job("gemma4_e2b_sft_ddp_e1", "/home/work/.data/qwen_sft/models/google__gemma-4-E2B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-367"),
    Job("gemma4_e2b_sft_hf_fsdp_e2", "/home/work/.data/qwen_sft/models/google__gemma-4-E2B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-5868"),
    Job("gemma4_e2b_sft_hf_fsdp_e1", "/home/work/.data/qwen_sft/models/google__gemma-4-E2B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-2934"),
    Job("gemma4_e4b_sft_ddp_e2", "/home/work/.data/qwen_sft/models/google__gemma-4-E4B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-2934"),
    Job("gemma4_e4b_sft_ddp_e1", "/home/work/.data/qwen_sft/models/google__gemma-4-E4B-it__terminal_sft_2epoch_ddp_4gpu/checkpoint-1467"),
    Job("gemma4_31b_sft_hf_fsdp_e2", "/home/work/.data/qwen_sft/models/google__gemma-4-31B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-2934"),
    Job("gemma4_31b_sft_hf_fsdp_e1", "/home/work/.data/qwen_sft/models/google__gemma-4-31B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-1467"),
    Job("ouro_2p6b_terminal_sft", "LLM-OS-Models/Ouro-2.6B-Terminal-SFT"),
    Job("ouro_1p4b_terminal_sft", "LLM-OS-Models/Ouro-1.4B-Terminal-SFT"),
    Job("bytedance_ouro_1p4b_base", "ByteDance/Ouro-1.4B"),
    Job("bytedance_ouro_2p6b_thinking", "ByteDance/Ouro-2.6B-Thinking"),
    Job("bytedance_ouro_2p6b_base", "ByteDance/Ouro-2.6B"),
    Job("bytedance_ouro_1p4b_thinking", "ByteDance/Ouro-1.4B-Thinking"),
    Job("ouro_1p4b_thinking_terminal_sft", "LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT"),
    Job("ouro_2p6b_thinking_terminal_sft", "LLM-OS-Models/Ouro-2.6B-Thinking-Terminal-SFT"),
]


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}"
    print(line, flush=True)
    with (LOG_DIR / "queue_python.log").open("a") as handle:
        handle.write(line + "\n")


def summarize() -> None:
    subprocess.run(
        [
            str(ROOT / ".liquid-sft-env/bin/python"),
            "tb2_lite/scripts/summarize_corrected_results.py",
            "--results-dir",
            str(OUT_DIR),
            "--output-path",
            str(OUT_DIR / "SUMMARY.md"),
            "--title",
            "README 모델 corrected TB2-lite vLLM 재평가 (corrected_readme_models_vllm)",
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if (OUT_DIR / "SUMMARY.md").exists():
        REPORT_PATH.write_text((OUT_DIR / "SUMMARY.md").read_text(), encoding="utf-8")


def env_for_gpu(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HF_HOME"] = env.get("HF_HOME", "/home/work/.data/huggingface")
    env["PYTHONNOUSERSITE"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = VLLM_SITE if not existing else f"{VLLM_SITE}:{existing}"
    return env


def launch(job: Job, gpu: int) -> subprocess.Popen:
    log_path = LOG_DIR / f"{job.short}.log"
    cmd = [
        VLLM_PY,
        "tb2_lite/scripts/replay_eval.py",
        "--model",
        job.model,
        "--model-short",
        job.short,
        "--gpu",
        str(gpu),
        "--eval-path",
        EVAL_PATH,
        "--output-dir",
        str(OUT_DIR),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        MAX_MODEL_LEN,
        "--max-tokens",
        MAX_TOKENS,
        "--temperature",
        "0.0",
        "--top-p",
        "1.0",
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--language-model-only",
        "--skip-if-exists",
        *job.extra,
    ]
    handle = log_path.open("w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env_for_gpu(gpu), stdout=handle, stderr=subprocess.STDOUT)
    setattr(proc, "_log_handle", handle)
    return proc


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending = [job for job in JOBS if not (OUT_DIR / f"{job.short}.json").exists()]
    log(f"queue start total_jobs={len(JOBS)} pending={len(pending)} gpu_count={GPU_COUNT}")
    running: dict[int, tuple[Job, subprocess.Popen]] = {}
    completed = 0
    failed = 0
    summarize()

    while pending or running:
        for gpu in range(GPU_COUNT):
            if gpu in running or not pending:
                continue
            job = pending.pop(0)
            proc = launch(job, gpu)
            running[gpu] = (job, proc)
            log(f"launch gpu={gpu} short={job.short} model={job.model}")

        time.sleep(POLL_SECONDS)

        for gpu, (job, proc) in list(running.items()):
            status = proc.poll()
            if status is None:
                continue
            handle = getattr(proc, "_log_handle", None)
            if handle:
                handle.close()
            del running[gpu]
            completed += 1
            if status == 0:
                log(f"done gpu={gpu} short={job.short}")
            else:
                failed += 1
                log(f"failed status={status} gpu={gpu} short={job.short} log={LOG_DIR / (job.short + '.log')}")
            summarize()
            log(f"progress completed={completed} running={len(running)} pending={len(pending)} failed={failed}")

    summarize()
    log(f"queue finished completed={completed} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
