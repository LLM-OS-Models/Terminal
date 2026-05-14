#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")
DATA_ROOT = Path("/home/work/.data/gemma4_native_sft")
LOG_DIR = DATA_ROOT / "logs"
STATE_PATH = DATA_ROOT / "monitor_state.json"


@dataclass(frozen=True)
class RunSpec:
    label: str
    model_id: str
    log_path: Path
    output_dir: Path


RUNS = [
    RunSpec(
        label="E2B-it native",
        model_id="google/gemma-4-E2B-it",
        log_path=LOG_DIR / "gemma4_e2b_it_native_2gpu_20260508T025603Z.log",
        output_dir=DATA_ROOT / "models/google__gemma-4-E2B-it__terminal_sft_native_liquid_2epoch",
    ),
    RunSpec(
        label="E2B base native",
        model_id="google/gemma-4-E2B",
        log_path=LOG_DIR / "gemma4_e2b_base_native_2gpu_20260508T025603Z.log",
        output_dir=DATA_ROOT / "models/google__gemma-4-E2B__terminal_sft_native_liquid_2epoch",
    ),
    RunSpec(
        label="E4B-it native",
        model_id="google/gemma-4-E4B-it",
        log_path=LOG_DIR / "gemma4_e4b_it_native_2gpu_20260508T025603Z.log",
        output_dir=DATA_ROOT / "models/google__gemma-4-E4B-it__terminal_sft_native_liquid_2epoch",
    ),
    RunSpec(
        label="E4B base native",
        model_id="google/gemma-4-E4B",
        log_path=LOG_DIR / "gemma4_e4b_base_native_2gpu_20260508T025603Z.log",
        output_dir=DATA_ROOT / "models/google__gemma-4-E4B__terminal_sft_native_liquid_2epoch",
    ),
]


PROGRESS_RE = re.compile(
    r"(?P<pct>\d+)%\|.*?\|\s*(?P<step>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[^<\]]+)<(?P<eta>[^,\]]+),\s*(?P<sec>[0-9.]+)s/it\]"
)
LOSS_RE = re.compile(r"\{'loss': '([^']+)'.*?'learning_rate': '([^']+)'.*?'epoch': '([^']+)'\}")


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(KST)


def parse_duration(value: str) -> timedelta | None:
    parts = value.strip().split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = [int(part) for part in parts]
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)
        if len(parts) == 2:
            minutes, seconds = [int(part) for part in parts]
            return timedelta(minutes=minutes, seconds=seconds)
    except ValueError:
        return None
    return None


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except Exception:
        return -1


def read_tail(path: Path, max_bytes: int = 2_000_000) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        return handle.read().decode("utf-8", errors="replace").replace("\r", "\n")


def latest_progress(spec: RunSpec) -> dict[str, object]:
    text = read_tail(spec.log_path)
    progress_matches = list(PROGRESS_RE.finditer(text))
    loss_matches = list(LOSS_RE.finditer(text))
    checkpoints = sorted(spec.output_dir.glob("checkpoint-*"), key=checkpoint_step) if spec.output_dir.exists() else []
    info: dict[str, object] = {
        "label": spec.label,
        "model_id": spec.model_id,
        "log": str(spec.log_path),
        "checkpoints": [path.name for path in checkpoints],
    }
    if progress_matches:
        match = progress_matches[-1]
        eta_delta = parse_duration(match.group("eta"))
        eta_kst = now_kst() + eta_delta if eta_delta is not None else None
        info.update(
            {
                "step": int(match.group("step")),
                "total": int(match.group("total")),
                "pct": int(match.group("pct")),
                "elapsed": match.group("elapsed"),
                "remaining": match.group("eta"),
                "sec_per_it": float(match.group("sec")),
                "eta_kst": eta_kst.strftime("%Y-%m-%d %H:%M:%S KST") if eta_kst else "unknown",
            }
        )
    if loss_matches:
        match = loss_matches[-1]
        info.update({"loss": match.group(1), "learning_rate": match.group(2), "epoch": match.group(3)})
    return info


def nvidia_smi() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        return [f"nvidia-smi unavailable: {exc}"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def process_lines() -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,stat,etime,cmd"],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        return [f"ps unavailable: {exc}"]
    needles = ("torchrun", "monitor_native_checkpoints", "update_hf_model_cards")
    lines = []
    for line in result.stdout.splitlines():
        if any(needle in line for needle in needles) and "report_native_status" not in line:
            lines.append(line.strip())
    return lines


def load_monitor_state() -> dict:
    if not STATE_PATH.exists():
        return {"published": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"published": {}}


def format_eval(value: object) -> str:
    if isinstance(value, dict):
        score = value.get("score")
        result = value.get("result")
        score_text = f"Score {float(score):.2f}" if isinstance(score, (int, float)) else "done"
        if result:
            return f"{score_text}, `{Path(str(result)).name}`"
        return score_text
    if value:
        return str(value)
    return "pending"


def build_report() -> str:
    stamp = now_kst().strftime("%Y-%m-%d %H:%M:%S KST")
    state = load_monitor_state()
    published = state.get("published", {})
    lines = [
        f"# Gemma 4 Native Pipeline Status - {stamp}",
        "",
        "## Training",
        "",
        "| Run | Step | Epoch | Loss | ETA(KST) | Remaining | Checkpoints |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for spec in RUNS:
        info = latest_progress(spec)
        step = info.get("step", "?")
        total = info.get("total", "?")
        epoch = info.get("epoch", "?")
        loss = info.get("loss", "?")
        eta = info.get("eta_kst", "unknown")
        remaining = info.get("remaining", "?")
        checkpoints = ", ".join(info.get("checkpoints", [])) or "-"
        lines.append(f"| {spec.label} | {step}/{total} | {epoch} | {loss} | {eta} | {remaining} | {checkpoints} |")

    lines.extend(
        [
            "",
            "## GPU",
            "",
            "| GPU | VRAM(MB) | Util(%) |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in nvidia_smi():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) == 3:
            lines.append(f"| {parts[0]} | {parts[1]} | {parts[2]} |")
        else:
            lines.append(f"| ? | ? | `{row}` |")

    lines.extend(["", "## Published Checkpoints", ""])
    if published:
        for item in sorted(published.values(), key=lambda x: str(x.get("published_at", ""))):
            lines.append(
                f"- `{item.get('repo_id')}` from `{Path(str(item.get('checkpoint', ''))).name}` "
                f"at `{item.get('published_at')}`, eval {format_eval(item.get('tb2_eval'))}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "1. Finish/upload small Gemma native checkpoints as they save.",
            "2. Evaluate small Gemma native checkpoints with vLLM when GPUs free.",
            "3. Update MODEL_EVALUATION_REPORT.md and HF model cards after each result.",
            "4. Lock root-cause analysis for old low Gemma scores.",
            "5. Start large Gemma native training: 26B-A4B-it, 26B-A4B, then 31B-it smoke, 31B-it full, 31B base full.",
            "6. Decide RL candidates only after large Gemma evaluation is complete.",
            "",
            "## Watch Processes",
            "",
        ]
    )
    for line in process_lines():
        lines.append(f"- `{line}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "logs/native_pipeline_status_latest.md")
    parser.add_argument("--append-log", type=Path, default=DATA_ROOT / "logs/native_pipeline_status_history.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    with args.append_log.open("a", encoding="utf-8") as handle:
        handle.write(report)
        handle.write("\n---\n\n")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
