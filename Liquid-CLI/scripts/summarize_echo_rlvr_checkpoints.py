#!/usr/bin/env python3
"""Summarize ECHO RLVR checkpoints and training metrics for later eval sweeps."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def iter_events(log_path: Path, event: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not log_path.exists():
        return rows
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("event") == event:
                rows.append(obj)
    return rows


def checkpoints(output_dir: Path) -> list[int]:
    if not output_dir.exists():
        return []
    values: list[int] = []
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir() and path.name.split("-")[-1].isdigit():
            values.append(int(path.name.split("-")[-1]))
    return sorted(values)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def window_stats(rows: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    window = [row for row in rows if start <= int(row.get("step", -1)) <= end]
    return {
        "steps": len(window),
        "reward_mean": mean([float(row["reward_mean"]) for row in window if "reward_mean" in row]),
        "verifier_reward_mean": mean(
            [float(row["verifier_reward_mean"]) for row in window if "verifier_reward_mean" in row]
        ),
        "world_loss_mean": mean([float(row["world_loss_mean"]) for row in window if "world_loss_mean" in row]),
        "action_tokens_mean": mean([float(row["action_tokens_mean"]) for row in window if "action_tokens_mean" in row]),
        "obs_tokens_mean": mean([float(row["obs_tokens_mean"]) for row in window if "obs_tokens_mean" in row]),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_markdown(run_dir: Path, output_dir: Path, train_steps: list[dict[str, Any]], ckpts: list[int]) -> str:
    latest_step = int(train_steps[-1]["step"]) if train_steps else None
    lines = [
        "# ECHO RLVR Checkpoint Eval Candidates",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Output dir: `{output_dir}`",
        f"- Latest logged train step: `{latest_step if latest_step is not None else 'none'}`",
        f"- Checkpoints found: `{', '.join('checkpoint-' + str(v) for v in ckpts) if ckpts else 'none'}`",
        "",
        "## How To Use",
        "",
        "Do not pick the best adapter from training reward alone. Run the same TB2-lite/TB2 evaluator over all saved checkpoints and select the best verified score.",
        "",
        "Recommended sweep order:",
        "",
        "1. Fast TB2-lite smoke on every checkpoint.",
        "2. Full TB2-lite on the top 3-5 checkpoints.",
        "3. Reserve full TerminalBench/TB2 final evaluation for the best few adapters.",
        "",
        "## Checkpoint Windows",
        "",
        "| checkpoint | train-step window | reward_mean | verifier_reward_mean | world_loss_mean | action_tokens_mean | obs_tokens_mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ckpt in ckpts:
        start = max(0, ckpt - 9)
        stats = window_stats(train_steps, start, ckpt)
        lines.append(
            "| "
            + f"`checkpoint-{ckpt}` | `{start}-{ckpt}` | {fmt(stats['reward_mean'])} | "
            + f"{fmt(stats['verifier_reward_mean'])} | {fmt(stats['world_loss_mean'])} | "
            + f"{fmt(stats['action_tokens_mean'])} | {fmt(stats['obs_tokens_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This run resumes adapter weights from an earlier LoRA checkpoint, but starts a fresh optimizer/scheduler.",
            "- Training reward is noisy because each step contains real terminal execution, timeouts, verifier outcomes, and ECHO world-model loss.",
            "- A checkpoint with lower immediate reward can still evaluate better if it improves command format, path handling, or later-step behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--write-md", default="")
    parser.add_argument("--write-json", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    train_steps = iter_events(run_dir / "train.log", "train_step")
    ckpts = checkpoints(output_dir)
    payload = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "latest_train_step": train_steps[-1] if train_steps else None,
        "checkpoints": [f"checkpoint-{step}" for step in ckpts],
        "checkpoint_windows": {
            f"checkpoint-{step}": window_stats(train_steps, max(0, step - 9), step) for step in ckpts
        },
    }
    if args.write_json:
        Path(args.write_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = build_markdown(run_dir, output_dir, train_steps, ckpts)
    if args.write_md:
        Path(args.write_md).write_text(markdown, encoding="utf-8")
    if not args.write_json and not args.write_md:
        print(markdown, end="")


if __name__ == "__main__":
    main()
