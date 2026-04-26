#!/usr/bin/env python3
"""Aggregate DeepSeek shard status files into a single progress view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-short", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    status_paths = sorted(results_dir.glob(f"{args.model_short}.status.shard*.json"))
    if not status_paths:
        raise SystemExit(f"No status files found in {results_dir} for {args.model_short}")

    statuses = [json.loads(path.read_text()) for path in status_paths]
    total_done = sum(int(s.get("completed_steps", 0)) for s in statuses)
    total_approx_done = sum(float(s.get("approx_completed_steps", s.get("completed_steps", 0)) or 0.0) for s in statuses)
    total_steps = sum(int(s.get("total_steps", 0)) for s in statuses)
    elapsed = max(float(s.get("elapsed_gen_sec", 0.0) or 0.0) for s in statuses)
    steps_per_sec = total_approx_done / elapsed if elapsed > 0 else 0.0
    remaining = max(total_steps - total_approx_done, 0)
    eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else None

    summary = {
        "model_short": args.model_short,
        "results_dir": str(results_dir),
        "shards_seen": len(statuses),
        "completed_steps": total_done,
        "approx_completed_steps": round(total_approx_done, 2),
        "total_steps": total_steps,
        "progress_pct": round(total_done / max(total_steps, 1) * 100, 1),
        "approx_progress_pct": round(total_approx_done / max(total_steps, 1) * 100, 1),
        "elapsed_gen_sec": round(elapsed, 1),
        "eta_sec": round(eta_sec, 1) if eta_sec is not None else None,
        "steps_per_sec": round(steps_per_sec, 4) if steps_per_sec > 0 else 0.0,
        "shards": statuses,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
