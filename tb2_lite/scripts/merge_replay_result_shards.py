#!/usr/bin/env python3
"""Merge TB2-lite replay result shards and recompute aggregate metrics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_metrics import aggregate_scores


def sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("sample_idx", 0)),
        int(row.get("step_idx", 0)),
        int(str(row.get("task_id", "task-0")).split("-")[-1]) if "-" in str(row.get("task_id", "")) else 0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    shard_results = [json.loads(path.read_text()) for path in args.inputs]
    per_step: list[dict[str, Any]] = []
    for result in shard_results:
        per_step.extend(result.get("per_step", []))
    per_step.sort(key=sort_key)

    first = shard_results[0]
    aggregate = aggregate_scores(per_step)
    gen_time = max(float(result.get("gen_time_sec", 0.0)) for result in shard_results)
    load_time = max(float(result.get("load_time_sec", 0.0)) for result in shard_results)
    total_input_steps = max(
        int(result.get("total_input_steps", result.get("total_steps", 0)))
        for result in shard_results
    )
    merged = {
        **{key: value for key, value in first.items() if key not in {"aggregate", "per_step"}},
        "timestamp": datetime.utcnow().isoformat(),
        "load_time_sec": round(load_time, 1),
        "gen_time_sec": round(gen_time, 1),
        "avg_sec_per_step": round(gen_time / max(len(per_step), 1), 3),
        "backend": f"{first.get('backend', 'unknown')}-merged",
        "complete": len(per_step) == total_input_steps,
        "generated_steps": len(per_step),
        "total_steps": len(per_step),
        "total_input_steps": total_input_steps,
        "merged_shards": [str(path) for path in args.inputs],
        "aggregate": aggregate,
        "per_step": per_step,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    score = round(100.0 * aggregate["avg_command_f1"], 2)
    print(json.dumps({"output_path": str(args.output), "score": score}, ensure_ascii=False))


if __name__ == "__main__":
    main()
