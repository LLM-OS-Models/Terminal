#!/usr/bin/env python3
"""Merge partial DeepSeek replay results from multiple tensor-parallel shards."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tb2_lite.scripts.replay_metrics import aggregate_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-short", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    part_paths = sorted(results_dir.glob(f"{args.model_short}.part*.json"))
    if not part_paths:
        raise SystemExit(f"No shard files found for {args.model_short} in {results_dir}")

    parts = [json.loads(path.read_text()) for path in part_paths]
    per_step: list[dict] = []
    for part in parts:
        per_step.extend(part["per_step"])

    aggregate = aggregate_scores(per_step)
    merged = {
        "model": parts[0]["model"],
        "model_path": parts[0]["model_path"],
        "model_short": parts[0]["model_short"],
        "gpu": [gpu for part in parts for gpu in part["gpu"]],
        "eval_path": parts[0]["eval_path"],
        "rows_total": parts[0]["rows_total"],
        "rows_in_shard": sum(part["rows_in_shard"] for part in parts),
        "shard_count": len(parts),
        "timestamp": datetime.now().isoformat(),
        "load_time_sec": round(max(part["load_time_sec"] for part in parts), 1),
        "gen_time_sec": round(max(part["gen_time_sec"] for part in parts), 1),
        "avg_sec_per_step": round(max(part["gen_time_sec"] for part in parts) / max(len(per_step), 1), 3),
        "sampling": {
            **parts[0]["sampling"],
            "parallel_shards": len(parts),
            "shard_rows": [part["rows_in_shard"] for part in parts],
            "shard_load_time_sec": [part["load_time_sec"] for part in parts],
            "shard_gen_time_sec": [part["gen_time_sec"] for part in parts],
        },
        "aggregate": aggregate,
        "per_step": per_step,
        "source_parts": [path.name for path in part_paths],
    }

    out_path = results_dir / f"{args.model_short}.json"
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"Merged {len(parts)} shard files into {out_path}")
    print(
        f"Score: {aggregate['next_action_score']} | "
        f"Cmd F1: {aggregate['avg_command_f1']} | "
        f"First cmd exact: {aggregate['first_cmd_exact_pct']}%"
    )


if __name__ == "__main__":
    main()
