#!/usr/bin/env python3
"""
Summarize evaluation results from all models.

Usage: python3 summarize.py [--results-dir DIR]
"""
import json
import os
import argparse
from datetime import datetime


def summarize(results_dir="results"):
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".json"))
    if not files:
        print("No result files found.")
        return

    rows = []
    for fname in files:
        path = os.path.join(results_dir, fname)
        with open(path) as f:
            data = json.load(f)

        agg = data.get("aggregate_metrics") or data.get("aggregate") or {}
        load_sec = data.get("load_time_sec", 0) or 0
        gen_sec = data.get("gen_time_sec", 0) or 0
        total_sec = data.get("total_time_sec")
        if total_sec is None:
            total_sec = load_sec + gen_sec

        rows.append({
            "model": data.get("model_short", fname.replace(".json", "")),
            "model_id": data.get("model", ""),
            "params_B": data.get("params_B", 0),
            "samples": data.get("num_samples", data.get("samples", 0)),
            "total_sec": total_sec,
            "avg_sec": data.get("avg_time_per_sample", data.get("avg_sec_per_sample", 0)),
            "has_cmds%": agg.get("pct_has_commands", agg.get("pct_has_cmds", 0)),
            "avg_cmds": agg.get("avg_command_count", agg.get("avg_cmds", 0)),
            "cmd_overlap": agg.get("avg_cmd_overlap", agg.get("avg_overlap", 0)),
            "thinking%": agg.get("pct_has_thinking", agg.get("pct_thinking", 0)),
            "avg_len": agg.get("avg_response_length", agg.get("avg_pred_len", 0)),
        })

    # Sort by command overlap (higher = better match with reference)
    rows.sort(key=lambda r: r["cmd_overlap"], reverse=True)

    # Print table
    hdr = f"{'Model':<45} {'B':>5} {'Samples':>7} {'Time':>8} {'Cmds%':>7} {'Overlap':>8} {'Think%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        total_sec = r["total_sec"] or 0
        print(
            f"{r['model']:<45} {r['params_B']:>5.1f} {r['samples']:>7} "
            f"{total_sec:>7.0f}s {r['has_cmds%']:>6.1f}% {r['cmd_overlap']:>7.4f} "
            f"{r['thinking%']:>6.1f}%"
        )

    # Save markdown table
    md_path = os.path.join(os.path.dirname(results_dir) or ".", "RESULTS_SUMMARY.md")
    with open(md_path, "w") as f:
        f.write(f"# Evaluation Results Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"Evaluated on {rows[0]['samples'] if rows else 0} terminal task samples.\n\n")
        f.write(f"Sorted by `cmd_overlap` (fraction of reference commands found in prediction).\n\n")
        f.write("| # | Model | Params | Time | Has Commands | Cmd Overlap | Thinking |\n")
        f.write("|---|-------|--------|------|-------------|-------------|----------|\n")
        for i, r in enumerate(rows, 1):
            f.write(
                f"| {i} | `{r['model']}` | {r['params_B']:.1f}B | "
                f"{(r['total_sec'] or 0):.0f}s | {r['has_cmds%']:.1f}% | "
                f"{r['cmd_overlap']:.4f} | {r['thinking%']:.1f}% |\n"
            )
    print(f"\nSaved: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    summarize(args.results_dir)
