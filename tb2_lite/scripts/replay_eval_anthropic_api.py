#!/usr/bin/env python3
"""Evaluate GLM-5.1 via Anthropic-compatible Messages API on TB2-lite replay."""
from __future__ import annotations

import argparse
import json
import sys
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_metrics import aggregate_scores, parse_prediction, score_commands, step_bucket


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def call_anthropic_api(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: float = 120.0,
    retries: int = 3,
) -> str:
    """Call Anthropic Messages API and return assistant content."""
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                content_blocks = data.get("content", [])
                text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
                return "".join(text_parts)
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                # Rate limit - wait longer
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    print(f"  Rate limited, waiting {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    time.sleep(min(2 ** attempt, 10))
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(f"API call failed after {retries} retries: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://api.z.ai/api/anthropic")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="glm-5.1")
    parser.add_argument("--model-short", default="GLM-5.1-API")
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    # Get API key from env if not provided
    import os
    api_key = args.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    if not api_key:
        print("ERROR: No API key. Set ANTHROPIC_AUTH_TOKEN or pass --api-key")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_short = args.model_short
    out_path = output_dir / f"{model_short}.json"
    if args.skip_if_exists and out_path.exists():
        print(f"SKIP: {out_path} exists")
        return

    rows = load_rows(Path(args.eval_path), args.limit)
    print(f"Loaded {len(rows)} steps from {args.eval_path}")

    gen_start = time.time()
    pred_texts = [""] * len(rows)
    errors = []

    def run_one(idx: int) -> tuple[int, str]:
        row = rows[idx]
        # Use messages from the row (already formatted)
        messages = row.get("messages", [])
        if not messages:
            # Fallback: treat prompt as single user message
            messages = [{"role": "user", "content": row.get("prompt", "")}]

        text = call_anthropic_api(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
        )
        return idx, text

    print(f"Running {len(rows)} steps with concurrency={args.concurrency}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(run_one, idx): idx for idx in range(len(rows))}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                done_idx, text = future.result()
                pred_texts[done_idx] = text
                done += 1
                if done % 10 == 0 or done == len(rows):
                    elapsed = time.time() - gen_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(rows) - done) / rate if rate > 0 else 0
                    print(f"  {done}/{len(rows)} done ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)
            except Exception as exc:
                errors.append({"idx": idx, "task_id": rows[idx].get("task_id"), "error": repr(exc)})
                done += 1

    gen_time = round(time.time() - gen_start, 1)
    print(f"\nGeneration done in {gen_time}s ({gen_time/max(len(rows),1):.2f}s/step)")

    if errors:
        print(f"WARNING: {len(errors)} errors:")
        for e in errors[:5]:
            print(f"  {e}")

    # Score
    per_step = []
    for i, (row, pred_text) in enumerate(zip(rows, pred_texts)):
        if not pred_text:
            # Use empty prediction for failed requests
            pred_text = '{"analysis":"","plan":"","commands":[]}'

        pred = parse_prediction(pred_text)
        ref_units = row["ref_command_units"]

        first_exact, precision, recall, f1 = score_commands(
            pred["command_units"], ref_units
        )
        pred_complete = pred.get("task_complete")
        ref_complete = row["ref_task_complete"]

        per_step.append({
            "task_id": row["task_id"],
            "sample_idx": row.get("sample_idx", 0),
            "step_idx": row["step_idx"],
            "total_steps": row["total_steps"],
            "bucket": step_bucket(row["step_idx"]),
            "source_group": row["source_group"],
            "valid_json": pred["valid_json"],
            "has_analysis": pred["has_analysis"],
            "has_plan": pred["has_plan"],
            "first_cmd_exact": first_exact,
            "ref_command_units": ref_units,
            "pred_command_units": pred["command_units"],
            "command_precision": round(precision, 4),
            "command_recall": round(recall, 4),
            "command_f1": round(f1, 4),
            "ref_task_complete": ref_complete,
            "pred_task_complete": pred_complete,
            "pred_task_complete_true": bool(pred_complete) and ref_complete,
            "pred_preview": pred_text[:800],
        })

    agg = aggregate_scores(per_step)

    result = {
        "model": args.model,
        "model_short": model_short,
        "eval_path": str(args.eval_path),
        "timestamp": datetime.utcnow().isoformat(),
        "gen_time_sec": gen_time,
        "avg_sec_per_step": round(gen_time / max(len(rows), 1), 3),
        "sampling": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "backend": "anthropic-api",
            "base_url": args.base_url,
            "concurrency": args.concurrency,
        },
        "aggregate": agg,
        "per_step": per_step,
        "errors": errors,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # Print summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {model_short}")
    print("=" * 70)
    print(f"Steps: {agg['steps']}, Tasks: {agg['tasks']}")
    print(f"Valid JSON:      {agg['valid_json_pct']}%")
    print(f"First cmd exact: {agg['first_cmd_exact_pct']}%")
    print(f"Precision:       {agg['avg_command_precision']:.4f}")
    print(f"Recall:          {agg['avg_command_recall']:.4f}")
    print(f"Cmd F1:          {agg['avg_command_f1']:.4f}")
    print(f"SCORE:           {100 * agg['avg_command_f1']:.2f}")
    print(f"Next Action:     {agg['next_action_score']}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
