"""Terminal-Bench 2.0 evaluation for Nemotron-Terminal-8B via vLLM."""

import json
import os
import time
import tomllib
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()

VLLM_URL = "http://localhost:8100/v1/chat/completions"
TASKS_DIR = Path("/home/work/.projects/Terminal/terminal-bench-2/terminal-bench")
RESULTS_DIR = Path("/home/work/.projects/Terminal/eval/results")

SYSTEM_PROMPT = """You are an autonomous terminal agent. Given a terminal task, you must analyze the situation and output a JSON object with the following structure:
{
  "analysis": "Analysis of the current terminal state...",
  "plan": "Step-by-step plan for the next command...",
  "commands": [
    {"keystrokes": "command here\\n", "duration": 0.1}
  ],
  "task_complete": false
}

Output ONLY valid JSON. Generate commands that would solve the task step by step."""


def load_tasks():
    tasks = []
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        instr_file = task_dir / "instruction.md"
        toml_file = task_dir / "task.toml"
        if not instr_file.exists():
            continue
        instruction = instr_file.read_text().strip()
        meta = {}
        if toml_file.exists():
            with open(toml_file, "rb") as f:
                meta = tomllib.load(f)
        tasks.append({
            "name": task_dir.name,
            "instruction": instruction,
            "category": meta.get("metadata", {}).get("category", "unknown"),
            "difficulty": meta.get("metadata", {}).get("difficulty", "unknown"),
        })
    return tasks


def query_model(instruction: str, max_retries=2):
    payload = {
        "model": "nemotron-terminal-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    for attempt in range(max_retries):
        try:
            start = time.time()
            resp = requests.post(VLLM_URL, json=payload, timeout=120)
            elapsed = time.time() - start
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "content": content,
                    "latency": round(elapsed, 2),
                    "tokens": usage.get("completion_tokens", 0),
                }
            else:
                return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}", "latency": round(elapsed, 2)}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return {"error": str(e), "latency": 0}
    return {"error": "max retries", "latency": 0}


def parse_response(content: str):
    """Check if the response is valid JSON with expected fields."""
    result = {"valid_json": False, "has_commands": False, "has_analysis": False,
              "task_complete_field": False, "num_commands": 0, "commands": []}
    try:
        # Try to find JSON in the response
        json_str = content
        if "{" in content:
            start = content.index("{")
            end = content.rfind("}") + 1
            json_str = content[start:end]
        data = json.loads(json_str)
        result["valid_json"] = True
        result["has_analysis"] = "analysis" in data
        result["task_complete_field"] = "task_complete" in data
        if "commands" in data and isinstance(data["commands"], list):
            result["has_commands"] = True
            result["num_commands"] = len(data["commands"])
            result["commands"] = [c.get("keystrokes", "") for c in data["commands"] if isinstance(c, dict)]
        if "plan" in data:
            result["has_plan"] = True
    except (json.JSONDecodeError, ValueError):
        pass
    return result


def evaluate():
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks from Terminal-Bench 2.0")

    results = []
    stats = {
        "total": len(tasks),
        "valid_json": 0,
        "has_commands": 0,
        "has_analysis": 0,
        "has_plan": 0,
        "task_complete_field": 0,
        "errors": 0,
        "total_latency": 0,
        "total_tokens": 0,
        "by_category": {},
        "by_difficulty": {},
    }

    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {task['name']} ({task['category']}, {task['difficulty']})...", end=" ", flush=True)
        resp = query_model(task["instruction"])

        if "error" in resp:
            print(f"ERROR: {resp['error'][:80]}")
            results.append({"task": task["name"], "category": task["category"],
                          "difficulty": task["difficulty"], "error": resp["error"]})
            stats["errors"] += 1
            continue

        parsed = parse_response(resp["content"])
        latency = resp["latency"]
        tokens = resp.get("tokens", 0)

        print(f"OK json={parsed['valid_json']} cmds={parsed['num_commands']} t={latency}s")

        result = {
            "task": task["name"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "valid_json": parsed["valid_json"],
            "has_commands": parsed["has_commands"],
            "has_analysis": parsed["has_analysis"],
            "has_plan": parsed.get("has_plan", False),
            "task_complete_field": parsed["task_complete_field"],
            "num_commands": parsed["num_commands"],
            "commands": parsed["commands"],
            "latency": latency,
            "tokens": tokens,
            "response": resp["content"][:500],
        }
        results.append(result)

        if parsed["valid_json"]:
            stats["valid_json"] += 1
        if parsed["has_commands"]:
            stats["has_commands"] += 1
        if parsed["has_analysis"]:
            stats["has_analysis"] += 1
        if parsed.get("has_plan"):
            stats["has_plan"] += 1
        if parsed["task_complete_field"]:
            stats["task_complete_field"] += 1
        stats["total_latency"] += latency
        stats["total_tokens"] += tokens

        cat = task["category"]
        if cat not in stats["by_category"]:
            stats["by_category"][cat] = {"total": 0, "valid_json": 0, "has_commands": 0}
        stats["by_category"][cat]["total"] += 1
        if parsed["valid_json"]:
            stats["by_category"][cat]["valid_json"] += 1
        if parsed["has_commands"]:
            stats["by_category"][cat]["has_commands"] += 1

        diff = task["difficulty"]
        if diff not in stats["by_difficulty"]:
            stats["by_difficulty"][diff] = {"total": 0, "valid_json": 0, "has_commands": 0}
        stats["by_difficulty"][diff]["total"] += 1
        if parsed["valid_json"]:
            stats["by_difficulty"][diff]["valid_json"] += 1
        if parsed["has_commands"]:
            stats["by_difficulty"][diff]["has_commands"] += 1

    # Summary
    n = stats["total"]
    summary = {
        "model": "nvidia/Nemotron-Terminal-8B",
        "benchmark": "Terminal-Bench 2.0",
        "framework": "vLLM 0.19.1 + custom evaluator (no Docker)",
        "total_tasks": n,
        "valid_json_pct": round(stats["valid_json"] / n * 100, 1),
        "has_commands_pct": round(stats["has_commands"] / n * 100, 1),
        "has_analysis_pct": round(stats["has_analysis"] / n * 100, 1),
        "has_plan_pct": round(stats["has_plan"] / n * 100, 1),
        "task_complete_field_pct": round(stats["task_complete_field"] / n * 100, 1),
        "avg_latency": round(stats["total_latency"] / max(n - stats["errors"], 1), 2),
        "total_tokens": stats["total_tokens"],
        "errors": stats["errors"],
        "by_category": stats["by_category"],
        "by_difficulty": stats["by_difficulty"],
        "note": "Without Docker, actual command execution cannot be verified. "
                "Metrics measure format compliance and output structure quality.",
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / "eval_details.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model:        {summary['model']}")
    print(f"Benchmark:    {summary['benchmark']}")
    print(f"Total tasks:  {summary['total_tasks']}")
    print(f"Valid JSON:   {summary['valid_json_pct']}%")
    print(f"Has commands: {summary['has_commands_pct']}%")
    print(f"Has analysis: {summary['has_analysis_pct']}%")
    print(f"Has plan:     {summary['has_plan_pct']}%")
    print(f"Errors:       {summary['errors']}")
    print(f"Avg latency:  {summary['avg_latency']}s")
    print()
    print("By Category:")
    for cat, s in sorted(stats["by_category"].items()):
        print(f"  {cat:30s} {s['total']:3d} tasks  JSON={s['valid_json']/s['total']*100:.0f}%  cmds={s['has_commands']/s['total']*100:.0f}%")
    print()
    print("By Difficulty:")
    for diff, s in sorted(stats["by_difficulty"].items()):
        print(f"  {diff:15s} {s['total']:3d} tasks  JSON={s['valid_json']/s['total']*100:.0f}%  cmds={s['has_commands']/s['total']*100:.0f}%")
    print()
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    evaluate()
