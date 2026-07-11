#!/usr/bin/env python3
"""Validate the three local LFM2.5 vLLM model IDs and JSON agent output."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000/v1").rstrip("/")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
EXPECTED_MODELS = (
    "lfm25-sft1",
    "lfm25-static-610",
    "lfm25-online-425",
)

TERMINUS_SMOKE_PROMPT = r"""
You are a terminal agent. Return one JSON object with these required fields:
analysis (string), plan (string), commands (array). Each command must contain
keystrokes (string ending in \n) and duration (number). task_complete is an
optional boolean. Do not use native function calling.

Task: Create /tmp/hello.txt containing exactly hello followed by a newline,
then print the file so it can be verified.

Current terminal state:
root@task:/app#
""".strip()


def api_request(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    request = urllib.request.Request(f"{API_BASE}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(f"vLLM HTTP {error.code}: {body[:500]}") from error


def first_balanced_json(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object, tolerating LFM <think> text."""
    start = -1
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                value = json.loads(text[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("first JSON value is not an object")
                return value
    raise ValueError("no balanced JSON object found")


def validate_agent_json(value: dict[str, Any]) -> None:
    for field in ("analysis", "plan", "commands"):
        if field not in value:
            raise ValueError(f"missing required field: {field}")
    if not isinstance(value["analysis"], str) or not isinstance(value["plan"], str):
        raise ValueError("analysis and plan must be strings")
    if not isinstance(value["commands"], list):
        raise ValueError("commands must be an array")
    if not value["commands"]:
        raise ValueError("commands must contain at least one command")
    for index, command in enumerate(value["commands"]):
        if not isinstance(command, dict):
            raise ValueError(f"command {index} is not an object")
        keystrokes = command.get("keystrokes")
        if not isinstance(keystrokes, str) or not keystrokes.endswith("\n"):
            raise ValueError(f"command {index} keystrokes must end in newline")
        if not isinstance(command.get("duration", 1.0), (int, float)):
            raise ValueError(f"command {index} duration must be numeric")


def main() -> int:
    model_response = api_request("/models")
    available = {
        entry["id"]
        for entry in model_response.get("data", [])
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    missing = sorted(set(EXPECTED_MODELS) - available)
    if missing:
        raise RuntimeError(f"missing served model IDs: {missing}; available={sorted(available)}")

    print(f"endpoint={API_BASE} models=ok")
    for model in EXPECTED_MODELS:
        response = api_request(
            "/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": TERMINUS_SMOKE_PROMPT}],
                "temperature": 0,
                "max_tokens": 4096,
            },
        )
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise RuntimeError(f"{model}: finish_reason={choice.get('finish_reason')!r}")
        content = choice["message"].get("content") or ""
        parsed = first_balanced_json(content)
        validate_agent_json(parsed)
        usage = response.get("usage") or {}
        print(
            f"model={model} finish=stop json=ok commands={len(parsed['commands'])} "
            f"completion_tokens={usage.get('completion_tokens', 'unknown')}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
