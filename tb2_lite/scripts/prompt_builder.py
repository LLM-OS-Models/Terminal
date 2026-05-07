from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptBuild:
    prompt: str
    status: str
    error: str | None = None


def sanitize_name(value: str) -> str:
    return value.rstrip("/").split("/")[-1].replace(" ", "-")


def row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in {"system", "user", "assistant", "tool"} and isinstance(content, str):
                normalized.append({"role": role, "content": content})
        if normalized:
            return normalized
    return [{"role": "user", "content": str(row.get("prompt", ""))}]


def build_prompt(tokenizer: Any, row: dict[str, Any]) -> PromptBuild:
    messages = row_messages(row)
    try:
        return PromptBuild(
            prompt=tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ),
            status="chat_template",
        )
    except Exception as exc:
        return PromptBuild(prompt=str(row.get("prompt", "")), status="raw_fallback", error=str(exc))


def build_prompts(tokenizer: Any, rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    builds = [build_prompt(tokenizer, row) for row in rows]
    counts = Counter(build.status for build in builds)
    errors = sorted({build.error for build in builds if build.error})[:3]
    return [build.prompt for build in builds], {
        "template_status_counts": dict(counts),
        "template_status": "chat_template" if counts == {"chat_template": len(builds)} else "mixed_or_raw",
        "rank_eligible": not counts.get("raw_fallback"),
        "raw_fallback_errors": errors,
    }
