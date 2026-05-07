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


def render_chatml(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = "assistant" if message["role"] == "assistant" else message["role"]
        if role == "tool":
            role = "user"
        parts.append(f"<|im_start|>{role}\n{message['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def render_gemma4_turn(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else message["role"]
        if role == "tool":
            role = "user"
        parts.append(f"<|turn|>{role}\n{message['content']}<|turn|>\n")
    parts.append("<|turn|>model\n")
    return "".join(parts)


def infer_fallback_style(model_name: str, tokenizer: Any) -> str | None:
    value = f"{model_name} {getattr(tokenizer, 'name_or_path', '')}".lower()
    if "gemma-4" in value:
        return "gemma4_turn"
    if any(marker in value for marker in ("qwen", "lfm", "nemotron-terminal", "ouro")):
        return "chatml"
    return None


def build_prompt(tokenizer: Any, row: dict[str, Any], model_name: str = "") -> PromptBuild:
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
        style = infer_fallback_style(model_name, tokenizer)
        if style == "chatml":
            return PromptBuild(prompt=render_chatml(messages), status="chatml_fallback", error=str(exc))
        if style == "gemma4_turn":
            return PromptBuild(prompt=render_gemma4_turn(messages), status="gemma4_fallback", error=str(exc))
        return PromptBuild(prompt=str(row.get("prompt", "")), status="raw_fallback", error=str(exc))


def build_prompts(tokenizer: Any, rows: list[dict[str, Any]], model_name: str = "") -> tuple[list[str], dict[str, Any]]:
    builds = [build_prompt(tokenizer, row, model_name=model_name) for row in rows]
    counts = Counter(build.status for build in builds)
    errors = sorted({build.error for build in builds if build.error})[:3]
    rank_eligible = not counts.get("raw_fallback")
    status = "chat_template" if counts == {"chat_template": len(builds)} else "model_specific_or_mixed"
    if counts.get("raw_fallback"):
        status = "mixed_or_raw"
    return [build.prompt for build in builds], {
        "template_status_counts": dict(counts),
        "template_status": status,
        "rank_eligible": rank_eligible,
        "raw_fallback_errors": errors,
    }
