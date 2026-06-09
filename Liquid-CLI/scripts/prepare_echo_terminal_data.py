#!/usr/bin/env python
"""Prepare ECHO-style terminal task data for local LFM RLVR.

The ECHO paper trains from Harbor/Docker task archives with rows shaped as:

    prompt, path, task_binary

Our no-Docker trainer executes the same task archives by unpacking them into a
local sandbox, so this script also emits:

    prompt, task_id, source, task_binary_b64, task_dir

The two outputs let us keep the data faithful to the ECHO schema while making
it immediately usable by the local LFM terminal ECHO trainer.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import os
import re
import sys
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]
ECHO_RL_DIR = REPO_DIR / "echo-rl"
if str(ECHO_RL_DIR) not in sys.path:
    sys.path.insert(0, str(ECHO_RL_DIR))


LFM_SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You can use bash commands to inspect files, transform data, write outputs, and verify your work.

Return JSON only, with this structure:

{
  "analysis": "Briefly describe the current state and your plan.",
  "commands": [
    {"keystrokes": "ONE_NON_INTERACTIVE_BASH_COMMAND\\n", "duration": 0.1}
  ],
  "task_complete": false
}

Rules:
- Use non-interactive commands only.
- Prefer one robust shell command or a short command batch.
- Commands execute from /workspace. Required outputs usually belong in /output.
- stdout, stderr, exit codes, files, and verifier results are used as training feedback.
- Do not touch files outside /workspace, /output, and /logs.
- Do not run GPU/CUDA/NVIDIA probes such as nvidia-smi, torch.cuda, nvcc, or
  CUDA_VISIBLE_DEVICES checks. GPUs are unavailable inside the task sandbox.
"""


QWEN35_INSTRUCTION_PREFIX = """You are given a task description and your goal is to solve the task by using shell commands and python code.
Start each response with a <think>...</think> section where you analyze the current state based on the terminal output and describe your plan for the next steps. Then, provide your commands using bash tool calls that specify the commands to execute. When you determine that the task is complete, use the done tool call to indicate completion.

Required tools:
- "bash": Call this to execute the specified bash command in the terminal and return the output to you in the next turn as terminal_output. You can use this to iteratively solve the task by analyzing the terminal output after each command. Example: <tool_call>
<function=bash>
<parameter=command>
....
</parameter>
<parameter=timeout>
....
</parameter>
</function>
</tool_call>

Optional tools:
- "done": Call this when you have verified the task is solved. Example: <tool_call>
<function=done>
</function>
</tool_call>

IMPORTANT:
- Only use <think> or <tool_call> to structure your response as described above.
- Exactly one <think> ... </think> block should be present. Failure to follow the response format will lead to parsing errors.
- Bash tool calls will be executed sequentially in the order they appear in the response.
- Each command will be executed in the same shell environment, so you can rely on side effects (e.g. file creation) across commands in the same batch.
- Each shell command string will be used completely verbatim. Write commands exactly as you want them sent to the terminal.
- Do not include extra whitespace before or after the commands unless it's part of the intended command
- The done tool can be included optionally at the end of the response if you determine that the task is complete. If included, it should be the last tool_call in the response. If not included, it is assumed to be false and the agent will continue with another turn.
- You may use the bash tool to verify and validate that the task is complete before calling the done tool. Verify that commands were executed successfully and that the expected output is present before calling done. The done tool should ONLY be called once you are confident that the task is fully solved."""


SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
SKIP_FILE_PATTERNS = [
    re.compile(r".*_summary\.json$"),
    re.compile(r".*leaderboard.*\.jsonl$"),
]


@dataclass
class SourceSummary:
    source: str
    rows: int
    percent: float
    skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--openthoughts-parquet",
        default="/home/work/.data/echo_terminal_data/raw/openthoughts_agent_v1_rl/tasks.parquet",
    )
    parser.add_argument(
        "--endless-root",
        default="/home/work/.data/echo_terminal_data/raw/endless_terminals",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/work/.data/echo_terminal_data/prepared",
    )
    parser.add_argument("--max-endless", type=int, default=0, help="0 means all available downloaded task dirs.")
    parser.add_argument("--max-openthoughts", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--tokenizer", default="", help="Optional tokenizer id/path for prompt token counts and ECHO tool system text.")
    parser.add_argument("--max-prompt-tokens", type=int, default=0, help="0 disables token filtering.")
    parser.add_argument("--emit-solution-reference", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_tokenizer(tokenizer_ref: str) -> Any | None:
    if not tokenizer_ref:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=True)


def b64_encode(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def make_lfm_prompt(instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": LFM_SYSTEM_PROMPT},
        {"role": "user", "content": f"Task:\n{instruction.strip()}"},
    ]


def make_echo_prompt(instruction: str, tokenizer: Any | None) -> list[dict[str, str]]:
    system_content = "You are a highly capable Linux terminal agent. Complete the user's task by running commands and verifying the result. When the task is complete, call done."
    if tokenizer is not None:
        try:
            from echo_rl.terminal_agent.tools import get_augmented_system_content

            system_content = get_augmented_system_content(tokenizer, "qwen35")
        except Exception:
            pass
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": QWEN35_INSTRUCTION_PREFIX + "\n\nTask: " + instruction.strip()},
    ]


def token_count(tokenizer: Any | None, prompt: list[dict[str, str]]) -> int | None:
    if tokenizer is None:
        return None
    try:
        ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
        return len(ids)
    except Exception:
        text = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in prompt)
        return len(tokenizer(text, add_special_tokens=False).input_ids)


def extract_instruction_from_tar(payload: bytes) -> str:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        names = tar.getnames()
        candidates = [name for name in names if name.endswith("instruction.md")]
        if not candidates:
            return ""
        extracted = tar.extractfile(tar.getmember(candidates[0]))
        if extracted is None:
            return ""
        return extracted.read().decode("utf-8", "replace").strip()


def extract_solution_from_tar(payload: bytes) -> str:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        names = tar.getnames()
        candidates = [name for name in names if name.endswith("solution/solve.sh")]
        if not candidates:
            return ""
        extracted = tar.extractfile(tar.getmember(candidates[0]))
        if extracted is None:
            return ""
        return extracted.read().decode("utf-8", "replace").strip()


def iter_endless_task_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if (root / "instruction.md").exists() and (root / "task.toml").exists():
        return [root]
    dirs = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "instruction.md").exists() and (path / "task.toml").exists():
            dirs.append(path)
    return dirs


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    rel = path.as_posix()
    return any(pattern.match(rel) for pattern in SKIP_FILE_PATTERNS)


def tar_task_dir(task_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for path in sorted(task_dir.rglob("*")):
                if should_skip(path.relative_to(task_dir)):
                    continue
                arcname = path.relative_to(task_dir).as_posix()
                info = tar.gettarinfo(str(path), arcname=arcname)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                if path.is_file():
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
                else:
                    tar.addfile(info)
    return buffer.getvalue()


def add_record(
    echo_rows: list[dict[str, Any]],
    lfm_rows: list[dict[str, Any]],
    source: str,
    task_id: str,
    task_binary: bytes,
    tokenizer: Any | None,
    max_prompt_tokens: int,
    solution_references: list[dict[str, Any]],
    emit_solution_reference: bool,
) -> bool:
    instruction = extract_instruction_from_tar(task_binary)
    if not instruction:
        return False
    lfm_prompt = make_lfm_prompt(instruction)
    echo_prompt = make_echo_prompt(instruction, tokenizer)
    lfm_tokens = token_count(tokenizer, lfm_prompt)
    echo_tokens = token_count(tokenizer, echo_prompt)
    if max_prompt_tokens and lfm_tokens is not None and lfm_tokens > max_prompt_tokens:
        return False
    path = f"{source}/{task_id}"
    echo_rows.append(
        {
            "prompt": echo_prompt,
            "path": path,
            "task_binary": task_binary,
            "instruction": instruction,
            "source": source,
            "_data_source": source,
            "prompt_tokens": echo_tokens,
        }
    )
    lfm_rows.append(
        {
            "prompt": lfm_prompt,
            "task_id": task_id,
            "source": source,
            "task_dir": "",
            "task_binary_b64": b64_encode(task_binary),
            "instruction": instruction,
            "echo_path": path,
            "prompt_tokens": lfm_tokens,
        }
    )
    if emit_solution_reference:
        solution = extract_solution_from_tar(task_binary)
        if solution:
            solution_references.append(
                {
                    "task_id": task_id,
                    "source": source,
                    "echo_path": path,
                    "instruction": instruction,
                    "solution_solve_sh": solution,
                    "note": "Reference solution script from task archive. Use for optional SFT/analysis, not as an on-policy RL trajectory.",
                }
            )
    return True


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_summaries(source_counts: Counter[str], skipped: Counter[str]) -> list[SourceSummary]:
    total = sum(source_counts.values())
    summaries = []
    for source, rows in sorted(source_counts.items()):
        percent = (rows / total * 100.0) if total else 0.0
        summaries.append(SourceSummary(source=source, rows=rows, percent=round(percent, 4), skipped=skipped[source]))
    return summaries


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)

    echo_rows: list[dict[str, Any]] = []
    lfm_rows: list[dict[str, Any]] = []
    solution_refs: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    openthoughts_path = Path(args.openthoughts_parquet)
    if openthoughts_path.exists():
        df = pd.read_parquet(openthoughts_path)
        if args.max_openthoughts:
            df = df.head(args.max_openthoughts)
        for row in df.itertuples(index=False):
            task_binary = bytes(getattr(row, "task_binary"))
            task_id = str(getattr(row, "path"))
            ok = add_record(
                echo_rows,
                lfm_rows,
                "openthoughts_agent_v1_rl",
                task_id,
                task_binary,
                tokenizer,
                args.max_prompt_tokens,
                solution_refs,
                args.emit_solution_reference,
            )
            source_counts["openthoughts_agent_v1_rl" if ok else "_skipped"] += int(ok)
            skipped["openthoughts_agent_v1_rl"] += int(not ok)

    endless_root = Path(args.endless_root)
    endless_dirs = iter_endless_task_dirs(endless_root)
    if args.max_endless:
        endless_dirs = endless_dirs[: args.max_endless]
    for task_dir in endless_dirs:
        try:
            task_binary = tar_task_dir(task_dir)
            ok = add_record(
                echo_rows,
                lfm_rows,
                "endless_terminals",
                task_dir.name,
                task_binary,
                tokenizer,
                args.max_prompt_tokens,
                solution_refs,
                args.emit_solution_reference,
            )
        except Exception:
            ok = False
        source_counts["endless_terminals" if ok else "_skipped"] += int(ok)
        skipped["endless_terminals"] += int(not ok)

    echo_df = pd.DataFrame(echo_rows)
    lfm_df = pd.DataFrame(lfm_rows)

    echo_parquet = output_dir / "echo_terminal_tasks_mixed.parquet"
    openthoughts_echo = output_dir / "echo_terminal_tasks_openthoughts.parquet"
    endless_echo = output_dir / "echo_terminal_tasks_endless.parquet"
    lfm_jsonl = output_dir / "lfm_live_tasks_mixed.jsonl"
    lfm_parquet = output_dir / "lfm_live_tasks_mixed.parquet"
    solution_jsonl = output_dir / "solution_references_mixed.jsonl"
    manifest_path = output_dir / "manifest.json"

    if not echo_df.empty:
        echo_df.to_parquet(echo_parquet, index=False)
        for source, path in [
            ("openthoughts_agent_v1_rl", openthoughts_echo),
            ("endless_terminals", endless_echo),
        ]:
            subset = echo_df[echo_df["source"] == source]
            if not subset.empty:
                subset.to_parquet(path, index=False)
    if not lfm_df.empty:
        lfm_df.to_parquet(lfm_parquet, index=False)
        write_jsonl(lfm_jsonl, lfm_rows)
    if solution_refs:
        write_jsonl(solution_jsonl, solution_refs)

    summaries = source_summaries(Counter(row["source"] for row in echo_rows), skipped)
    manifest = {
        "created_by": "Liquid-CLI/scripts/prepare_echo_terminal_data.py",
        "paper": "ECHO: Terminal Agents Learn World Models for Free, arXiv:2605.24517v1",
        "paper_training_corpus_note": "The paper reports 8870 tasks: 1977 Endless Terminals, 723 OpenThoughts-Agent-v1-RL, and 6170 additional modified-Endless generated tasks; train8770/val100.",
        "local_scope_note": "This manifest covers public/local data available in this workspace. The paper's private generated 6170-task Harbor export is not included unless separately supplied.",
        "outputs": {
            "echo_mixed_parquet": str(echo_parquet),
            "echo_openthoughts_parquet": str(openthoughts_echo),
            "echo_endless_parquet": str(endless_echo),
            "lfm_mixed_jsonl": str(lfm_jsonl),
            "lfm_mixed_parquet": str(lfm_parquet),
            "solution_references_jsonl": str(solution_jsonl),
        },
        "inputs": {
            "openthoughts_parquet": str(openthoughts_path),
            "endless_root": str(endless_root),
        },
        "row_count": len(echo_rows),
        "source_summaries": [asdict(summary) for summary in summaries],
        "skipped": dict(skipped),
        "tokenizer": args.tokenizer,
        "max_prompt_tokens": args.max_prompt_tokens,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
