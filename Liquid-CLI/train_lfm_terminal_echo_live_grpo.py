#!/usr/bin/env python
"""No-Docker live terminal ECHO-style RLVR.

This script is intentionally direct:
  1. sample terminal tasks,
  2. let the model issue shell commands over multiple turns,
  3. execute those commands in a local sandbox,
  4. feed stdout/stderr/exit-code observations back into the next turn,
  5. optimize LoRA adapters with verifier reward + ECHO observation CE.

It does not use Docker. Task archives are unpacked into per-rollout local
workspace/output/log directories. This is appropriate for our local experiments,
but it is weaker isolation than Harbor/Docker in the ECHO paper.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_SCRIPT = Path(__file__).resolve().with_name("train_lfm_terminal_live_rlvr_grpo.py")


def load_base_module():
    spec = importlib.util.spec_from_file_location("live_terminal_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_terminal_base"] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


SYSTEM_PROMPT = base.SYSTEM_PROMPT


@dataclass
class Trajectory:
    task_id: str
    source: str
    input_ids: list[int]
    action_mask: list[int]
    obs_mask: list[int]
    reward: float
    verifier_reward: float
    turns: int
    command_count: int
    trace: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch")
    parser.add_argument(
        "--output-dir",
        default="/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_r32",
    )
    parser.add_argument("--sandbox-root", default="/home/work/.data/liquid_cli_sft/live_terminal_echo/sandboxes")
    parser.add_argument("--trace-dir", default="/home/work/.data/liquid_cli_sft/live_terminal_echo/traces")
    parser.add_argument("--hf-cache-dir", default="/home/work/.data/liquid_cli_sft/live_terminal_rlvr/hf")
    parser.add_argument("--hf-rl-dataset", default="open-thoughts/OpenThoughts-Agent-v1-RL")
    parser.add_argument("--hf-rl-parquet", default="")
    parser.add_argument("--endless-repo", default="obiwan96/endless-terminals")
    parser.add_argument("--tb-dev-repo", default="open-thoughts/OpenThoughts-TB-dev")
    parser.add_argument("--tblite-repo", default="open-thoughts/OpenThoughts-TBLite")
    parser.add_argument("--include-openthoughts-rl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-endless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-tb-dev", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-tblite-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-task-dir", action="append", default=[])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=17)

    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--max-wall-time-hours",
        type=float,
        default=0.0,
        help="Gracefully stop after this many hours and save final_lora. 0 disables.",
    )
    parser.add_argument("--prompts-per-rank", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=0,
        help="Parallel rollout workers per rank for vLLM/terminal execution. 0 means num_generations.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=0.2)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.03)
    parser.add_argument("--command-timeout", type=float, default=20.0)
    parser.add_argument("--verifier-timeout", type=float, default=60.0)
    parser.add_argument("--max-commands-per-turn", type=int, default=3)
    parser.add_argument("--max-command-chars", type=int, default=4000)
    parser.add_argument("--max-terminal-output-chars", type=int, default=12000)
    parser.add_argument("--observation-role", choices=["user", "tool"], default="user")
    parser.add_argument("--rollout-backend", choices=["hf", "vllm_http"], default="hf")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8123/v1")
    parser.add_argument("--vllm-served-model", default="")
    parser.add_argument("--vllm-request-timeout", type=float, default=180.0)
    parser.add_argument("--vllm-stop", action="append", default=["<|im_end|>"])

    parser.add_argument("--world-model-coeff", type=float, default=0.05)
    parser.add_argument("--policy-coeff", type=float, default=1.0)
    parser.add_argument(
        "--single-generation-advantage",
        choices=["zero", "reward", "centered_reward"],
        default="centered_reward",
        help="Fallback policy advantage when num_generations=1 or group reward std is zero.",
    )
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-success-bonus", type=float, default=0.0)
    parser.add_argument("--format-penalty", type=float, default=0.05)
    parser.add_argument("--command-bonus", type=float, default=0.02)

    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate")
    parser.add_argument("--sft-adapter-path", default="")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-sandboxes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test-env", action="store_true")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        init_method = os.environ.get("DIST_INIT_METHOD", "")
        if init_method:
            dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
        else:
            dist.init_process_group("nccl")
    return rank, local_rank, world_size


def is_rank0(rank: int) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def common_prefix_len(a: list[int], b: list[int]) -> int:
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
    )


def apply_chat_template_text(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        ids = apply_chat_template(tokenizer, messages, add_generation_prompt=add_generation_prompt)
        return tokenizer.decode(ids, skip_special_tokens=False)


def append_message_with_mask(
    tokenizer: Any,
    messages: list[dict[str, str]],
    full_ids: list[int],
    action_mask: list[int],
    obs_mask: list[int],
    role: str,
    content: str,
    mask_kind: str,
) -> tuple[list[dict[str, str]], list[int], list[int], list[int]]:
    before = list(full_ids)
    new_messages = messages + [{"role": role, "content": content}]
    after = apply_chat_template(tokenizer, new_messages, add_generation_prompt=False)
    prefix = common_prefix_len(before, after)
    delta = after[prefix:]
    if prefix < len(before):
        # Chat templates can rewrite the final assistant turn when a new message
        # is appended. Rebuild from the full after sequence and keep existing
        # masks for the stable prefix only.
        full_ids = before[:prefix]
        action_mask = action_mask[:prefix]
        obs_mask = obs_mask[:prefix]
    full_ids.extend(delta)
    if mask_kind == "action":
        action_mask.extend([1] * len(delta))
        obs_mask.extend([0] * len(delta))
    elif mask_kind == "observation":
        action_mask.extend([0] * len(delta))
        obs_mask.extend([1] * len(delta))
    else:
        action_mask.extend([0] * len(delta))
        obs_mask.extend([0] * len(delta))
    return new_messages, full_ids, action_mask, obs_mask


def format_observation(events: list[dict[str, Any]], warnings: list[str]) -> str:
    parts: list[str] = []
    if warnings:
        parts.append("WARNINGS:\n" + "\n".join(f"- {w}" for w in warnings))
    for i, event in enumerate(events, 1):
        parts.append(
            "\n".join(
                [
                    f"COMMAND {i}: {event.get('raw_command') or event.get('command')}",
                    f"EXIT_CODE: {event.get('exit_code')}",
                    "STDOUT:",
                    str(event.get("stdout") or ""),
                    "STDERR:",
                    str(event.get("stderr") or ""),
                ]
            ).strip()
        )
    if not parts:
        return "No command was executed. Provide valid shell commands or mark the task complete."
    return "TERMINAL OUTPUT:\n" + "\n\n".join(parts)


def run_commands_in_sandbox(commands: list[str], sandbox: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], bool]:
    events: list[dict[str, Any]] = []
    blocked = False
    for raw_command in commands[: args.max_commands_per_turn]:
        if len(raw_command) > args.max_command_chars:
            events.append({"blocked": True, "reason": "command_too_long", "raw_command": raw_command[:500]})
            blocked = True
            break
        if base.is_unsafe_command(raw_command):
            events.append({"blocked": True, "reason": "unsafe_pattern", "raw_command": raw_command[:500]})
            blocked = True
            break
        command = base.rewrite_paths(raw_command, sandbox)
        event = base.run_subprocess(command, sandbox / "workspace", args.command_timeout)
        event["raw_command"] = raw_command
        events.append(event)
        if event.get("timeout"):
            break
    return events, blocked


@torch.no_grad()
def generate_assistant_text_vllm_http(
    tokenizer: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    seed: int,
) -> str:
    prompt = apply_chat_template_text(tokenizer, messages, add_generation_prompt=True)
    payload: dict[str, Any] = {
        "model": args.vllm_served_model or args.model_path,
        "prompt": prompt,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "seed": seed,
    }
    stops = [s for s in args.vllm_stop if s]
    if stops:
        payload["stop"] = stops
    request = urllib.request.Request(
        args.vllm_base_url.rstrip("/") + "/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.vllm_request_timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {body[:2000]}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"vLLM response had no choices: {data}")
    return str(choices[0].get("text") or "").strip()


@torch.no_grad()
def generate_assistant_text(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> str:
    if args.rollout_backend == "vllm_http":
        return generate_assistant_text_vllm_http(tokenizer, messages, args, seed)

    prompt_ids = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    # Some remote-code model classes reject the `generator=` kwarg in
    # `generate`. Seed the active CUDA RNG instead; this is sufficient for
    # independent rollout samples in our local on-policy loop.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    old_use_cache = getattr(model.config, "use_cache", None)
    if old_use_cache is not None:
        model.config.use_cache = True
    try:
        output = model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
    new_ids = output[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text.strip()


def rollout_one(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    rollout_seed: int,
) -> Trajectory:
    task_id = str(row["task_id"])
    source = str(row["source"])
    sandbox: Path | None = None
    messages = list(row["prompt"])
    full_ids = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    action_mask = [0] * len(full_ids)
    obs_mask = [0] * len(full_ids)
    trace: dict[str, Any] = {
        "task_id": task_id,
        "source": source,
        "turns": [],
        "rollout_seed": rollout_seed,
        "rank": rank,
    }
    command_count = 0
    blocked_any = False
    parse_errors = 0
    verifier_reward = 0.0
    reward = 0.0

    try:
        sandbox, _task_root = base.setup_sandbox(
            str(row.get("task_binary_b64", "")),
            str(row.get("task_dir", "")),
            re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:80],
        )
        for turn in range(args.max_turns):
            if len(full_ids) >= args.max_seq_length - args.max_new_tokens - 256:
                trace["stop_reason"] = "max_seq_length"
                break
            text = generate_assistant_text(
                model,
                tokenizer,
                messages,
                args,
                device,
                seed=rollout_seed + turn * 7919,
            )
            messages, full_ids, action_mask, obs_mask = append_message_with_mask(
                tokenizer, messages, full_ids, action_mask, obs_mask, "assistant", text, "action"
            )
            commands, is_done, parse_ok, warnings = base.parse_commands(text)
            if not parse_ok:
                parse_errors += 1
                obs_text = "FORMAT ERROR:\nNo valid terminal command was parsed. Use the requested shell-command format."
                messages, full_ids, action_mask, obs_mask = append_message_with_mask(
                    tokenizer, messages, full_ids, action_mask, obs_mask, args.observation_role, obs_text, "observation"
                )
                trace["turns"].append({"turn": turn, "assistant": text[:4000], "parse_ok": False, "warnings": warnings})
                continue
            if is_done:
                trace["turns"].append({"turn": turn, "assistant": text[:4000], "parse_ok": True, "done": True})
                trace["stop_reason"] = "done"
                break
            events, blocked = run_commands_in_sandbox(commands, sandbox, args)
            command_count += len([e for e in events if not e.get("blocked")])
            blocked_any = blocked_any or blocked
            obs_text = format_observation(events, warnings)
            if len(obs_text) > args.max_terminal_output_chars:
                obs_text = obs_text[-args.max_terminal_output_chars :]
            messages, full_ids, action_mask, obs_mask = append_message_with_mask(
                tokenizer, messages, full_ids, action_mask, obs_mask, args.observation_role, obs_text, "observation"
            )
            trace["turns"].append(
                {
                    "turn": turn,
                    "assistant": text[:4000],
                    "parse_ok": True,
                    "done": False,
                    "commands": commands[: args.max_commands_per_turn],
                    "events": events,
                }
            )
        else:
            trace["stop_reason"] = "max_turns"

        verifier_reward, verifier_event = base.run_verifier(sandbox)
        reward = verifier_reward * args.reward_scale
        if verifier_reward >= 1.0:
            reward += args.reward_success_bonus
        reward += min(command_count, 3) * args.command_bonus
        reward -= parse_errors * args.format_penalty
        if blocked_any:
            reward -= 0.25
        reward = float(max(-1.0, min(2.0, reward)))
        trace["verifier_reward"] = verifier_reward
        trace["reward"] = reward
        trace["verifier"] = verifier_event
        trace["command_count"] = command_count
        trace["parse_errors"] = parse_errors
        trace["blocked"] = blocked_any
    except Exception as exc:
        trace["exception"] = f"{type(exc).__name__}: {exc}"
        reward = -0.5
    finally:
        if sandbox is not None and not args.keep_sandboxes:
            shutil.rmtree(sandbox, ignore_errors=True)

    if len(full_ids) > args.max_seq_length:
        full_ids = full_ids[-args.max_seq_length :]
        action_mask = action_mask[-args.max_seq_length :]
        obs_mask = obs_mask[-args.max_seq_length :]

    return Trajectory(
        task_id=task_id,
        source=source,
        input_ids=full_ids,
        action_mask=action_mask,
        obs_mask=obs_mask,
        reward=reward,
        verifier_reward=verifier_reward,
        turns=len(trace.get("turns", [])),
        command_count=command_count,
        trace=trace,
    )


def trajectory_loss(
    model: Any,
    traj: Trajectory,
    advantage: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    ids = torch.tensor(traj.input_ids, dtype=torch.long, device=device).unsqueeze(0)
    action_mask = torch.tensor(traj.action_mask, dtype=torch.float, device=device).unsqueeze(0)
    obs_mask = torch.tensor(traj.obs_mask, dtype=torch.float, device=device).unsqueeze(0)
    if ids.shape[1] < 3:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"policy_loss": 0.0, "world_loss": 0.0, "action_tokens": 0.0, "obs_tokens": 0.0}

    out = model(input_ids=ids[:, :-1], use_cache=False)
    logits = out.logits
    labels = ids[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    action_next = action_mask[:, 1:]
    obs_next = obs_mask[:, 1:]
    action_tokens = action_next.sum().clamp_min(1.0)
    obs_tokens = obs_next.sum().clamp_min(1.0)

    adv = torch.tensor(float(advantage), device=device)
    policy_loss = -(token_logp * action_next).sum() / action_tokens * adv
    world_loss = -(token_logp * obs_next).sum() / obs_tokens
    loss = args.policy_coeff * policy_loss + args.world_model_coeff * world_loss
    if args.entropy_coeff:
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        loss = loss - args.entropy_coeff * (entropy * action_next).sum() / action_tokens
    return loss, {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "world_loss": float(world_loss.detach().cpu().item()),
        "action_tokens": float(action_next.sum().detach().cpu().item()),
        "obs_tokens": float(obs_next.sum().detach().cpu().item()),
    }


def append_trace(trace_dir: Path, rank: int, trace: dict[str, Any]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / f"echo_live_rollouts_rank{rank}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")


def select_rows(dataset: Any, step: int, rank: int, world_size: int, prompts_per_rank: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = step * world_size * prompts_per_rank + rank * prompts_per_rank
    for i in range(prompts_per_rank):
        rows.append(dataset[(start + i) % len(dataset)])
    return rows


def rollout_group(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    step: int,
    row_i: int,
    trace_dir: Path,
) -> list[Trajectory]:
    work: list[tuple[int, int]] = []
    for gen_i in range(args.num_generations):
        seed = args.seed + step * 1_000_003 + rank * 10_007 + row_i * 101 + gen_i
        work.append((gen_i, seed))

    def run_one(item: tuple[int, int]) -> tuple[int, Trajectory]:
        gen_i, seed = item
        traj = rollout_one(model, tokenizer, row, args, device, rank, seed)
        append_trace(trace_dir, rank, traj.trace)
        return gen_i, traj

    if args.rollout_backend == "vllm_http" and len(work) > 1:
        workers = args.rollout_workers or args.num_generations
        workers = max(1, min(workers, len(work)))
        results: list[tuple[int, Trajectory]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, item) for item in work]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item[0])
        return [traj for _gen_i, traj in results]

    return [run_one(item)[1] for item in work]


def set_base_globals(args: argparse.Namespace) -> None:
    base.GLOBAL_CONFIG.update(
        {
            "sandbox_root": args.sandbox_root,
            "trace_dir": args.trace_dir,
            "command_timeout": args.command_timeout,
            "verifier_timeout": args.verifier_timeout,
            "max_commands_per_completion": args.max_commands_per_turn,
            "max_command_chars": args.max_command_chars,
            "max_terminal_output_chars": args.max_terminal_output_chars,
            "keep_sandboxes": args.keep_sandboxes,
        }
    )


def save_checkpoint(model: Any, output_dir: Path, name: str, rank: int, tokenizer: Any) -> None:
    if rank != 0:
        return
    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    module.save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))


def log_event(rank: int, event: str, **payload: Any) -> None:
    if rank != 0:
        return
    message = {"event": event, **payload}
    print(json.dumps(message, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    wall_start = time.monotonic()
    rank, local_rank, world_size = setup_distributed()
    set_base_globals(args)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    output_dir = Path(args.output_dir)
    trace_dir = Path(args.trace_dir)
    if is_rank0(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)

    log_event(rank, "startup", world_size=world_size, local_rank=local_rank, device=str(device))
    log_event(rank, "load_tokenizer_start", model_path=args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log_event(rank, "load_tokenizer_done", pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    log_event(rank, "build_dataset_start")
    dataset = base.build_dataset(tokenizer, args)
    log_event(rank, "build_dataset_done", rows=len(dataset))
    if args.smoke_test_env:
        row = dataset[0]
        sandbox, _ = base.setup_sandbox(str(row.get("task_binary_b64", "")), str(row.get("task_dir", "")), "smoke")
        event = base.run_subprocess("pwd && ls -la && find . -maxdepth 2 -type f | sort", sandbox / "workspace", 20)
        verifier_reward, verifier_event = base.run_verifier(sandbox)
        if not args.keep_sandboxes:
            shutil.rmtree(sandbox, ignore_errors=True)
        if is_rank0(rank):
            print(json.dumps({"event": "smoke_env", "cmd": event, "verifier_reward": verifier_reward, "verifier": verifier_event}, ensure_ascii=False, indent=2))
    if args.dry_run:
        if is_rank0(rank):
            print(json.dumps({"event": "dry_run_ok", "rows": len(dataset)}, ensure_ascii=False))
        return

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    log_event(rank, "load_model_start", model_path=args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    log_event(rank, "load_model_done")
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    log_event(rank, "model_to_device_start", device=str(device))
    model.to(device)
    log_event(rank, "model_to_device_done")
    if args.sft_adapter_path:
        log_event(rank, "load_adapter_start", adapter_path=args.sft_adapter_path)
        model = PeftModel.from_pretrained(model, args.sft_adapter_path, is_trainable=True)
        log_event(rank, "load_adapter_done")
    else:
        log_event(rank, "create_lora_start", lora_rank=args.lora_rank, target_modules=args.target_modules)
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=args.lora_rank,
                target_modules=base.parse_target_modules(args.target_modules),
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
            ),
        )
        log_event(rank, "create_lora_done")
    for candidate in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        if candidate is not None and not hasattr(candidate, "warnings_issued"):
            candidate.warnings_issued = {}
    if world_size > 1:
        log_event(rank, "ddp_wrap_start", world_size=world_size)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        log_event(rank, "ddp_wrap_done")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    log_event(rank, "optimizer_ready", trainable_parameters=sum(p.numel() for p in trainable))

    if is_rank0(rank):
        print(
            json.dumps(
                {
                    "task": "echo_live_terminal_grpo_no_docker",
                    "paper_reference": "ECHO objective: verifier RL + CE on terminal observation tokens.",
                    "model_path": args.model_path,
                    "dataset_rows": len(dataset),
                    "output_dir": str(output_dir),
                    "world_size": world_size,
                    "prompts_per_rank": args.prompts_per_rank,
                    "num_generations": args.num_generations,
                    "rollout_workers": args.rollout_workers or args.num_generations,
                    "global_rollouts_per_step": world_size * args.prompts_per_rank * args.num_generations,
                    "max_steps": args.max_steps,
                    "max_wall_time_hours": args.max_wall_time_hours,
                    "world_model_coeff": args.world_model_coeff,
                    "learning_rate": args.learning_rate,
                    "trace_dir": str(trace_dir),
                    "no_docker": True,
                    "rollout_backend": args.rollout_backend,
                    "vllm_base_url": args.vllm_base_url if args.rollout_backend == "vllm_http" else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    global_update = 0
    for step in range(args.max_steps):
        model.eval()
        rows = select_rows(dataset, step, rank, world_size, args.prompts_per_rank)
        all_trajs: list[Trajectory] = []
        for row_i, row in enumerate(rows):
            group = rollout_group(
                model.module if hasattr(model, "module") else model,
                tokenizer,
                row,
                args,
                device,
                rank,
                step,
                row_i,
                trace_dir,
            )
            rewards = torch.tensor([t.reward for t in group], dtype=torch.float)
            mean = float(rewards.mean().item())
            std = float(rewards.std(unbiased=False).item())
            for traj in group:
                traj.trace["group_reward_mean"] = mean
                traj.trace["group_reward_std"] = std
                all_trajs.append(traj)

        model.train()
        step_loss_value = 0.0
        metrics_sum = {"policy_loss": 0.0, "world_loss": 0.0, "action_tokens": 0.0, "obs_tokens": 0.0}
        rewards_by_prompt: list[list[float]] = []
        cursor = 0
        for _row in rows:
            group = all_trajs[cursor : cursor + args.num_generations]
            cursor += args.num_generations
            rewards_by_prompt.append([t.reward for t in group])
            values = torch.tensor([t.reward for t in group], dtype=torch.float)
            mean = float(values.mean().item())
            std = float(values.std(unbiased=False).item())
            denom = std if std > 1e-6 else 1.0
            for traj in group:
                if len(group) > 1 and std > 1e-6:
                    advantage = (traj.reward - mean) / denom
                elif args.single_generation_advantage == "reward":
                    advantage = traj.reward
                elif args.single_generation_advantage == "centered_reward":
                    advantage = traj.reward - 0.5
                else:
                    advantage = 0.0
                loss, lm = trajectory_loss(model, traj, advantage, args, device)
                scaled_loss = loss / max(len(all_trajs), 1) / args.gradient_accumulation_steps
                step_loss_value += float(scaled_loss.detach().cpu().item())
                scaled_loss.backward()
                for k in metrics_sum:
                    metrics_sum[k] += lm[k]
                del loss, scaled_loss

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            if global_update < args.warmup_steps:
                lr_scale = float(global_update + 1) / max(args.warmup_steps, 1)
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * lr_scale
            else:
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1

        local_reward_mean = sum(t.reward for t in all_trajs) / max(len(all_trajs), 1)
        global_reward_mean = all_reduce_mean(local_reward_mean, device)
        local_vreward_mean = sum(t.verifier_reward for t in all_trajs) / max(len(all_trajs), 1)
        global_vreward_mean = all_reduce_mean(local_vreward_mean, device)

        if is_rank0(rank) and (step % args.logging_steps == 0 or step == args.max_steps - 1):
            denom = max(len(all_trajs), 1)
            log = {
                "event": "train_step",
                "step": step,
                "loss": step_loss_value,
                "reward_mean": global_reward_mean,
                "verifier_reward_mean": global_vreward_mean,
                "local_reward_groups": rewards_by_prompt[:4],
                "policy_loss_mean": metrics_sum["policy_loss"] / denom,
                "world_loss_mean": metrics_sum["world_loss"] / denom,
                "action_tokens_mean": metrics_sum["action_tokens"] / denom,
                "obs_tokens_mean": metrics_sum["obs_tokens"] / denom,
                "lr": optimizer.param_groups[0]["lr"],
            }
            print(json.dumps(log, ensure_ascii=False), flush=True)

        if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
            save_checkpoint(model, output_dir, f"checkpoint-{step + 1}", rank, tokenizer)
            barrier()

        if args.max_wall_time_hours > 0:
            elapsed_hours = (time.monotonic() - wall_start) / 3600.0
            stop_tensor = torch.tensor(
                1 if elapsed_hours >= args.max_wall_time_hours else 0,
                dtype=torch.int,
                device=device,
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            if int(stop_tensor.item()) == 1:
                log_event(
                    rank,
                    "wall_time_stop",
                    step=step,
                    elapsed_hours=elapsed_hours,
                    max_wall_time_hours=args.max_wall_time_hours,
                )
                break

    save_checkpoint(model, output_dir, "final_lora", rank, tokenizer)
    barrier()
    if is_rank0(rank):
        (output_dir / "TRAINING_NOTES.txt").write_text(
            "\n".join(
                [
                    "ECHO live terminal RLVR LoRA adapter.",
                    "No Docker was used.",
                    "The model generated commands, commands were executed in local sandboxes,",
                    "terminal observations were fed back into later turns, and observation",
                    "tokens received auxiliary CE loss with verifier reward policy loss.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"event": "training_complete", "output_dir": str(output_dir / "final_lora")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
