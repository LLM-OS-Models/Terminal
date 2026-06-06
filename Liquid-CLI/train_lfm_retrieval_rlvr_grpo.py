#!/usr/bin/env python
"""GRPO RLVR for Harness-1 style retrieval curation on LFM2.5.

The reward is fully programmatic: parse `curated_doc_ids` from the completion,
compare with gold/evidence document IDs, and reward high-recall curation. This
keeps the RL loop independent from OpenAI/Tinker while preserving the core
Harness-1 learning target: find evidence, keep the right documents, and avoid
dropping found answer docs.
"""

from __future__ import annotations

import argparse
import builtins
import importlib.machinery
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if not hasattr(builtins, "PreTrainedConfig"):
    if hasattr(transformers, "PreTrainedConfig"):
        builtins.PreTrainedConfig = transformers.PreTrainedConfig
    elif hasattr(transformers, "PretrainedConfig"):
        builtins.PreTrainedConfig = transformers.PretrainedConfig

# TRL imports judge callbacks on GRPO import. The local RLVR path does not use
# pairwise judges, and the optional llm-blender package is incompatible with the
# installed Transformers version, so provide the tiny surface TRL needs.
if "llm_blender" not in sys.modules:
    llm_blender_stub = types.ModuleType("llm_blender")
    llm_blender_stub.__spec__ = importlib.machinery.ModuleSpec("llm_blender", loader=None)

    class _UnusedBlender:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("llm_blender is not used by retrieval RLVR.")

    llm_blender_stub.Blender = _UnusedBlender
    sys.modules["llm_blender"] = llm_blender_stub

# Some TRL builds detect Weave from stale package metadata and then import it
# unconditionally while loading optional callbacks. Retrieval RLVR does not use
# Weave logging, so provide a no-op import surface instead of depending on that
# optional package on shared training nodes.
if "weave" not in sys.modules:
    weave_stub = types.ModuleType("weave")
    weave_stub.__spec__ = importlib.machinery.ModuleSpec("weave", loader=None)

    class _UnusedEvaluationLogger:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("weave logging is not used by retrieval RLVR.")

    def _unused_weave_init(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("weave logging is not used by retrieval RLVR.")

    weave_stub.EvaluationLogger = _UnusedEvaluationLogger
    weave_stub.init = _unused_weave_init
    trace_stub = types.ModuleType("weave.trace")
    trace_stub.__spec__ = importlib.machinery.ModuleSpec("weave.trace", loader=None)
    context_stub = types.ModuleType("weave.trace.context")
    context_stub.__spec__ = importlib.machinery.ModuleSpec("weave.trace.context", loader=None)

    class _UnusedWeaveClientContext:
        @staticmethod
        def get_weave_client() -> None:
            return None

    context_stub.weave_client_context = _UnusedWeaveClientContext
    sys.modules["weave"] = weave_stub
    sys.modules["weave.trace"] = trace_stub
    sys.modules["weave.trace.context"] = context_stub

from trl import GRPOConfig, GRPOTrainer


FORMAT_ERROR_REWARD = -1.0
NO_CURATE_REWARD = -0.25


def parse_target_modules(value: str) -> str | list[str]:
    value = value.strip()
    if value == "all-linear":
        return value
    if value.startswith("regex:"):
        return value.removeprefix("regex:")
    modules = [module.strip() for module in value.split(",") if module.strip()]
    return modules if len(modules) > 1 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument("--sft-adapter-path", default=None)
    parser.add_argument(
        "--dataset-path",
        default="/home/work/.data/liquid_cli_sft/datasets/lfm25_harness1_retrieval_rlvr_browsecomp_v1",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B__harness1_retrieval_rlvr_grpo_r32",
    )
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--max-prompt-length", type=int, default=12288)
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,out_proj,in_proj,w1,w2,w3,gate",
    )
    parser.add_argument("--random-state", type=int, default=3407)
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    return parser.parse_args()


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    return str(completion)


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def normalize_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        pieces = re.split(r"[\s,;]+", value.strip())
    elif isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, dict):
                item = item.get("doc_id") or item.get("docid") or item.get("id")
            pieces.append(str(item))
    else:
        pieces = [str(value)]

    out: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        doc = str(piece).strip().strip("\"'`[](){}")
        if not doc or doc in seen:
            continue
        seen.add(doc)
        out.append(doc)
    return out


def parse_curated_ids(text: str) -> tuple[list[str], bool]:
    parsed = extract_json_object(text)
    if parsed is None:
        return [], False
    for key in ("curated_doc_ids", "doc_ids", "selected_doc_ids", "evidence_doc_ids"):
        if key in parsed:
            return normalize_ids(parsed.get(key)), True
    return [], False


def extract_candidate_ids_from_text(text: str, candidate_ids: list[str]) -> list[str]:
    """Fallback signal when the model mentions IDs but fails strict JSON."""
    out: list[str] = []
    seen: set[str] = set()
    for cid in sorted((str(item) for item in candidate_ids), key=len, reverse=True):
        if cid and cid not in seen and cid in text:
            seen.add(cid)
            out.append(cid)
    return out


def f_beta(recall: float, precision: float, beta: float = 2.0) -> float:
    if recall <= 0.0 and precision <= 0.0:
        return 0.0
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


def reward_retrieval_curation(completions, **kwargs) -> list[float]:
    gold_lists = kwargs.get("gold_doc_ids", [])
    answer_lists = kwargs.get("answer_doc_ids", [])
    candidate_lists = kwargs.get("candidate_doc_ids", [])
    scores: list[float] = []

    for completion, gold_ids, answer_ids, candidate_ids in zip(
        completions, gold_lists, answer_lists, candidate_lists
    ):
        text = completion_text(completion)
        curated_ids, valid_json = parse_curated_ids(text)
        if not valid_json:
            fallback_ids = extract_candidate_ids_from_text(text, candidate_ids)
            if not fallback_ids:
                scores.append(FORMAT_ERROR_REWARD)
                continue
            curated_ids = fallback_ids
        if not curated_ids:
            scores.append(NO_CURATE_REWARD)
            continue

        gold = set(str(x) for x in gold_ids)
        answer = set(str(x) for x in answer_ids) or gold
        candidates = set(str(x) for x in candidate_ids)
        selected = [doc for doc in curated_ids if doc in candidates]
        selected_set = set(selected)
        invalid_count = max(0, len(curated_ids) - len(selected))

        if not selected_set:
            scores.append(-0.5 - 0.05 * invalid_count)
            continue

        recall = len(selected_set & gold) / max(len(gold), 1)
        precision = len(selected_set & gold) / max(len(selected_set), 1)
        answer_recall = len(selected_set & answer) / max(len(answer), 1)
        fb = f_beta(recall, precision, beta=2.0)
        over_select = max(0, len(selected_set) - max(len(gold) + 3, 8))

        score = (
            0.65 * fb
            + 0.75 * recall
            + 0.85 * answer_recall
            - 0.05 * invalid_count
            - 0.025 * over_select
        )
        if not valid_json:
            score -= 0.45
        scores.append(max(-1.0, score))
    return scores


def reward_strict_json(completions, **kwargs) -> list[float]:
    scores: list[float] = []
    for completion in completions:
        text = completion_text(completion)
        ids, ok = parse_curated_ids(text)
        if not ok:
            scores.append(-0.5)
            continue
        parsed = extract_json_object(text) or {}
        score = 0.25
        if isinstance(parsed.get("reasoning"), str) and 8 <= len(parsed["reasoning"]) <= 600:
            score += 0.15
        if ids:
            score += 0.15
        if text.strip().startswith("{") and text.strip().endswith("}"):
            score += 0.10
        scores.append(score)
    return scores


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    target_modules = parse_target_modules(args.target_modules)
    dataset = load_from_disk(args.dataset_path)

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --push-to-hub is enabled.")
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if torch.cuda.is_available():
        model.to(torch.device("cuda", local_rank))

    if args.sft_adapter_path:
        model = PeftModel.from_pretrained(model, args.sft_adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(
            model,
            lora_config,
        )

    for candidate in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        if candidate is not None and not hasattr(candidate, "warnings_issued"):
            candidate.warnings_issued = {}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        print(
            json.dumps(
                {
                    "task": "harness1_retrieval_rlvr_grpo",
                    "model_path": args.model_path,
                    "sft_adapter_path": args.sft_adapter_path,
                    "dataset_path": args.dataset_path,
                    "dataset_rows": len(dataset),
                    "output_dir": args.output_dir,
                    "world_size": world_size,
                    "max_seq_length": args.max_seq_length,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    "num_generations": args.num_generations,
                    "effective_batch_size": world_size
                    * args.per_device_train_batch_size
                    * args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "max_steps": args.max_steps,
                    "beta": args.beta,
                    "target_modules": target_modules,
                    "use_vllm": args.use_vllm,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    train_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=1.05,
        use_vllm=args.use_vllm,
        remove_unused_columns=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_strategy="every_save" if args.push_to_hub else "checkpoint",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_retrieval_curation, reward_strict_json],
        args=train_args,
        train_dataset=dataset,
    )
    trainer.train()

    final_dir = Path(args.output_dir) / "final_lora"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if local_rank == 0:
        (final_dir / "RLVR_NOTES.md").write_text(
            "\n".join(
                [
                    "# Harness-1 Retrieval RLVR Adapter",
                    "",
                    "This adapter was trained with programmatic rewards over Harness-style",
                    "candidate pools. Rewards compare `curated_doc_ids` against gold and",
                    "answer evidence document IDs. No OpenAI verifier is required for this",
                    "local RLVR objective.",
                    "",
                ]
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
