#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.models.gemma4 import Gemma4ForCausalLM


BASE_TO_TEMPLATE_MODEL = {
    "google/gemma-4-E2B": "google/gemma-4-E2B-it",
    "google/gemma-4-E4B": "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B": "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B": "google/gemma-4-31B-it",
}


def resolve_model_path(model_id_or_path: str) -> str:
    path = Path(model_id_or_path)
    if path.exists():
        return str(path)
    return snapshot_download(
        repo_id=model_id_or_path,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "model-*.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "processor_config.json",
        ],
    )


def fsdp_uses_sharded_state_dict(fsdp_config_path: str | None) -> bool:
    if not fsdp_config_path:
        return False
    try:
        with open(fsdp_config_path) as handle:
            config = json.load(handle)
    except Exception:
        return False
    return str(config.get("fsdp_state_dict_type", "")).upper() == "SHARDED_STATE_DICT"


def load_tokenizer(model_id: str, resolved_model_path: str, template_model_id: str | None) -> tuple[Any, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_has_template = bool(getattr(tokenizer, "chat_template", None))
    inferred_template_model = template_model_id or BASE_TO_TEMPLATE_MODEL.get(model_id, model_id)
    template_was_injected = False
    if not original_has_template or inferred_template_model != model_id:
        template_source_path = resolve_model_path(inferred_template_model)
        template_tokenizer = AutoTokenizer.from_pretrained(template_source_path, trust_remote_code=False)
        if not getattr(template_tokenizer, "chat_template", None):
            raise RuntimeError(f"Template tokenizer has no chat_template: {inferred_template_model}")
        tokenizer.chat_template = template_tokenizer.chat_template
        template_was_injected = True

    return tokenizer, {
        "model_tokenizer_had_template": original_has_template,
        "template_model_id": inferred_template_model,
        "template_was_injected": template_was_injected,
    }


def load_gemma4_text_only_model(model_path: str) -> Gemma4ForCausalLM:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        if str(getattr(config, "model_type", "")).startswith("gemma4"):
            text_config = config
        else:
            raise RuntimeError("Gemma 4 checkpoint is missing text_config")

    model = Gemma4ForCausalLM(text_config)
    model_root = Path(model_path)
    index_path = model_root / "model.safetensors.index.json"
    single_path = model_root / "model.safetensors"

    def mapped_key(key: str) -> str | None:
        if key.startswith("model.language_model."):
            return "model." + key[len("model.language_model.") :]
        if key.startswith("language_model."):
            return "model." + key[len("language_model.") :]
        if key.startswith("model.") or key == "lm_head.weight":
            return key
        return None

    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_to_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for old_key, shard_name in weight_map.items():
            new_key = mapped_key(old_key)
            if new_key is not None:
                shard_to_keys[shard_name].append((old_key, new_key))
        for shard_name, key_pairs in sorted(shard_to_keys.items()):
            shard_state = {}
            with safe_open(str(model_root / shard_name), framework="pt", device="cpu") as handle:
                for old_key, new_key in key_pairs:
                    shard_state[new_key] = handle.get_tensor(old_key)
            model.load_state_dict(shard_state, strict=False)
            del shard_state
    elif single_path.exists():
        shard_state = {}
        with safe_open(str(single_path), framework="pt", device="cpu") as handle:
            for old_key in handle.keys():
                new_key = mapped_key(old_key)
                if new_key is not None:
                    shard_state[new_key] = handle.get_tensor(old_key)
        model.load_state_dict(shard_state, strict=False)
        del shard_state
    else:
        raise FileNotFoundError(f"No safetensors weights found in {model_path}")

    model.tie_weights()
    return model


class CausalLMCollator:
    def __init__(self, tokenizer: Any, needs_mm_token_type_ids: bool = False):
        self.tokenizer = tokenizer
        self.needs_mm_token_type_ids = needs_mm_token_type_ids

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["labels"])
            labels.append(feature["labels"] + ([-100] * pad_len))
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        if self.needs_mm_token_type_ids:
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--template-model-id", default=None)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fsdp", default=None)
    parser.add_argument("--fsdp-config", default=None)
    parser.add_argument("--save-only-model", action="store_true")
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--skip-final-save", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    resolved_model_path = resolve_model_path(args.model_id)
    tokenizer, tokenizer_meta = load_tokenizer(args.model_id, resolved_model_path, args.template_model_id)
    dataset = load_from_disk(args.dataset_path)

    model = load_gemma4_text_only_model(resolved_model_path).to(dtype=torch.bfloat16)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if local_rank == 0:
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "resolved_model_path": resolved_model_path,
                    "dataset_path": args.dataset_path,
                    "dataset_rows": len(dataset),
                    "output_dir": args.output_dir,
                    "world_size": world_size,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "optim": args.optim,
                    "num_train_epochs": args.num_train_epochs,
                    "max_steps": args.max_steps,
                    "fsdp": args.fsdp,
                    "fsdp_config": args.fsdp_config,
                    **tokenizer_meta,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim=args.optim,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=min(8, os.cpu_count() or 1),
        fsdp=args.fsdp,
        fsdp_config=args.fsdp_config,
        save_only_model=args.save_only_model,
        save_total_limit=args.save_total_limit,
        ddp_find_unused_parameters=False if world_size > 1 else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer, needs_mm_token_type_ids=True),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if distributed:
        torch.distributed.barrier()

    final_dir = Path(args.output_dir) / "final"
    if local_rank == 0:
        if args.skip_final_save or fsdp_uses_sharded_state_dict(args.fsdp_config):
            checkpoints = sorted(
                [path for path in Path(args.output_dir).glob("checkpoint-*") if path.is_dir()],
                key=lambda path: int(path.name.split("-")[-1]),
            )
            if final_dir.exists() or final_dir.is_symlink():
                if final_dir.is_symlink() or final_dir.is_file():
                    final_dir.unlink()
                else:
                    shutil.rmtree(final_dir)
            if checkpoints:
                final_dir.symlink_to(checkpoints[-1].name)
                tokenizer.save_pretrained(str(checkpoints[-1]))
        else:
            trainer.save_model(str(final_dir))
            tokenizer.save_pretrained(str(final_dir))

    if distributed:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
