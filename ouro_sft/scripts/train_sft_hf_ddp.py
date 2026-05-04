#!/usr/bin/env python3
"""SFT training for Ouro models using HuggingFace Trainer + DDP."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Patch ROPE_INIT_FUNCTIONS for Ouro compatibility with transformers >= 4.56
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if "default" not in ROPE_INIT_FUNCTIONS:
    def _compute_default_rope_parameters(config, device):
        base = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        attention_scaling = 1.0
        return inv_freq, attention_scaling
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

# Patch _get_tied_weight_keys for Ouro compatibility (list vs dict)
import torch.nn as nn
import transformers.modeling_utils as _mu
_orig_get_tied = _mu._get_tied_weight_keys
def _patched_get_tied_weight_keys(module: nn.Module) -> list:
    tied_weight_keys: list = []
    for name, submodule in module.named_modules():
        tied = getattr(submodule, "_tied_weights_keys", {}) or {}
        if isinstance(tied, list):
            tied_weight_keys.extend([f"{name}.{k}" if name else k for k in tied])
        else:
            tied_weight_keys.extend([f"{name}.{k}" if name else k for k in tied.keys()])
    return tied_weight_keys
_mu._get_tied_weight_keys = _patched_get_tied_weight_keys


def processed_dataset_ready(processed_path: Path) -> bool:
    return (
        processed_path.exists()
        and (processed_path / "dataset_info.json").exists()
        and (processed_path / "state.json").exists()
    )


def wait_for_processed_dataset(processed_path: Path, *, timeout_seconds: int = 3600) -> None:
    start = time.time()
    while time.time() - start < timeout_seconds:
        if processed_dataset_ready(processed_path):
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for processed dataset at {processed_path}")


def prepare_processed_dataset(
    *,
    raw_data_path: str,
    processed_data_path: str,
    model_path: str,
    num_proc: int,
    max_seq_length: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_from_disk(raw_data_path)
    raw_rows = len(raw_dataset)

    turn_examples: list[dict] = []
    for row in raw_dataset:
        conversations = row["conversations"]
        for idx in range(1, len(conversations)):
            prev_msg = conversations[idx - 1]
            cur_msg = conversations[idx]
            if prev_msg.get("role") != "user" or cur_msg.get("role") != "assistant":
                continue
            assistant_text = cur_msg.get("content", "")
            has_commands = '"keystrokes"' in assistant_text or '"commands"' in assistant_text
            empty_commands = '"commands": []' in assistant_text or '"commands":[]' in assistant_text
            if not has_commands or empty_commands:
                continue
            turn_examples.append(
                {
                    "conversations": [
                        {"role": "user", "content": prev_msg.get("content", "")},
                        {"role": "assistant", "content": assistant_text},
                    ]
                }
            )

    dataset = Dataset.from_list(turn_examples)

    def tokenize_fn(examples):
        texts = tokenizer.apply_chat_template(
            examples["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        model_inputs = tokenizer(
            texts,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = [list(ids) for ids in model_inputs["input_ids"]]
        return model_inputs

    source_columns = list(dataset.column_names)
    processed = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=source_columns,
        desc="tokenize",
    )

    print(
        json.dumps(
            {
                "raw_rows": raw_rows,
                "train_rows": len(dataset),
                "processed_rows": len(processed),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    processed.save_to_disk(processed_data_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ByteDance/Ouro-1.4B")
    parser.add_argument(
        "--data-path",
        default="/home/work/.data/liquid_cli_sft/datasets/sft_data",
    )
    parser.add_argument(
        "--processed-data-path",
        default="/home/work/.data/ouro_sft/datasets/ouro_1_4b_processed",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/work/.data/ouro_sft/models/ByteDance__Ouro-1.4B__terminal_sft_2epoch",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    distributed = True

    if local_rank == 0:
        print(
            json.dumps(
                {
                    "model_path": args.model_path,
                    "data_path": args.data_path,
                    "output_dir": args.output_dir,
                    "world_size": world_size,
                    "max_seq_length": args.max_seq_length,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "num_train_epochs": args.num_train_epochs,
                    "learning_rate": args.learning_rate,
                    "save_strategy": args.save_strategy,
                    "gradient_checkpointing": args.gradient_checkpointing,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    processed_path = Path(args.processed_data_path)
    if local_rank == 0 and processed_path.exists() and not processed_dataset_ready(processed_path):
        shutil.rmtree(processed_path, ignore_errors=True)

    if local_rank == 0 and not processed_dataset_ready(processed_path):
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        prepare_processed_dataset(
            raw_data_path=args.data_path,
            processed_data_path=args.processed_data_path,
            model_path=args.model_path,
            num_proc=min(16, os.cpu_count() or 1),
            max_seq_length=args.max_seq_length,
        )

    if local_rank != 0:
        wait_for_processed_dataset(processed_path)

    if distributed:
        torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id
    # Fix rope_scaling for transformers >= 4.56 compatibility
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        if config.rope_scaling.get("rope_type") == "default":
            config.rope_scaling = {"rope_type": "linear", "factor": 1.0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": f"cuda:{local_rank}"},
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    dataset = load_from_disk(args.processed_data_path)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else 500,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        dataloader_num_workers=min(8, os.cpu_count() or 1),
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_seq_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if distributed:
        torch.distributed.barrier()

    if local_rank == 0:
        final_dir = Path(args.output_dir) / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        print(f"Model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
