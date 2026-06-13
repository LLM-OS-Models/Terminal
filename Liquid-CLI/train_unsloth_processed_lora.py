#!/usr/bin/env python
"""LoRA SFT on already-normalized LFM conversation datasets."""

from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from huggingface_hub import login

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if not hasattr(builtins, "PreTrainedConfig"):
    if hasattr(transformers, "PreTrainedConfig"):
        builtins.PreTrainedConfig = transformers.PreTrainedConfig
    elif hasattr(transformers, "PretrainedConfig"):
        builtins.PreTrainedConfig = transformers.PretrainedConfig

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer

from train_unsloth_processed import (
    build_processed_meta,
    dataset_artifact_paths,
    metadata_matches,
    metadata_path,
    prepare_processed_dataset,
    remove_dataset_artifact,
    save_dataset_atomic,
    wait_for_dataset,
    write_metadata,
)
from terminal_sft_utils import load_holdout_keys, stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="LiquidAI/LFM2.5-8B-A1B")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--processed-data-path", required=True)
    parser.add_argument("--train-ready-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="steps")
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--dataset-num-proc", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--holdout-path", default="/home/work/.data/liquid_cli_sft/datasets/empty_holdout.jsonl")
    parser.add_argument("--instruction-part", default="<|im_start|>user\n")
    parser.add_argument("--response-part", default="<|im_start|>assistant\n")
    parser.add_argument("--overwrite-processed-data", action="store_true")
    parser.add_argument("--overwrite-train-ready-data", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--random-state", type=int, default=3407)
    return parser.parse_args()


def parse_target_modules(value: str) -> str | list[str]:
    value = value.strip()
    if value == "all-linear":
        return value
    if value.startswith("regex:"):
        return value.removeprefix("regex:")
    modules = [module.strip() for module in value.split(",") if module.strip()]
    return modules if len(modules) > 1 else value


def main() -> None:
    args = parse_args()
    target_modules = parse_target_modules(args.target_modules)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)

    processed_path = Path(args.processed_data_path)
    train_ready_path = Path(args.train_ready_data_path)

    if local_rank == 0:
        if args.overwrite_processed_data:
            remove_dataset_artifact(args.processed_data_path)
        if args.overwrite_train_ready_data:
            remove_dataset_artifact(args.train_ready_data_path)

        tokenizer_for_meta = transformers.AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
        holdout_keys = load_holdout_keys(args.holdout_path)
        expected_processed_meta = build_processed_meta(
            tokenizer=tokenizer_for_meta,
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            holdout_path=args.holdout_path,
            holdout_keys=holdout_keys,
            response_part=args.response_part,
        )
        _, _, ready_path = dataset_artifact_paths(args.processed_data_path)
        if processed_path.exists() and ready_path.exists() and not metadata_matches(
            args.processed_data_path,
            expected_processed_meta,
        ):
            remove_dataset_artifact(args.processed_data_path)
            remove_dataset_artifact(args.train_ready_data_path)

        _, _, ready_path = dataset_artifact_paths(args.processed_data_path)
        if not processed_path.exists() or not ready_path.exists():
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            prepare_processed_dataset(
                dataset_path=args.dataset_path,
                dataset_name=args.dataset_name,
                dataset_split=args.dataset_split,
                processed_data_path=args.processed_data_path,
                model_path=args.model_path,
                num_proc=args.dataset_num_proc,
                holdout_path=args.holdout_path,
                response_part=args.response_part,
            )
    wait_for_dataset(args.processed_data_path)

    processed_meta = json.loads(metadata_path(args.processed_data_path).read_text())
    expected_train_ready_meta = {
        "source_mode": "unsloth_train_on_responses_only",
        "processed_meta_hash": stable_hash(json.dumps(processed_meta, sort_keys=True, default=str)),
        "instruction_part": args.instruction_part,
        "response_part": args.response_part,
        "max_seq_length": args.max_seq_length,
    }
    if local_rank == 0:
        _, _, train_ready_ready_path = dataset_artifact_paths(args.train_ready_data_path)
        if train_ready_path.exists() and train_ready_ready_path.exists() and not metadata_matches(
            args.train_ready_data_path,
            expected_train_ready_meta,
        ):
            remove_dataset_artifact(args.train_ready_data_path)

    _, _, train_ready_ready_path = dataset_artifact_paths(args.train_ready_data_path)
    use_train_ready_cache = (
        train_ready_path.exists()
        and train_ready_ready_path.exists()
        and metadata_matches(args.train_ready_data_path, expected_train_ready_meta)
    )

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --push-to-hub is enabled.")
        login(token=token, add_to_git_credential=False)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{local_rank}"},
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha or args.lora_rank * 2,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.random_state,
    )

    dataset = load_from_disk(
        args.train_ready_data_path if use_train_ready_cache else args.processed_data_path
    )

    if local_rank == 0:
        print(
            json.dumps(
                {
                    "model_path": args.model_path,
                    "dataset_path": args.dataset_path,
                    "dataset_name": args.dataset_name,
                    "processed_data_path": args.processed_data_path,
                    "train_ready_data_path": args.train_ready_data_path,
                    "using_train_ready_cache": use_train_ready_cache,
                    "output_dir": args.output_dir,
                    "world_size": world_size,
                    "max_seq_length": args.max_seq_length,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_batch_size": world_size
                    * args.per_device_train_batch_size
                    * args.gradient_accumulation_steps,
                    "learning_rate": args.learning_rate,
                    "num_train_epochs": args.num_train_epochs,
                    "save_strategy": args.save_strategy,
                    "save_steps": args.save_steps if args.save_strategy == "steps" else None,
                    "dataset_rows": len(dataset),
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha or args.lora_rank * 2,
                    "target_modules": target_modules,
                    "push_to_hub": args.push_to_hub,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    train_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        packing=not use_train_ready_cache,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=True if world_size > 1 else None,
        dataset_num_proc=args.dataset_num_proc,
        dataset_kwargs={"skip_prepare_dataset": True} if use_train_ready_cache else None,
        dataloader_num_workers=min(8, os.cpu_count() or 1),
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_strategy="every_save" if args.push_to_hub else "checkpoint",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=not use_train_ready_cache,
        args=train_args,
    )

    if not use_train_ready_cache:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=args.instruction_part,
            response_part=args.response_part,
        )
        if local_rank == 0:
            save_dataset_atomic(trainer.train_dataset, args.train_ready_data_path)
            write_metadata(args.train_ready_data_path, expected_train_ready_meta)
        wait_for_dataset(args.train_ready_data_path)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = Path(args.output_dir) / "final_lora"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
