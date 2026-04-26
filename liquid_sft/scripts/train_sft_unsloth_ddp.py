#!/usr/bin/env python3
from __future__ import annotations

import argparse
import builtins
import json
import os
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from huggingface_hub import login

# Liquid-CLI 원본과 동일하게 PreTrainedConfig를 먼저 주입해야 LFM2 patch가 안정적으로 잡힌다.
if not hasattr(builtins, "PreTrainedConfig"):
    if hasattr(transformers, "PreTrainedConfig"):
        builtins.PreTrainedConfig = transformers.PreTrainedConfig
    elif hasattr(transformers, "PretrainedConfig"):
        builtins.PreTrainedConfig = transformers.PretrainedConfig

from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats, train_on_responses_only
from trl import SFTConfig, SFTTrainer


def prepare_processed_dataset(
    *,
    raw_data_path: str,
    processed_data_path: str,
    model_path: str,
    num_proc: int,
) -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(raw_data_path)
    dataset = standardize_data_formats(dataset)
    source_columns = list(dataset.column_names)
    bos = tokenizer.bos_token or ""

    def formatting_prompts_func(examples):
        texts = tokenizer.apply_chat_template(
            examples["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        if bos:
            texts = [text[len(bos):] if text.startswith(bos) else text for text in texts]
        return {"text": texts}

    processed = dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=num_proc,
        remove_columns=source_columns,
        desc="format_chat_text",
    )
    processed.save_to_disk(processed_data_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="LiquidAI/LFM2-8B-A1B")
    parser.add_argument("--data-path", default="/home/work/.data/liquid_cli_sft/datasets/sft_data")
    parser.add_argument(
        "--processed-data-path",
        default="/home/work/.data/liquid_cli_sft/datasets/sft_data_unsloth_processed",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local",
    )
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default="LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    processed_path = Path(args.processed_data_path)
    if local_rank == 0 and not processed_path.exists():
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        prepare_processed_dataset(
            raw_data_path=args.data_path,
            processed_data_path=args.processed_data_path,
            model_path=args.model_path,
            num_proc=min(16, os.cpu_count() or 1),
        )

    if distributed:
        torch.distributed.barrier()

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --push-to-hub is enabled.")
        login(token=token, add_to_git_credential=False)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        full_finetuning=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{local_rank}"},
    )

    dataset = load_from_disk(args.processed_data_path)

    if local_rank == 0:
        print(
            json.dumps(
                {
                    "model_path": args.model_path,
                    "data_path": args.data_path,
                    "processed_data_path": args.processed_data_path,
                    "output_dir": args.output_dir,
                    "world_size": world_size,
                    "max_seq_length": args.max_seq_length,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "num_train_epochs": args.num_train_epochs,
                    "save_strategy": args.save_strategy,
                    "save_steps": args.save_steps if args.save_strategy == "steps" else None,
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
        packing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        dataloader_num_workers=min(8, os.cpu_count() or 1),
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,
        args=train_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
