#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_data_formats, train_on_responses_only
import transformers
from datasets import Dataset, load_from_disk
from huggingface_hub import login
from trl import SFTConfig, SFTTrainer
from terminal_sft_utils import is_holdout_row, load_holdout_keys, stable_hash, tokenizer_metadata


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


def prepare_meta_path(processed_path: Path) -> Path:
    return processed_path / "prepare_meta.json"


def processed_meta_matches(processed_path: Path, expected: dict) -> bool:
    path = prepare_meta_path(processed_path)
    if not path.exists():
        return False
    try:
        current = json.loads(path.read_text())
    except Exception:
        return False
    return all(current.get(key) == value for key, value in expected.items())


def prepare_processed_dataset(
    *,
    raw_data_path: str,
    processed_data_path: str,
    model_path: str,
    num_proc: int,
    conversation_mode: str,
    holdout_path: str | None,
) -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_from_disk(raw_data_path)
    raw_rows = len(raw_dataset)
    holdout_keys = load_holdout_keys(holdout_path)
    if holdout_keys:
        raw_dataset = raw_dataset.filter(
            lambda example: not is_holdout_row(example, holdout_keys),
            num_proc=num_proc,
            desc="exclude_eval_holdout",
        )
    dataset = standardize_data_formats(raw_dataset)

    if conversation_mode == "turn_pairs":
        turn_examples: list[dict[str, list[dict[str, str]]]] = []

        for row in dataset:
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
                            {
                                "role": "user",
                                "content": prev_msg.get("content", ""),
                            },
                            {
                                "role": "assistant",
                                "content": assistant_text,
                            },
                        ]
                    }
                )

        dataset = Dataset.from_list(turn_examples)
    elif conversation_mode != "full_conversations":
        raise ValueError(f"Unsupported conversation mode: {conversation_mode}")

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
    print(
        json.dumps(
            {
                "conversation_mode": conversation_mode,
                "raw_rows": raw_rows,
                "train_rows": len(dataset),
                "processed_rows": len(processed),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    processed.save_to_disk(processed_data_path)
    meta = {
        "source_mode": "raw_conversations_template_text",
        "raw_data_path": raw_data_path,
        "model_path": model_path,
        "conversation_mode": conversation_mode,
        "holdout_path": holdout_path,
        "holdout_hash": stable_hash(json.dumps(sorted(holdout_keys))),
        "raw_rows": raw_rows,
        "skipped_holdout_rows": raw_rows - len(raw_dataset),
        "train_rows": len(dataset),
        "processed_rows": len(processed),
        **tokenizer_metadata(tokenizer, model_path),
    }
    prepare_meta_path(Path(processed_data_path)).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--data-path",
        default="/home/work/.data/liquid_cli_sft/datasets/sft_data",
    )
    parser.add_argument(
        "--processed-data-path",
        default="/home/work/.data/qwen_sft/datasets/qwen35_2b_processed",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_unsloth",
    )
    parser.add_argument(
        "--conversation-mode",
        choices=["turn_pairs", "full_conversations"],
        default="turn_pairs",
    )
    parser.add_argument("--train-mode", choices=["full", "lora"], default="full")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--holdout-path", default="eval/eval_dataset.jsonl")
    parser.add_argument("--overwrite-processed-data", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable Unsloth gradient checkpointing. Disabled by default for Qwen3.5 full finetuning stability.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--fsdp", default=None)
    parser.add_argument("--fsdp-config", default=None)
    parser.add_argument("--return-logits", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--hub-model-id",
        default="LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    if args.return_logits:
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    processed_path = Path(args.processed_data_path)
    if local_rank == 0:
        holdout_keys = load_holdout_keys(args.holdout_path)
        expected_meta = {
            "source_mode": "raw_conversations_template_text",
            "raw_data_path": args.data_path,
            "model_path": args.model_path,
            "conversation_mode": args.conversation_mode,
            "holdout_path": args.holdout_path,
            "holdout_hash": stable_hash(json.dumps(sorted(holdout_keys))),
        }
        if args.overwrite_processed_data and processed_path.exists():
            shutil.rmtree(processed_path, ignore_errors=True)
        if processed_dataset_ready(processed_path) and not processed_meta_matches(processed_path, expected_meta):
            shutil.rmtree(processed_path, ignore_errors=True)
    if local_rank == 0 and processed_path.exists() and not processed_dataset_ready(processed_path):
        shutil.rmtree(processed_path, ignore_errors=True)

    if local_rank == 0 and not processed_dataset_ready(processed_path):
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        prepare_processed_dataset(
            raw_data_path=args.data_path,
            processed_data_path=args.processed_data_path,
            model_path=args.model_path,
            num_proc=min(16, os.cpu_count() or 1),
            conversation_mode=args.conversation_mode,
            holdout_path=args.holdout_path,
        )

    if local_rank != 0:
        wait_for_processed_dataset(processed_path)

    if distributed:
        torch.distributed.barrier()

    if args.push_to_hub:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --push-to-hub is enabled.")
        login(token=token, add_to_git_credential=False)

    use_unsloth_gc = "unsloth" if args.gradient_checkpointing else False
    if args.train_mode == "full":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_length,
            load_in_4bit=False,
            full_finetuning=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            use_gradient_checkpointing=use_unsloth_gc,
            device_map={"": f"cuda:{local_rank}"},
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
            load_in_16bit=not args.load_in_4bit,
            full_finetuning=False,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            use_gradient_checkpointing=use_unsloth_gc,
            device_map={"": f"cuda:{local_rank}"},
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing=use_unsloth_gc,
            max_seq_length=args.max_seq_length,
        )

    dataset = load_from_disk(args.processed_data_path)

    if local_rank == 0:
        print(
            json.dumps(
                {
                    "model_path": args.model_path,
                    "data_path": args.data_path,
                    "holdout_path": args.holdout_path,
                    "processed_data_path": args.processed_data_path,
                    "output_dir": args.output_dir,
                    "conversation_mode": args.conversation_mode,
                    "train_mode": args.train_mode,
                    "world_size": world_size,
                    "max_seq_length": args.max_seq_length,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "num_train_epochs": args.num_train_epochs,
                    "save_strategy": args.save_strategy,
                    "save_steps": args.save_steps if args.save_strategy == "steps" else None,
                    "fsdp": args.fsdp,
                    "fsdp_config": args.fsdp_config,
                    "return_logits": args.return_logits,
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
        gradient_checkpointing=args.gradient_checkpointing,
        # Qwen3.5 conditional-generation models can keep multimodal branches inactive
        # on text-only batches, so DDP must tolerate unused parameters in full FT.
        ddp_find_unused_parameters=(True if (world_size > 1 and args.train_mode == "full") else (False if world_size > 1 else None)),
        fsdp=args.fsdp,
        fsdp_config=args.fsdp_config,
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

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if distributed:
        torch.distributed.barrier()

    final_dir = Path(args.output_dir) / "final"
    if local_rank == 0:
        if args.train_mode == "lora":
            model.save_pretrained_merged(str(final_dir), tokenizer, save_method="merged_16bit")
        else:
            trainer.save_model(str(final_dir))
            tokenizer.save_pretrained(str(final_dir))

    if distributed:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
