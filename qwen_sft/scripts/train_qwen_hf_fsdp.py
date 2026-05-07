#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from datasets import Dataset, load_from_disk
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.models.gemma4 import Gemma4ForCausalLM

from terminal_sft_utils import (
    assert_assistant_label_audit,
    build_turn_pair_examples as build_sft_turn_pair_examples,
    encode_assistant_only,
    load_holdout_keys,
    stable_hash,
    tokenizer_metadata,
)


def processed_dataset_ready(processed_path: Path) -> bool:
    return (
        processed_path.exists()
        and (processed_path / "dataset_info.json").exists()
        and (processed_path / "state.json").exists()
    )


def fsdp_uses_sharded_state_dict(fsdp_config_path: str | None) -> bool:
    if not fsdp_config_path:
        return False
    try:
        with open(fsdp_config_path) as f:
            config = json.load(f)
    except Exception:
        return False
    return str(config.get("fsdp_state_dict_type", "")).upper() == "SHARDED_STATE_DICT"


def wait_for_processed_dataset(processed_path: Path, *, timeout_seconds: int = 3600) -> None:
    start = time.time()
    while time.time() - start < timeout_seconds:
        if processed_dataset_ready(processed_path):
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for processed dataset at {processed_path}")


def prepare_meta_path(processed_path: Path) -> Path:
    return processed_path / "prepare_meta.json"


def processed_meta_matches(processed_path: Path, expected: dict[str, object]) -> bool:
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
    max_seq_length: int,
    trust_remote_code: bool = False,
    holdout_path: str | None = None,
    allow_text_cache: bool = False,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_from_disk(raw_data_path)
    holdout_keys = load_holdout_keys(holdout_path)

    if "conversations" not in raw_dataset.column_names:
        if "text" in raw_dataset.column_names:
            legacy_note = " --allow-text-cache was set, but legacy text-cache training is disabled after the template audit."
            raise RuntimeError(
                "Refusing to train from a preformatted text cache. "
                "Use raw conversations so the current model chat_template and assistant-only labels are applied."
                + (legacy_note if allow_text_cache else "")
            )
        raise RuntimeError(f"Unsupported dataset columns: {raw_dataset.column_names}")

    examples, skipped_holdout = build_sft_turn_pair_examples(raw_dataset, holdout_keys)

    rows: list[dict[str, list[int] | int]] = []
    skipped_empty_assistant = 0
    for ex in examples:
        encoded = encode_assistant_only(
            tokenizer=tokenizer,
            user=ex["user"],
            assistant=ex["assistant"],
            max_seq_length=max_seq_length,
        )
        if encoded is None:
            skipped_empty_assistant += 1
            continue
        rows.append(encoded)

    audit = assert_assistant_label_audit(rows)
    train_rows = [
        {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
        }
        for row in rows
    ]
    processed = Dataset.from_list(train_rows)
    meta = {
        "source_mode": "raw_conversations_template_masked",
        "raw_data_path": raw_data_path,
        "model_path": model_path,
        "raw_rows": len(raw_dataset),
        "skipped_holdout_conversations": skipped_holdout,
        "turn_pair_rows": len(examples),
        "skipped_empty_assistant_rows": skipped_empty_assistant,
        "processed_rows": len(processed),
        "max_seq_length": max_seq_length,
        "holdout_path": holdout_path,
        "holdout_hash": stable_hash(json.dumps(sorted(holdout_keys))),
        **audit,
        **tokenizer_metadata(tokenizer, model_path),
    }
    print(
        json.dumps(meta, ensure_ascii=False),
        flush=True,
    )
    processed.save_to_disk(processed_data_path)
    prepare_meta_path(Path(processed_data_path)).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )


class CausalLMCollator:
    def __init__(self, tokenizer, needs_mm_token_type_ids: bool = False):
        self.tokenizer = tokenizer
        self.needs_mm_token_type_ids = needs_mm_token_type_ids

    def __call__(self, features):
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            pad_len = max_len - len(f["labels"])
            labels.append(f["labels"] + ([-100] * pad_len))
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        if self.needs_mm_token_type_ids:
            # Text-only Gemma 4 training still requires mm_token_type_ids.
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"], dtype=torch.long)
        return batch


def load_gemma4_text_only_model(model_path: str) -> tuple[Gemma4ForCausalLM, str]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        raise RuntimeError("Gemma 4 checkpoint is missing text_config")

    model = Gemma4ForCausalLM(text_cfg)

    model_root = Path(model_path)
    index_path = model_root / "model.safetensors.index.json"
    single_path = model_root / "model.safetensors"

    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_to_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for old_key, shard_name in weight_map.items():
            if not old_key.startswith("model.language_model."):
                continue
            new_key = "model." + old_key[len("model.language_model.") :]
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
                if not old_key.startswith("model.language_model."):
                    continue
                new_key = "model." + old_key[len("model.language_model.") :]
                shard_state[new_key] = handle.get_tensor(old_key)
        model.load_state_dict(shard_state, strict=False)
        del shard_state
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")

    model.tie_weights()
    return model, str(getattr(text_cfg, "model_type", "gemma4_text"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--processed-data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fsdp", default=None)
    parser.add_argument("--fsdp-config", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--save-only-model", action="store_true")
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--skip-final-save", action="store_true")
    parser.add_argument("--overwrite-processed-data", action="store_true")
    parser.add_argument("--allow-text-cache", action="store_true")
    parser.add_argument("--holdout-path", default="eval/eval_dataset.jsonl")
    parser.add_argument(
        "--ddp-find-unused-parameters",
        choices=["auto", "true", "false"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    processed_path = Path(args.processed_data_path)
    if local_rank == 0 and args.overwrite_processed_data and processed_path.exists():
        shutil.rmtree(processed_path, ignore_errors=True)
    if local_rank == 0 and processed_dataset_ready(processed_path):
        holdout_keys = load_holdout_keys(args.holdout_path)
        expected_meta = {
            "source_mode": "raw_conversations_template_masked",
            "raw_data_path": args.data_path,
            "model_path": args.model_path,
            "max_seq_length": args.max_seq_length,
            "holdout_path": args.holdout_path,
            "holdout_hash": stable_hash(json.dumps(sorted(holdout_keys))),
        }
        if not processed_meta_matches(processed_path, expected_meta):
            shutil.rmtree(processed_path, ignore_errors=True)
    if local_rank == 0 and processed_path.exists() and not processed_dataset_ready(processed_path):
        shutil.rmtree(processed_path, ignore_errors=True)

    if local_rank == 0 and not processed_dataset_ready(processed_path):
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        prepare_processed_dataset(
            raw_data_path=args.data_path,
            processed_data_path=args.processed_data_path,
            model_path=args.model_path,
            max_seq_length=args.max_seq_length,
            trust_remote_code=args.trust_remote_code,
            holdout_path=args.holdout_path,
            allow_text_cache=args.allow_text_cache,
        )

    if local_rank != 0:
        wait_for_processed_dataset(processed_path)

    if distributed:
        torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(args.processed_data_path)

    if local_rank == 0:
        print(
            json.dumps(
                {
                    "model_path": args.model_path,
                    "processed_data_path": args.processed_data_path,
                    "output_dir": args.output_dir,
                    "world_size": world_size,
                    "max_seq_length": args.max_seq_length,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "num_train_epochs": args.num_train_epochs,
                    "fsdp": args.fsdp,
                    "fsdp_config": args.fsdp_config,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "save_only_model": args.save_only_model,
                    "save_total_limit": args.save_total_limit,
                    "skip_final_save": args.skip_final_save,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type.startswith("gemma4"):
        model, loaded_model_type = load_gemma4_text_only_model(args.model_path)
        model = model.to(dtype=torch.bfloat16)
        model_type = str(loaded_model_type).lower()
        # FSDP cannot handle tied weights — give lm_head its own copy
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16,
            config=config,
        )
        # FSDP cannot handle tied weights — break the tie if present
        if getattr(config, "tie_word_embeddings", False) and hasattr(model, "lm_head"):
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.ddp_find_unused_parameters == "true":
        ddp_find_unused_parameters = True
    elif args.ddp_find_unused_parameters == "false":
        ddp_find_unused_parameters = False
    else:
        # Gemma 4 instruct checkpoints include multimodal towers that are unused in
        # text-only SFT, so DDP must tolerate unused parameters.
        ddp_find_unused_parameters = world_size > 1 and model_type.startswith("gemma4")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=ddp_find_unused_parameters if world_size > 1 else None,
        dataloader_num_workers=min(8, os.cpu_count() or 1),
        fsdp=args.fsdp,
        fsdp_config=args.fsdp_config,
        save_only_model=args.save_only_model,
        save_total_limit=args.save_total_limit,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer, needs_mm_token_type_ids=model_type.startswith("gemma4")),
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
        else:
            trainer.save_model(str(final_dir))
            tokenizer.save_pretrained(str(final_dir))

    if distributed:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
