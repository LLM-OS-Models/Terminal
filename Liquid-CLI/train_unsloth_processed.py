import builtins
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import transformers
from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from terminal_sft_utils import is_holdout_row, load_holdout_keys, stable_hash, tokenizer_metadata


# Unsloth 내부 exec() 환경에서도 작동하도록 builtins에 강제 주입
if not hasattr(builtins, "PreTrainedConfig"):
    if hasattr(transformers, "PreTrainedConfig"):
        builtins.PreTrainedConfig = transformers.PreTrainedConfig
    elif hasattr(transformers, "PretrainedConfig"):
        builtins.PreTrainedConfig = transformers.PretrainedConfig


from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {
      "keystrokes": "ls -la\n",
      "duration": 0.1
    },
    {
      "keystrokes": "cd project\n",
      "duration": 0.1
    }
  ],
  "task_complete": true
}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- Most bash commands should end with a newline (\n) to cause them to execute
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {"keystrokes": "", "duration": 10.0} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command
- Extra text before or after the JSON will generate warnings but be tolerated
- The JSON must be valid - use proper escaping for quotes and special characters within strings
- Commands array can be empty if you want to wait without taking action"""


def _normalize_content(content):
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _normalize_tool_calls(tool_calls):
    if not tool_calls:
        return None
    if isinstance(tool_calls, str):
        tool_calls = tool_calls.strip()
        if not tool_calls:
            return None
        try:
            tool_calls = json.loads(tool_calls)
        except json.JSONDecodeError:
            return None
    if not isinstance(tool_calls, list):
        return None

    normalized = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments", {})
        if not name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"input": arguments}
        if not isinstance(arguments, dict):
            arguments = {"input": arguments}
        normalized.append(
            {
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": str(name),
                    "arguments": arguments,
                },
            }
        )
    return normalized or None


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        return None

    normalized = []
    first = messages[0]
    first_role = first.get("role")
    first_content = _normalize_content(first.get("content", ""))
    start_idx = 0

    if first_role == "system":
        normalized.append({"role": "system", "content": first_content})
        start_idx = 1
    elif first_role == "user" and first_content.startswith(SYSTEM_PROMPT):
        user_content = first_content[len(SYSTEM_PROMPT) :].lstrip()
        normalized.append({"role": "system", "content": SYSTEM_PROMPT})
        if user_content:
            normalized.append({"role": "user", "content": user_content})
        start_idx = 1
    else:
        normalized.append({"role": "system", "content": SYSTEM_PROMPT})
        if first_role == "user":
            normalized.append({"role": "user", "content": first_content})
            start_idx = 1

    for message in messages[start_idx:]:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        row = {
            "role": role,
            "content": _normalize_content(message.get("content", "")),
        }
        if role == "assistant":
            tool_calls = _normalize_tool_calls(message.get("tool_calls"))
            if tool_calls:
                row["tool_calls"] = tool_calls
        normalized.append(row)

    return normalized


def load_source_dataset(*, dataset_path, dataset_name, dataset_split):
    if dataset_path and Path(dataset_path).exists():
        return load_from_disk(dataset_path)
    if dataset_name:
        return load_dataset(dataset_name, split=dataset_split)
    raise ValueError("Either --dataset-path or --dataset-name must be provided.")


def dataset_artifact_paths(dataset_path):
    final_path = Path(dataset_path)
    tmp_path = final_path.parent / f"{final_path.name}.tmp"
    ready_path = final_path.parent / f"{final_path.name}.ready"
    return final_path, tmp_path, ready_path


def remove_dataset_artifact(dataset_path):
    final_path, tmp_path, ready_path = dataset_artifact_paths(dataset_path)
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if ready_path.exists():
        ready_path.unlink()
    if final_path.exists():
        shutil.rmtree(final_path)


def save_dataset_atomic(dataset, dataset_path):
    final_path, tmp_path, ready_path = dataset_artifact_paths(dataset_path)
    remove_dataset_artifact(dataset_path)
    dataset.save_to_disk(str(tmp_path))
    tmp_path.rename(final_path)
    ready_path.write_text("ok\n")


def metadata_path(dataset_path):
    final_path, _, _ = dataset_artifact_paths(dataset_path)
    return final_path / "prepare_meta.json"


def metadata_matches(dataset_path, expected_meta):
    path = metadata_path(dataset_path)
    if not path.exists():
        return False
    try:
        current = json.loads(path.read_text())
    except Exception:
        return False
    for key, value in expected_meta.items():
        if current.get(key) != value:
            return False
    return True


def write_metadata(dataset_path, meta):
    metadata_path(dataset_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def source_dataset_id(*, dataset_path, dataset_name, dataset_split):
    if dataset_path:
        return f"path:{dataset_path}"
    return f"hf:{dataset_name}:{dataset_split}"


def build_processed_meta(
    *,
    tokenizer,
    model_path,
    dataset_path,
    dataset_name,
    dataset_split,
    holdout_path,
    holdout_keys,
    response_part,
):
    return {
        "source_mode": "raw_conversations_template_text",
        "source_dataset": source_dataset_id(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
        ),
        "model_path": model_path,
        "holdout_path": holdout_path,
        "holdout_hash": stable_hash(json.dumps(sorted(holdout_keys))),
        "response_part": response_part,
        **tokenizer_metadata(tokenizer, model_path),
    }


def prepare_processed_dataset(
    *,
    dataset_path,
    dataset_name,
    dataset_split,
    processed_data_path,
    model_path,
    num_proc,
    holdout_path,
    response_part,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_source_dataset(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
    )
    holdout_keys = load_holdout_keys(holdout_path)
    raw_rows = len(dataset)
    if holdout_keys:
        dataset = dataset.filter(
            lambda example: not is_holdout_row(example, holdout_keys),
            num_proc=num_proc,
            desc="exclude_eval_holdout",
        )
    source_columns = list(dataset.column_names)
    bos = tokenizer.bos_token or ""

    def format_batch(examples):
        rows = []
        for conversation in examples["conversations"]:
            messages = normalize_messages(conversation)
            if not messages:
                rows.append("")
                continue
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if bos and text.startswith(bos):
                text = text[len(bos) :]
            rows.append(text)
        return {"text": rows}

    processed = dataset.map(
        format_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=source_columns,
        desc="format_lfm_chatml",
    )
    sample_count = min(len(processed), 100)
    if sample_count:
        marker_hits = sum(response_part in row["text"] for row in processed.select(range(sample_count)))
        if marker_hits == 0:
            raise RuntimeError(
                f"response_part marker was not found in formatted samples: {response_part!r}"
            )
    save_dataset_atomic(processed, processed_data_path)
    meta = build_processed_meta(
        tokenizer=tokenizer,
        model_path=model_path,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        holdout_path=holdout_path,
        holdout_keys=holdout_keys,
        response_part=response_part,
    )
    meta.update(
        {
            "raw_rows": raw_rows,
            "skipped_holdout_rows": raw_rows - len(dataset),
            "processed_rows": len(processed),
        }
    )
    write_metadata(processed_data_path, meta)


def wait_for_dataset(dataset_path, timeout_sec=7200):
    final_path, _, ready_path = dataset_artifact_paths(dataset_path)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if final_path.exists() and ready_path.exists():
            state_file = final_path / "state.json"
            info_file = final_path / "dataset_info.json"
            if state_file.exists() and info_file.exists():
                return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for dataset artifact: {dataset_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="LiquidAI/LFM2-8B-A1B",
    )
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--dataset-name",
        default="gyung/LFM2-Terminal-SFT-Processed",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--processed-data-path",
        default="/home/work/.data/liquid_cli_sft/datasets/gyung__LFM2-Terminal-SFT-Processed__lfm2_chatml",
    )
    parser.add_argument(
        "--train-ready-data-path",
        default="/home/work/.data/liquid_cli_sft/datasets/gyung__LFM2-Terminal-SFT-Processed__lfm2_chatml_train_ready",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/work/.data/liquid_cli_sft/models/LFM2-8B-A1B__terminal_sft_h200_7gpu_processed",
    )
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=24)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--dataset-num-proc", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--holdout-path", default="eval/eval_dataset.jsonl")
    parser.add_argument("--instruction-part", default="<|im_start|>user\n")
    parser.add_argument("--response-part", default="<|im_start|>assistant\n")
    parser.add_argument("--overwrite-processed-data", action="store_true")
    parser.add_argument("--overwrite-train-ready-data", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default="LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth-7GPU")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
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
        full_finetuning=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": f"cuda:{local_rank}"},
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
                    "dataset_split": args.dataset_split,
                    "holdout_path": args.holdout_path,
                    "processed_data_path": args.processed_data_path,
                    "train_ready_data_path": args.train_ready_data_path,
                    "using_train_ready_cache": use_train_ready_cache,
                    "output_dir": args.output_dir,
                    "world_size": world_size,
                    "max_seq_length": args.max_seq_length,
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "num_train_epochs": args.num_train_epochs,
                    "save_strategy": args.save_strategy,
                    "save_steps": args.save_steps if args.save_strategy == "steps" else None,
                    "dataset_rows": len(dataset),
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
        ddp_find_unused_parameters=False if world_size > 1 else None,
        dataset_num_proc=args.dataset_num_proc,
        dataset_kwargs={"skip_prepare_dataset": True} if use_train_ready_cache else None,
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

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
