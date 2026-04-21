#!/usr/bin/env python3
"""
Terminal Agent Model Evaluator

Usage:
    python run_eval.py --model <hf_model_id> --gpu <gpu_id> [--max-samples N] [--output-dir DIR]

Example:
    # Single model on GPU 0
    python run_eval.py --model Qwen/Qwen3.5-2B --gpu 0

    # Test with 10 samples first
    python run_eval.py --model google/gemma-4-E2B-it --gpu 1 --max-samples 10
"""

import argparse
import json
import os
import re
import time
import torch
from pathlib import Path
from datetime import datetime


def load_model(model_name, gpu_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
    import gc

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[GPU {gpu_id}] AutoModelForCausalLM failed: {e}, trying trust_remote_code=True with device_map=auto")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[GPU {gpu_id}] Loaded {model_name} ({n_params:.1f}B params)")
    return model, tokenizer, n_params


def extract_task_prompt(conversations):
    first_user = None
    for msg in conversations:
        if msg["role"] == "user":
            first_user = msg["content"]
            break
    return first_user


def extract_reference_response(conversations):
    assistant_msgs = [m for m in conversations if m["role"] == "assistant"]
    if not assistant_msgs:
        return ""
    return assistant_msgs[0]["content"]


def extract_commands(text):
    patterns = [
        r"```(?:bash|sh|shell)?\s*\n(.*?)```",
        r"`([^`]+)`",
        r"^\s*[$>]\s*(.+)$",
        r"(?:run|execute|use|type|enter|command):\s*`?([^`\n]+)`?",
    ]
    commands = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
        commands.extend(matches)
    return [c.strip() for c in commands if c.strip()]


def compute_metrics(prediction, reference, conversations):
    if not prediction or not reference:
        return {
            "has_response": bool(prediction),
            "response_length": len(prediction) if prediction else 0,
            "has_commands": False,
            "command_count": 0,
            "has_thinking": False,
            "reference_cmd_overlap": 0.0,
        }

    pred_cmds = extract_commands(prediction)
    ref_cmds = extract_commands(reference)

    overlap = 0
    if ref_cmds:
        for rc in ref_cmds:
            rc_lower = rc.lower().strip()
            for pc in pred_cmds:
                if rc_lower in pc.lower() or pc.lower() in rc_lower:
                    overlap += 1
                    break
        overlap = overlap / len(ref_cmds)

    return {
        "has_response": True,
        "response_length": len(prediction),
        "has_commands": len(pred_cmds) > 0,
        "command_count": len(pred_cmds),
        "has_thinking": "<think" in prediction[:200] or "Okay," in prediction[:100],
        "reference_cmd_overlap": round(overlap, 4),
        "pred_cmds_sample": pred_cmds[:5],
    }


def generate_response(model, tokenizer, prompt, model_name, max_new_tokens=2048):
    if "gemma" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif "qwen" in model_name.lower():
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"User: {prompt}\nAssistant: "

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate(model_name, gpu_id, eval_path, max_samples=None, output_dir="results"):
    model, tokenizer, n_params = load_model(model_name, gpu_id)

    with open(eval_path) as f:
        samples = [json.loads(line) for line in f]

    if max_samples:
        samples = samples[:max_samples]

    results = []
    total_time = 0

    for i, sample in enumerate(samples):
        task_prompt = extract_task_prompt(sample["conversations"])
        reference = extract_reference_response(sample["conversations"])

        if not task_prompt:
            continue

        t0 = time.time()
        prediction = generate_response(model, tokenizer, task_prompt, model_name)
        elapsed = time.time() - t0
        total_time += elapsed

        metrics = compute_metrics(prediction, reference, sample["conversations"])
        metrics["generation_time_sec"] = round(elapsed, 2)

        results.append({
            "sample_idx": i,
            "source_file": sample.get("_source_file", ""),
            "task": sample.get("task", ""),
            "prediction_preview": prediction[:500] if prediction else "",
            "reference_preview": reference[:500] if reference else "",
            "metrics": metrics,
        })

        if (i + 1) % 10 == 0:
            print(f"[GPU {gpu_id}] {model_name}: {i+1}/{len(samples)} done ({elapsed:.1f}s last)")

    model_short = model_name.split("/")[-1]
    summary = {
        "model": model_name,
        "model_short": model_short,
        "params_B": round(n_params, 1),
        "gpu": gpu_id,
        "num_samples": len(results),
        "total_time_sec": round(total_time, 1),
        "avg_time_per_sample": round(total_time / max(len(results), 1), 2),
        "timestamp": datetime.now().isoformat(),
        "aggregate_metrics": {
            "avg_response_length": round(sum(r["metrics"]["response_length"] for r in results) / max(len(results), 1), 0),
            "pct_has_commands": round(sum(1 for r in results if r["metrics"]["has_commands"]) / max(len(results), 1) * 100, 1),
            "avg_command_count": round(sum(r["metrics"]["command_count"] for r in results) / max(len(results), 1), 1),
            "avg_cmd_overlap": round(sum(r["metrics"]["reference_cmd_overlap"] for r in results) / max(len(results), 1), 4),
            "pct_has_thinking": round(sum(1 for r in results if r["metrics"]["has_thinking"]) / max(len(results), 1) * 100, 1),
        },
        "per_sample_results": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_short}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[GPU {gpu_id}] {model_name} done!")
    print(f"  Samples: {len(results)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time/sample: {total_time/max(len(results),1):.2f}s")
    print(f"  Has commands: {summary['aggregate_metrics']['pct_has_commands']}%")
    print(f"  Avg cmd overlap: {summary['aggregate_metrics']['avg_cmd_overlap']}")
    print(f"  Saved: {out_path}")

    del model
    torch.cuda.empty_cache()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on terminal tasks")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--gpu", type=int, required=True, help="GPU device ID")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--eval-path", default="eval_dataset.jsonl", help="Path to eval JSONL")
    args = parser.parse_args()

    evaluate(args.model, args.gpu, args.eval_path, args.max_samples, args.output_dir)
