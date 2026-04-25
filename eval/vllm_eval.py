#!/usr/bin/env python3
"""Fast vLLM-based evaluator. One model, one GPU, all 50 samples."""
import json, os, re, time, argparse
from datetime import datetime
from vllm import LLM, SamplingParams

def extract_task(conversations):
    for m in conversations:
        if m["role"] == "user":
            return m["content"]
    return ""

def extract_ref(conversations):
    for m in conversations:
        if m["role"] == "assistant":
            return m["content"]
    return ""


def build_prompt(tokenizer, model_name, task):
    msgs = [{"role": "user", "content": task}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        if "lfm" in model_name.lower():
            return f"<|user|>\n{task}<|end_of_text|>\n<|assistant|>\n"
        return f"User: {task}\nAssistant: "

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gdn-triton", action="store_true")
    parser.add_argument("--eval-path", default="eval_dataset.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Don't override CUDA_VISIBLE_DEVICES - it's set externally per process
    gpu_label = os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu))

    with open(args.eval_path) as f:
        samples = [json.loads(l) for l in f]

    print(f"[GPU {gpu_label}] Loading {args.model} with vLLM...")
    t0 = time.time()
    kwargs = dict(
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        disable_log_stats=True,
    )
    if args.gdn_triton:
        kwargs["additional_config"] = {"gdn_prefill_backend": "triton"}
    llm = LLM(model=args.model, **kwargs)
    load_time = time.time() - t0
    print(f"[GPU {gpu_label}] Loaded in {load_time:.0f}s")

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0)

    # Build prompts
    tokenizer = llm.get_tokenizer()
    prompts = []
    for s in samples:
        task = extract_task(s["conversations"])
        prompts.append(build_prompt(tokenizer, args.model, task))

    print(f"[GPU {gpu_label}] Generating {len(prompts)} responses...")
    t1 = time.time()
    outputs = llm.generate(prompts, sampling)
    gen_time = time.time() - t1

    # Compute metrics
    results = []
    for i, (out, sample) in enumerate(zip(outputs, samples)):
        pred = out.outputs[0].text
        ref = extract_ref(sample["conversations"])

        # Count code blocks / commands in prediction
        pred_cmds = re.findall(r"```(?:bash|sh|shell)?\s*\n(.*?)```", pred, re.DOTALL)
        pred_cmds += re.findall(r"`([^`\n]+)`", pred)
        ref_cmds = re.findall(r"```(?:bash|sh|shell)?\s*\n(.*?)```", ref, re.DOTALL)
        ref_cmds += re.findall(r"`([^`\n]+)`", ref)

        overlap = 0
        if ref_cmds:
            for rc in ref_cmds:
                rc_s = rc.strip().lower()
                if any(rc_s in pc.lower() or pc.lower() in rc_s for pc in pred_cmds):
                    overlap += 1
            overlap /= len(ref_cmds)

        results.append({
            "idx": i,
            "source": sample.get("_source_file", ""),
            "pred_preview": pred[:300],
            "has_cmds": len(pred_cmds) > 0,
            "n_cmds": len(pred_cmds),
            "cmd_overlap": round(overlap, 4),
            "thinking": "<think" in pred[:200],
            "pred_len": len(pred),
        })

    n = len(results)
    agg = {
        "avg_pred_len": round(sum(r["pred_len"] for r in results) / n),
        "pct_has_cmds": round(sum(r["has_cmds"] for r in results) / n * 100, 1),
        "avg_cmds": round(sum(r["n_cmds"] for r in results) / n, 1),
        "avg_overlap": round(sum(r["cmd_overlap"] for r in results) / n, 4),
        "pct_thinking": round(sum(r["thinking"] for r in results) / n * 100, 1),
    }

    model_short = args.model.split("/")[-1]
    summary = {
        "model": args.model, "model_short": model_short,
        "gpu": gpu_label, "samples": n,
        "load_time_sec": round(load_time), "gen_time_sec": round(gen_time, 1),
        "avg_sec_per_sample": round(gen_time / n, 2),
        "timestamp": datetime.now().isoformat(),
        "aggregate": agg, "per_sample": results,
    }

    out_path = os.path.join(args.output_dir, f"{model_short}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[GPU {gpu_label}] DONE {model_short}")
    print(f"  Load: {load_time:.0f}s | Gen: {gen_time:.1f}s | {gen_time/n:.2f}s/sample")
    print(f"  Has cmds: {agg['pct_has_cmds']}% | Overlap: {agg['avg_overlap']} | Thinking: {agg['pct_thinking']}%")
    print(f"  Saved: {out_path}")

if __name__ == "__main__":
    main()
