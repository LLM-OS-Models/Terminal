#!/usr/bin/env python3
"""Fast single-GPU evaluator using transformers."""
import json, os, re, time, argparse, torch, sys
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

TB2_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "tb2_lite" / "scripts"
if str(TB2_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(TB2_SCRIPT_DIR))

from prompt_builder import build_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--eval-path", default="eval_dataset.jsonl")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    with open(args.eval_path) as f:
        samples = [json.loads(l) for l in f]

    print(f"[GPU {args.gpu}] Loading {args.model}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    load_time = time.time() - t0
    print(f"[GPU {args.gpu}] Loaded {n_params:.1f}B in {load_time:.0f}s")

    results = []
    t1 = time.time()

    for i, sample in enumerate(samples):
        convs = sample["conversations"]
        task = next((m["content"] for m in convs if m["role"] == "user"), "")
        ref = next((m["content"] for m in convs if m["role"] == "assistant"), "")

        text = build_prompt(tok, {"prompt": task}).prompt

        inputs = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        pred = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Metrics
        p_cmds = re.findall(r"```(?:bash|sh)?\s*\n(.*?)```", pred, re.DOTALL)
        p_cmds += re.findall(r"`([^`\n]+)`", pred)
        r_cmds = re.findall(r"```(?:bash|sh)?\s*\n(.*?)```", ref, re.DOTALL)
        r_cmds += re.findall(r"`([^`\n]+)`", ref)
        overlap = 0
        if r_cmds:
            for rc in r_cmds:
                rc_l = rc.strip().lower()
                if any(rc_l in pc.lower() or pc.lower() in rc_l for pc in p_cmds):
                    overlap += 1
            overlap /= len(r_cmds)

        results.append({
            "idx": i, "source": sample.get("_source_file", ""),
            "pred_preview": pred[:200], "has_cmds": len(p_cmds) > 0,
            "n_cmds": len(p_cmds), "cmd_overlap": round(overlap, 4),
            "thinking": "<think" in pred[:200], "pred_len": len(pred),
        })

        if (i+1) % 10 == 0:
            print(f"[GPU {args.gpu}] {i+1}/{len(samples)} done")

    gen_time = time.time() - t1
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
        "params_B": round(n_params, 1), "gpu": args.gpu,
        "samples": n, "load_time_sec": round(load_time),
        "gen_time_sec": round(gen_time, 1),
        "avg_sec_per_sample": round(gen_time / n, 2),
        "timestamp": datetime.now().isoformat(),
        "aggregate": agg, "per_sample": results,
    }

    out_path = os.path.join(args.output_dir, f"{model_short}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[GPU {args.gpu}] DONE {model_short}")
    print(f"  Load:{load_time:.0f}s Gen:{gen_time:.1f}s {gen_time/n:.2f}s/sample")
    print(f"  cmds:{agg['pct_has_cmds']}% overlap:{agg['avg_overlap']} think:{agg['pct_thinking']}%")

if __name__ == "__main__":
    main()
