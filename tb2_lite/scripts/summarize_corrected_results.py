#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


DISPLAY_NAME_BY_SHORT = {
    "gyung__LFM2_8B_Terminal_SFT_Unsloth_corrected303": "gyung/LFM2-8B-Terminal-SFT-Unsloth",
    "gyung_lfm2_8b_terminal_sft_unsloth": "gyung/LFM2-8B-Terminal-SFT-Unsloth",
    "nemotron_terminal_8b": "nvidia/Nemotron-Terminal-8B",
    "nemotron_terminal_14b": "nvidia/Nemotron-Terminal-14B",
    "nemotron_terminal_32b": "nvidia/Nemotron-Terminal-32B",
    "qwen35_2b_base": "Qwen/Qwen3.5-2B",
    "qwen35_4b_base": "Qwen/Qwen3.5-4B",
    "qwen35_9b_base": "Qwen/Qwen3.5-9B",
    "qwen35_27b_base": "Qwen/Qwen3.5-27B",
    "qwen36_27b_base": "Qwen/Qwen3.6-27B",
    "qwen36_35b_a3b_fp8_base": "Qwen/Qwen3.6-35B-A3B-FP8",
    "liquid_lfm25_1p2b_instruct_base": "LiquidAI/LFM2.5-1.2B-Instruct",
    "liquid_lfm2_2p6b_base": "LiquidAI/LFM2-2.6B",
    "liquid_lfm2_8b_a1b_base": "LiquidAI/LFM2-8B-A1B",
    "liquid_lfm2_24b_a2b_base": "LiquidAI/LFM2-24B-A2B",
    "qwen35_2b_sft_samecount_e1": "LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount",
    "qwen35_2b_sft_samecount_e2": "LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount",
    "qwen35_2b_sft_unsloth_e2": "LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth",
    "qwen35_4b_sft_2bdata_e1": "LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData",
    "qwen35_4b_sft_2bdata_e2": "LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData",
    "qwen35_9b_sft_2bdata_e1": "LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-1Epoch-FullFT-2BData",
    "qwen35_9b_sft_2bdata_e2": "LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData",
    "qwen35_27b_sft_hf_fsdp_e1": "LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData",
    "qwen35_27b_sft_hf_fsdp_e2": "LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData",
    "qwen35_35b_a3b_sft_hf_fsdp_e1": "LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData",
    "qwen35_35b_a3b_sft_hf_fsdp_e2": "LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData",
    "qwen36_27b_sft_hf_fsdp_e1": "LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData",
    "qwen36_27b_sft_hf_fsdp_e2": "LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData",
    "qwen36_35b_a3b_sft_hf_fsdp_e1": "LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData",
    "qwen36_35b_a3b_sft_hf_fsdp_e2": "LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData",
    "lfm25_1p2b_sft_unsloth_e2": "LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth",
    "lfm2_8b_sft_unsloth_e2": "LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth",
    "lfm2_2p6b_sft_unsloth_e2": "LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth",
    "lfm2_24b_a2b_sft_hf_fsdp_e1": "LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-2BData",
    "lfm2_24b_a2b_sft_hf_fsdp_e2": "LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-2BData",
    "gemma4_26b_a4b_it_base": "google/gemma-4-26B-A4B-it",
    "gemma4_31b_it_base": "google/gemma-4-31B-it",
    "gemma4_e4b_it_base": "google/gemma-4-E4B-it",
    "gemma4_e2b_it_base": "google/gemma-4-E2B-it",
    "gemma4_26b_a4b_sft_hf_fsdp_e1": "LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData",
    "gemma4_26b_a4b_sft_hf_fsdp_e2": "LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData",
    "gemma4_31b_sft_hf_fsdp_e1": "LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData",
    "gemma4_31b_sft_hf_fsdp_e2": "LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData",
    "gemma4_e2b_sft_ddp_e1": "LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-1Epoch-DDP-4GPU",
    "gemma4_e2b_sft_ddp_e2": "LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-DDP-4GPU",
    "gemma4_e2b_sft_hf_fsdp_e1": "LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData",
    "gemma4_e2b_sft_hf_fsdp_e2": "LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData",
    "gemma4_e4b_sft_ddp_e1": "LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-1Epoch-DDP-4GPU",
    "gemma4_e4b_sft_ddp_e2": "LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-2Epoch-DDP-4GPU",
    "ouro_2p6b_terminal_sft": "LLM-OS-Models/Ouro-2.6B-Terminal-SFT",
    "ouro_1p4b_terminal_sft": "LLM-OS-Models/Ouro-1.4B-Terminal-SFT",
    "ouro_1p4b_thinking_terminal_sft": "LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT",
    "ouro_2p6b_thinking_terminal_sft": "LLM-OS-Models/Ouro-2.6B-Thinking-Terminal-SFT",
    "bytedance_ouro_1p4b_base": "ByteDance/Ouro-1.4B",
    "bytedance_ouro_2p6b_base": "ByteDance/Ouro-2.6B",
    "bytedance_ouro_1p4b_thinking": "ByteDance/Ouro-1.4B-Thinking",
    "bytedance_ouro_2p6b_thinking": "ByteDance/Ouro-2.6B-Thinking",
    "jackrong_qwen35_27b_claude_opus_distill": "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
    "sft_h200_7gpu_processed_template_holdout_checkpoint_830_lfm_eval": "LLM-OS-Models/LFM2-8B-A1B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout",
    "sft_h200_7gpu_processed_template_holdout_checkpoint_1660_lfm_eval": "LLM-OS-Models/LFM2-8B-A1B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout",
    "sft_h200_8gpu_lfm25_1p2b_processed_template_holdout_checkpoint_545_lfm_eval": "LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout",
    "sft_h200_8gpu_lfm25_1p2b_processed_template_holdout_checkpoint_1090_lfm_eval": "LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout",
    "sft_h200_8gpu_lfm2_2p6b_processed_template_holdout_checkpoint_545_lfm_eval": "LLM-OS-Models/LFM2-2.6B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout",
    "sft_h200_8gpu_lfm2_2p6b_processed_template_holdout_checkpoint_1090_lfm_eval": "LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout",
    "lfm2_24b_a2b_hf_fsdp_8gpu_checkpoint_730_lfm_eval": "LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-TemplateMasked",
    "lfm2_24b_a2b_hf_fsdp_8gpu_checkpoint_1460_lfm_eval": "LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-TemplateMasked",
}


def display_name(model_short: str, model_path: str) -> str:
    mapped = DISPLAY_NAME_BY_SHORT.get(model_short)
    if mapped:
        return mapped
    if model_path and not model_path.startswith("/"):
        return model_path
    if model_path:
        path = Path(model_path)
        checkpoint = path.name if path.name.startswith("checkpoint-") else ""
        parent = path.parent.name if checkpoint else path.name
        suffix = f" ({checkpoint})" if checkpoint else ""
        return f"{parent}{suffix}"
    return model_short


def load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            agg = data["aggregate"]
        except Exception as exc:
            rows.append({
                "model_short": path.stem,
                "status": "parse_error",
                "error": str(exc),
            })
            continue
        prompt_meta = data.get("prompt_template", {})
        model_short = data.get("model_short", path.stem)
        model_path = data.get("model_path", "")
        rows.append({
            "model_short": model_short,
            "model_name": display_name(model_short, model_path),
            "model_path": model_path,
            "timestamp": data.get("timestamp", ""),
            "status": "ok",
            "score": 100.0 * float(agg.get("avg_command_f1", 0.0)),
            "cmd_f1": float(agg.get("avg_command_f1", 0.0)),
            "precision": float(agg.get("avg_command_precision", 0.0)),
            "recall": float(agg.get("avg_command_recall", 0.0)),
            "first_exact": float(agg.get("first_cmd_exact_pct", 0.0)),
            "valid_json": float(agg.get("valid_json_pct", 0.0)),
            "old_weighted": float(agg.get("next_action_score", 0.0)),
            "steps": int(agg.get("steps", 0)),
            "tasks": int(agg.get("tasks", 0)),
            "sec_per_step": float(data.get("avg_sec_per_step", 0.0)),
            "load_time": float(data.get("load_time_sec", 0.0)),
            "template_status": prompt_meta.get("template_status", "unknown"),
            "rank_eligible": bool(prompt_meta.get("rank_eligible", True)),
            "by_bucket": agg.get("by_bucket", {}),
        })
    rows = dedupe_rows(rows)
    rows.sort(key=lambda row: (row.get("status") != "ok", not row.get("rank_eligible", False), -row.get("score", 0.0)))
    return rows


def timestamp_key(row: dict) -> str:
    timestamp = row.get("timestamp")
    return timestamp if isinstance(timestamp, str) else ""


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """Keep one row per displayed HF repository/checkpoint name."""
    selected: dict[str, dict] = {}
    passthrough: list[dict] = []
    for row in rows:
        if row.get("status") != "ok":
            passthrough.append(row)
            continue
        key = str(row.get("model_name") or row.get("model_short"))
        current = selected.get(key)
        if current is None or timestamp_key(row) >= timestamp_key(current):
            selected[key] = row
    return list(selected.values()) + passthrough


def build_markdown(rows: list[dict], results_dir: Path, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"생성 시각: `{datetime.now(UTC).isoformat()}`",
        "",
        "이 문서는 corrected 303-step TB2-lite 평가 JSON을 다시 읽어서 정리한 결과다.",
        "",
        "점수 기준:",
        "",
        "```text",
        "score = 100 * avg_command_f1",
        "```",
        "",
        "`first_cmd_exact_pct`는 순위에 직접 섞지 않고 보조 지표로만 기록한다.",
        "",
        f"결과 디렉터리: `{results_dir}`",
        "",
        "## 전체 순위",
        "",
        "| 순위 | 모델(HF 저장소명) | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Template | Sec/Step | Load(s) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    rank = 0
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("rank_eligible"):
            rank += 1
            rank_text = str(rank)
        else:
            rank_text = "제외"
        lines.append(
            "| "
            f"{rank_text} | `{row['model_name']}` | {row['score']:.2f} | {row['cmd_f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['first_exact']:.1f}% | "
            f"{row['valid_json']:.1f}% | {row['template_status']} | "
            f"{row['sec_per_step']:.3f} | {row['load_time']:.1f} |"
        )

    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        lines.extend(["", "## 파싱 실패 파일", ""])
        for row in failures:
            lines.append(f"- `{row['model_short']}`: {row.get('error', '')}")

    lines.extend([
        "",
        "## 해석 기준",
        "",
        "- `rank_eligible=false` 또는 `Template=raw_fallback`인 결과는 정상 chat template 평가가 아니므로 순위에서 제외한다.",
        "- 같은 모델의 epoch/checkpoint 비교는 `Score`를 우선 보고, 거의 같으면 `Recall`, `Precision`, `Valid JSON`을 함께 본다.",
        "- README legacy 386-step 표와 직접 비교하지 않는다.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--title", default="Corrected TB2-lite 모델 평가 결과")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = load_rows(results_dir)
    markdown = build_markdown(rows, results_dir, args.title)
    Path(args.output_path).write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
