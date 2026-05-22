#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tb2_lite.scripts.summarize_corrected_results import load_rows  # noqa: E402


DEFAULT_RESULTS_DIR = Path("/home/work/.data/tb2_lite_eval/corrected_readme_models_vllm")
DEFAULT_OUTPUT_DIR = Path("/home/work/.data/hf_model_cards/LLM-OS-Models")
DEFAULT_STATE_PATH = DEFAULT_OUTPUT_DIR / "uploaded_state.json"
DEFAULT_ENV_PATH = ROOT / ".env"
CARD_VERSION = "tb2-corrected-card-2026-05-09-v5-native-gemma"


REPO_ALIASES = {
    "llm-os-models/ouro-1.4b-terminal-sft": "LLM-OS-Models/Ouro-1.4B-terminal-sft",
    "llm-os-models/ouro-2.6b-terminal-sft": "LLM-OS-Models/Ouro-2.6B-terminal-sft",
    "llm-os-models/ouro-1.4b-thinking-terminal-sft": "LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT",
    "llm-os-models/ouro-2.6b-thinking-terminal-sft": "LLM-OS-Models/Ouro-2.6B-Thinking-Terminal-SFT",
}


def load_env_token(path: Path) -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return os.environ["HUGGING_FACE_HUB_TOKEN"]
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        if key.strip() not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
            continue
        value = value.strip().strip("'\"")
        if value:
            return value
    return None


def normalize_repo(repo_id: str) -> str:
    return repo_id.strip().lower()


def safe_filename(repo_id: str) -> str:
    return repo_id.replace("/", "__") + ".md"


def infer_base_model(repo_id: str) -> str:
    name = repo_id.split("/", 1)[-1]
    if name.startswith("LFM2.5-1.2B"):
        return "LiquidAI/LFM2.5-1.2B-Instruct"
    if name.startswith("LFM2-2.6B"):
        return "LiquidAI/LFM2-2.6B"
    if name.startswith("LFM2-8B-A1B"):
        return "LiquidAI/LFM2-8B-A1B"
    if name.startswith("LFM2-8B"):
        return "LiquidAI/LFM2-8B"
    if name.startswith("LFM2-24B-A2B"):
        return "LiquidAI/LFM2-24B-A2B"
    if name.startswith("Qwen3.5-35B-A3B"):
        return "Qwen/Qwen3.5-35B-A3B"
    if name.startswith("Qwen3.5-"):
        size = name.split("-Terminal-", 1)[0].replace("Qwen3.5-", "")
        return f"Qwen/Qwen3.5-{size}"
    if name.startswith("Qwen3.6-35B-A3B"):
        return "Qwen/Qwen3.6-35B-A3B"
    if name.startswith("Qwen3.6-"):
        size = name.split("-Terminal-", 1)[0].replace("Qwen3.6-", "")
        return f"Qwen/Qwen3.6-{size}"
    if name.startswith("gemma-4-"):
        base = name.split("-Terminal-", 1)[0]
        return f"google/{base}"
    if name.startswith("Ouro-1.4B-Thinking"):
        return "ByteDance/Ouro-1.4B-Thinking"
    if name.startswith("Ouro-2.6B-Thinking"):
        return "ByteDance/Ouro-2.6B-Thinking"
    if name.startswith("Ouro-1.4B"):
        return "ByteDance/Ouro-1.4B"
    if name.startswith("Ouro-2.6B"):
        return "ByteDance/Ouro-2.6B"
    return "unknown"


def infer_training_method(repo_id: str) -> str:
    name = repo_id.split("/", 1)[-1]
    bits: list[str] = []
    if "1Epoch" in name:
        bits.append("1 epoch")
    elif "2Epoch" in name:
        bits.append("2 epochs")
    if "LiquidCLI" in name:
        bits.append("Liquid-CLI style preprocessing/training")
    if "Native-Liquid" in name:
        bits.append("Gemma native Liquid preprocessing")
    if "TemplateHoldout" in name:
        bits.append("chat-template aligned holdout split")
    if "TemplateMasked" in name:
        bits.append("assistant-only template-masked labels")
    if "Unsloth" in name:
        bits.append("Unsloth SFT")
    if "HF-FSDP" in name:
        bits.append("HF FSDP full fine-tuning")
    if "FullFT" in name:
        bits.append("full fine-tuning")
    if "DDP" in name:
        bits.append("DDP fine-tuning")
    if "SameCount" in name:
        bits.append("same-count data setting")
    if "2BData" in name:
        bits.append("2BData setting")
    return ", ".join(bits) if bits else "Terminal SFT"


def model_family(repo_id: str) -> str:
    name = repo_id.split("/", 1)[-1].lower()
    if name.startswith("qwen"):
        return "qwen"
    if name.startswith("lfm"):
        return "lfm"
    if name.startswith("gemma"):
        return "gemma"
    if name.startswith("ouro"):
        return "ouro"
    return "other"


def strengths_and_limits(row: dict[str, Any], repo_id: str) -> tuple[list[str], list[str]]:
    family = model_family(repo_id)
    score = float(row.get("score", 0.0))
    precision = float(row.get("precision", 0.0))
    recall = float(row.get("recall", 0.0))
    valid_json = float(row.get("valid_json", 0.0))
    strengths: list[str] = []
    limits: list[str] = []

    if score >= 38:
        strengths.append("현재 corrected TB2-lite 기준 상위권 점수이며, 터미널 명령 재현 안정성이 높습니다.")
    elif score >= 33:
        strengths.append("중상위권 점수로, 기본적인 터미널 next-action imitation은 비교적 안정적입니다.")
    else:
        strengths.append("특정 크기/가속 경로에서 비용 대비 빠른 추론을 기대할 수 있습니다.")

    if precision >= recall + 0.03:
        strengths.append("잘못된 명령을 많이 내기보다 보수적으로 맞는 명령을 내는 경향이 있습니다.")
        limits.append("recall이 상대적으로 낮아 필요한 명령 일부를 빠뜨릴 수 있습니다.")
    elif recall >= precision + 0.03:
        strengths.append("필요한 명령을 넓게 회수하려는 경향이 있습니다.")
        limits.append("precision이 상대적으로 낮아 불필요하거나 다른 명령이 섞일 수 있습니다.")
    else:
        strengths.append("precision/recall 균형이 비교적 맞는 편입니다.")

    if valid_json >= 95:
        strengths.append("평가 중 JSON 형식 유지율이 높습니다.")
    else:
        limits.append("JSON 형식 실패가 있어 실행 전에 파싱 검증/재시도가 필요합니다.")

    if family == "lfm":
        strengths.append("LFM 계열은 Liquid chat template과 터미널 SFT 포맷을 맞춘 경량/효율 실험에 유리합니다.")
        limits.append("Qwen 상위권 대비 command F1이 낮게 나온 결과는 지능 차이와 함께 포맷, 토크나이저, 학습 경로 차이가 섞인 값입니다.")
    elif family == "qwen":
        strengths.append("Qwen 계열은 이번 평가에서 명령 JSON 안정성과 command F1이 전반적으로 강했습니다.")
    elif family == "gemma":
        limits.append("Gemma 계열은 학습/평가 chat template 불일치에 민감하므로 vLLM chat_template 경로로만 비교해야 합니다.")
    elif family == "ouro":
        limits.append("Ouro 계열은 assistant-only masking 및 prompt template 일치 여부가 성능 해석에 큰 영향을 줍니다.")

    limits.append("이 모델은 자동 터미널 조작 보조용 SFT 모델이며, 일반 대화/범용 추론 성능을 보장하지 않습니다.")
    limits.append("생성 명령은 실제 실행 전에 sandbox, allowlist, human review 같은 안전장치를 거쳐야 합니다.")
    return strengths, limits


def family_interpretation(row: dict[str, Any], repo_id: str) -> list[str]:
    family = model_family(repo_id)
    score = float(row.get("score", 0.0))
    sec_per_step = float(row.get("sec_per_step", 0.0))
    notes: list[str] = []

    if family == "qwen":
        notes.append(
            "Qwen 계열은 base prior 자체가 강하고, 이번 corrected 평가에서도 chat template 경로가 정상 적용된 상태에서 최상위권 점수를 냈습니다."
        )
        notes.append(
            "평가 코드는 모델명을 보고 가산하지 않으며 `100 * avg_command_f1`만 순위 점수로 사용합니다. 높은 점수는 Qwen에 특화된 코드라기보다 터미널 next-action 포맷과 base/SFT 조합이 잘 맞은 결과로 해석합니다."
        )
    elif family == "lfm":
        notes.append(
            "LFM 계열은 base 점수 대비 SFT 상승폭이 크고 sec/step이 낮아, 반복 평가와 RL 실험을 돌리기 좋은 효율형 후보입니다."
        )
        notes.append(
            "다음 단계에서는 valid JSON, command precision, premature complete를 reward/penalty로 직접 잡는 RL이 가장 실용적입니다."
        )
    elif family == "ouro":
        notes.append(
            "Ouro 계열은 Thinking SFT 쪽에서 점수가 잘 올라가지만, 같은 평가 기준에서 LFM/Qwen 대비 sec/step이 커서 RL 대량 반복에는 비용이 큽니다."
        )
        notes.append(
            "점수는 의미 있으나 속도 병목이 있어, 주력보다는 상위 후보 확인용 ablation에 더 적합합니다."
        )
    elif family == "gemma":
        if "native-liquid" in repo_id.lower():
            notes.append(
                "이 repo는 Gemma 4 전용 chat template, thinking history 제거, assistant JSON-only target, base 모델 template 주입 정책으로 다시 학습한 native Liquid 결과입니다."
            )
            notes.append(
                "기존 Gemma SFT의 낮은 점수는 template/target 포맷 충돌과 일부 checkpoint/export 문제를 포함했으므로, 이 native 결과를 새 기준으로 봐야 합니다."
            )
        else:
            notes.append(
                "Gemma 계열은 chat template과 thinking marker 처리에 민감합니다. 현재 결과는 corrected vLLM 평가 스냅샷이며, Gemma native style 재학습/재평가 결과가 나오면 교체해야 합니다."
            )
            notes.append(
                "특히 기존 Gemma SFT 일부는 template 불일치 또는 로드 실패 이슈가 있었으므로, native Liquid/Gemma 전처리 결과를 별도 기준으로 봐야 합니다."
            )
    else:
        notes.append("이 결과는 TB2-lite 터미널 next-action JSON 재현 기준의 스냅샷입니다.")

    if sec_per_step > 0:
        if sec_per_step <= 2.0:
            notes.append(f"속도는 `{sec_per_step:.3f}` sec/step 수준으로 빠른 편입니다.")
        elif sec_per_step >= 5.0:
            notes.append(f"속도는 `{sec_per_step:.3f}` sec/step 수준으로 느린 편이라 배치 평가/RL 비용을 따로 봐야 합니다.")
        else:
            notes.append(f"속도는 `{sec_per_step:.3f}` sec/step 수준입니다.")

    if score >= 38:
        notes.append("RL 후보성: top-tier SFT로 reward tuning/GRPO 비교의 기준선 후보입니다.")
    elif family == "lfm" and score >= 31:
        notes.append("RL 후보성: 효율과 SFT 반응성이 좋아 Qwen 다음 주력 후보로 둘 수 있습니다.")
    elif family == "ouro" and score >= 35:
        notes.append("RL 후보성: 점수는 충분하지만 속도 비용 때문에 소규모 비교 후보로 두는 편이 안전합니다.")
    elif family == "gemma" and score >= 32:
        notes.append("RL 후보성: native 재평가에서도 이 수준 이상이면 Gemma 후보 1개를 추가할 만합니다.")
    else:
        notes.append("RL 후보성: 현재 점수만으로는 주력 후보보다 보조/비교군에 가깝습니다.")
    return notes


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def format_float(value: float) -> str:
    return f"{value:.4f}"


def result_fingerprint(row: dict[str, Any], repo_id: str, total_results: int) -> str:
    payload = {
        "card_version": CARD_VERSION,
        "repo_id": repo_id,
        "model_short": row.get("model_short"),
        "timestamp": row.get("timestamp"),
        "score": row.get("score"),
        "steps": row.get("steps"),
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def pending_fingerprint(repo_id: str, total_results: int) -> str:
    payload = {
        "card_version": CARD_VERSION,
        "repo_id": repo_id,
        "status": "pending",
        "total_results": total_results,
    }
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def recommended_tp(repo_id: str) -> int:
    name = repo_id.lower()
    if any(marker in name for marker in ("35b", "31b", "27b", "26b", "24b")):
        return 2
    return 1


def quickstart_lines(repo_id: str, model_short: str | None = None) -> list[str]:
    tp = recommended_tp(repo_id)
    short = model_short or safe_filename(repo_id).removesuffix(".md")
    gemma_extra = [
        "  --thinking-mode off \\",
        "  --strip-thinking-history auto \\",
        "  --gemma4-empty-thought-channel auto \\",
    ] if "gemma-4" in repo_id.lower() else []
    eval_command = [
        "python tb2_lite/scripts/replay_eval.py \\",
        f"  --model {repo_id} \\",
        f"  --model-short {short} \\",
        "  --eval-path tb2_lite/data/replay_full.jsonl \\",
        "  --output-dir /home/work/.data/tb2_lite_eval/corrected_readme_models_vllm \\",
        "  --dtype bfloat16 \\",
        f"  --tp {tp} \\",
        "  --max-model-len 49152 \\",
        "  --max-tokens 1024 \\",
        "  --temperature 0.0 \\",
        "  --top-p 1.0 \\",
        "  --gpu-memory-utilization 0.92 \\",
    ]
    eval_command.extend(gemma_extra)
    eval_command.append("  --language-model-only")

    return [
        "",
        "## Quickstart",
        "",
        "설치와 로그인:",
        "",
        "```bash",
        "pip install -U vllm transformers huggingface_hub",
        "huggingface-cli login",
        "```",
        "",
        "관련 코드:",
        "",
        "- GitHub: https://github.com/LLM-OS-Models/Terminal",
        "- vLLM 평가 실행: `tb2_lite/scripts/replay_eval.py`",
        "- chat template/fallback 생성: `tb2_lite/scripts/prompt_builder.py`",
        "- JSON/command 채점: `tb2_lite/scripts/replay_metrics.py`",
        "",
        "vLLM 직접 실행 예시. 평가 코드와 동일하게 chat template을 우선 사용하고, template이 없으면 ChatML/Gemma fallback을 사용합니다.",
        "",
        "```python",
        "from transformers import AutoTokenizer",
        "from vllm import LLM, SamplingParams",
        "",
        f"model_id = \"{repo_id}\"",
        f"tp = {tp}",
        "",
        "tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)",
        "llm = LLM(",
        "    model=model_id,",
        "    tokenizer=model_id,",
        "    trust_remote_code=True,",
        "    dtype=\"bfloat16\",",
        "    tensor_parallel_size=tp,",
        "    max_model_len=49152,",
        "    gpu_memory_utilization=0.92,",
        ")",
        "",
        "messages = [",
        "    {\"role\": \"system\", \"content\": \"You are a terminal automation assistant. Return JSON only.\"},",
        "    {\"role\": \"user\", \"content\": \"Inspect the current directory and list Python files.\"},",
        "]",
        "",
        "def render_chatml(messages):",
        "    parts = []",
        "    for message in messages:",
        "        role = \"assistant\" if message[\"role\"] == \"assistant\" else message[\"role\"]",
        "        if role == \"tool\":",
        "            role = \"user\"",
        "        parts.append(f\"<|im_start|>{role}\\n{message['content']}<|im_end|>\\n\")",
        "    parts.append(\"<|im_start|>assistant\\n\")",
        "    return \"\".join(parts)",
        "",
        "def render_gemma4_turn(messages, empty_thought_channel=False):",
        "    parts = [\"<bos>\"]",
        "    for message in messages:",
        "        role = \"model\" if message[\"role\"] == \"assistant\" else message[\"role\"]",
        "        if role == \"tool\":",
        "            role = \"user\"",
        "        parts.append(f\"<|turn>{role}\\n{message['content'].strip()}<turn|>\\n\")",
        "    parts.append(\"<|turn>model\\n\")",
        "    if empty_thought_channel:",
        "        parts.append(\"<|channel>thought\\n<channel|>\")",
        "    return \"\".join(parts)",
        "",
        "def render_prompt(model_id, tokenizer, messages):",
        "    model_key = model_id.lower()",
        "    if \"gemma-4\" in model_key:",
        "        try:",
        "            return tokenizer.apply_chat_template(",
        "                messages,",
        "                tokenize=False,",
        "                add_generation_prompt=True,",
        "                enable_thinking=False,",
        "            )",
        "        except Exception:",
        "            return render_gemma4_turn(",
        "                messages,",
        "                empty_thought_channel=(\"26b\" in model_key or \"31b\" in model_key),",
        "            )",
        "    try:",
        "        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
        "    except Exception:",
        "        return render_chatml(messages)",
        "",
        "prompt = render_prompt(model_id, tokenizer, messages)",
        "sampling = SamplingParams(",
        "    temperature=0.0,",
        "    top_p=1.0,",
        "    max_tokens=1024,",
        "    repetition_penalty=1.0,",
        ")",
        "outputs = llm.generate([prompt], sampling_params=sampling)",
        "print(outputs[0].outputs[0].text)",
        "```",
        "",
        "권장 출력 형식:",
        "",
        "```json",
        "{",
        "  \"analysis\": \"brief reasoning about the next terminal action\",",
        "  \"plan\": \"short execution plan\",",
        "  \"commands\": [",
        "    {\"keystrokes\": \"ls -la\\n\", \"duration\": 0.1}",
        "  ],",
        "  \"task_complete\": false",
        "}",
        "```",
        "",
        "평가와 동일한 replay 명령:",
        "",
        "```bash",
        *eval_command,
        "```",
        "",
        f"- 기본 권장 tensor parallel: `{tp}`. OOM이면 `--tp`와 `tensor_parallel_size`를 2/4/8로 올리세요.",
        "- corrected TB2-lite 평가는 `temperature=0.0`, `top_p=1.0`, `max_tokens=1024`로 고정했습니다.",
        "- Gemma 4는 JSON 출력을 위해 `enable_thinking=False`를 사용하고, 26B/31B 계열은 평가 코드에서 empty thought channel 처리를 자동 적용합니다.",
    ]


def build_rank_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    rank = 0
    ranks: dict[str, int] = {}
    for row in sorted(rows, key=lambda r: (-float(r.get("score", 0.0)), str(r.get("model_name", "")))):
        if row.get("status") != "ok" or not row.get("rank_eligible", False):
            continue
        rank += 1
        ranks[str(row.get("model_name", "")).lower()] = rank
    return ranks


def build_model_card(row: dict[str, Any], repo_id: str, rank: int | None, total_ranked: int, total_results: int) -> str:
    base_model = infer_base_model(repo_id)
    training_method = infer_training_method(repo_id)
    strengths, limits = strengths_and_limits(row, repo_id)
    family_notes = family_interpretation(row, repo_id)
    prompt_meta = {
        "template_status": row.get("template_status", "unknown"),
        "rank_eligible": row.get("rank_eligible", False),
    }
    if row.get("steps"):
        prompt_meta["steps"] = row.get("steps")
    if row.get("tasks"):
        prompt_meta["tasks"] = row.get("tasks")

    rank_text = f"{rank} / {total_ranked}" if rank is not None else "순위 제외"
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    eval_timestamp = row.get("timestamp") or "unknown"
    model_short = row.get("model_short", "")

    lines = [
        "---",
        "language:",
        "- en",
        "- ko",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "- terminal",
        "- sft",
        "- vllm",
        "- tb2-lite",
        f"base_model: {base_model}",
        "---",
        "",
        f"# {repo_id}",
        "",
        "터미널 작업 자동화를 위한 Terminal SFT 모델입니다. 입력된 작업/이전 터미널 상태를 보고 다음에 실행할 명령을 JSON 형태로 생성하는 용도로 학습했습니다.",
        "",
        "## 모델 요약",
        "",
        f"- Base model: `{base_model}`",
        f"- Training setup: `{training_method}`",
        f"- Evaluation snapshot: `{timestamp}`",
        f"- Evaluation result id: `{model_short}`",
        *quickstart_lines(repo_id, str(model_short or "")),
        "",
        "## 평가 결과",
        "",
        "평가는 corrected TB2-lite replay set에서 vLLM으로 수행했습니다. 순위 점수는 `100 * avg_command_f1`만 사용하고, `first_cmd_exact_pct`는 보조 지표로만 봅니다.",
        "",
        f"- Rank: `{rank_text}`",
        f"- Score: `{float(row.get('score', 0.0)):.2f}`",
        f"- Command F1: `{format_float(float(row.get('cmd_f1', 0.0)))}`",
        f"- Command precision: `{format_float(float(row.get('precision', 0.0)))}`",
        f"- Command recall: `{format_float(float(row.get('recall', 0.0)))}`",
        f"- First command exact: `{format_pct(float(row.get('first_exact', 0.0)))}`",
        f"- Valid JSON: `{format_pct(float(row.get('valid_json', 0.0)))}`",
        f"- Steps / tasks: `{row.get('steps', 0)} / {row.get('tasks', 0)}`",
        f"- Sec/step: `{float(row.get('sec_per_step', 0.0)):.3f}`",
        f"- Load time: `{float(row.get('load_time', 0.0)):.1f}s`",
        f"- Template status: `{row.get('template_status', 'unknown')}`",
        f"- Rank eligible: `{row.get('rank_eligible', False)}`",
        f"- Eval timestamp: `{eval_timestamp}`",
        f"- 현재 집계된 평가 결과 수: `{total_results}`",
        "",
        "Prompt/template audit:",
        "",
        "```json",
        json.dumps(prompt_meta, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 장점",
        "",
    ]
    lines.extend(f"- {item}" for item in strengths)
    lines.extend([
        "",
        "## 모델군 해석",
        "",
    ])
    lines.extend(f"- {item}" for item in family_notes)
    lines.extend([
        "",
        "## 한계와 주의사항",
        "",
    ])
    lines.extend(f"- {item}" for item in limits)
    lines.extend([
        "",
        "## 해석 메모",
        "",
        "TB2-lite 점수는 일반 지능 벤치마크가 아니라 터미널 next-action JSON 재현 능력을 측정합니다. 따라서 모델 크기, chat template 일치, assistant-only masking, tokenizer, 학습 데이터 holdout 여부가 모두 점수에 영향을 줍니다.",
        "",
        "루트 README.md의 값이 더 최신이면 해당 값을 우선 확인하세요. 이 모델카드는 완료된 평가 JSON을 기준으로 개별 저장소에 빠르게 반영한 스냅샷입니다.",
    ])
    return "\n".join(lines) + "\n"


def build_pending_model_card(repo_id: str, total_results: int) -> str:
    base_model = infer_base_model(repo_id)
    training_method = infer_training_method(repo_id)
    family = model_family(repo_id)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    if family == "qwen":
        family_note = "Qwen 계열은 현재 corrected TB2-lite에서 가장 강한 기준선입니다. 이 repo는 아직 현재 집계 JSON과 직접 매칭되는 점수가 없어 별도 평가가 필요합니다."
    elif family == "lfm":
        family_note = "LFM 계열은 빠른 sec/step과 큰 SFT 반응성이 장점입니다. 이 repo는 아직 현재 집계 JSON과 직접 매칭되는 점수가 없어 별도 평가가 필요합니다."
    elif family == "ouro":
        family_note = "Ouro 계열은 Thinking SFT 점수가 좋지만 속도 비용을 같이 봐야 합니다. 이 repo는 아직 현재 집계 JSON과 직접 매칭되는 점수가 없어 별도 평가가 필요합니다."
    elif family == "gemma":
        family_note = "Gemma 계열은 native Gemma/Liquid 전처리와 chat template 처리가 중요합니다. 이 repo는 corrected 평가가 끝나면 점수 카드로 교체합니다."
    else:
        family_note = "이 repo는 아직 현재 corrected TB2-lite 집계 JSON과 직접 매칭되는 점수가 없습니다."

    lines = [
        "---",
        "language:",
        "- en",
        "- ko",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "- terminal",
        "- sft",
        "- vllm",
        "- tb2-lite",
        "- evaluation-pending",
        f"base_model: {base_model}",
        "---",
        "",
        f"# {repo_id}",
        "",
        "터미널 작업 자동화를 위한 Terminal SFT 모델입니다. 입력된 작업/이전 터미널 상태를 보고 다음에 실행할 명령을 JSON 형태로 생성하는 용도로 학습했습니다.",
        "",
        "## 모델 요약",
        "",
        f"- Base model: `{base_model}`",
        f"- Training setup: `{training_method}`",
        f"- Model card snapshot: `{timestamp}`",
        f"- Corrected TB2-lite evaluated results currently indexed: `{total_results}`",
        "- Corrected TB2-lite score: `pending / not matched in current result directory`",
        *quickstart_lines(repo_id),
        "",
        "## 평가 상태",
        "",
        "- Current corrected TB2-lite score: `pending`",
        "- Reason: 현재 `/home/work/.data/tb2_lite_eval/corrected_readme_models_vllm` 집계 결과와 이 HF repo명이 직접 매칭되지 않았습니다.",
        "- Next step: 동일한 `tb2_lite/scripts/replay_eval.py` 경로로 평가를 돌린 뒤 점수 카드로 자동 교체합니다.",
        "",
        "## 모델군 해석",
        "",
        f"- {family_note}",
        "- TB2-lite 점수는 일반 지능 벤치마크가 아니라 터미널 next-action JSON 재현 능력을 측정합니다.",
        "- 생성 명령은 실제 실행 전에 sandbox, allowlist, human review 같은 안전장치를 거쳐야 합니다.",
    ]
    return "\n".join(lines) + "\n"


def load_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_org_repos(api: HfApi, org: str) -> dict[str, str]:
    repos = {}
    for model in api.list_models(author=org, limit=500):
        repos[normalize_repo(model.id)] = model.id
    for key, repo in REPO_ALIASES.items():
        repos.setdefault(key, repo)
    return repos


def matching_rows_by_repo(rows: list[dict[str, Any]], org_repos: dict[str, str]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        model_name = str(row.get("model_name", ""))
        key = normalize_repo(model_name)
        if key not in org_repos:
            continue
        repo_id = org_repos[key]
        current = selected.get(repo_id)
        if current is None or str(row.get("timestamp", "")) >= str(current.get("timestamp", "")):
            selected[repo_id] = row
    return selected


def upload_one(
    api: HfApi,
    repo_id: str,
    row: dict[str, Any],
    rank: int | None,
    total_ranked: int,
    total_results: int,
    output_dir: Path,
    dry_run: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    card = build_model_card(row, repo_id, rank, total_ranked, total_results)
    path = output_dir / safe_filename(repo_id)
    path.write_text(card, encoding="utf-8")
    if not dry_run:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update model card with corrected TB2-lite evaluation",
        )
    return path


def upload_pending(
    api: HfApi,
    repo_id: str,
    total_results: int,
    output_dir: Path,
    dry_run: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    card = build_pending_model_card(repo_id, total_results)
    path = output_dir / safe_filename(repo_id)
    path.write_text(card, encoding="utf-8")
    if not dry_run:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update model card with pending TB2-lite evaluation status",
        )
    return path


def run_once(args: argparse.Namespace, api: HfApi, state: dict[str, str]) -> int:
    rows = load_rows(args.results_dir)
    org_repos = get_org_repos(api, args.org)
    selected = matching_rows_by_repo(rows, org_repos)
    if args.repo:
        wanted = {normalize_repo(repo_id) for repo_id in args.repo}
        selected = {repo_id: row for repo_id, row in selected.items() if normalize_repo(repo_id) in wanted}
    rank_map = build_rank_map(rows)
    total_ranked = len(rank_map)
    total_results = sum(1 for row in rows if row.get("status") == "ok")
    changed = 0

    scored_items = sorted(
        selected.items(),
        key=lambda item: (-float(item[1].get("score", 0.0)), item[0].lower()),
    )
    for repo_id, row in scored_items:
        fingerprint = result_fingerprint(row, repo_id, total_results)
        if not args.force and state.get(repo_id) == fingerprint:
            continue
        rank = rank_map.get(str(row.get("model_name", "")).lower())
        path = upload_one(
            api=api,
            repo_id=repo_id,
            row=row,
            rank=rank,
            total_ranked=total_ranked,
            total_results=total_results,
            output_dir=args.output_dir,
            dry_run=not args.upload,
        )
        changed += 1
        action = "uploaded" if args.upload else "wrote"
        print(f"{action} {repo_id} score={float(row.get('score', 0.0)):.2f} path={path}", flush=True)
        if args.upload:
            state[repo_id] = fingerprint
            save_state(args.state_path, state)
        if args.limit and changed >= args.limit:
            break

    if args.include_pending and not args.repo and not (args.limit and changed >= args.limit):
        for repo_id in sorted(set(org_repos.values())):
            if repo_id in selected:
                continue
            if "Native-Liquid" in repo_id:
                continue
            fingerprint = pending_fingerprint(repo_id, total_results)
            if not args.force and state.get(repo_id) == fingerprint:
                continue
            path = upload_pending(
                api=api,
                repo_id=repo_id,
                total_results=total_results,
                output_dir=args.output_dir,
                dry_run=not args.upload,
            )
            changed += 1
            action = "uploaded-pending" if args.upload else "wrote-pending"
            print(f"{action} {repo_id} path={path}", flush=True)
            if args.upload:
                state[repo_id] = fingerprint
                save_state(args.state_path, state)
            if args.limit and changed >= args.limit:
                break
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update LLM-OS-Models Hugging Face model cards from corrected TB2-lite results.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--org", default="LLM-OS-Models")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--include-pending", action="store_true", help="Also update org repos without a matched corrected result JSON.")
    parser.add_argument("--repo", action="append", default=[], help="Only update this exact repo id. Can be passed multiple times.")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = load_env_token(args.env_path)
    if args.upload and not token:
        print("HF token not found. Put HF_TOKEN in environment or .env.", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    state = load_state(args.state_path)

    while True:
        try:
            uploaded = run_once(args, api, state)
            if uploaded == 0:
                print(f"no new completed model cards at {datetime.now(UTC).isoformat()}", flush=True)
        except Exception as exc:
            sanitized = re.sub(r"hf_[A-Za-z0-9_\\-]+", "hf_***", str(exc))
            print(f"model-card update error: {type(exc).__name__}: {sanitized}", file=sys.stderr, flush=True)
            if not args.watch:
                return 1
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
