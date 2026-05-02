# Terminal Benchmark 2 (TB2-lite) 평가 핸드오프 문서

> **마지막 업데이트:** 2026-05-02 14:20
> **목적:** 대화 클리어 후 이 문서만 보고 TB2-lite 평가를 처음부터 끝까지 수행할 수 있도록 모든 정보를 담음

---

## 1. 현재 상태 (14:20 기준)

### 실행 중인 평가

| GPU | VRAM | 하는 일 | 시작 | 예상 완료 |
|-----|------|--------|------|-----------|
| 0 | 65GB | **31B e2** transformers | 14:10 | ~16:30 |
| 1 | 65GB | **31B e1** transformers | 14:10 | ~16:30 |
| 2 | 17GB | **E4B e1** transformers | 14:10 | ~14:50 |
| 3 | 17GB | **E4B e2** transformers | 14:10 | ~14:50 |
| 4-7 | 비어있음 | — | — | — |

### 학습 상태
- **모든 SFT 학습 완료**
- E4B HF+FSDP 마지막으로 완료 (2026-05-02 14:00)

### 평가 완료 vs 미완료
- **완료 (21개):** 26B e1/e2 (+rp), LFM2 e1/e2 (+rp/min_p), E2B e1/e2 (+rp), 31B vLLM (전부 garbage)
- **실행 중 (4개):** 31B e1/e2 transformers, E4B e1/e2 HF+FSDP
- **README 반영:** 완료된 건 전부 반영됨. 위 4개 완료 후 추가 반영 필요

---

## 2. 프로젝트 개요

**TB2-lite (Terminal Benchmark 2 lite)** — LLM 모델이 터미널 명령어를 얼마나 정확하게 예측하는지 평가하는 벤치마크.

- 386개 스텝(50개 태스크)의 터미널 세션 리플레이 데이터
- 채점 공식: `next_action_score = 100 * (0.7 * avg_command_f1 + 0.3 * first_cmd_exact_pct / 100)`
- 모델이 `{"analysis": "...", "plan": "...", "command": "..."}` JSON 형태로 다음 명령어 예측

---

## 3. 하드웨어 및 환경

- **GPU:** 8x NVIDIA H200 (143GB VRAM each)
- **CUDA:** 570.86.10 / CUDA 12.9

### Python 가상환경

| 환경 | vLLM | 용도 |
|---|---|---|
| `.vllm-014` | 0.14.1 | LFM2 (torch 2.12+cu128, transformers 5.5.4) |
| `.vllm-0_19_1` | 0.19.1 | Gemma4-26B MoE + **transformers 평가** (accelerate 1.6.0 포함) |
| `.vllm-nightly` | 0.20.1rc1 cu129 | Gemma4 dense 시도용 (결국 불가 → transformers 사용) |

**주의:** `PYTHONNOUSERSITE=1` 필수. 안 하면 user-site torch 충돌.

---

## 4. 평가 스크립트

```
tb2_lite/scripts/
├── replay_eval.py              # vLLM 기반 (빠름, ~30-70초)
├── replay_eval_transformers.py # transformers 기반 (느림, 31B은 ~2-3시간)
└── replay_metrics.py           # 채점 로직
```

평가 데이터: `tb2_lite/data/replay_full.jsonl` (386 스텝)
결과 출력: `/home/work/.data/tb2_lite_eval/YYYYMMDDTHHMMSSZ/`

---

## 5. 모델 체크포인트

경로 prefix: `/home/work/.data/qwen_sft/models/`

| 모델 | 크기 | e1 | e2 | vLLM | 평가 상태 |
|---|---|---|---|---|---|
| Gemma4-26B-A4B | 26B(4B 활성) | ckpt-734 | ckpt-1468 | 0.19.1 | **완료** (18.12) |
| Gemma4-31B | 31B | ckpt-1467 | ckpt-2934 | X (transformers) | **실행 중** |
| Gemma4-E2B | 2B | ckpt-2934 | ckpt-5868 | X (transformers) | **완료** (6.54) |
| Gemma4-E4B | 4B | ckpt-2934 | ckpt-5868 | X (transformers) | **실행 중** |
| LFM2-24B-A2B | 24B(2B 활성) | ckpt-734 | ckpt-1468 | 0.14.1 | **완료** (15.41) |

디렉토리 이름 패턴: `{org}__{model}__terminal_sft_2epoch_hf_fsdp`

---

## 6. vLLM 호환성

| 모델 | 0.14.1 | 0.19.1 | 0.20.1 cu129 | Transformers |
|---|---|---|---|---|
| LFM2-24B-A2B | **O** | X | X | O |
| Gemma4-26B-A4B | X | **O** (`--language-model-only`) | X | O |
| Gemma4-31B | X | X (k_eq_v) | X (반복 붕괴) | **O** |
| Gemma4-E2B | X | X (k_eq_v) | 미확인 | **O** |
| Gemma4-E4B | X | X (k_eq_v) | 미확인 | **O** |

이슈:
- Gemma4 dense k_eq_v 버그 (PR #41253): 엔진 초기화 실패
- Gemma4 반복 붕괴 (gemma#622): "de la la" 반복 출력, 모델 레벨 버그

---

## 7. 완료된 평가 결과

> `/home/work/.data/tb2_lite_eval/20260502T012745Z/` 기준

| 모델 | Epoch | 백엔드 | 샘플링 | Score | F1 | Exact% | Valid JSON% |
|---|---|---|---|---:|---:|---:|---:|
| Gemma4-26B-A4B | e2 | vLLM 0.19.1 | 기본 | **18.12** | 0.2135 | 10.6 | 22.5 |
| Gemma4-26B-A4B | e2 | vLLM 0.19.1 | rp=1.05 | 17.95 | 0.2132 | 10.1 | 19.9 |
| Gemma4-26B-A4B | e1 | vLLM 0.19.1 | 기본 | 16.76 | 0.1961 | 10.1 | 12.4 |
| Gemma4-26B-A4B | e1 | vLLM 0.19.1 | rp=1.05 | 16.01 | 0.1975 | 7.3 | 13.5 |
| LFM2-24B-A2B | e2 | vLLM 0.14.1 | rp=1.05, min_p=0.15 | **15.41** | 0.1734 | 10.9 | 53.6 |
| LFM2-24B-A2B | e2 | vLLM 0.14.1 | 기본 | 14.08 | 0.1621 | 9.1 | 47.7 |
| LFM2-24B-A2B | e1 | vLLM 0.14.1 | rp=1.05, min_p=0.15 | 13.24 | 0.1592 | 7.0 | 54.4 |
| LFM2-24B-A2B | e1 | vLLM 0.14.1 | 기본 | 13.09 | 0.157 | 7.0 | 54.7 |
| Gemma4-E2B | e2 | transformers | 기본 | 6.54 | 0.0656 | 6.5 | 8.5 |
| Gemma4-E2B | e2 | transformers | rp=1.05 | 6.54 | 0.0656 | 6.5 | 8.5 |
| Gemma4-E2B | e1 | transformers | 기본 | 6.51 | 0.0652 | 6.5 | 4.1 |
| Gemma4-E2B | e1 | transformers | rp=1.05 | 6.51 | 0.0652 | 6.5 | 4.1 |
| Gemma4-31B | e1/e2 | vLLM 모든 버전 | — | 6.49 | 0.0648 | 6.5 | 0.0 |

---

## 8. 전체 평가 자동화 스크립트

GPU 8개를 빈틈없이 사용. 작업 큐에서 꺼내서 빈 GPU에 즉시 할당.

```bash
#!/bin/bash
# run_all_evals.sh — TB2-lite 전체 평가 (GPU 8개 풀활용)
# 사용법: bash run_all_evals.sh

set -euo pipefail
cd /home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/scripts

EVAL_DATA="/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/data/replay_full.jsonl"
OUTPUT_DIR="/home/work/.data/tb2_lite_eval/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTPUT_DIR"
echo "[$(date +%T)] Output: $OUTPUT_DIR"

MODELS="/home/work/.data/qwen_sft/models"
V014="../../.vllm-014/bin/python"
V014_PP="../../.vllm-014/lib/python3.12/site-packages"
V019="../../.vllm-0_19_1/bin/python"
V019_PP="../../.vllm-0_19_1/lib/python3.12/site-packages"

# ============================================
# 작업 큐 정의 (순서대로 실행)
# ============================================
declare -a CMD_QUEUE=()

# --- vLLM 0.14.1 (LFM2) — 빠름 (~30초) ---
CMD_QUEUE+=("V014|0|$MODELS/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-1468|lfm2_24b_a2b_e2_chat")
CMD_QUEUE+=("V014|0|$MODELS/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-734|lfm2_24b_a2b_e1_chat")
CMD_QUEUE+=("V014_RP|0|$MODELS/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-1468|lfm2_24b_a2b_e2_rp105")
CMD_QUEUE+=("V014_RP|0|$MODELS/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-734|lfm2_24b_a2b_e1_rp105")

# --- vLLM 0.19.1 (Gemma4-26B MoE) — 빠름 (~70초) ---
CMD_QUEUE+=("V019_LM|0|$MODELS/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-1468|gemma4_26b_a4b_e2_chat")
CMD_QUEUE+=("V019_LM|0|$MODELS/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-734|gemma4_26b_a4b_e1_chat")
CMD_QUEUE+=("V019_LM_RP|0|$MODELS/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-1468|gemma4_26b_a4b_e2_rp105")
CMD_QUEUE+=("V019_LM_RP|0|$MODELS/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-734|gemma4_26b_a4b_e1_rp105")

# --- Transformers (Gemma4 E2B) — 중간 (~30-50초) ---
CMD_QUEUE+=("TF|0|$MODELS/google__gemma-4-E2B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-5868|gemma4_e2b_e2_hf_fsdp")
CMD_QUEUE+=("TF|0|$MODELS/google__gemma-4-E2B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-2934|gemma4_e2b_e1_hf_fsdp")

# --- Transformers (Gemma4 E4B) — 느림 (~40-50분) ---
CMD_QUEUE+=("TF|0|$MODELS/google__gemma-4-E4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-5868|gemma4_e4b_e2_hf_fsdp")
CMD_QUEUE+=("TF|0|$MODELS/google__gemma-4-E4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-2934|gemma4_e4b_e1_hf_fsdp")

# --- Transformers (Gemma4 31B) — 매우 느림 (~2-3시간) ---
CMD_QUEUE+=("TF|0|$MODELS/google__gemma-4-31B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-2934|gemma4_31b_e2_transformers")
CMD_QUEUE+=("TF|0|$MODELS/google__gemma-4-31B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-1467|gemma4_31b_e1_transformers")

# ============================================
# GPU 풀 관리
# ============================================
NUM_GPUS=8
declare -A GPU_BUSY  # gpu_id -> pid

find_free_gpu() {
    for g in $(seq 0 $((NUM_GPUS-1))); do
        if [ -z "${GPU_BUSY[$g]:-}" ]; then
            echo "$g"
            return
        fi
    done
    echo ""
}

wait_for_free_gpu() {
    while true; do
        free=$(find_free_gpu)
        if [ -n "$free" ]; then
            echo "$free"
            return
        fi
        # 완료된 프로세스 정리
        for g in "${!GPU_BUSY[@]}"; do
            pid="${GPU_BUSY[$g]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "[$(date +%T)] GPU $g 완료 (pid $pid)" >&2
                unset GPU_BUSY[$g]
            fi
        done
        sleep 5
    done
}

wait_all_done() {
    while [ ${#GPU_BUSY[@]} -gt 0 ]; do
        for g in "${!GPU_BUSY[@]}"; do
            pid="${GPU_BUSY[$g]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "[$(date +%T)] GPU $g 완료 (pid $pid)" >&2
                unset GPU_BUSY[$g]
            fi
        done
        sleep 5
    done
}

# ============================================
# 작업 실행
# ============================================
QUEUE_IDX=0
TOTAL=${#CMD_QUEUE[@]}

for entry in "${CMD_QUEUE[@]}"; do
    QUEUE_IDX=$((QUEUE_IDX+1))
    IFS='|' read -r TYPE _GPU MODEL NAME <<< "$entry"

    gpu=$(wait_for_free_gpu)
    echo "[$(date +%T)] [$QUEUE_IDX/$TOTAL] GPU $gpu: $NAME 시작"

    case "$TYPE" in
        V014)
            CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 \
                PYTHONPATH=$V014_PP $V014 replay_eval.py \
                --model "$MODEL" --model-short "$NAME" \
                --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
                --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
                --max-tokens 1024 --temperature 0.0 &>/dev/null &
            ;;
        V014_RP)
            CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 \
                PYTHONPATH=$V014_PP $V014 replay_eval.py \
                --model "$MODEL" --model-short "$NAME" \
                --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
                --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
                --max-tokens 1024 --temperature 0.0 \
                --repetition-penalty 1.05 --min-p 0.15 &>/dev/null &
            ;;
        V019_LM)
            CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 \
                PYTHONPATH=$V019_PP $V019 replay_eval.py \
                --model "$MODEL" --model-short "$NAME" \
                --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
                --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
                --max-tokens 1024 --temperature 0.0 --language-model-only &>/dev/null &
            ;;
        V019_LM_RP)
            CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 \
                PYTHONPATH=$V019_PP $V019 replay_eval.py \
                --model "$MODEL" --model-short "$NAME" \
                --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
                --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
                --max-tokens 1024 --temperature 0.0 --language-model-only \
                --repetition-penalty 1.05 &>/dev/null &
            ;;
        TF)
            CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 \
                PYTHONPATH=$V019_PP $V019 replay_eval_transformers.py \
                --model "$MODEL" --model-short "$NAME" \
                --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
                --dtype bfloat16 --max-model-len 8192 --max-tokens 1024 --temperature 0.0 &>/dev/null &
            ;;
    esac

    GPU_BUSY[$gpu]=$!
done

echo "[$(date +%T)] 모든 작업 큐 소진. 완료 대기..."
wait_all_done

echo "[$(date +%T)] 전체 평가 완료!"
echo "결과: $OUTPUT_DIR"

# 결과 요약
echo ""
echo "=== 결과 요약 ==="
python3 -c "
import json, glob
for f in sorted(glob.glob('$OUTPUT_DIR/*.json')):
    name = f.split('/')[-1].replace('.json','')
    if name == 'summary': continue
    try:
        d = json.load(open(f))
        agg = d.get('aggregate', {})
        score = agg.get('next_action_score', 'N/A')
        f1 = agg.get('avg_command_f1', 'N/A')
        exact = agg.get('first_cmd_exact_pct', 'N/A')
        gen = d.get('gen_time_sec', '?')
        print(f'{name:40s} score={score:>6}  f1={f1}  exact={exact}%  gen={gen}s')
    except Exception as e:
        print(f'{name:40s} ERROR: {e}')
"
```

---

## 9. TODO

1. **실행 중인 4개 평가 완료 대기** → 결과 수집 → README 반영
2. **HuggingFace 업로드** — 26B, LFM2, E4B HF+FSDP 체크포인트 (토큰: 별도 전달)
3. **E4B/E2B vLLM nightly 시도** — 0.20.1 cu129에서 가능한지 아직 미확인

---

## 10. 참고 문서

- vLLM 공식: https://docs.vllm.ai/en/latest/
- Gemma4 vLLM: https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- LFM2 vLLM: https://docs.liquid.ai/deployment/gpu-inference/vllm
- LFM2 tool-use: https://docs.liquid.ai/lfm/key-concepts/tool-use
- Gemma4 반복 붕괴: https://github.com/google-deepmind/gemma/issues/622
- vLLM k_eq_v PR: https://github.com/vllm-project/vllm/pull/41253
