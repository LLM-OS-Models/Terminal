# Terminal Benchmark 2 (TB2-lite) 평가 핸드오프 문서

> **마지막 업데이트:** 2026-05-02 14:05
> **목적:** 대화 클리어 후 이 문서만 보고 TB2-lite 평가를 처음부터 끝까지 수행할 수 있도록 모든 정보를 담음

---

## 1. 프로젝트 개요

**TB2-lite (Terminal Benchmark 2 lite)** — LLM 모델이 터미널 명령어를 얼마나 정확하게 예측하는지 평가하는 벤치마크.

### 평가 방식
- 386개 스텝(50개 태스크)의 터미널 세션 리플레이 데이터
- 모델이 현재 상태를 보고 **다음에 실행할 명령어**를 예측
- JSON 형태로 `{"analysis": "...", "plan": "...", "command": "..."}` 출력 기대

### 채점 공식
```
next_action_score = 100 * (0.7 * avg_command_f1 + 0.3 * first_cmd_exact_pct / 100)
```
- `avg_command_f1`: 예측 명령어와 정답 명령어의 토큰 단위 F1 평균
- `first_cmd_exact_pct`: 첫 번째 명령어가 완전 일치하는 비율

---

## 2. 하드웨어 및 환경

### 서버 사양
- **GPU:** 8x NVIDIA H200 (143GB VRAM each)
- **CUDA Driver:** 570.86.10 / CUDA 12.9
- **OS:** Linux 5.15.0-151-generic

### Python 가상환경 (3개 사용)

| 환경 디렉토리 | vLLM 버전 | 용도 | 비고 |
|---|---|---|---|
| `.vllm-014` | 0.14.1 | LFM2 모델 전용 | torch 2.12.0+cu128, transformers 5.5.4 |
| `.vllm-0_19_1` | 0.19.1 | Gemma4-26B MoE + transformers 평가 | torch 2.12.0+cu128, transformers 5.5.4, accelerate 1.6.0 |
| `.vllm-nightly` | 0.20.1rc1 cu129 | Gemma4 dense 모델 시도용 (31B/E4B) | **결국 vLLM으로는 31B/E4B 불가 → transformers 사용** |

### 중요: PYTHONNOUSERSITE
모든 실행 시 `PYTHONNOUSERSITE=1` 필수. 안 하면 user-site torch가 충돌.

---

## 3. 평가 스크립트

### 스크립트 위치
```
tb2_lite/scripts/
├── replay_eval.py              # vLLM 기반 평가
├── replay_eval_transformers.py # transformers 기반 평가 (vLLM 안 되는 모델용)
└── replay_metrics.py           # 채점 로직
```

### 평가 데이터
```
tb2_lite/data/replay_full.jsonl  # 386 스텝, 50 태스크
```

### 결과 출력 디렉토리
```
/home/work/.data/tb2_lite_eval/YYYYMMDDTHHMMSSZ/  # 타임스탬프별
```

---

## 4. 평가 실행 커맨드

### 4-1. vLLM 기반 평가 (LFM2)

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/scripts

# LFM2 (vLLM 0.14.1) — 기본
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-014/lib/python3.12/site-packages \
  ../../.vllm-014/bin/python replay_eval.py \
  --model $MODEL_PATH \
  --model-short $NAME \
  --eval-path /home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/data/replay_full.jsonl \
  --output-dir /home/work/.data/tb2_lite_eval/20260502T012745Z \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --max-tokens 1024 --temperature 0.0

# LFM2 + repetition_penalty + min_p (Liquid 공식 권장)
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-014/lib/python3.12/site-packages \
  ../../.vllm-014/bin/python replay_eval.py \
  --model $MODEL_PATH \
  --model-short $NAME \
  --eval-path /home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/data/replay_full.jsonl \
  --output-dir /home/work/.data/tb2_lite_eval/20260502T012745Z \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --max-tokens 1024 --temperature 0.0 \
  --repetition-penalty 1.05 --min-p 0.15
```

### 4-2. vLLM 기반 평가 (Gemma4-26B MoE)

```bash
# Gemma4 MoE (vLLM 0.19.1) — --language-model-only 필수
CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-0_19_1/lib/python3.12/site-packages \
  ../../.vllm-0_19_1/bin/python replay_eval.py \
  --model $MODEL_PATH \
  --model-short $NAME \
  --eval-path /home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/data/replay_full.jsonl \
  --output-dir /home/work/.data/tb2_lite_eval/20260502T012745Z \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --max-tokens 1024 --temperature 0.0 \
  --language-model-only
```

### 4-3. Transformers 기반 평가 (Gemma4 31B, E4B — vLLM 불가)

```bash
# Gemma4 dense 모델 (transformers 직접 추론)
# 31B: GPU 2개 필요 (~72GB VRAM), E4B: GPU 1개로 가능
CUDA_VISIBLE_DEVICES=$GPUS PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-0_19_1/lib/python3.12/site-packages \
  ../../.vllm-0_19_1/bin/python replay_eval_transformers.py \
  --model $MODEL_PATH \
  --model-short $NAME \
  --eval-path /home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/data/replay_full.jsonl \
  --output-dir /home/work/.data/tb2_lite_eval/20260502T012745Z \
  --dtype bfloat16 --max-model-len 8192 --max-tokens 1024 --temperature 0.0
```

### 속도 참고
- vLLM: 386 스텝 ~30-70초
- Transformers 31B (2 GPU): ~45-50분
- Transformers E4B (1 GPU): ~40-45분

---

## 5. 모델 체크포인트

### SFT 학습 완료 모델 (모두 학습 완료)

| 모델 | 베이스 모델 크기 | 체크포인트 경로 | e1 체크포인트 | e2 체크포인트 |
|---|---|---|---|---|
| **Gemma4-26B-A4B** (MoE) | 26B (활성 4B) | `.../google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/` | checkpoint-734 | checkpoint-1468 |
| **Gemma4-31B** (dense) | 31B | `.../google__gemma-4-31B-it__terminal_sft_2epoch_hf_fsdp/` | checkpoint-1467 | checkpoint-2934 |
| **Gemma4-E2B** | 2B | `.../google__gemma-4-E2B-it__terminal_sft_2epoch_hf_fsdp/` | checkpoint-2934 | checkpoint-5868 |
| **Gemma4-E4B** | 4B | `.../google__gemma-4-E4B-it__terminal_sft_2epoch_hf_fsdp/` | checkpoint-2934 | checkpoint-5868 |
| **LFM2-24B-A2B** (MoE) | 24B (활성 2B) | `.../LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/` | checkpoint-734 | checkpoint-1468 |

공통 경로 prefix: `/home/work/.data/qwen_sft/models/`

### 기타 학습된 모델 (참고용)
- `Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/` — checkpoint-55
- `Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata/` — checkpoint-960, checkpoint-1920
- `Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata/` — checkpoint-2193, checkpoint-4386
- `Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp/` — checkpoint-1917, checkpoint-3834
- `Qwen__Qwen3.6-35B-A3B__terminal_sft_2epoch_hf_fsdp/` — checkpoint-2934, checkpoint-5868
- `google__gemma-4-E2B-it__terminal_sft_2epoch_ddp_4gpu/` — checkpoint-367, checkpoint-734
- `google__gemma-4-E4B-it__terminal_sft_2epoch_ddp_4gpu/` — checkpoint-1467, checkpoint-2934

---

## 6. vLLM 호환성 정리

| 모델 | vLLM 0.14.1 | vLLM 0.19.1 | vLLM 0.20.1 cu129 | Transformers |
|---|---|---|---|---|
| LFM2-24B-A2B | **O** | X | X | O |
| Gemma4-26B-A4B | X | **O** (`--language-model-only`) | X | O |
| Gemma4-31B | X | X (k_eq_v 버그) | X (반복 붕괴) | **O** |
| Gemma4-E2B | X | X (k_eq_v 버그) | 미확인 | **O** |
| Gemma4-E4B | X | X (k_eq_v 버그) | 미확인 | **O** |

### 알려진 이슈
1. **Gemma4 dense k_eq_v 버그** (vLLM PR #41253): vLLM 0.19.1에서 Gemma4 dense 모델(31B, E2B, E4B) 엔진 초기화 실패
2. **Gemma4 반복 붕괴** (google-deepmind/gemma#622): vLLM nightly로는 엔진은 뜨지만 "de la la" 반복 출력. 긴 프롬프트에서 발생하는 모델 레벨 버그
3. **PYTHONNOUSERSITE=1 필수**: user-site torch와 충돌 방지

---

## 7. 현재까지 완료된 평가 결과

> `/home/work/.data/tb2_lite_eval/20260502T012745Z/` 기준

### 결과 테이블

| 모델 | 체크포인트 | 백엔드 | 샘플링 | Score | F1 | Exact% | gen시간 |
|---|---|---|---|---|---|---|---|
| Gemma4-26B-A4B | e2 (1468) | vLLM 0.19.1 | 기본 | **18.12** | 0.2135 | 10.6 | 69.5s |
| Gemma4-26B-A4B | e2 (1468) | vLLM 0.19.1 | rp=1.05 | 17.95 | 0.2132 | 10.1 | 72.2s |
| Gemma4-26B-A4B | e1 (734) | vLLM 0.19.1 | 기본 | 16.76 | 0.1961 | 10.1 | 70.3s |
| Gemma4-26B-A4B | e1 (734) | vLLM 0.19.1 | rp=1.05 | 16.01 | 0.1975 | 7.3 | 73.2s |
| LFM2-24B-A2B | e2 (1468) | vLLM 0.14.1 | rp=1.05, min_p=0.15 | **15.41** | 0.1734 | 10.9 | 36.3s |
| LFM2-24B-A2B | e2 (1468) | vLLM 0.14.1 | 기본 | 14.08 | 0.1621 | 9.1 | 30.4s |
| LFM2-24B-A2B | e1 (734) | vLLM 0.14.1 | rp=1.05, min_p=0.15 | 13.24 | 0.1592 | 7.0 | 38.9s |
| LFM2-24B-A2B | e1 (734) | vLLM 0.14.1 | 기본 | 13.09 | 0.157 | 7.0 | 31.0s |
| Gemma4-E2B | e2 (5868) | transformers | 기본 | 6.54 | 0.0656 | 6.5 | 45.9s |
| Gemma4-E2B | e2 (5868) | transformers | rp=1.05 | 6.54 | 0.0656 | 6.5 | 29.3s |
| Gemma4-E2B | e1 (2934) | transformers | 기본 | 6.51 | 0.0652 | 6.5 | 47.5s |
| Gemma4-E2B | e1 (2934) | transformers | rp=1.05 | 6.51 | 0.0652 | 6.5 | 29.5s |
| Gemma4-31B | e2 (2934) | vLLM nightly | 기본 | 6.49 | 0.0648 | 6.5 | 327.7s |
| Gemma4-31B | e1 (1467) | vLLM 0.19.1 | 기본 | 6.49 | 0.0648 | 6.5 | 315.0s |

### 결과 없는 체크포인트 (non-chat 프롬프트 결과만 있음)
| Gemma4-26B-A4B | e1 | vLLM 0.19.1 | non-chat | 13.33 | 0.1549 | 8.3 | 71.5s |
| Gemma4-26B-A4B | e2 | vLLM 0.19.1 | non-chat | 12.73 | 0.1454 | 8.5 | 71.7s |
| Gemma4-31B | e2 | vLLM 0.19.1 | non-chat | 6.49 | 0.0648 | 6.5 | 315.2s |
| LFM2-24B-A2B | e1 | vLLM 0.14.1 | non-chat | 6.49 | 0.0648 | 6.5 | 21.2s |
| LFM2-24B-A2B | e2 | vLLM 0.14.1 | non-chat | 6.49 | 0.0648 | 6.5 | 20.0s |

---

## 8. 현재 진행 중인 평가 (2026-05-02 14:00 기준)

| 작업 | GPU | 상태 | 시작시간 |
|---|---|---|---|
| 31B e2 transformers | 5, 6 | **실행 중** (~50분 경과) | 13:15 |
| E4B e1 transformers | 7 | **실행 중** (~45분 경과) | 13:16 |
| E4B e2 transformers | 0-3 | **실행 중** (방금 시작) | 14:02 |
| 31B e1 transformers | 4 | **실행 중** (방금 시작) | 14:02 |

---

## 9. 아직 안 한 것 (TODO)

### 즉시 해야 할 것
1. **위 4개 실행 중인 평가 완료 대기** → 결과 수집
2. **README 업데이트** — 새 결과(transformers 기반 31B e1/e2, E4B e1/e2) 반영
3. **HuggingFace 업로드** — 체크포인트 업로드 (토큰: 별도 전달)

### 추가 고려사항
4. **Gemma4 큰 모델 저성능 원인 분석** — 이미 README에 상세 분석 있음 (SFT 데이터 포맷 불일치, Gemma4 출력 스타일 차이)
5. **LFM2 tool-use 프롬프트 개선** — https://docs.liquid.ai/lfm/key-concepts/tool-use 참고
6. **E4B vLLM nightly 시도** — 0.20.1 cu129에서 E4B가 되는지 아직 안 해봄 (시도할 가치 있음)

### Qwen 계열 모델 평가 (이전에 이미 평가했을 수 있음)
- 이전 세션에서 Qwen3.5-2B/4B/9B/27B, Qwen3.6-35B-A3B 등은 이미 평가 완료했을 가능성
- README에 결과 있는지 확인 후 미평가 모델만 추가 평가

---

## 10. 완전 자동화 평가 스크립트 (복붙해서 실행)

### GPU 할당 계획 (병렬 실행)

```bash
#!/bin/bash
# TB2-lite 전체 모델 평가 스크립트
# GPU 8개를 모두 활용하여 병렬로 평가

cd /home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/scripts

EVAL_DATA="/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/data/replay_full.jsonl"
OUTPUT_DIR="/home/work/.data/tb2_lite_eval/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTPUT_DIR"
echo "Output dir: $OUTPUT_DIR"

# ============================================
# 빠른 것부터 (vLLM, GPU 1개씩)
# ============================================

# GPU 0: LFM2 e2 (vLLM 0.14.1)
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-014/lib/python3.12/site-packages \
  ../../.vllm-014/bin/python replay_eval.py \
  --model /home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-1468 \
  --model-short lfm2_24b_a2b_e2 \
  --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --max-tokens 1024 --temperature 0.0 &

# GPU 1: LFM2 e1 (vLLM 0.14.1)
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-014/lib/python3.12/site-packages \
  ../../.vllm-014/bin/python replay_eval.py \
  --model /home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-734 \
  --model-short lfm2_24b_a2b_e1 \
  --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --max-tokens 1024 --temperature 0.0 &

# GPU 2: Gemma4-26B e2 (vLLM 0.19.1)
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-0_19_1/lib/python3.12/site-packages \
  ../../.vllm-0_19_1/bin/python replay_eval.py \
  --model /home/work/.data/qwen_sft/models/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-1468 \
  --model-short gemma4_26b_a4b_e2 \
  --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --max-tokens 1024 --temperature 0.0 --language-model-only &

# GPU 3: Gemma4-26B e1 (vLLM 0.19.1)
CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-0_19_1/lib/python3.12/site-packages \
  ../../.vllm-0_19_1/bin/python replay_eval.py \
  --model /home/work/.data/qwen_sft/models/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-734 \
  --model-short gemma4_26b_a4b_e1 \
  --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --max-tokens 1024 --temperature 0.0 --language-model-only &

# GPU 4: E4B e1 (transformers)
CUDA_VISIBLE_DEVICES=4 PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-0_19_1/lib/python3.12/site-packages \
  ../../.vllm-0_19_1/bin/python replay_eval_transformers.py \
  --model /home/work/.data/qwen_sft/models/google__gemma-4-E4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-2934 \
  --model-short gemma4_e4b_e1 \
  --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
  --dtype bfloat16 --max-model-len 8192 --max-tokens 1024 --temperature 0.0 &

# GPU 5: E4B e2 (transformers)
CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-0_19_1/lib/python3.12/site-packages \
  ../../.vllm-0_19_1/bin/python replay_eval_transformers.py \
  --model /home/work/.data/qwen_sft/models/google__gemma-4-E4B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-5868 \
  --model-short gemma4_e4b_e2 \
  --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
  --dtype bfloat16 --max-model-len 8192 --max-tokens 1024 --temperature 0.0 &

# GPU 6+7: 31B e2 (transformers, 2 GPU 필요)
CUDA_VISIBLE_DEVICES=6,7 PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-0_19_1/lib/python3.12/site-packages \
  ../../.vllm-0_19_1/bin/python replay_eval_transformers.py \
  --model /home/work/.data/qwen_sft/models/google__gemma-4-31B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-2934 \
  --model-short gemma4_31b_e2 \
  --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
  --dtype bfloat16 --max-model-len 8192 --max-tokens 1024 --temperature 0.0 &

echo "All 7 evals launched. Waiting..."
wait

# ============================================
# 31B e1 (위 완료 후 GPU 6+7로)
# ============================================
CUDA_VISIBLE_DEVICES=6,7 PYTHONNOUSERSITE=1 \
  PYTHONPATH=../../.vllm-0_19_1/lib/python3.12/site-packages \
  ../../.vllm-0_19_1/bin/python replay_eval_transformers.py \
  --model /home/work/.data/qwen_sft/models/google__gemma-4-31B-it__terminal_sft_2epoch_hf_fsdp/checkpoint-1467 \
  --model-short gemma4_31b_e1 \
  --eval-path "$EVAL_DATA" --output-dir "$OUTPUT_DIR" \
  --dtype bfloat16 --max-model-len 8192 --max-tokens 1024 --temperature 0.0

echo "All evaluations complete!"
echo "Results in: $OUTPUT_DIR"
```

### 결과 요약 추출

```bash
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

## 11. 참고 문서 링크

- vLLM 공식: https://docs.vllm.ai/en/latest/
- Gemma4 vLLM 레시피: https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- Qwen3.5 vLLM 레시피: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
- Liquid LFM2 vLLM 배포: https://docs.liquid.ai/deployment/gpu-inference/vllm
- Liquid LFM2 tool-use: https://docs.liquid.ai/lfm/key-concepts/tool-use
- Gemma4 반복 붕괴 이슈: https://github.com/google-deepmind/gemma/issues/622
- vLLM Gemma4 k_eq_v PR: https://github.com/vllm-project/vllm/pull/41253
