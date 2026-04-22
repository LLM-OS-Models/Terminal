# Terminal Agent

NVIDIA Nemotron-Terminal-Corpus 데이터셋으로 터미널 에이전트 모델을 학습하고, Terminal-Bench 2.0 기준으로 평가하는 프로젝트.

## 프로젝트 구조

```
Terminal/
├── README.md                  # 이 파일
├── pyproject.toml              # uv 프로젝트 설정
├── dataset/                    # 학습 데이터 (git 추적 안 함, 7.7GB)
│   ├── dataset_adapters/       #   code, math, swe (~226k)
│   └── synthetic_tasks/        #   easy, medium, mixed (~140k)
├── docs/                       # 문서
│   ├── EVALUATION.md           #   평가 계획서 (TB2.0 기준, 샘플 선정 이유)
│   └── NEXT_EVAL_PLAN.md       #   다음 평가 계획 (vLLM 12개 모델)
├── eval/                       # 평가 파이프라인
│   ├── eval_dataset.jsonl      #   평가용 50샘플 (학습에서 제외)
│   ├── fast_eval.py            #   단일 GPU 평가 (transformers, greedy)
│   ├── vllm_eval.py            #   vLLM 직접 로딩 평가 (배치 생성)
│   ├── tb2_eval.py             #   TB2.0 전체 태스크 평가 (vLLM API 호출)
│   ├── run_all.sh              #   8개 모델 병렬 실행 스크립트
│   ├── run_eval.py             #   run_all.sh용 평가 스크립트 (1차 시도, 실패)
│   ├── summarize.py            #   결과 요약 테이블 생성
│   ├── EVAL_RESULTS_2026-04-21.md  # 1차 평가 결과 보고서
│   ├── results/                #   모델별 평가 결과 JSON
│   └── logs/
│       ├── run1_failed/        #   1차 시도 로그 (run_eval.py, 전부 실패)
│       └── run2_*.log          #   2차 성공 로그 (fast_eval.py, 6개 완료)
├── terminal-bench-2/           # TB2.0 태스크 정의 (89개)
└── vllm-env/                   # vLLM 가상환경
```

## 데이터셋

**원본**: [NVIDIA/Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) (~366k 샘플)

| 카테고리 | 샘플 수 | 난이도 |
|----------|---------|--------|
| dataset_adapters/code | ~32k | - |
| dataset_adapters/math | ~163k | - |
| dataset_adapters/swe | ~32k | - |
| synthetic_tasks (10개 도메인) | ~140k | easy/medium/mixed |

평가용 50샘플은 Terminal-Bench 2.0 카테고리 분포에 비례하게 추출하여 학습 데이터에서 제거.

## 평가

### 내부 평가 (eval/)

```bash
# transformers 기반 단일 모델 평가
cd eval
CUDA_VISIBLE_DEVICES=0 python3 fast_eval.py --model Qwen/Qwen3.5-2B --gpu 0

# vLLM 직접 로딩 평가 (배치 생성, 더 빠름)
CUDA_VISIBLE_DEVICES=0 python3 vllm_eval.py --model Qwen/Qwen3.5-2B --gpu 0

# 결과 요약
python3 summarize.py
```

### Terminal-Bench 2.0 최종 평가

```bash
# 내부 평가 (vLLM API, Docker 없이 포맷만 검증)
python3 tb2_eval.py

# 공식 평가 (Docker 컨테이너에서 실제 실행)
harbor run -d terminal-bench@2.0 -a "agent" -m "model" -k 5
```

- 총 89개 태스크, 16개 카테고리
- 리더보드: https://www.tbench.ai/leaderboard/terminal-bench/2.0

## 벤치마크 대상 모델

### 1차 평가 (완료 2026-04-21, transformers 기반)

| 모델 | 파라미터 | cmd_overlap | 비고 |
|------|---------|-------------|------|
| gemma-4-E2B-it | 5.1B | **0.0421** | 1위, SFT 1순위 |
| Qwen3.5-4B | 4.2B | 0.0385 | 2위 |
| gemma-4-E4B-it | 7.9B | 0.0359 | 3위 |
| Qwen3.5-2B | 1.9B | 0.0323 | 경량 최고 |
| Qwen3.5-9B | 9.0B | 0.0308 | 기대 이하 |
| gemma-4-E4B-it-OBLITERATED | 7.9B | 0.0000 | 제외 |
| ~~Qwen3.6-35B-A3B-FP8~~ | 35B | - | OOM 실패 |
| ~~Qwen3.5-27B-Claude-Distilled~~ | 27B | - | 로더 에러 실패 |

상세 결과: `eval/EVAL_RESULTS_2026-04-21.md`

### 2차 평가 (예정, vLLM 기반)

| 모델 | 파라미터 | 비고 |
|------|---------|------|
| nvidia/Nemotron-Terminal-8B | 8B | 터미널 특화 SFT 모델 |
| nvidia/Nemotron-Terminal-14B | 14B | |
| nvidia/Nemotron-Terminal-32B | 32B | |
| principled-intelligence/gemma-4-E2B-it-text-only | 5.1B | 비전 헤드 제거 |
| principled-intelligence/gemma-4-E4B-it-text-only | 7.9B | 비전 헤드 제거 |
| principled-intelligence/Qwen3.5-2B-text-only | 1.9B | 비전 헤드 제거 |
| principled-intelligence/Qwen3.5-4B-text-only | 4.2B | 비전 헤드 제거 |
| principled-intelligence/Qwen3.5-9B-text-only | 9.0B | 비전 헤드 제거 |
| google/gemma-4-26B-A4B-it | 26B (4B active) | MoE |
| google/gemma-4-31B-it | 31B | |
| Jiunsong/supergemma4-26b-abliterated-multimodal | 26B | **평가 불가** (vLLM 미지원 아키텍처) |
| Qwen3.6-27B | 27B | Qwen3.6, triton GDN backend |
| Qwen/Qwen3.6-35B-A3B-FP8 | 35B (3B active) | 1차 실패 → vLLM으로 재시도 |
| Jackrong/Qwen3.5-27B-Claude-Distilled | 27B | 1차 실패 → vLLM으로 재시도 |

## 하드웨어

- GPU: 8× NVIDIA H200 (각 143.7GB VRAM, 총 ~1.15TB)
- CUDA 12.9, Driver 570.86.10

## 워크플로우

1. **1차 베이스라인 평가** (완료 2026-04-21) — transformers 기반 8개 모델 zero-shot 평가 (6개 완료, 2개 실패). 결과: `eval/EVAL_RESULTS_2026-04-21.md`
2. **2차 평가** (예정) — vLLM 기반 12개 모델 평가. 계획: `docs/NEXT_EVAL_PLAN.md`
   - NVIDIA Nemotron-Terminal (8B/14B/32B) — 터미널 특화 SFT 모델
   - text-only 경량 모델 5개 — 비전 헤드 제거 버전
   - 대형 모델 4개 (gemma-4-26B/31B, Qwen3.6-35B, Qwen3.5-27B-Distilled)
3. **학습** — Nemotron-Terminal-Corpus로 SFT (베이스 모델 선택 후)
4. **내부 평가** — 50샘플 eval set으로 체크포인트별 성능 추적
5. **최종 평가** — Terminal-Bench 2.0 제출

### 다음에 할 일

> **2차 평가 준비물**: `docs/NEXT_EVAL_PLAN.md` 참고
> - GPU 6 기존 vLLM 서버 종료
> - `eval/vllm_eval.py` 확인/작성
> - 페이즈 1 (소형 8개, GPU당 1개) → 페이즈 2 (대형 4개, 2-GPU TP) 순서로 실행
> - 완료 후 `summarize.py` 로 결과 비교

- Phase 3 (LFM): 스크립트 준비 완료 (`eval/run_phase3.sh`), LFM2-24B-A2B, LFM2-8B-A1B, LFM2-2.6B, LFM2.5-1.2B-Instruct 평가 예정

## llm-os-eval-core 연동

Terminal 평가는 `llm-os-eval-core` 프레임워크를 통해 CLI로 실행할 수도 있다:

```bash
llm-os-eval run terminal \
  --model Qwen/Qwen3-4B \
  --samples eval/eval_dataset.jsonl \
  --output eval/results/Qwen3-4B_v0.jsonl \
  --base-url http://localhost:8001/v1
```

## 벤치마크 결과 (2026-04-23, Round 3)

llm-os-eval-core 기반 8개 모델 병렬 평가 결과 (8x H200 GPU):

| 모델 | Size | 계열 | 성공률 |
|------|------|------|--------|
| Qwen3-4B | 4B | Qwen3 | 8% |
| gemma-4-E2B-it | MoE | Gemma | 8% |
| Qwen3-8B | 8B | Qwen3 | 8% |
| Qwen2.5-14B-Instruct | 14B | Qwen2.5 | 8% |
| gemma-4-31B-it | 31B | Gemma | 8% |
| Qwen3-0.6B | 0.6B | Qwen3 | 0% |
| Llama-3.1-8B-Instruct | 8B | Meta | 0% |
| Nemotron-Terminal-8B | 8B | NVIDIA | 0% |

Nemotron-Terminal-8B가 터미널 특화 SFT 모델임에도 0% 성공률을 기록했다. 이는 llm-os-eval-core의 TerminalEvaluator가 TB2.0이 아닌 내부 평가 포맷을 사용하기 때문이다. TB2.0 평가는 `tb2_eval.py`로 별도 실행한다.

## 참고

- 논문: Pi et al., "On Data Engineering for Scaling LLM Terminal Capabilities" (arXiv:2602.21193)
- 데이터셋: https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus
- 벤치마크: https://www.tbench.ai
