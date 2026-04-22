# vLLM 기반 12개 모델 평가 계획

**목표**: 12개 모델을 vLLM으로 최대한 빠르게 평가. 속도가 1순위.

## 평가 대상 모델

### 그룹 A: NVIDIA Nemotron-Terminal (터미널 특화 모델)

| # | 모델 | 파라미터 | 비고 |
|---|------|---------|------|
| 1 | nvidia/Nemotron-Terminal-8B | 8B | SFT 완료된 터미널 특화 모델 |
| 2 | nvidia/Nemotron-Terminal-14B | 14B | |
| 3 | nvidia/Nemotron-Terminal-32B | 32B | 가장 큰 터미널 특화 모델 |

### 그룹 B: text-only (비전 제거 경량 모델)

| # | 모델 | 원본 | 비고 |
|---|------|------|------|
| 4 | principled-intelligence/gemma-4-E2B-it-text-only | gemma-4-E2B-it | 비전 헤드 제거 |
| 5 | principled-intelligence/gemma-4-E4B-it-text-only | gemma-4-E4B-it | 비전 헤드 제거 |
| 6 | principled-intelligence/Qwen3.5-2B-text-only | Qwen3.5-2B | 비전 헤드 제거 |
| 7 | principled-intelligence/Qwen3.5-4B-text-only | Qwen3.5-4B | 비전 헤드 제거 |
| 8 | principled-intelligence/Qwen3.5-9B-text-only | Qwen3.5-9B | 비전 헤드 제거 |

### 그룹 C: 대형 모델 + 이전 실패 모델 재시도

| # | 모델 | 파라미터 | 비고 |
|---|------|---------|------|
| 9 | google/gemma-4-26B-A4B-it | 26B (4B active) | MoE, 신규 |
| 10 | google/gemma-4-31B-it | 31B | 신규 |
| 11 | Qwen/Qwen3.6-35B-A3B-FP8 | 35B (3B active) | MoE FP8, 이전 OOM → vLLM으로 재시도 |
| 12 | Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled | 27B | 이전 로더 에러 → vLLM으로 재시도 |

### 그룹 D: Abliterated 모델 (평가 불가)

| # | 모델 | 파라미터 | 비고 |
|---|------|---------|------|
| 13 | Jiunsong/supergemma4-26b-abliterated-multimodal | 26B | **평가 불가** - vLLM weight key 불일치 |
| 14 | Jiunsong/supergemma4-26b-uncensored-gguf-v2 | 26B | **평가 불가** - gemma4 아키텍처 GGUF 미지원 |
| 15 | Jiunsong/SuperGemma4-31b-abliterated-GGUF | 31B | **평가 불가** - 동일 |

## 리소스 및 배치 전략

### H200 8-GPU (각 143.7GB, 총 ~1.15TB)

속도 최우선이므로 **vLLM 멀티 인스턴스 병렬 실행**이 핵심.

#### 페이즈 1: 소형 모델 (1GPU당 1개, 8개 동시 실행)

| GPU | 모델 | 예상 VRAM | 포트 |
|-----|------|----------|------|
| 0 | principled-intelligence/Qwen3.5-2B-text-only | ~4GB | 8100 |
| 1 | principled-intelligence/gemma-4-E2B-it-text-only | ~10GB | 8101 |
| 2 | principled-intelligence/Qwen3.5-4B-text-only | ~9GB | 8102 |
| 3 | principled-intelligence/gemma-4-E4B-it-text-only | ~16GB | 8103 |
| 4 | principled-intelligence/Qwen3.5-9B-text-only | ~19GB | 8104 |
| 5 | nvidia/Nemotron-Terminal-8B | ~16GB | 8105 |
| 6 | nvidia/Nemotron-Terminal-14B | ~28GB | 8106 |
| 7 | google/gemma-4-26B-A4B-it | ~52GB (MoE) | 8107 |

→ 8개 모델 동시에 띄우고, API로 병렬 호출하여 평가. 각각 수 분 내 완료 예상.

#### 페이즈 2: 대형 모델 (1 GPU, 이전 페이즈 완료 후)

| GPU | 모델 | 예상 VRAM | 포트 | 비고 |
|-----|------|----------|------|------|
| 0 | nvidia/Nemotron-Terminal-32B | ~64GB | 8100 | TP=1 (H200 143GB 충분) |
| 1 | google/gemma-4-31B-it | ~62GB | 8101 | 완료 (overlap=0.04) |
| 2 | Qwen/Qwen3.6-35B-A3B-FP8 | ~36GB | 8102 | flashinfer 이슈 |
| 3 | Jackrong/Qwen3.5-27B-Claude-Distilled | ~54GB | 8103 | NCCL/flashinfer 이슈 |

> Jiunsong abliterated 모델은 vLLM/transformers 미지원으로 평가 불가.

## 실행 스크립트

### 1. vLLM 서버 띄우기 (페이즈 1)

```bash
# 터미널 8개에 분산 실행 또는 nohup/background로 일괄 실행
CUDA_VISIBLE_DEVICES=0 vllm serve principled-intelligence/Qwen3.5-2B-text-only --port 8100 --trust-remote-code &
CUDA_VISIBLE_DEVICES=1 vllm serve principled-intelligence/gemma-4-E2B-it-text-only --port 8101 --trust-remote-code &
CUDA_VISIBLE_DEVICES=2 vllm serve principled-intelligence/Qwen3.5-4B-text-only --port 8102 --trust-remote-code &
CUDA_VISIBLE_DEVICES=3 vllm serve principled-intelligence/gemma-4-E4B-it-text-only --port 8103 --trust-remote-code &
CUDA_VISIBLE_DEVICES=4 vllm serve principled-intelligence/Qwen3.5-9B-text-only --port 8104 --trust-remote-code &
CUDA_VISIBLE_DEVICES=5 vllm serve nvidia/Nemotron-Terminal-8B --port 8105 --trust-remote-code &
CUDA_VISIBLE_DEVICES=6 vllm serve nvidia/Nemotron-Terminal-14B --port 8106 --trust-remote-code &
CUDA_VISIBLE_DEVICES=7 vllm serve google/gemma-4-26B-A4B-it --port 8107 --trust-remote-code &

# 모든 서버 ready 대기
for port in $(seq 8100 8107); do
  until curl -s http://localhost:$port/v1/models > /dev/null 2>&1; do sleep 2; done
  echo "Port $port ready"
done
echo "=== Phase 1: All servers ready ==="
```

### 2. 평가 실행

```bash
# eval/vllm_eval.py 가 이미 있으면 사용, 없으면 아래 참고
# 모델-포트 매핑으로 병렬 호출

python3 eval/vllm_eval.py --model principled-intelligence/Qwen3.5-2B-text-only --port 8100 &
python3 eval/vllm_eval.py --model principled-intelligence/gemma-4-E2B-it-text-only --port 8101 &
python3 eval/vllm_eval.py --model principled-intelligence/Qwen3.5-4B-text-only --port 8102 &
python3 eval/vllm_eval.py --model principled-intelligence/gemma-4-E4B-it-text-only --port 8103 &
python3 eval/vllm_eval.py --model principled-intelligence/Qwen3.5-9B-text-only --port 8104 &
python3 eval/vllm_eval.py --model nvidia/Nemotron-Terminal-8B --port 8105 &
python3 eval/vllm_eval.py --model nvidia/Nemotron-Terminal-14B --port 8106 &
python3 eval/vllm_eval.py --model google/gemma-4-26B-A4B-it --port 8107 &
wait
```

### 3. 페이즈 1 서버 종료 후 페이즈 2

```bash
# 페이즈 1 서버 전부 종료
pkill -f "vllm serve"

# 페이즈 2: 대형 모델 + Abliterated
CUDA_VISIBLE_DEVICES=0,1 vllm serve nvidia/Nemotron-Terminal-32B --port 8100 --tensor-parallel-size 2 --trust-remote-code &
CUDA_VISIBLE_DEVICES=2,3 vllm serve google/gemma-4-31B-it --port 8101 --tensor-parallel-size 2 --trust-remote-code &
CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8102 --trust-remote-code &
CUDA_VISIBLE_DEVICES=5 vllm serve Jiunsong/supergemma4-26b-abliterated-multimodal --port 8103 --trust-remote-code &
CUDA_VISIBLE_DEVICES=6,7 vllm serve Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled --port 8104 --tensor-parallel-size 2 --trust-remote-code &

# ready 대기 후 동일하게 평가
for port in 8100 8101 8102 8103 8104; do
  until curl -s http://localhost:$port/v1/models > /dev/null 2>&1; do sleep 2; done
done

python3 eval/vllm_eval.py --model nvidia/Nemotron-Terminal-32B --port 8100 &
python3 eval/vllm_eval.py --model google/gemma-4-31B-it --port 8101 &
python3 eval/vllm_eval.py --model Qwen/Qwen3.6-35B-A3B-FP8 --port 8102 &
python3 eval/vllm_eval.py --model Jiunsong/supergemma4-26b-abliterated-multimodal --port 8103 &
python3 eval/vllm_eval.py --model Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled --port 8104 &
wait
```

## vllm_eval.py 필요 사양

`eval/vllm_eval.py`가 없거나 미완이면 아래 기준으로 작성:

1. **입력**: `--model`, `--port`, `--eval-path`(기본 eval_dataset.jsonl), `--output-dir`(기본 results)
2. **동작**: OpenAI-compatible API로 `http://localhost:{port}/v1/chat/completions` 호출
3. **파라미터**: temperature=0, max_tokens=1024 (fast_eval.py와 동일 조건)
4. **측정**: has_cmds, avg_cmds, cmd_overlap, thinking, avg_pred_len (fast_eval.py와 동일 지표)
5. **출력**: results/{model_short}.json (fast_eval.py 결과와 동일 포맷 → summarize.py로 통합 비교 가능)
6. **속도**: vLLM의 continuous batching 활용, 필요시 여러 요청 동시 전송

## 체크리스트

실행 전 확인:

- [ ] GPU 6에서 돌고 있는 기존 vLLM 서버 종료 (`kill` PID 3732668)
- [ ] 페이즈 1 모델 다운로드 여부 확인 (없으면 vLLM이 자동 다운로드하므로 첫 실행 시 시간 소요)
- [ ] `eval/vllm_eval.py` 존재 및 정상 동작 확인
- [ ] 디스크 여유 공간 확인 (12개 모델, 약 200GB+ 필요)
- [ ] 기존 results/ 파일 백업 (덮어쓰기 방지 — 모델명이 다르므로 안전)

실행 후:

- [ ] 12개 결과 파일 존재 확인
- [ ] `python3 eval/summarize.py` 로 전체 비교표 생성
- [ ] `eval/EVAL_RESULTS_2026-04-21.md` 에 결과 갱신 또는 새 날짜로 생성
- [ ] SFT 베이스 모델 최종 선정

## 예상 소요 시간

| 페이즈 | 모델 수 | GPU 시간 | 실제 시간 (병렬) |
|--------|---------|---------|-----------------|
| 페이즈 1 (소형 8개) | 8 | ~40분×8 | **~40분** (8 GPU 병렬) |
| 페이즈 2 (대형 4개) | 4 | ~30분×4 | **~30분** (4×2 GPU 병렬) |
| 모델 로딩 | - | - | **~10분** |
| **총 예상** | **12** | | **~1시간 20분** |

이전 fast_eval.py로 6개에 40분 걸린 것과 비교하면, vLLM + 병렬로 12개를 비슷한 시간에 완료 가능.

## Phase 3: LFM 모델 평가 (신규)

Phase 1/2 완료 후 LFM (Liquid Foundation Models) 계열 평가를 위한 Phase 3 스크립트 추가.

### 실행
```bash
bash eval/run_phase3.sh
```

### 평가 모델 (4-GPU 병렬)

| GPU | 모델 | 파라미터 | 타입 |
|-----|------|---------|------|
| 0 | LiquidAI/LFM2-24B-A2B | 23.84B | MoE (2B active) |
| 1 | LiquidAI/LFM2-8B-A1B | 8.34B | MoE (1B active) |
| 2 | LiquidAI/LFM2-2.6B | 2.57B | Dense |
| 3 | LiquidAI/LFM2.5-1.2B-Instruct | 1.17B | Dense (Instruct) |

### 벤치마크 기대 성능

LFM 계열은 속도에서는 우수하나(동일 규모 대비 2-4배 빠름), 품질 벤치마크에서는:
- LFM2-2.6B: MMLU 61.71, 5태스크 평균 61.02 (LFM 최고 품질)
- LFM2-24B-A2B: MMLU 69.24, HellaSwag 45.40 (MoE 비효율)
- LFM2.5-1.2B-Instruct: Instruction-tuned, 프롬프트 따르기에 유리
