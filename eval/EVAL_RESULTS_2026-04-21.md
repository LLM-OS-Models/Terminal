# Zero-Shot Baseline Evaluation — 2026-04-21

NVIDIA Nemotron-Terminal-Corpus 기반 SFT 대상 베이스 모델 선정을 위한 zero-shot 평가.
8개 모델 중 6개 완료, 2개 실패. H200 8-GPU 환경에서 50샘플 내부 평가셋으로 병렬 평가.

## 평가 방법

- **스크립트**: `eval/fast_eval.py` (transformers 기반, greedy decoding)
- **평가셋**: `eval/eval_dataset.jsonl` (50샘플, 학습 데이터에서 제외)
  - medium 34, mixed 8, code 3, swe 3, math 2
- **측정 지표**:
  - `has_cmds`: 응답에 터미널 명령어가 포함된 비율
  - `avg_cmds`: 응답당 평균 명령어 수
  - `cmd_overlap`: 정답 명령어와의 겹침 비율 (0~1, 핵심 지표)
  - `thinking`: `<think` 태그 사용 여부
- **입력**: 사용자 질문만 제공 (single-turn zero-shot, 멀티턴 context 없음)

## 전체 결과 요약

| 모델 | 파라미터 | has_cmds | avg_cmds | **cmd_overlap** | avg_len | sec/sample | 비고 |
|------|---------|----------|----------|-----------------|---------|------------|------|
| gemma-4-E2B-it | 5.1B | **94.0%** | **9.5** | **0.0421** | 2454 | 37.0s | 명령어 생성량 최다 |
| Qwen3.5-4B | 4.2B | 74.0% | 4.3 | 0.0385 | 3519 | 47.3s | SWE 강점 |
| gemma-4-E4B-it | 7.9B | **94.0%** | 7.4 | 0.0359 | 2363 | 46.1s | 안정적 응답 |
| Qwen3.5-2B | 1.9B | 70.0% | 3.5 | 0.0323 | 3322 | 34.7s | 경량 모델 최고 |
| Qwen3.5-9B | 9.0B | 80.0% | 3.6 | 0.0308 | 3493 | 47.4s | 크기 대비 실망 |
| gemma-4-E4B-it-OBLITERATED | 7.9B | 2.0% | 0.1 | 0.0000 | 873 | 19.9s | **제외** |

## 카테고리별 상세 분석

### cmd_overlap (핵심 지표) 카테고리별 비교

| 모델 | swe (3) | medium (34) | mixed (8) | code (3) | math (2) |
|------|---------|-------------|-----------|----------|----------|
| gemma-4-E2B-it | **0.6410** | 0.0053 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-4B | **0.6410** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| gemma-4-E4B-it | 0.5385 | 0.0053 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B | 0.2051 | 0.0294 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-9B | 0.5128 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| OBLITERATED | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### has_cmds 카테고리별 비교

| 모델 | swe (3) | medium (34) | mixed (8) | code (3) | math (2) |
|------|---------|-------------|-----------|----------|----------|
| gemma-4-E2B-it | 3/3 | 34/34 | 8/8 | 0/3 | 2/2 |
| gemma-4-E4B-it | 3/3 | 33/34 | 8/8 | 1/3 | 2/2 |
| Qwen3.5-9B | 3/3 | 28/34 | 8/8 | 1/3 | 0/2 |
| Qwen3.5-4B | 3/3 | 26/34 | 8/8 | 0/3 | 0/2 |
| Qwen3.5-2B | 2/3 | 25/34 | 8/8 | 0/3 | 0/2 |
| OBLITERATED | 1/3 | 0/34 | 0/8 | 0/3 | 0/2 |

## 모델별 분석

### gemma-4-E2B-it (5.1B) — 종합 1위

- overlap 0.0421, has_cmds 94%, 평균 명령어 9.5개로 전체 최고 성능
- SWE 카테고리에서 overlap 0.6410로 정답에 가장 근접한 명령어 생성
- 5.1B 파라미터임에도 9.0B Qwen3.5-9B를 압도 — 파라미터 효율성 우수
- 단점: medium 난이도에서 불필요하게 많은 명령어를 생성 (avg 10.5개, overlap은 0.005로 낮음). 노이즈가 많은 편
- **SFT 대상 후보 1순위** — 베이스 성능이 가장 좋고, 학습 데이터로 패턴 정제 가능

### Qwen3.5-4B (4.2B) — 종합 2위

- overlap 0.0385로 2위, SWE에서 gemma-4-E2B-it와 동일한 0.6410 기록
- 응답 길이 3519자로 가장 길고 상세한 분석을 생성하는 경향
- 하지만 medium/mixed/code/math에서 overlap 0 — 범용성은 떨어짐
- sec/sample 47.3초로 2B(34.7초)보다 36% 느림
- **SFT 대상 후보** — Qwen 계열은 학습 데이터(Nemotron-Terminal-Corpus)와 호환성이 높을 가능성

### gemma-4-E4B-it (7.9B) — 종합 3위

- has_cmds 94%로 명령어 생성은 안정적이나, overlap 0.0359로 E2B보다 낮음
- OBLITERATED 버전과 비교해 정상 버전이 확실히 우수 → safety filter가 모델 능력을 해치지 않음을 확인
- E2B(5.1B)보다 파라미터가 55% 많은데 성능은 비슷하거나 약간 낮음 — 크기 대비 효율은 E2B가 더 좋음
- **SFT 대상 후보** — 안정적인 베이스라인이지만, E2B로 SFT하는 것이 비용 효율적일 수 있음

### Qwen3.5-2B (1.9B) — 경량 모델 중 최고

- overlap 0.0323, has_cmds 70%로 경량 모델 치고는 선전
- medium 카테고리에서 0.0294 overlap으로 다른 모델보다 높음 — 범용 태스크에서 의외로 강함
- sec/sample 34.7초로 가장 빠름, 추론 비용 최소
- **SFT 대상** — 리소스 제약 시나리오용. SFT 후에도 한계가 있을 수 있으나 실험 가치 있음

### Qwen3.5-9B (9.0B) — 기대 이하

- overlap 0.0308로 4B(0.0385)보다 낮음. 파라미터 2배인데 성능은 역전
- has_cmds 80%로 4B(74%)보다 약간 높으나, avg_cmds 3.6으로 4B(4.3)보다 적음
- medium 카테고리에서 overlap 0.0000 — 4B(0.0000)과 같지만, 2B(0.0294)보다도 낮음
- **원인 추정**: 9B 모델이 "분석"에 치중하고 실제 명령어 실행을 덜 시도하는 경향. 더 큰 모델일수록 더 신중하게 행동하려다 보니 명령어 생성이 보수적일 수 있음
- **SFT 대상 신중 검토** — zero-shot에서 4B보다 못한 점이 우려. SFT로 패턴을 가르치면 개선될 수 있으나, 4B를 먼저 SFT해보고 필요시 9B 실험

### gemma-4-E4B-it-OBLITERATED (7.9B) — 제외

- overlap 0.0000, has_cmds 2%로 사실상 무능력
- 응답이 평균 873자로 매우 짧고, Ansible 플레이북이나 JSON 분석만 출력
- safety filter 제거(OBLITERATED)가 지시사항 따르기 능력 자체를 파괴한 것으로 보임
- **SFT 대상에서 제외**

## 공통 패턴

1. **SWE 카테고리가 유일한 변별점**: 6개 모델 중 5개가 SWE에서만 유의미한 overlap을 기록. medium/mixed/code/math에서는 사실상 전부 0에 가까움
2. **code/math에서 전멸**: 대부분의 모델이 code와 math 카테고리에서 명령어를 생성하지 못함. zero-shot으로는 이런 특수 영역의 명령어를 생성하기 어려움 → SFT로 해결해야 할 핵심 영역
3. **JSON 분석 우선**: 모든 모델이 ````json\n{"analysis": ...` 형식으로 응답을 시작. 이는 프롬프트에 터미널 명령어를 직접 실행하라는 지시가 명확하지 않아서일 수 있음 → SFT 데이터로 패턴 교정 필요
4. **thinking 미사용**: 6개 모델 모두 `<think` 태그를 사용하지 않음. reasoning 능력이 zero-shot에서는 발현되지 않음

## 실패 모델 (2/8)

### Qwen3.5-27B-Claude-Distilled (27B, GPU 6)

- **에러**: `ValueError: Unrecognized configuration class Qwen3_5Config for AutoModelForSeq2SeqLM`
- **원인**: `run_eval.py`가 `AutoModelForSeq2SeqLM`을 사용. Qwen3.5는 Causal LM이라 호환 안 됨
- **해결**: `fast_eval.py` 기반으로 재실행하면 됨 (이미 `AutoModelForCausalLM` 사용 중). GPU 6에서 vLLM 서버가 돌고 있으므로 종료 후 실행

### Qwen3.6-35B-A3B-FP8 (35B MoE, GPU 7)

- **에러**: `torch.OutOfMemoryError: Tried to allocate 50.10 GiB`
- **원인**: FP8 MoE 모델의 KV 캐시 + 활성 파라미터 로딩에 단일 H200(143GB)으로 부족
- **해결**: (1) vLLM 서빙 후 API 호출, (2) 2-GPU tensor parallel, (3) 평가 제외

## 순위별 SFT 대상 추천

| 우선순위 | 모델 | 이유 |
|----------|------|------|
| 1순위 | **gemma-4-E2B-it** | zero-shot 최고 성능, 파라미터 효율 우수, SFT로 노이즈 정제 가능 |
| 2순위 | **Qwen3.5-4B** | SWE 강점, Qwen 계열은 학습 데이터 포맷 호환 가능성 |
| 3순위 | **Qwen3.5-2B** | 경량 baseline, 리소스 제약 시나리오 |
| 보류 | **gemma-4-E4B-it** | E2B와 큰 차이 없어 비용 효율성 의문 |
| 보류 | **Qwen3.5-9B** | 4B보다 못한 zero-shot 성능, SFT 효과 검증 필요 |
| 제외 | **OBLITERATED** | 지시사항 이해 불가 |

## 다음 단계

1. **Qwen3.5-27B-Claude-Distilled** — `fast_eval.py`로 GPU 6에서 재실행 (vLLM 서버 종료 후)
2. **Qwen3.6-35B-A3B-FP8** — vLLM 서빙으로 평가하거나 제외 결정
3. SFT 베이스 모델 확정 후 Nemotron-Terminal-Corpus로 학습
4. 내부 50샘플 평가셋으로 체크포인트별 성능 추적
5. Terminal-Bench 2.0 최종 제출

## Terminal-Bench 2.0 (vLLM, Nemotron-Terminal-8B)

`tb2_eval.py`로 vLLM API 호출하여 89개 태스크 전체 평가 (Docker 없이 포맷만 검증).

| 모델 | 태스크 | valid_json | has_commands | has_plan | avg_latency |
|------|--------|------------|--------------|----------|-------------|
| nvidia/Nemotron-Terminal-8B | 89 | 74.2% | 74.2% | 74.2% | 5.39s |

결과: `results/eval_summary.json`, `results/eval_details.json`

## 메모

- overlap 지표가 전반적으로 매우 낮음(최대 0.04) — zero-shot 기대치. SFT 후 크게 개선될 것으로 예상
- 평가 방식이 reference 명령어와의 문자열 매칭이므로, 기능적으로 정답이어도 다른 명령어를 쓰면 0점으로 측정됨
- 내부 평가는 모델 간 상대 비교가 목적, 최종 성능은 Terminal-Bench 2.0에서 확인
- 평가 시작: 2026-04-21 21:33, 6개 모델 완료: 2026-04-21 ~22:12 (약 40분 소요)
