# 전체 평가 총정리 (2026-04-25)

`eval/eval_dataset.jsonl` 50샘플 기준으로, 기존 결과(`2026-04-21`, `2026-04-22`)와 이번 결과(`2026-04-25`)를 합친 통합 요약.

- 개별 보고서:
  - [eval/EVAL_RESULTS_2026-04-21.md](/home/work/.projects/LLM-OS-Models/Terminal/eval/EVAL_RESULTS_2026-04-21.md)
  - [eval/EVAL_RESULTS_2026-04-22.md](/home/work/.projects/LLM-OS-Models/Terminal/eval/EVAL_RESULTS_2026-04-22.md)
  - [eval/EVAL_RESULTS_2026-04-25.md](/home/work/.projects/LLM-OS-Models/Terminal/eval/EVAL_RESULTS_2026-04-25.md)
- 이번 런 결과 디렉토리: [eval/results/20260425T094140Z](/home/work/.projects/LLM-OS-Models/Terminal/eval/results/20260425T094140Z)
- 문제 모델 기록: [eval/VLLM_EVAL_ISSUES_2026-04-25.md](/home/work/.projects/LLM-OS-Models/Terminal/eval/VLLM_EVAL_ISSUES_2026-04-25.md)

중복 모델은 최신 결과를 우선 사용했다.

## 전체 통합 순위

| # | 모델 | 출처 | cmd_overlap | cmds% | think% | avg_len |
|---|------|------|-------------|-------|--------|---------|
| 1 | **LFM2-8B-A1B** | 2026-04-25 | **0.0436** | 40% | 0% | 4055 |
| 1 | **gemma-4-E2B-it** | 2026-04-22 | **0.0436** | 94% | 0% | 3605 |
| 3 | gemma-4-E4B-it | 2026-04-22 | 0.0421 | 94% | 0% | 3853 |
| 4 | Nemotron-Terminal-14B | 2026-04-25 | 0.0400 | 8% | 100% | 2081 |
| 4 | Nemotron-Terminal-32B | 2026-04-25 | 0.0400 | 10% | 100% | 2031 |
| 4 | Nemotron-Terminal-8B | 2026-04-25 | 0.0400 | 8% | 100% | 2371 |
| 4 | Qwen3.6-35B-A3B-FP8 | 2026-04-25 | 0.0400 | 52% | 0% | 1638 |
| 4 | gemma-4-31B-it | 2026-04-25 | 0.0400 | 86% | 0% | 1703 |
| 9 | LFM2-2.6B | 2026-04-25 | 0.0385 | 58% | 0% | 4780 |
| 9 | Qwen3.5-4B | 2026-04-21 | 0.0385 | 74% | 0% | 3519 |
| 9 | Qwen3.6-27B | 2026-04-25 | 0.0385 | 50% | 0% | 1747 |
| 12 | Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled | 2026-04-25 | 0.0323 | 62% | 0% | 1812 |
| 12 | Qwen3.5-2B | 2026-04-21 | 0.0323 | 70% | 0% | 3322 |
| 12 | gemma-4-26B-A4B-it | 2026-04-25 | 0.0323 | 78% | 0% | 1280 |
| 15 | Qwen3.5-9B | 2026-04-21 | 0.0308 | 80% | 0% | 3493 |
| 16 | LFM2-24B-A2B | 2026-04-25 | 0.0209 | 30% | 0% | 2537 |
| 17 | LFM2-8B-Terminal-SFT-Unsloth | 2026-04-25 | 0.0000 | 2% | 100% | 1619 |
| 17 | LFM2.5-1.2B-Instruct | 2026-04-25 | 0.0000 | 10% | 0% | 1581 |
| 17 | gemma-4-E4B-it-OBLITERATED | 2026-04-22 | 0.0000 | 2% | 0% | 873 |

## 핵심 결론

### 1. 전체 최고는 공동 1위
- `LFM2-8B-A1B`와 `gemma-4-E2B-it`이 모두 `cmd_overlap 0.0436`.
- 이번 신규 실험에서 **기존 최고를 넘긴 모델은 없지만**, `LFM2-8B-A1B`가 같은 급으로 올라왔다.

### 2. 작은 Gemma가 아직 강하다
- `gemma-4-E2B-it 5.1B`, `gemma-4-E4B-it 7.9B`가 여전히 최상위권이다.
- `31B`, `32B`, `35B`급 대형 모델들도 이 둘을 못 넘었다.

### 3. 큰 모델이라고 더 좋지 않다
- `gemma-4-31B-it`, `Qwen3.6-35B-A3B-FP8`, `Nemotron-Terminal-32B` 모두 `0.0400`.
- 상위권이긴 하지만, **성능 우위가 아니라 비용 증가**에 더 가깝다.

### 4. Nemotron은 크기보다 포맷 특성이 두드러진다
- `8B`, `14B`, `32B` 모두 `0.0400`.
- `thinking 100%`, `cmds 8~10%`로 패턴도 거의 같다.

### 5. LFM 계열은 편차가 매우 크다
- `LFM2-8B-A1B`: 최고권
- `LFM2-2.6B`: 준수
- `LFM2-24B-A2B`: 기대 이하
- `LFM2.5-1.2B-Instruct`, `gyung/LFM2-8B-Terminal-SFT-Unsloth`: 현재 evaluator 기준 약함

## 속도 총평

`2026-04-21` transformers baseline은 대략 `34.7~47.4 sec/sample`였다.  
이번 vLLM 런은 generation 기준으로:

| 모델 | gen time | sec/sample |
|------|----------|------------|
| LFM2.5-1.2B-Instruct | 3.6s / 50 | 0.07 |
| LFM2-2.6B | 7.8s / 50 | 0.16 |
| LFM2-8B-A1B | 12.6s / 50 | 0.25 |
| Qwen3.6-35B-A3B-FP8 | 17.8s / 50 | 0.36 |
| Nemotron-Terminal-32B | 27.3s / 50 | 0.55 |
| gemma-4-31B-it | 34.3s / 50 | 0.69 |

정리하면:
- **생성 속도는 vLLM이 확실히 빠르다**
- 다만 첫 실행은 download/load/`torch.compile`/CUDA graph capture 때문에 총 wall-clock은 커질 수 있다

## 추천 메모

| 우선순위 | 모델 | 이유 |
|---------|------|------|
| 1순위 | **LFM2-8B-A1B** | 최신 런 공동 1위 |
| 2순위 | **gemma-4-E2B-it** | 전체적으로 가장 안정적인 기존 강자 |
| 3순위 | **gemma-4-E4B-it** | 여전히 높은 상위권 |
| 4순위 | **Nemotron-Terminal-8B / 14B** | 터미널 특화 prior + 0.0400 유지 |

## 평가 불가 / 제외 메모

- `principled-intelligence/*-text-only` 5개: vLLM 호환 문제
- `Jiunsong/supergemma4-26b-abliterated-multimodal`: weight mapping `KeyError`
- GGUF 계열: 이번 런에서는 제외. vLLM GGUF는 experimental 경로

한 줄 결론:

**성능만 보면 작은/중간급 강자들이 아직 안 죽었고, 속도는 vLLM이 확실히 이겼다.**
