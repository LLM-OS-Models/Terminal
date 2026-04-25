# TB2-lite Replay 결과 정리 (2026-04-25)

## 실험 범위

이번 문서는 아래 3개 run을 합친 **TB2-lite 멀티턴 replay** 결과 정리입니다.

- 메인 run: [20260425T101719Z](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T101719Z)
- 보너스 run: [20260425T102110Z_bonus](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T102110Z_bonus)
- remaining run: [20260425T193000Z_remaining](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T193000Z_remaining)

이번 평가에서 성공적으로 점수가 나온 모델은 총 18개입니다.  
`OBLITERATUS/gemma-4-E4B-it-OBLITERATED` 는 processor/feature extractor 문제로 최종 스킵했습니다.

## 평가 지표

- `Cmd F1`
  - 정답 명령과 예측 명령 유사도
- `First Cmd Exact`
  - 첫 명령을 정확히 맞춘 비율
- `Score`
  - `0.7 * Cmd F1 + 0.3 * First Cmd Exact`
- `Sec/Step`
  - replay step 1개당 평균 생성 시간
- `Load`
  - 모델 로딩 시간

중요한 건 `Cmd F1`, `First Cmd Exact`, 그리고 `Sec/Step` 입니다.

## 전체 통합 순위

| 순위 | 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gyung/LFM2-8B-Terminal-SFT-Unsloth` | 30.14 | 0.2965 | 31.3% | 0.029 | 46.6 |
| 2 | `nvidia/Nemotron-Terminal-8B` | 30.02 | 0.2969 | 30.8% | 0.078 | 49.8 |
| 3 | `nvidia/Nemotron-Terminal-32B` | 29.13 | 0.2872 | 30.1% | 0.281 | 99.1 |
| 4 | `nvidia/Nemotron-Terminal-14B` | 27.72 | 0.2751 | 28.2% | 0.108 | 68.9 |
| 5 | `Qwen/Qwen3.5-9B` | 27.04 | 0.2808 | 24.6% | 0.072 | 90.6 |
| 6 | `Qwen/Qwen3.5-2B` | 26.52 | 0.2743 | 24.4% | 0.024 | 84.7 |
| 7 | `Qwen/Qwen3.5-4B` | 26.36 | 0.2745 | 23.8% | 0.055 | 77.2 |
| 8 | `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | 26.35 | 0.2744 | 23.8% | 0.282 | 113.4 |
| 9 | `google/gemma-4-26B-A4B-it` | 25.95 | 0.2631 | 25.1% | 0.094 | 123.9 |
| 10 | `Qwen/Qwen3.6-27B` | 25.60 | 0.2702 | 22.3% | 0.282 | 118.5 |
| 11 | `Qwen/Qwen3.6-35B-A3B-FP8` | 25.21 | 0.2658 | 22.0% | 0.082 | 120.5 |
| 12 | `google/gemma-4-31B-it` | 24.70 | 0.2594 | 21.8% | 0.404 | 101.4 |
| 13 | `LiquidAI/LFM2-2.6B` | 24.12 | 0.2379 | 24.9% | 0.032 | 36.4 |
| 14 | `google/gemma-4-E4B-it` | 23.43 | 0.2391 | 22.3% | 0.051 | 129.7 |
| 15 | `LiquidAI/LFM2.5-1.2B-Instruct` | 23.36 | 0.2381 | 22.3% | 0.021 | 31.7 |
| 16 | `LiquidAI/LFM2-8B-A1B` | 23.19 | 0.2336 | 22.8% | 0.025 | 58.2 |
| 17 | `google/gemma-4-E2B-it` | 23.05 | 0.2359 | 21.8% | 0.032 | 116.0 |
| 18 | `LiquidAI/LFM2-24B-A2B` | 22.80 | 0.2323 | 21.8% | 0.050 | 81.6 |

## 핵심 해석

### 1. terminal 계열이 가장 강했다

상위 4개 중 4개가 사실상 terminal-tuned / terminal-specialized 계열입니다.

- `gyung/LFM2-8B-Terminal-SFT-Unsloth`
- `Nemotron-Terminal-8B`
- `Nemotron-Terminal-32B`
- `Nemotron-Terminal-14B`

즉, 이번 replay 기준에서는 **일반 instruct보다 terminal 계열이 더 잘 맞았습니다.**

### 2. 작은 Qwen 3.5 계열이 생각보다 강했다

이번에 새로 넣은 작은 Qwen 3.5가 꽤 잘 나왔습니다.

- `Qwen3.5-9B`: 5위
- `Qwen3.5-2B`: 6위
- `Qwen3.5-4B`: 7위

특히 `Qwen3.5-2B`는 작지만 `Sec/Step 0.024`로 매우 빠르고 점수도 높았습니다.  
즉 **작은 모델 baseline 중에서는 꽤 유력한 실험용 모델**입니다.

### 3. Gemma는 replay 기준에서 생각보다 강하지 않았다

기존 단발 프록시에서는 `gemma-4-E2B-it`가 가장 좋아 보였지만, 이번 replay에서는 아래처럼 나왔습니다.

- `gemma-4-26B-A4B-it`: 9위
- `gemma-4-31B-it`: 12위
- `gemma-4-E4B-it`: 14위
- `gemma-4-E2B-it`: 17위

즉 **단발 프록시와 멀티턴 replay 순위가 크게 다릅니다.**

### 4. Liquid 계열은 이번 기준에서 상대적으로 약했다

- `LFM2-2.6B`: 13위
- `LFM2.5-1.2B-Instruct`: 15위
- `LFM2-8B-A1B`: 16위
- `LFM2-24B-A2B`: 18위

예외는 `gyung/LFM2-8B-Terminal-SFT-Unsloth` 입니다.  
즉 **Liquid base 자체보다 terminal SFT 버전이 훨씬 중요하게 보입니다.**

## 속도 해석

빠른 모델:

- `LFM2.5-1.2B-Instruct`: `0.021 sec/step`
- `Qwen3.5-2B`: `0.024 sec/step`
- `LFM2-8B-A1B`: `0.025 sec/step`
- `gyung/LFM2-8B-Terminal-SFT-Unsloth`: `0.029 sec/step`

느린 모델:

- `gemma-4-31B-it`: `0.404 sec/step`
- `Qwen3.6-27B`: `0.282 sec/step`
- `Jackrong/...Reasoning-Distilled`: `0.282 sec/step`
- `Nemotron-Terminal-32B`: `0.281 sec/step`

속도와 점수를 같이 보면 특히 눈에 띄는 건:

- `gyung/LFM2-8B-Terminal-SFT-Unsloth`
- `nvidia/Nemotron-Terminal-8B`
- `Qwen/Qwen3.5-2B`

입니다.

## 이전 단발 eval 대비 속도

대략적인 비교는 이렇습니다.

- 기존 단발 `eval` 평균: 약 `0.422 sec/sample`
- 현재 `tb2_lite` 평균: 약 `0.146 sec/step`
- replay는 평균 `7.72 step/task`
- task 기준 환산 시 약 `1.124 sec/task`

즉 **단발 eval보다는 대략 2.7배 정도 느리지만**, full TB2 실실행보다는 훨씬 가볍고 충분히 계속 돌릴 만한 속도입니다.

## 결론

이번 결과 기준으로 가장 먼저 계속 밀어볼 모델은 아래입니다.

1. `gyung/LFM2-8B-Terminal-SFT-Unsloth`
2. `nvidia/Nemotron-Terminal-8B`
3. `nvidia/Nemotron-Terminal-32B`
4. `nvidia/Nemotron-Terminal-14B`

작은 빠른 baseline으로는:

- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`

이 세 개가 꽤 괜찮았습니다.

한 줄로 요약하면:  
**이번 TB2-lite replay에서는 terminal-specialized 계열이 제일 강했고, 작은 Qwen 3.5도 예상보다 훨씬 괜찮았습니다.**
