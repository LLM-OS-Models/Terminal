# 4차 평가 결과 (2026-04-25)

vLLM 기반 12개 요청 모델 평가 완료. `eval/eval_dataset.jsonl` 50샘플 기준.

- 실행 결과: [eval/results/20260425T094140Z](/home/work/.projects/LLM-OS-Models/Terminal/eval/results/20260425T094140Z)
- 이슈 로그: [eval/VLLM_EVAL_ISSUES_2026-04-25.md](/home/work/.projects/LLM-OS-Models/Terminal/eval/VLLM_EVAL_ISSUES_2026-04-25.md)
- 참고 비교:
  - [eval/EVAL_RESULTS_2026-04-21.md](/home/work/.projects/LLM-OS-Models/Terminal/eval/EVAL_RESULTS_2026-04-21.md)
  - [eval/EVAL_RESULTS_2026-04-22.md](/home/work/.projects/LLM-OS-Models/Terminal/eval/EVAL_RESULTS_2026-04-22.md)

중복 모델은 **최신 결과(2026-04-25)** 를 우선 사용했다.

## 전체 통합 순위 (기존 포함)

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

## 이번 런 순위

| # | 모델 | 파라미터 | cmd_overlap | cmds% | think% | avg_len |
|---|------|---------|-------------|-------|--------|---------|
| 1 | **LFM2-8B-A1B** | 8.3B (1B active) | **0.0436** | 40% | 0% | 4055 |
| 2 | Nemotron-Terminal-14B | 14B | 0.0400 | 8% | 100% | 2081 |
| 2 | Nemotron-Terminal-32B | 32B | 0.0400 | 10% | 100% | 2031 |
| 2 | Nemotron-Terminal-8B | 8B | 0.0400 | 8% | 100% | 2371 |
| 2 | Qwen3.6-35B-A3B-FP8 | 35B (3B active) | 0.0400 | 52% | 0% | 1638 |
| 2 | gemma-4-31B-it | 31B | 0.0400 | 86% | 0% | 1703 |
| 7 | LFM2-2.6B | 2.6B | 0.0385 | 58% | 0% | 4780 |
| 8 | Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled | 27B | 0.0323 | 62% | 0% | 1812 |
| 8 | gemma-4-26B-A4B-it | 26B (4B active) | 0.0323 | 78% | 0% | 1280 |
| 10 | LFM2-24B-A2B | 23.8B (2B active) | 0.0209 | 30% | 0% | 2537 |
| 11 | LFM2-8B-Terminal-SFT-Unsloth | 8.3B (SFT) | 0.0000 | 2% | 100% | 1619 |
| 11 | LFM2.5-1.2B-Instruct | 1.2B | 0.0000 | 10% | 0% | 1581 |

## 주요 발견

### 1. 전체 통합 1위는 공동 1위
- `LFM2-8B-A1B`와 기존 `gemma-4-E2B-it`이 모두 `cmd_overlap 0.0436`.
- 따라서 현재 통합 기준 최고는 **공동 1위**다.
- 새 모델이 기존 최고를 넘진 못했지만, `LFM2-8B-A1B`는 최소한 같은 급까지 올라왔다.

### 2. 더 큰 모델이라고 더 좋지 않았다
- `gemma-4-31B-it`, `Qwen3.6-35B-A3B-FP8`, `Nemotron-Terminal-32B` 모두 `0.0400`.
- 모두 상위권이긴 하지만 기존 `gemma-4-E2B-it 0.0436`, `gemma-4-E4B-it 0.0421`을 넘지 못했다.
- 이번에도 **스케일링이 terminal 내부 지표에서 바로 성능 우위로 이어지지 않았다.**

### 3. Nemotron 계열은 크기와 무관하게 비슷한 패턴
- `Nemotron-Terminal-8B/14B/32B` 모두 `0.0400`.
- 세 모델 모두 `thinking 100%`, `cmds 8~10%`로 응답 포맷 경향도 거의 같다.
- 크기를 32B까지 키워도 내부 지표상 이득은 아직 안 보인다.

### 4. LFM 계열은 편차가 컸다
- `LFM2-8B-A1B`: 최고 성능 (`0.0436`)
- `LFM2-2.6B`: 준수 (`0.0385`)
- `LFM2-24B-A2B`: 기대 이하 (`0.0209`)
- `LFM2.5-1.2B-Instruct`: 사실상 약함 (`0.0000`)

### 5. gyung LFM2-8B-Terminal-SFT-Unsloth는 현재 evaluator 기준 성능이 낮다
- `cmd_overlap 0.0000`, `cmds 2%`, `thinking 100%`
- 이번 evaluator/prompt 조건에서는 기대만큼 command generation으로 연결되지 않았다.
- 터미널 특화 SFT 모델이라는 점을 감안하면, **현재 포맷 적합성이나 프롬프트 alignment를 별도로 확인할 가치가 있다.**

### 6. vLLM은 확실히 빨라졌다
- `2026-04-21` transformers baseline은 `34.7~47.4 sec/sample`.
- 이번 vLLM 런은 generation만 보면:
  - `LFM2.5-1.2B-Instruct`: `3.6s / 50` → `0.07 sec/sample`
  - `LFM2-8B-A1B`: `12.6s / 50` → `0.25 sec/sample`
  - `Qwen3.6-35B-A3B-FP8`: `17.8s / 50` → `0.36 sec/sample`
  - `Nemotron-Terminal-32B`: `27.3s / 50` → `0.55 sec/sample`
  - `gemma-4-31B-it`: `34.3s / 50` → `0.69 sec/sample`
- 다만 첫 실행은 download/load/`torch.compile`/CUDA graph capture가 커서 총 시간은 load가 더 크게 잡힌다.

## 기존 결과와 비교

| 비교 기준 | 최고 모델 | cmd_overlap |
|----------|-----------|-------------|
| 2026-04-21 | gemma-4-E2B-it | 0.0421 |
| 2026-04-22 | gemma-4-E2B-it | 0.0436 |
| 2026-04-25 | LFM2-8B-A1B | 0.0436 |

정리하면:
- `2026-04-25` 최고 성능은 **기존 최고와 동률**
- 통합 Top 3는 `LFM2-8B-A1B`, `gemma-4-E2B-it`, `gemma-4-E4B-it`
- `2026-04-25`의 대형 신규 모델들(`31B`, `32B`, `35B`)은 **기존 소형/중형 Gemma를 넘지 못함**
- **속도는 vLLM이 확실히 우세**, 성능은 아직 “작은 강자”가 유지

## 평가 불가 / 스킵 모델

| 모델 | 사유 |
|------|------|
| principled-intelligence/Qwen3.5-2B-text-only | `Qwen3_5TextConfig`와 vLLM config 타입 불일치 |
| principled-intelligence/Qwen3.5-4B-text-only | 동일 |
| principled-intelligence/Qwen3.5-9B-text-only | 동일 |
| principled-intelligence/gemma-4-E2B-it-text-only | vLLM이 pooling/embed 모델로 해석 |
| principled-intelligence/gemma-4-E4B-it-text-only | 동일 |
| Jiunsong/supergemma4-26b-abliterated-multimodal | weight load 중 `KeyError` |
| Jiunsong/supergemma4-26b-uncensored-gguf-v2 | 이번 런 스킵 — GGUF는 vLLM에서 experimental 경로 |
| Jiunsong/SuperGemma4-31b-abliterated-GGUF | 이번 런 스킵 — 동일 |

## 추가 baseline (요청 외, spare GPU)

| 모델 | cmd_overlap | cmds% | think% | load | gen |
|------|-------------|-------|--------|------|-----|
| Qwen3.6-27B | 0.0385 | 50% | 0% | 130s | 42.3s |

## SFT 베이스 관점 메모

| 우선순위 | 모델 | 근거 |
|---------|------|------|
| 1순위 | **LFM2-8B-A1B** | 이번 런 1위, 기존 최고와 동률 |
| 2순위 | **gemma-4-E2B-it** | 기존 전체 최고권 유지, 아직 안 무너짐 |
| 3순위 | **Nemotron-Terminal-8B / 14B** | 0.0400 유지, 터미널 특화 prior 보유 |

이번 라운드만 놓고 보면, **성능 최고는 LFM2-8B-A1B**, **안정적 기존 강자는 gemma-4-E2B-it**, **대형 모델은 생각보다 이득이 작다**가 핵심 결론이다.
