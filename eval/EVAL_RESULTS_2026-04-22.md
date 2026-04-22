# 2차 평가 결과 (2026-04-22)

vLLM 기반 10개 모델 평가 완료. eval_dataset.jsonl 50샘플 기준.

## 종합 순위

| # | 모델 | 파라미터 | cmd_overlap | cmds% | think% | avg_len |
|---|------|---------|-------------|-------|--------|---------|
| 1 | **gemma-4-E2B-it** | 5.1B | **0.0436** | 94% | 0% | 3605 |
| 2 | gemma-4-E4B-it | 7.9B | 0.0421 | 94% | 0% | 3853 |
| 3 | Nemotron-Terminal-14B | 14B | 0.0400 | 8% | 100% | 2081 |
| 3 | Nemotron-Terminal-8B | 8B | 0.0400 | 8% | 100% | 2371 |
| 3 | gemma-4-31B-it | 31B | 0.0400 | 86% | 0% | 1706 |
| 6 | Qwen3.5-4B | 4.2B | 0.0385 | 74% | 0% | 3519 |
| 7 | Qwen3.5-2B | 1.9B | 0.0323 | 70% | 0% | 3322 |
| 7 | gemma-4-26B-A4B-it | 26B(4B active) | 0.0323 | 78% | 0% | 1280 |
| 9 | Qwen3.5-9B | 9.0B | 0.0308 | 80% | 0% | 3493 |
| 10 | gemma-4-E4B-it-OBLITERATED | 7.9B | 0.0000 | 2% | 0% | 873 |

## 주요 발견

### 1. Gemma-4 E2B이 여전히 최강 (cmd_overlap 0.0436)
- 1차 평가와 동일하게 gemma-4-E2B-it이 1위
- gemma-4 시리즈(E2B, E4B, 31B)가 상위 1, 2, 5위를 차지
- 파라미터 수와 성능이 비례하지 않음 — E2B(5.1B) > E4B(7.9B) > 31B

### 2. Nemotron-Terminal 특이점: 높은 overlap + 100% thinking
- Nemotron-8B, 14B 모두 overlap=0.04 (3위 타이)
- **thinking 토큰 100%** — 모든 응답에 <think...> 포함
- cmds%는 8%로 매우 낮음 (코드 블록 대신 thinking에 집중)
- 실제 명령어 품질은 overlap으로만 보면 준수하지만, 응답 포맷이 다름

### 3. 모델 크기 vs 성능 역설
- **작은 모델이 큰 모델보다 나음**: E2B(5.1B) > E4B(7.9B) > 31B
- **Qwen3.5**: 4B > 2B > 9B (9B가 가장 낮음)
- MoE(gemma-4-26B-A4B, 4B active)는 순수 4B와 비슷한 수준
- 스케일링이 terminal 태스크에서는 효과 없음

### 4. Abliteration = 성능 제로
- gemma-4-E4B-it-OBLITERATED: overlap 0.0, cmds 2%
- 안전 필터 제거가 터미널 명령어 생성 능력까지 파괴

### 5. 응답 길이 분석
- gemma-4-31B: 1706자 (짧고 정확)
- gemma-4-E4B: 3853자 (길고 상세)
- Nemotron-14B: 2081자 (thinking 포함, 실제 명령어는 적음)
- OBLITERATED: 873자 (거의 응답 없음)

## 평가 불가 모델

| 모델 | 사유 |
|------|------|
| Qwen3.6-35B-A3B-FP8 | flashinfer.gdn_prefill 누락 (Qwen Mamba 아키텍처) |
| supergemma4-26b-abliterated | vLLM 미지원 아키텍처 (weight key 불일치) |
| Qwen3.5-27B-Claude-Distilled | NCCL 에러 (TP=2), Qwen 아키텍처 동일 이슈 예상 |
| Nemotron-Terminal-32B | torch.compile 10분+ 타임아웃 |

## SFT 베이스 모델 추천

| 우선순위 | 모델 | 근거 |
|---------|------|------|
| **1순위** | gemma-4-E2B-it (5.1B) | cmd_overlap 최고, 가벼움, 빠른 추론 |
| **2순위** | gemma-4-E4B-it (7.9B) | 1위와 근소한 차이, 더 많은 용량 |
| **3순위** | Nemotron-Terminal-8B | 터미널 특화 SFT 모델, thinking 능력 |

E2B가 파라미터 효율성/성능/속도에서 최적의 베이스. Nemotron-Terminal-8B는 터미널 특화라 SFT 시너지 가능.
