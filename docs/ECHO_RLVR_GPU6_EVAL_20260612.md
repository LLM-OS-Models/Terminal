# LFM2.5 ECHO RLVR GPU6 평가 노트

업데이트: 2026-06-12 22:31 UTC / 2026-06-13 07:31 KST

상세한 실행/데이터/학습 방법/해석은 [`docs/LFM25_ECHO_RLVR_RUNBOOK_KO_20260612.md`](LFM25_ECHO_RLVR_RUNBOOK_KO_20260612.md)에 정리했다. 현재 활성 run 상태는 [`docs/LFM25_ECHO_RLVR_CURRENT_STATUS_KO_20260612.md`](LFM25_ECHO_RLVR_CURRENT_STATUS_KO_20260612.md)에 따로 정리한다. 이 문서는 GPU6 TB2-lite replay 평가 결과만 짧게 추적한다.

## 평가 설정

- Base model for parent/continuation eval: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- Base model for raw clean-start eval: `LiquidAI/LFM2.5-8B-A1B`
- Active raw training run: `run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k`
- Evaluated checkpoint pools: parent run, continuation run, previous turbo run, and current raw clean-start run as checkpoints appear
- Evaluation GPU: `6`
- Evaluation set: `tb2_lite/data/replay_full.jsonl`
- Evaluation size: 303 replay steps
- Score: `100 * avg_command_f1`
- Result directory: `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612`
- HF dataset sync path: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts/eval/tb2_lite_gpu6/lfm25_echo_rlvr_gpu6_eval_20260612`

## 현재 비교 기준

- SFT 1Epoch: Score `52.30`
- SFT 2Epoch: Score `50.48`
- LiquidAI base: Score `36.53`
- Raw LFM2.5 rerun: Score `39.92`
- parent ECHO RLVR standalone `checkpoint-1880`: Score `50.05`
- parentrun sweep best so far: `checkpoint-610`, Score `54.05`
- continuation sweep best so far: `checkpoint-220`, Score `53.26`
- raw clean-start run best so far: `checkpoint-1100`, Score `44.84`

## 현재 결론

RLVR이 완전히 무의미하다는 결론은 아니다. GPU6 sweep 기준 현재 최고는 parentrun `checkpoint-610`이며, Score `54.05`로 SFT 1Epoch baseline `52.30`을 `+1.75` 넘겼다.

하지만 long-run aha moment는 아직 확인되지 않았다. 여러 checkpoint가 SFT baseline을 넘지만, 최신/final checkpoint가 자동으로 최고가 되는 흐름은 아니다. 따라서 이 run은 final checkpoint보다 best checkpoint selection이 중요하다.

중요: `parentrun checkpoint-N`, `continue checkpoint-M`, `turbo checkpoint-K`, `raw checkpoint-R`는 서로 다른 축이다. continuation run은 parent `checkpoint-1880`에서 이어서 시작했으므로, continuation `checkpoint-250`은 대략 누적 `2130` step 지점이다. raw clean-start run은 SFT adapter 없이 `LiquidAI/LFM2.5-8B-A1B`에 새 LoRA를 붙여 시작한 별도 run이다.

현재 GPU6 sweep 상태:

- 평가 완료 JSON: `281`개
- 현재 best: parentrun `checkpoint-610`, Score `54.05`
- GPU6 watcher는 `parentrun`, `continuation`, `turbo`, `raw` checkpoint 디렉터리를 모두 보도록 수정했다.
- README 점수 계산 버그를 수정했다. 점수는 `next_action_score`가 아니라 `100 * avg_command_f1`이다.
- 점수 그래프: [`docs/assets/echo_rlvr_score_vs_epoch_20260612.png`](assets/echo_rlvr_score_vs_epoch_20260612.png)

## Latest Top Results

| Rank | Checkpoint | Score | next_action_score | First Cmd | Valid JSON |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `lfm25-echo-rlvr-parentrun-checkpoint-610` | `54.05` | `54.18` | `54.5%` | `77.9%` |
| 2 | `lfm25-echo-rlvr-parentrun-checkpoint-490` | `53.76` | `53.17` | `51.8%` | `77.2%` |
| 3 | `lfm25-echo-rlvr-parentrun-checkpoint-650` | `53.65` | `52.91` | `51.2%` | `76.2%` |
| 4 | `lfm25-echo-rlvr-parentrun-checkpoint-230` | `53.43` | `52.76` | `51.2%` | `77.9%` |
| 5 | `lfm25-echo-rlvr-parentrun-checkpoint-440` | `53.32` | `53.16` | `52.8%` | `75.2%` |

## Raw Clean-Start Results

이번 raw clean-start run은 SFT 1Epoch adapter 없이 순수 `LiquidAI/LFM2.5-8B-A1B`에서 시작했다. 1175 step까지 평가한 현재 기준으로 raw base rerun `39.92`보다 분명히 오른 구간은 있다. 최고점은 checkpoint-1100 Score `44.84`로 raw base 대비 `+4.92`다. 다만 SFT 1Epoch baseline `52.30`에는 아직 `-7.46` 낮고, 최신 checkpoint-1175는 Score `43.29`로 최고점보다 내려왔다.

| Checkpoint | Score | next_action_score | First Cmd | Valid JSON | Note |
| ---: | ---: | ---: | ---: | ---: | --- |
| raw base rerun | 39.92 | - | - | - | no RLVR |
| raw 25 | 37.89 | 39.00 | 41.6% | 59.4% | below raw base |
| raw 50 | 40.34 | 40.90 | 42.2% | 58.1% | above raw base |
| raw 75 | 39.19 | 39.22 | 39.3% | 57.1% | below raw base |
| raw 100 | 40.34 | 40.81 | 41.9% | 58.4% | above raw base |
| raw 125 | 41.01 | 41.67 | 43.2% | 57.4% | previous raw best |
| raw 150 | 40.43 | 40.39 | 40.3% | 59.4% | above raw base |
| raw 175 | 40.02 | 40.19 | 40.6% | 58.1% | roughly flat |
| raw 200 | 39.67 | 39.95 | 40.6% | 59.7% | not best |
| raw 225 | 41.06 | 40.92 | 40.6% | 57.4% | early raw best |
| raw 250 | 39.95 | 40.84 | 42.9% | 59.1% | falls back near raw base |
| raw 300 | 41.52 | 41.72 | 42.2% | 59.1% | above raw base |
| raw 325 | 42.08 | 42.12 | 42.2% | 59.4% | new raw best |
| raw 425 | 41.93 | 42.31 | 43.2% | 61.1% | above raw base |
| raw 475 | 43.41 | 43.77 | 44.6% | 59.7% | clear jump |
| raw 550 | 43.46 | 43.38 | 43.2% | 64.7% | valid JSON improves |
| raw 700 | 42.73 | 42.57 | 42.2% | 63.7% | stable above base |
| raw 800 | 43.94 | 43.42 | 42.2% | 61.7% | high raw checkpoint |
| raw 900 | 43.78 | 43.73 | 43.6% | 60.7% | high raw checkpoint |
| raw 925 | 43.83 | 43.46 | 42.6% | 64.7% | high raw checkpoint |
| raw 975 | 44.47 | 44.39 | 44.2% | 63.7% | near current raw best |
| raw 1000 | 42.72 | 43.37 | 44.9% | 65.7% | local drop |
| raw 1075 | 43.77 | 44.20 | 45.2% | 68.6% | best valid JSON so far |
| raw 1100 | 44.84 | 45.16 | 45.9% | 66.3% | current raw best |
| raw 1125 | 43.77 | 43.81 | 43.9% | 64.4% | below raw best |
| raw 1150 | 44.10 | 43.35 | 41.6% | 65.3% | high but not best |
| raw 1175 | 43.29 | 43.56 | 44.2% | 63.7% | latest evaluated |

해석: raw clean-start에서도 terminal feedback 신호가 먹히고 있다. 1100 step에서 raw base rerun 대비 `+4.92`까지 올라갔고, first command exact도 `45.9%`까지 오른다. 다만 SFT 1Epoch의 구조화된 출력 습관을 따라잡지는 못했다. 현재 결론은 "ECHO RLVR은 raw 모델도 끌어올리지만, 이 TB2-lite 기준에서는 대량 SFT를 대체하기보다는 SFT 이후 미세 개선에 더 강하다"에 가깝다.

논문 재현성 주의: 현재 run은 ECHO의 핵심 objective인 action-token policy loss + terminal-observation CE loss를 구현했지만, Microsoft ECHO 논문의 Harbor/Docker/SkyRL/FSDP 재현은 아니다. 원 논문은 8870 train tasks, Docker/Harbor, 최대 16턴, 2048 generated tokens/turn, 8 B200, 500 GRPO steps 설정이다. 현재 run은 no-Docker local sandbox, 1500 prepared tasks, 최대 4턴, 256 generated tokens/turn, LoRA 2GPU 학습 + vLLM 4GPU rollout이다. 그래서 결과 해석은 "ECHO-style local adaptation"으로 해야 한다.

## Checkpoint Results

| Checkpoint | Score | Cmd F1 | First Cmd | Valid JSON | Note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 50.46 | 0.5075 | 49.8% | 76.2% | early |
| 20 | 51.78 | 0.5276 | 49.5% | 77.6% | early high |
| 30 | 51.36 | 0.5203 | 49.8% | 76.6% | early |
| 40 | 51.61 | 0.5209 | 50.5% | 77.2% | early |
| 50 | 51.59 | 0.5163 | 51.5% | 76.2% | early |
| 60 | 51.44 | 0.5214 | 49.8% | 76.9% | early |
| 70 | 50.95 | 0.5187 | 48.8% | 77.9% | early |
| 80 | 51.78 | 0.5276 | 49.5% | 76.6% | early high |
| 90 | 51.49 | 0.5162 | 51.2% | 74.6% | early |
| 100 | 52.02 | 0.5156 | 53.1% | 77.6% | near baseline |
| 110 | 51.52 | 0.5238 | 49.5% | 75.6% | early |
| 120 | 51.21 | 0.5109 | 51.5% | 75.2% | early |
| 130 | 50.77 | 0.5187 | 48.2% | 78.2% | early |
| 140 | 51.48 | 0.5177 | 50.8% | 76.9% | early |
| 150 | 51.16 | 0.5187 | 49.5% | 76.6% | early |
| 160 | 51.30 | 0.5237 | 48.8% | 76.9% | early |
| 170 | 50.84 | 0.5111 | 50.2% | 75.2% | early |
| 180 | 51.86 | 0.5176 | 52.1% | 76.9% | early |
| 200 | 50.73 | 0.5169 | 48.5% | 75.6% | lower |
| 250 | 52.88 | 0.5305 | 52.5% | 74.9% | continuation best |
| 300 | 49.87 | 0.5058 | 48.2% | 76.2% | regression |
| 350 | 50.60 | 0.5094 | 49.8% | 76.6% | lower |
| 400 | 51.47 | 0.5218 | 49.8% | 77.6% | partial recovery |
| 450 | 51.13 | 0.5072 | 52.1% | 76.2% | lower |
| 500 | 51.41 | 0.5150 | 51.2% | 75.6% | lower |
| 540 | 51.12 | 0.5152 | 50.2% | 76.9% | latest-window sample |
| 550 | 49.38 | 0.4946 | 49.2% | 74.3% | low point |
| 560 | 51.55 | 0.5243 | 49.5% | 76.9% | partial recovery |
| 570 | 50.44 | 0.5097 | 49.2% | 75.9% | lower |
| 580 | 51.12 | 0.5182 | 49.5% | 77.2% | lower |
| 590 | 49.95 | 0.5015 | 49.5% | 74.3% | lower |
| 600 | 51.04 | 0.5085 | 51.5% | 74.9% | lower |
| parentrun 610 | 54.05 | 0.5405 | 54.5% | 77.9% | current best |
| 620 | 50.68 | 0.5162 | 48.5% | 75.6% | lower |
| 630 | 49.85 | 0.5086 | 47.5% | 76.6% | latest checked |
| parent standalone 1880 | 50.05 | 0.5114 | 47.5% | 74.9% | previous-run adapter |
| parentrun 10 | 51.14 | 0.5154 | 50.2% | 76.6% | parent sweep start |
| parentrun 1830 | 51.94 | 0.5269 | 50.2% | 76.9% | parent late |
| parentrun 1880 | 51.86 | 0.5215 | 51.2% | 76.9% | parent final prefix |

## 남은 평가

GPU6 watcher는 10-step dense checkpoint와 최신 checkpoint를 계속 평가한다. 새 JSON은 `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612`에 저장되고, 15분 간격으로 HF dataset eval path에 sync된다. 현재는 parent run의 초반부터 전수 스윕하는 단계다.
