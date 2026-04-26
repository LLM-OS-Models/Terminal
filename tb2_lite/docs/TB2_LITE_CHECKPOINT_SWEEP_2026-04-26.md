# TB2-lite Checkpoint Sweep 결과 정리 (2026-04-26)

## 왜 이 문서를 만들었나

소형 Liquid 계열과 체크포인트 시점을 빠르게 비교하기 위해 `tb2_lite` full replay를 병렬로 돌린 결과를 정리했습니다.

이번 sweep은 아래 4개를 비교합니다.

- `LiquidAI/LFM2.5-1.2B-Base` `final`
- `LiquidAI/LFM2.5-1.2B-Base` `checkpoint-27` (`1 epoch`)
- `LiquidAI/LFM2-2.6B` `checkpoint-27` (`1 epoch`)
- `LiquidAI/LFM2-2.6B` `final`
- 기존 로컬 `8B`의 `checkpoint-27` (`1 epoch`)

결과 경로:

- [20260426T_tb2lite_small_ckpts_parallel](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_small_ckpts_parallel)

## Sweep 결과

| 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 28.78 | 0.2821 | 30.1% | 0.025 | 30.3 |
| `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.32 | 0.2756 | 30.1% | 0.030 | 34.0 |
| `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 28.32 | 0.2781 | 29.5% | 0.029 | 29.1 |
| `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.30 | 0.2778 | 29.5% | 0.025 | 30.3 |
| `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.15 | 0.2779 | 29.0% | 0.027 | 40.3 |

## 전체 모델 비교에서의 위치

현재 점수가 있는 전체 `24개` 기준으로 보면:

| 전체 순위 | 모델 | Score |
| --- | --- | ---: |
| 4 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 28.78 |
| 5 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 28.51 |
| 6 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.32 |
| 7 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 28.32 |
| 8 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.30 |
| 9 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.15 |

## 해석

- 이번 sweep 최고는 `1.2B final`입니다.
- `1.2B final`은 기존 로컬 `8B final (28.51)`도 이겼습니다.
- `1.2B final`과 `1.2B checkpoint-27` 차이는 `+0.48`이라, 2 epoch까지 가는 이득이 실제로 있습니다.
- `2.6B checkpoint-27`과 `2.6B final`은 점수는 같고, final이 `Cmd F1`만 약간 더 높았습니다.
- `2.6B`는 1 epoch 이후 추가 이득이 크지 않았습니다.
- 속도는 `1.2B`가 가장 좋고, `2.6B`도 충분히 빠른 편입니다.

한 줄 결론:

**현재 추가 실험 중에서는 `LiquidAI/LFM2.5-1.2B-Base final`이 가장 강했고, 작은데도 성능이 매우 잘 나왔습니다.**
