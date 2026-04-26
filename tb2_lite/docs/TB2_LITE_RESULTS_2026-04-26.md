# TB2-lite Replay 결과 정리 (2026-04-26)

## 왜 이 문서를 만들었나

`Liquid-CLI + Unsloth`와 `Qwen3.5 + Unsloth`로 학습한 로컬 SFT 모델을 `tb2_lite` 기준으로 한 번에 비교하기 위해 만들었습니다.

이번 문서는 아래 run들을 통합한 결과입니다.

- [20260426T_tb2lite_h200_sft_final_1gpu](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_h200_sft_final_1gpu)
- [20260426T_tb2lite_qwen35_2b_sft_final_1gpu_vllm](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_sft_final_1gpu_vllm)
- [20260426T_tb2lite_qwen35_2b_fullft_samecount_final_vllmfix3](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_final_vllmfix3)
- [20260426T_tb2lite_qwen35_2b_fullft_samecount_ckpt55_final_vllmfix3](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_ckpt55_final_vllmfix3)

## 핵심 결과

이번 날짜 기준 최고 추가 성능은 아래 둘입니다.

| 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) | Gen(s) | 전체 순위 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 29.77 | 0.2912 | 31.3% | 0.030 | 52.6 | 11.6 | 3 |
| `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 29.66 | 0.2917 | 30.8% | 0.029 | 52.2 | 11.2 | 4 |

한 줄 요약:

- `Qwen3.5-2B`는 LoRA 경로에서는 `22.84`로 실패했지만,
- `same-count full FT`에서는 `29.77`까지 올라가면서 **전체 3위**가 됐습니다.

## LFM 로컬 SFT 결과

| 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) | Gen(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 28.51 | 0.2796 | 29.8% | 0.025 | 32.5 | 9.5 |

관련 JSON:

- [LFM2-8B-Terminal-SFT-Unsloth-H200-local-final.json](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_h200_sft_final_1gpu/LFM2-8B-Terminal-SFT-Unsloth-H200-local-final.json)

## Qwen3.5-2B LoRA 결과

| 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) | Gen(s) | 전체 순위 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` | 22.84 | 0.2586 | 15.8% | 0.028 | 89.9 | 10.9 | 26 |

관련 JSON:

- [Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth.json](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_sft_final_1gpu_vllm/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth.json)

## Qwen3.5-2B Full FT same-count 결과

학습 산출물:

- output root:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount`
- eval-ready final:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/final-vllmfix3`
- eval-ready 1epoch:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/checkpoint-55-vllmfix3`

결과 JSON:

- [Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount-vllmfix3.json](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_final_vllmfix3/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount-vllmfix3.json)
- [Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount-vllmfix3.json](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_ckpt55_final_vllmfix3/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount-vllmfix3.json)

| 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) | Gen(s) | 전체 순위 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 29.77 | 0.2912 | 31.3% | 0.030 | 52.6 | 11.6 | 3 |
| `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 29.66 | 0.2917 | 30.8% | 0.029 | 52.2 | 11.2 | 4 |

핵심 비교:

- base `Qwen/Qwen3.5-2B (26.52)` 대비
  - `1 epoch`: `+3.14`
  - `2 epoch final`: `+3.25`
- LoRA final `22.84` 대비
  - `1 epoch`: `+6.82`
  - `2 epoch final`: `+6.93`
- `2 epoch final`이 `1 epoch`보다 `+0.11` 높았습니다.

## 왜 `vllmfix3`가 필요했나

`trainer.save_model()`로 저장된 full FT 산출물은 `vLLM`이 바로 읽는 표준 HF `Qwen3.5-VL` 키가 아니었습니다.

실제 문제:

- 원본 full FT key 예:
  `model.language_model.language_model.language_model.layers...`
  `model.language_model.visual...`
- `vLLM`의 `Qwen3-VL` 매퍼가 기대하는 HF key:
  `model.language_model...`
  `model.visual...`

그래서 아래 스크립트로 prefix를 표준 HF 형식으로 다시 썼습니다.

- [fix_fullft_export_prefix.py](/home/work/.projects/LLM-OS-Models/Terminal/qwen_sft/scripts/fix_fullft_export_prefix.py)

이 변환을 거친 `vllmfix3` 디렉터리에서 `tb2_lite` 평가가 정상 동작했습니다.

## 전체 모델 비교

현재 `tb2_lite` full replay 기준 점수가 있는 모델 `27개` 전체 비교는 아래와 같습니다.

| 순위 | 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gyung/LFM2-8B-Terminal-SFT-Unsloth` | 30.14 | 0.2965 | 31.3% | 0.029 | 46.6 |
| 2 | `nvidia/Nemotron-Terminal-8B` | 30.02 | 0.2969 | 30.8% | 0.078 | 49.8 |
| 3 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 29.77 | 0.2912 | 31.3% | 0.030 | 52.6 |
| 4 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 29.66 | 0.2917 | 30.8% | 0.029 | 52.2 |
| 5 | `nvidia/Nemotron-Terminal-32B` | 29.13 | 0.2872 | 30.1% | 0.281 | 99.1 |
| 6 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 28.78 | 0.2821 | 30.1% | 0.025 | 30.3 |
| 7 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 28.51 | 0.2796 | 29.8% | 0.025 | 32.5 |
| 8 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.32 | 0.2756 | 30.1% | 0.030 | 34.0 |
| 9 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 28.32 | 0.2781 | 29.5% | 0.029 | 29.1 |
| 10 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.30 | 0.2778 | 29.5% | 0.025 | 30.3 |
| 11 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.15 | 0.2779 | 29.0% | 0.027 | 40.3 |
| 12 | `nvidia/Nemotron-Terminal-14B` | 27.72 | 0.2751 | 28.2% | 0.108 | 68.9 |
| 13 | `Qwen/Qwen3.5-9B` | 27.04 | 0.2808 | 24.6% | 0.072 | 90.6 |
| 14 | `Qwen/Qwen3.5-2B` | 26.52 | 0.2743 | 24.4% | 0.024 | 84.7 |
| 15 | `Qwen/Qwen3.5-4B` | 26.36 | 0.2745 | 23.8% | 0.055 | 77.2 |
| 16 | `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | 26.35 | 0.2744 | 23.8% | 0.282 | 113.4 |
| 17 | `google/gemma-4-26B-A4B-it` | 25.95 | 0.2631 | 25.1% | 0.094 | 123.9 |
| 18 | `Qwen/Qwen3.6-27B` | 25.60 | 0.2702 | 22.3% | 0.282 | 118.5 |
| 19 | `Qwen/Qwen3.6-35B-A3B-FP8` | 25.21 | 0.2658 | 22.0% | 0.082 | 120.5 |
| 20 | `google/gemma-4-31B-it` | 24.70 | 0.2594 | 21.8% | 0.404 | 101.4 |
| 21 | `LiquidAI/LFM2-2.6B` | 24.12 | 0.2379 | 24.9% | 0.032 | 36.4 |
| 22 | `google/gemma-4-E4B-it` | 23.43 | 0.2391 | 22.3% | 0.051 | 129.7 |
| 23 | `LiquidAI/LFM2.5-1.2B-Instruct` | 23.36 | 0.2381 | 22.3% | 0.021 | 31.7 |
| 24 | `LiquidAI/LFM2-8B-A1B` | 23.19 | 0.2336 | 22.8% | 0.025 | 58.2 |
| 25 | `google/gemma-4-E2B-it` | 23.05 | 0.2359 | 21.8% | 0.032 | 116.0 |
| 26 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` | 22.84 | 0.2586 | 15.8% | 0.028 | 89.9 |
| 27 | `LiquidAI/LFM2-24B-A2B` | 22.80 | 0.2323 | 21.8% | 0.050 | 81.6 |

## 결론

- `Qwen3.5-2B`는 이번 실험에서 **LoRA보다 full FT가 압도적으로 좋았습니다.**
- `same-count full FT final`은 전체 `27개 중 3위`입니다.
- 공개 최고 `gyung/LFM2-8B-Terminal-SFT-Unsloth`는 아직 못 넘었지만, 차이는 `-0.37`까지 줄었습니다.
- `1 epoch`도 `29.66`이라 거의 비슷해서, 이 설정에서는 이미 1 epoch에서 대부분의 성능이 올라온다고 보는 게 맞습니다.
