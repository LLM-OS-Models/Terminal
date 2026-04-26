# Performance Summary (2026-04-26)

프로젝트 루트에서 바로 보기 위한 성능 요약 문서입니다.

## 오늘 핵심 결론

- 오늘 새로 확인한 최고 성능은 `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`
- 점수는 `29.77`
- 전체 `31개` 결과 기준 `3위`
- `1 epoch`도 `29.66`으로 바로 아래 `4위`
- 오늘 추가 완료한 `Qwen3.5-4B` full FT는 `1 epoch 28.89`, `2 epoch final 26.80`
- 오늘 추가 완료한 `Qwen3.5-9B` full FT는 `1 epoch 27.45`, `2 epoch final 28.80`
- 즉 `4B`는 `1 epoch`가 더 좋았고, `2 epoch`에서 성능이 내려갔습니다
- 반대로 `9B`는 `2 epoch final`이 더 좋았습니다
- base `Qwen/Qwen3.5-2B (26.52)` 대비 `+3.25`
- 이전 `Qwen3.5-2B` LoRA final `22.84` 대비 `+6.93`
- 아직 전체 최고는 공개 `gyung/LFM2-8B-Terminal-SFT-Unsloth (30.14)`

## 최신 실험 결과

| 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) | 전체 순위 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 29.77 | 0.2912 | 31.3% | 0.030 | 52.6 | 3 |
| `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 29.66 | 0.2917 | 30.8% | 0.029 | 52.2 | 4 |
| `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData` | 28.89 | 0.3185 | 22.0% | 0.063 | 85.2 | 6 |
| `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData` | 28.80 | 0.3094 | 23.8% | 0.084 | 73.9 | 7 |
| `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 28.78 | 0.2821 | 30.1% | 0.025 | 30.3 | 7 |
| `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 28.51 | 0.2796 | 29.8% | 0.025 | 32.5 | 8 |
| `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.32 | 0.2756 | 30.1% | 0.030 | 34.0 | 9 |
| `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 28.32 | 0.2781 | 29.5% | 0.029 | 29.1 | 10 |
| `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.30 | 0.2778 | 29.5% | 0.025 | 30.3 | 11 |
| `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.15 | 0.2779 | 29.0% | 0.027 | 40.3 | 12 |
| `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-1Epoch-FullFT-2BData` | 27.45 | 0.3021 | 21.0% | 0.083 | 86.5 | 15 |
| `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData` | 26.80 | 0.2907 | 21.5% | 0.065 | 60.3 | 17 |
| `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` | 22.84 | 0.2586 | 15.8% | 0.028 | 89.9 | 30 |

## 현재 전체 상위권

| 순위 | 모델 | Score | Sec/Step |
| --- | --- | ---: | ---: |
| 1 | `gyung/LFM2-8B-Terminal-SFT-Unsloth` | 30.14 | 0.029 |
| 2 | `nvidia/Nemotron-Terminal-8B` | 30.02 | 0.078 |
| 3 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 29.77 | 0.030 |
| 4 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 29.66 | 0.029 |
| 5 | `nvidia/Nemotron-Terminal-32B` | 29.13 | 0.281 |
| 6 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData` | 28.89 | 0.063 |
| 7 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData` | 28.80 | 0.084 |
| 8 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 28.78 | 0.025 |
| 9 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 28.51 | 0.025 |
| 10 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 28.32 | 0.029 |

## 해석

- `Qwen3.5-2B`는 LoRA 경로에선 실패했지만, `same-count full FT`에서는 완전히 달랐습니다.
- `1 epoch`와 `2 epoch` 차이는 `29.66 -> 29.77`로 `+0.11`입니다.
- 즉 이번 설정에서는 `1 epoch`만으로도 거의 다 올라왔고, `2 epoch`가 소폭 더 좋았습니다.
- `Qwen3.5-4B`는 `1 epoch 28.89 -> 2 epoch 26.80`으로 오히려 내려갔습니다.
- `Qwen3.5-9B`는 `1 epoch 27.45 -> 2 epoch 28.80`으로 올라갔습니다.
- base `Qwen/Qwen3.5-9B (27.04)` 대비 `1 epoch +0.41`, `2 epoch final +1.76`입니다.
- 즉 `4B`에선 현재 설정상 `1 epoch checkpoint`를 쓰는 편이 맞습니다.
- `Qwen3.5-4B 1 epoch`는 base `Qwen/Qwen3.5-4B (26.36)` 대비 `+2.53`이고, `final`은 `+0.44`에 그쳤습니다.
- `Qwen2B full FT final`은 공개 `gyung` SFT보다 `-0.37`, `Nemotron-Terminal-8B`보다 `-0.25` 낮습니다.
- 반대로 `Nemotron-Terminal-32B`보다 `+0.64`, `LFM 1.2B final`보다 `+0.99`, `LFM 8B final`보다 `+1.26` 높습니다.
- `Qwen3.5-2B` LoRA final은 여전히 하위권이라, 이번 케이스에서는 **LoRA보다 full FT가 압도적으로 낫다**고 보는 게 맞습니다.

## 저장 경로

학습 결과:

- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_unsloth_lora`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata`
- `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2.5-1.2B-Base__terminal_sft_h200_4gpu`
- `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2-2.6B__terminal_sft_h200_4gpu`
- `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local`

평가 결과:

- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_final_vllmfix3`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_ckpt55_final_vllmfix3`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_4b_fullft_2bdata_ckpt960_vllmfix4`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_4b_fullft_2bdata_final_vllmfix4`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_9b_fullft_2bdata_ckpt2193_vllmfix9`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_9b_fullft_2bdata_final_vllmfix9`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_small_ckpts_parallel`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_h200_sft_final_1gpu`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_sft_final_1gpu_vllm`

허깅페이스:

- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`
- `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData`
- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth`

세부 문서:

- [TB2_LITE_RESULTS_2026-04-26.md](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/docs/TB2_LITE_RESULTS_2026-04-26.md)
- [QWEN_SFT_STATUS_2026-04-26.md](/home/work/.projects/LLM-OS-Models/Terminal/qwen_sft/docs/QWEN_SFT_STATUS_2026-04-26.md)
- [TB2_LITE_CHECKPOINT_SWEEP_2026-04-26.md](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/docs/TB2_LITE_CHECKPOINT_SWEEP_2026-04-26.md)
