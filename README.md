# Terminal Agent

Nemotron-Terminal-Corpus 기반 터미널 에이전트 학습과 평가를 위한 작업 폴더입니다.

현재 이 저장소는 크게 세 축으로 운영됩니다.

1. `eval/` 기반 단발 프록시 평가
2. `tb2_lite/` 기반 멀티턴 replay 평가
3. `Liquid-CLI + Unsloth` / `Qwen3.5 + Unsloth` / `Qwen3.5 + HF+FSDP` 기반 SFT 및 학습

기준 날짜: `2026-04-28`

## 오늘 기준 핵심 상태

### 1. TB2-lite 평가

`tb2_lite`는 full Terminal-Bench 2.0보다 훨씬 빠르게 모델을 걸러내기 위한 replay 평가 레이어입니다.

주요 문서:

- 루트 성능 요약:
  [PERFORMANCE_SUMMARY_2026-04-26.md](./PERFORMANCE_SUMMARY_2026-04-26.md)
- 통합 결과:
  [TB2_LITE_RESULTS_2026-04-26.md](./tb2_lite/docs/TB2_LITE_RESULTS_2026-04-26.md)
- 소형/체크포인트 스윕:
  [TB2_LITE_CHECKPOINT_SWEEP_2026-04-26.md](./tb2_lite/docs/TB2_LITE_CHECKPOINT_SWEEP_2026-04-26.md)
- 미실험/스킵:
  [TB2_LITE_UNTESTED_2026-04-25.md](./tb2_lite/docs/TB2_LITE_UNTESTED_2026-04-25.md)
- `vLLM` 이슈:
  [TB2_LITE_VLLM_ISSUES_2026-04-26.md](./tb2_lite/docs/TB2_LITE_VLLM_ISSUES_2026-04-26.md)

오늘 핵심 결과:

- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`
  - `Score 29.77`
  - `Cmd F1 0.2912`
  - `First Cmd Exact 31.3%`
  - `0.030 sec/step`
  - `Load 52.6s`
  - 전체 `33개 중 3위`
- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount`
  - `Score 29.66`
  - `Cmd F1 0.2917`
  - `First Cmd Exact 30.8%`
  - `0.029 sec/step`
  - `Load 52.2s`
  - 전체 `33개 중 4위`
- `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData`
  - `Score 28.89`
  - `Cmd F1 0.3185`
  - `First Cmd Exact 22.0%`
  - `0.063 sec/step`
  - `Load 85.2s`
  - 전체 `33개 중 6위`
- `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData`
  - `Score 28.80`
  - `Cmd F1 0.3094`
  - `First Cmd Exact 23.8%`
  - `0.084 sec/step`
  - `Load 73.9s`
  - 전체 `33개 중 7위`
- `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData`
  - `Score 26.75`
  - `Cmd F1 0.2956`
  - `First Cmd Exact 20.2%`
  - `0.226 sec/step`
  - `Load 108.5s`
  - 전체 `33개 중 18위`
- `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData`
  - `Score 24.41`
  - `Cmd F1 0.2720`
  - `First Cmd Exact 17.9%`
  - `0.222 sec/step`
  - `Load 103.6s`
  - 전체 `35개 중 27위`
- `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData`
  - `Score 26.20`
  - `Cmd F1 0.2865`
  - `First Cmd Exact 20.5%`
  - `0.060 sec/step`
  - `Load 439.0s`
  - 전체 `35개 중 22위`
- `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData`
  - `Score 22.40`
  - `Cmd F1 0.2433`
  - `First Cmd Exact 17.9%`
  - `0.060 sec/step`
  - `Load 302.8s`
  - 전체 `35개 중 35위`
- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
  - `Score 28.51`
  - `Cmd F1 0.2796`
  - `First Cmd Exact 29.8%`
  - `0.025 sec/step`
  - `Load 32.5s`
  - 전체 `33개 중 9위`
- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` (LoRA)
  - `Score 22.84`
  - `Cmd F1 0.2586`
  - `First Cmd Exact 15.8%`
  - `0.028 sec/step`
  - `Load 89.9s`
  - 전체 `33개 중 32위`

현재 `tb2_lite` 점수 확정 모델은 `35개`입니다.

- 이 표에는 `점수가 실제로 나온 모델만` 넣었습니다.
- `Qwen3.5-27B` `1 epoch`와 `2 epoch final`도 이제 포함했습니다.
- `Qwen3.5-35B-A3B` `1 epoch` / `2 epoch` 평가도 이제 포함했습니다.

현재 `tb2_lite` 전체 비교 순위:

| 순위 | 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gyung/LFM2-8B-Terminal-SFT-Unsloth` | 30.14 | 0.2965 | 31.3% | 0.029 | 46.6 |
| 2 | `nvidia/Nemotron-Terminal-8B` | 30.02 | 0.2969 | 30.8% | 0.078 | 49.8 |
| 3 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 29.77 | 0.2912 | 31.3% | 0.030 | 52.6 |
| 4 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 29.66 | 0.2917 | 30.8% | 0.029 | 52.2 |
| 5 | `nvidia/Nemotron-Terminal-32B` | 29.13 | 0.2872 | 30.1% | 0.281 | 99.1 |
| 6 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData` | 28.89 | 0.3185 | 22.0% | 0.063 | 85.2 |
| 7 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData` | 28.80 | 0.3094 | 23.8% | 0.084 | 73.9 |
| 8 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 28.78 | 0.2821 | 30.1% | 0.025 | 30.3 |
| 9 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 28.51 | 0.2796 | 29.8% | 0.025 | 32.5 |
| 10 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 28.32 | 0.2781 | 29.5% | 0.029 | 29.1 |
| 11 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.32 | 0.2756 | 30.1% | 0.030 | 34.0 |
| 12 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.30 | 0.2778 | 29.5% | 0.025 | 30.3 |
| 13 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.15 | 0.2779 | 29.0% | 0.027 | 40.3 |
| 14 | `nvidia/Nemotron-Terminal-14B` | 27.72 | 0.2751 | 28.2% | 0.108 | 68.9 |
| 15 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-1Epoch-FullFT-2BData` | 27.45 | 0.3021 | 21.0% | 0.083 | 86.5 |
| 16 | `Qwen/Qwen3.5-9B` | 27.04 | 0.2808 | 24.6% | 0.072 | 90.6 |
| 17 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData` | 26.80 | 0.2907 | 21.5% | 0.065 | 60.3 |
| 18 | `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 26.75 | 0.2956 | 20.2% | 0.226 | 108.5 |
| 19 | `Qwen/Qwen3.5-2B` | 26.52 | 0.2743 | 24.4% | 0.024 | 84.7 |
| 20 | `Qwen/Qwen3.5-4B` | 26.36 | 0.2745 | 23.8% | 0.055 | 77.2 |
| 21 | `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | 26.35 | 0.2744 | 23.8% | 0.282 | 113.4 |
| 22 | `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 26.20 | 0.2865 | 20.5% | 0.060 | 439.0 |
| 23 | `google/gemma-4-26B-A4B-it` | 25.95 | 0.2631 | 25.1% | 0.094 | 123.9 |
| 24 | `Qwen/Qwen3.6-27B` | 25.60 | 0.2702 | 22.3% | 0.282 | 118.5 |
| 25 | `Qwen/Qwen3.6-35B-A3B-FP8` | 25.21 | 0.2658 | 22.0% | 0.082 | 120.5 |
| 26 | `google/gemma-4-31B-it` | 24.70 | 0.2594 | 21.8% | 0.404 | 101.4 |
| 27 | `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 24.41 | 0.2720 | 17.9% | 0.222 | 103.6 |
| 28 | `LiquidAI/LFM2-2.6B` | 24.12 | 0.2379 | 24.9% | 0.032 | 36.4 |
| 29 | `google/gemma-4-E4B-it` | 23.43 | 0.2391 | 22.3% | 0.051 | 129.7 |
| 30 | `LiquidAI/LFM2.5-1.2B-Instruct` | 23.36 | 0.2381 | 22.3% | 0.021 | 31.7 |
| 31 | `LiquidAI/LFM2-8B-A1B` | 23.19 | 0.2336 | 22.8% | 0.025 | 58.2 |
| 32 | `google/gemma-4-E2B-it` | 23.05 | 0.2359 | 21.8% | 0.032 | 116.0 |
| 33 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` | 22.84 | 0.2586 | 15.8% | 0.028 | 89.9 |
| 34 | `LiquidAI/LFM2-24B-A2B` | 22.80 | 0.2323 | 21.8% | 0.050 | 81.6 |
| 35 | `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 22.40 | 0.2433 | 17.9% | 0.060 | 302.8 |

핵심 해석:

- `Qwen3.5-2B`는 LoRA보다 full FT가 압도적으로 좋았습니다.
- `Qwen3.5-4B`는 `1 epoch`가 좋았고, `2 epoch final`은 오히려 내려갔습니다.
- `Qwen3.5-9B`는 반대로 `2 epoch final`이 더 좋았습니다.
- `Qwen3.5-27B`는 학습은 잘 됐지만, 점수는 `2 epoch 26.75`, `1 epoch 24.41`로 기대보다 낮았습니다.
- `Qwen3.5-35B-A3B`도 학습/평가는 성공했지만, `2 epoch 26.20`, `1 epoch 22.40`으로 기대보다 낮았습니다.

큰 모델이 오히려 떨어진 이유:

- 이 평가는 `다음 한두 개 명령을 얼마나 정확하게 고르느냐`에 크게 반응합니다. `next_action_score`가 `Cmd F1 70% + first command exact 30%` 구조라, 큰 모델이 길게 설명하거나 여러 명령을 제안해도 첫 행동이 살짝 빗나가면 손해가 큽니다.
- 단순 스타일 문제만은 아닙니다. `Qwen3.5-27B`는 `valid_json 56.7%`로 형식은 가장 잘 맞췄는데도 `Score 26.75`였습니다. 즉 포맷보다 **실제 첫 명령 선택**이 약했습니다.
- `Qwen3.5-2B`는 `first_cmd_exact 31.3%`였는데, `27B`는 `20.2%`, `35B`는 `20.5%`였습니다. 이 차이가 점수 차이의 핵심입니다.
- `35B-A3B`는 특히 `early bucket`이 약했습니다. `2B early F1 0.3942`, `27B 0.3996`인데 `35B 0.3140`이라, 초반 탐색/진입 명령을 더 자주 틀렸습니다.
- `27B`는 평균 예측 명령 수가 `5.11`로 너무 많았습니다. `2B`는 `0.43`, `35B`는 `1.49`였습니다. `27B`는 실제로는 꽤 맞는 명령을 여러 개 내지만, 이 벤치는 **과한 계획/과한 행동열**보다 **짧고 맞는 첫 행동**을 더 좋게 칩니다.
- `35B-A3B`는 MoE 특성까지 겹쳐서 `math 0.1955`, `system_administration 0.1997`, `model_training 0.2462`처럼 특정 터미널 작업군에서 약하게 나왔습니다. 단순히 크기만 키운다고 이 벤치가 좋아지지 않았습니다.

### 2. Qwen3.5-27B

학습:

- `HF + FSDP`, `2 epoch`, `8 GPU`
- 총 학습 시간: `19시간 53분 57초`
- 최종 train loss: `0.5311`

저장:

- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-1917`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-3834`

허깅페이스:

- `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData`

평가 결과:

- `/tmp/tb2_lite_results/20260427T_tb2lite_qwen35_27b_hf_fsdp_ckpt3834_vllmfix_tp1_lmonly`
- `/tmp/tb2_lite_results/20260427T_tb2lite_qwen35_27b_hf_fsdp_ckpt1917_vllmfix_tp1_lmonly`

### 3. Qwen3.5-35B-A3B

학습:

- `HF + FSDP`, `2 epoch`, `8 GPU`, `model-only checkpoint`
- 총 학습 시간: `4시간 43분 19초`
- 최종 train loss: `0.5427`

저장:

- `/tmp/qwen_sft/models/Qwen__Qwen3.5-35B-A3B__terminal_sft_2epoch_hf_fsdp_modelonly/checkpoint-1917`
- `/tmp/qwen_sft/models/Qwen__Qwen3.5-35B-A3B__terminal_sft_2epoch_hf_fsdp_modelonly/checkpoint-3834`
- `/tmp/qwen_sft/models/Qwen__Qwen3.5-35B-A3B__terminal_sft_2epoch_hf_fsdp_modelonly/final -> checkpoint-3834`

크기:

- `checkpoint-1917`: `129.15 GiB`
- `checkpoint-3834`: `129.15 GiB`

평가 결과:

- `1 epoch`: `Score 22.40`, `Cmd F1 0.2433`, `First Cmd Exact 17.9%`
- `2 epoch`: `Score 26.20`, `Cmd F1 0.2865`, `First Cmd Exact 20.5%`

허깅페이스:

- `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData`

평가 메모:

- 체크포인트가 `qwen3_5_moe_text`로 저장돼서 `vLLM` 기본 경로에서 바로 안 떴습니다.
- 그래서 `base config + text-only vllmfix` 방식으로 `27B`와 같은 우회 경로를 적용했습니다.

### 4. Liquid SFT

원본/준비 코드 경로:

- [Liquid-CLI](./Liquid-CLI)
- `unsloth-src/`
- [liquid_sft](./liquid_sft)
- 상태 문서:
  [SFT_PREP_STATUS_2026-04-25.md](./liquid_sft/docs/SFT_PREP_STATUS_2026-04-25.md)

완료된 모델:

- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth`

### 5. Qwen SFT

별도 경로:

- [qwen_sft](./qwen_sft)
- 상태 문서:
  [QWEN_SFT_STATUS_2026-04-26.md](./qwen_sft/docs/QWEN_SFT_STATUS_2026-04-26.md)

핵심:

- `Qwen3.5-2B` LoRA: 성능 실패
- `Qwen3.5-2B` full FT same-count: 성공
- `Qwen3.5-4B` full FT 2BData: `1 epoch`가 best
- `Qwen3.5-9B` full FT 2BData: `2 epoch final`이 best
- `Qwen3.5-27B` HF+FSDP: 학습/평가 완료
- `Qwen3.5-35B-A3B` HF+FSDP: 학습 완료, 평가 완료

## 저장 경로

학습 결과:

- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_unsloth_lora`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp`
- `/tmp/qwen_sft/models/Qwen__Qwen3.5-35B-A3B__terminal_sft_2epoch_hf_fsdp_modelonly`
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
- `/tmp/tb2_lite_results/20260427T_tb2lite_qwen35_27b_hf_fsdp_ckpt3834_vllmfix_tp1_lmonly`
- `/tmp/tb2_lite_results/20260427T_tb2lite_qwen35_27b_hf_fsdp_ckpt1917_vllmfix_tp1_lmonly`
- `/tmp/tb2_lite_results/20260428T_tb2lite_qwen35_35b_a3b_ckpt3834_vllmfix_tp1_lmonly`
- `/tmp/tb2_lite_results/20260428T_tb2lite_qwen35_35b_a3b_ckpt1917_vllmfix_tp1_lmonly`

허깅페이스:

- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`
- `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData`
- `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData`
- `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData`
- `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData`
- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth`
