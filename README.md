# Terminal Agent

Nemotron-Terminal-Corpus 기반 터미널 에이전트 학습과 평가를 위한 작업 폴더입니다.

현재 이 저장소는 크게 세 축으로 운영됩니다.

1. `eval/` 기반 단발 프록시 평가
2. `tb2_lite/` 기반 멀티턴 replay 평가
3. `Liquid-CLI + Unsloth` / `Qwen3.5 + Unsloth` / `Qwen3.5~3.6 + HF+FSDP` 기반 SFT 및 학습

기준 날짜: `2026-04-30`

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
  - 전체 `37개 중 6위`
- `LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData`
  - `Score 28.84`
  - `Cmd F1 0.3065`
  - `First Cmd Exact 24.6%`
  - `0.170 sec/step`
  - `Load 99.7s`
  - 전체 `37개 중 7위`
- `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData`
  - `Score 28.80`
  - `Cmd F1 0.3094`
  - `First Cmd Exact 23.8%`
  - `0.084 sec/step`
  - `Load 73.9s`
  - 전체 `37개 중 8위`
- `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData`
  - `Score 26.75`
  - `Cmd F1 0.2956`
  - `First Cmd Exact 20.2%`
  - `0.226 sec/step`
  - `Load 108.5s`
  - 전체 `37개 중 19위`
- `LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData`
  - `Score 25.74`
  - `Cmd F1 0.2721`
  - `First Cmd Exact 22.3%`
  - `0.166 sec/step`
  - `Load 99.9s`
  - 전체 `37개 중 25위`
- `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData`
  - `Score 24.41`
  - `Cmd F1 0.2720`
  - `First Cmd Exact 17.9%`
  - `0.222 sec/step`
  - `Load 103.6s`
  - 전체 `37개 중 29위`
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
  - 전체 `37개 중 37위`
- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
  - `Score 28.51`
  - `Cmd F1 0.2796`
  - `First Cmd Exact 29.8%`
  - `0.025 sec/step`
  - `Load 32.5s`
  - 전체 `37개 중 10위`
- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` (LoRA)
  - `Score 22.84`
  - `Cmd F1 0.2586`
  - `First Cmd Exact 15.8%`
  - `0.028 sec/step`
  - `Load 89.9s`
  - 전체 `37개 중 35위`

현재 `tb2_lite` 점수 확정 모델은 `37개`입니다.

- 이 표에는 `점수가 실제로 나온 모델만` 넣었습니다.
- `Qwen3.5-27B` `1 epoch`와 `2 epoch final`도 이제 포함했습니다.
- `Qwen3.5-35B-A3B` `1 epoch` / `2 epoch` 평가도 이제 포함했습니다.
- `Qwen3.6-27B` `1 epoch` / `2 epoch final`도 이제 포함했습니다.
- `Qwen3.6-35B-A3B` `1 epoch` / `2 epoch final`도 이제 포함했습니다.

현재 `tb2_lite` 전체 비교 순위:

| 순위 | 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gyung/LFM2-8B-Terminal-SFT-Unsloth` | 30.14 | 0.2965 | 31.3% | 0.029 | 46.6 |
| 2 | `nvidia/Nemotron-Terminal-8B` | 30.02 | 0.2969 | 30.8% | 0.078 | 49.8 |
| 3 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 29.77 | 0.2912 | 31.3% | 0.030 | 52.6 |
| 4 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 29.66 | 0.2917 | 30.8% | 0.029 | 52.2 |
| 5 | `nvidia/Nemotron-Terminal-32B` | 29.13 | 0.2872 | 30.1% | 0.281 | 99.1 |
| 6 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData` | 28.89 | 0.3185 | 22.0% | 0.063 | 85.2 |
| 7 | `LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 28.84 | 0.3065 | 24.6% | 0.170 | 99.7 |
| 8 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData` | 28.80 | 0.3094 | 23.8% | 0.084 | 73.9 |
| 9 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 28.78 | 0.2821 | 30.1% | 0.025 | 30.3 |
| 10 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 28.51 | 0.2796 | 29.8% | 0.025 | 32.5 |
| 11 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 28.32 | 0.2781 | 29.5% | 0.029 | 29.1 |
| 12 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.32 | 0.2756 | 30.1% | 0.030 | 34.0 |
| 13 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.30 | 0.2778 | 29.5% | 0.025 | 30.3 |
| 14 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.15 | 0.2779 | 29.0% | 0.027 | 40.3 |
| 15 | `nvidia/Nemotron-Terminal-14B` | 27.72 | 0.2751 | 28.2% | 0.108 | 68.9 |
| 16 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-1Epoch-FullFT-2BData` | 27.45 | 0.3021 | 21.0% | 0.083 | 86.5 |
| 17 | `Qwen/Qwen3.5-9B` | 27.04 | 0.2808 | 24.6% | 0.072 | 90.6 |
| 18 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData` | 26.80 | 0.2907 | 21.5% | 0.065 | 60.3 |
| 19 | `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 26.75 | 0.2956 | 20.2% | 0.226 | 108.5 |
| 20 | `Qwen/Qwen3.5-2B` | 26.52 | 0.2743 | 24.4% | 0.024 | 84.7 |
| 21 | `Qwen/Qwen3.5-4B` | 26.36 | 0.2745 | 23.8% | 0.055 | 77.2 |
| 22 | `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | 26.35 | 0.2744 | 23.8% | 0.282 | 113.4 |
| 23 | `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 26.20 | 0.2865 | 20.5% | 0.060 | 439.0 |
| 24 | `google/gemma-4-26B-A4B-it` | 25.95 | 0.2631 | 25.1% | 0.094 | 123.9 |
| 25 | `LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 25.74 | 0.2721 | 22.3% | 0.166 | 99.9 |
| 26 | `LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 25.74 | 0.2700 | 22.8% | 0.065 | 768.9 |
| 27 | `Qwen/Qwen3.6-27B` | 25.60 | 0.2702 | 22.3% | 0.282 | 118.5 |
| 28 | `LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 25.58 | 0.2698 | 22.3% | 0.066 | 547.6 |
| 29 | `Qwen/Qwen3.6-35B-A3B-FP8` | 25.21 | 0.2658 | 22.0% | 0.082 | 120.5 |
| 30 | `google/gemma-4-31B-it` | 24.70 | 0.2594 | 21.8% | 0.404 | 101.4 |
| 31 | `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 24.41 | 0.2720 | 17.9% | 0.222 | 103.6 |
| 32 | `LiquidAI/LFM2-2.6B` | 24.12 | 0.2379 | 24.9% | 0.032 | 36.4 |
| 33 | `google/gemma-4-E4B-it` | 23.43 | 0.2391 | 22.3% | 0.051 | 129.7 |
| 34 | `LiquidAI/LFM2.5-1.2B-Instruct` | 23.36 | 0.2381 | 22.3% | 0.021 | 31.7 |
| 35 | `LiquidAI/LFM2-8B-A1B` | 23.19 | 0.2336 | 22.8% | 0.025 | 58.2 |
| 36 | `google/gemma-4-E2B-it` | 23.05 | 0.2359 | 21.8% | 0.032 | 116.0 |
| 37 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` | 22.84 | 0.2586 | 15.8% | 0.028 | 89.9 |
| 38 | `LiquidAI/LFM2-24B-A2B` | 22.80 | 0.2323 | 21.8% | 0.050 | 81.6 |
| 39 | `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 22.40 | 0.2433 | 17.9% | 0.060 | 302.8 |

핵심 해석:

- `Qwen3.5-2B`는 LoRA보다 full FT가 압도적으로 좋았습니다.
- `Qwen3.5-4B`는 `1 epoch`가 좋았고, `2 epoch final`은 오히려 내려갔습니다.
- `Qwen3.5-9B`는 반대로 `2 epoch final`이 더 좋았습니다.
- `Qwen3.5-27B`는 학습은 잘 됐지만, 점수는 `2 epoch 26.75`, `1 epoch 24.41`로 기대보다 낮았습니다.
- `Qwen3.5-35B-A3B`도 학습/평가는 성공했지만, `2 epoch 26.20`, `1 epoch 22.40`으로 기대보다 낮았습니다.
- `Qwen3.6-27B`는 `2 epoch 28.84`로 크게 회복됐고, 현재 전체 `7위`입니다. 다만 `1 epoch 25.74`는 아직 낮아서, 이 모델도 `2 epoch`까지 가야 성능이 붙었습니다.
- `Qwen3.6-35B-A3B`는 `1 epoch 25.58`, `2 epoch 25.74`로 아주 조금만 좋아졌고, 현재 전체 `26위` 수준입니다. 즉 `3.6`으로 올려도 `35B-A3B` 계열은 여전히 큰 개선이 없었습니다.

큰 모델이 오히려 떨어진 이유:

- 이 평가는 `다음 한두 개 명령을 얼마나 정확하게 고르느냐`에 크게 반응합니다. `next_action_score`가 `Cmd F1 70% + first command exact 30%` 구조라, 큰 모델이 길게 설명하거나 여러 명령을 제안해도 첫 행동이 살짝 빗나가면 손해가 큽니다.
- 단순 스타일 문제만은 아닙니다. `Qwen3.5-27B`는 `valid_json 56.7%`로 형식은 가장 잘 맞췄는데도 `Score 26.75`였습니다. 즉 포맷보다 **실제 첫 명령 선택**이 약했습니다.
- `Qwen3.5-2B`는 `first_cmd_exact 31.3%`였는데, `27B`는 `20.2%`, `35B`는 `20.5%`였습니다. 이 차이가 점수 차이의 핵심입니다.
- `35B-A3B`는 특히 `early bucket`이 약했습니다. `2B early F1 0.3942`, `27B 0.3996`인데 `35B 0.3140`이라, 초반 탐색/진입 명령을 더 자주 틀렸습니다.
- `27B`는 평균 예측 명령 수가 `5.11`로 너무 많았습니다. `2B`는 `0.43`, `35B`는 `1.49`였습니다. `27B`는 실제로는 꽤 맞는 명령을 여러 개 내지만, 이 벤치는 **과한 계획/과한 행동열**보다 **짧고 맞는 첫 행동**을 더 좋게 칩니다.
- `35B-A3B`는 MoE 특성까지 겹쳐서 `math 0.1955`, `system_administration 0.1997`, `model_training 0.2462`처럼 특정 터미널 작업군에서 약하게 나왔습니다. 단순히 크기만 키운다고 이 벤치가 좋아지지 않았습니다.
- `Qwen3.6-27B`는 `Qwen3.5-27B`보다 확실히 좋아졌지만, 여전히 `Qwen3.5-2B full FT`보다는 약합니다. 수치로 보면 `first_cmd_exact 24.6% vs 31.3%`, `early bucket F1 0.3443 vs 0.3942`, `avg_pred_cmds 1.54 vs 0.43`입니다. 즉 `3.6-27B`는 많이 회복됐어도 **첫 행동이 아직 덜 정확하고, 필요 이상으로 더 말하거나 더 움직입니다.**
- `Qwen3.6-35B-A3B`는 `Qwen3.6-27B`보다도 더 약했습니다. `2 epoch` 기준 `first_cmd_exact 22.8%`, `Cmd F1 0.2700`이라, `27B 2 epoch`의 `24.6% / 0.3065`보다 분명히 낮습니다. 즉 모델을 더 키운다고 바로 이 벤치가 좋아지지 않았고, `35B-A3B`는 여전히 **첫 액션을 짧고 정확하게 고르는 능력**에서 밀렸습니다.
- 카테고리별로 보면 큰 모델 약점이 더 명확합니다. `Qwen3.6-27B`는 `swe 0.1292`, `system_administration 0.2254`가 특히 낮고, `Qwen3.5-35B-A3B`는 `system_administration 0.1997`, `model_training 0.2462`, `math 0.1955`가 낮습니다. 즉 큰 모델은 전반적으로 다 나쁜 게 아니라, **실제 터미널에서 바로 다음 탐색 명령을 골라야 하는 작업군**에서 더 자주 미끄러집니다.
- 실제 출력 패턴도 비슷합니다. 작은 상위 모델은 바로 JSON 안에 짧은 명령을 넣는 경우가 많은데, 큰 모델은 문제 재서술, 장황한 계획, 잘못된 하위 과제 설정이 먼저 나오는 비율이 높았습니다. 예를 들어 `swe` 샘플에서 큰 모델은 `rm` 동작 수정 이슈를 보자마자 긴 설명과 계획을 먼저 쓰고, 어떤 경우엔 보안 리포트 작성 같은 다른 문제로 프레이밍하기도 했습니다. 이 경우 나중에 맞는 명령이 일부 섞여 있어도, **첫 행동이 틀리면 점수가 크게 깎입니다.**
- 그래서 해석은 두 층으로 해야 합니다. 현재 `TB2-lite`는 분명 **짧고 정확한 첫 액션**에 유리한 평가입니다. 하지만 동시에 큰 모델들이 실제로도 그 첫 액션을 더 자주 틀리고, 불필요한 분석/계획 토큰을 더 많이 쓰고 있습니다. 즉 **평가 방식의 편향이 일부 있지만, 실제 약점도 분명히 존재합니다.**

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

### 4. Qwen3.6-27B

학습:

- `HF + FSDP`, `2 epoch`, `8 GPU`
- 총 학습 시간: `약 22시간 10분`
- 최종 train loss: `0.277`

저장:

- `/tmp/qwen_sft/models/Qwen__Qwen3.6-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-2934`
- `/tmp/qwen_sft/models/Qwen__Qwen3.6-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-5868`
- `/tmp/qwen_sft/models/Qwen__Qwen3.6-27B__terminal_sft_2epoch_hf_fsdp/final -> checkpoint-5868`

허깅페이스:

- `LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData`

평가 결과:

- `1 epoch`: `Score 25.74`, `Cmd F1 0.2721`, `First Cmd Exact 22.3%`, 전체 `37개 중 25위`
- `2 epoch`: `Score 28.84`, `Cmd F1 0.3065`, `First Cmd Exact 24.6%`, 전체 `37개 중 7위`

평가 경로:

- `/tmp/tb2_lite_results/20260430T_tb2lite_qwen36_27b_ckpt2934_vllmfix_tp1_lmonly`
- `/tmp/tb2_lite_results/20260430T_tb2lite_qwen36_27b_ckpt5868_vllmfix_tp1_lmonly`

### 5. Qwen3.6-35B-A3B

학습:

- `HF + FSDP`, `2 epoch`, `8 GPU`, `model-only checkpoint`
- 총 학습 시간: `약 6시간 46분 40초`
- 최종 train loss: `0.2811`

저장:

- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.6-35B-A3B__terminal_sft_2epoch_hf_fsdp/checkpoint-2934`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.6-35B-A3B__terminal_sft_2epoch_hf_fsdp/checkpoint-5868`

허깅페이스:

- `LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData`

평가 결과:

- `1 epoch`: `Score 25.58`, `Cmd F1 0.2698`, `First Cmd Exact 22.3%`, 전체 `39개 중 28위`
- `2 epoch`: `Score 25.74`, `Cmd F1 0.2700`, `First Cmd Exact 22.8%`, 전체 `39개 중 26위`

평가 경로:

- `/home/work/.data/tb2_lite_results/20260430T_tb2lite_qwen36_35b_a3b_ckpt2934_vllmfix_tp1_lmonly`
- `/home/work/.data/tb2_lite_results/20260430T_tb2lite_qwen36_35b_a3b_ckpt5868_vllmfix_tp1_lmonly`

### 6. Liquid SFT

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

### 7. Qwen SFT

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
- `Qwen3.6-27B` HF+FSDP: 학습/평가 완료, HF 업로드 완료
- `Qwen3.6-35B-A3B` HF+FSDP: 학습/평가 완료, HF 업로드 진행 중

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
