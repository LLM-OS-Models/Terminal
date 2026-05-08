# README 모델 corrected TB2-lite vLLM 재평가 (corrected_readme_models_vllm)

생성 시각: `2026-05-07T23:05:11.149061+00:00`

이 문서는 corrected 303-step TB2-lite 평가 JSON을 다시 읽어서 정리한 결과다.

점수 기준:

```text
score = 100 * avg_command_f1
```

`first_cmd_exact_pct`는 순위에 직접 섞지 않고 보조 지표로만 기록한다.

결과 디렉터리: `/home/work/.data/tb2_lite_eval/corrected_readme_models_vllm`

## 전체 순위

| 순위 | 모델(HF 저장소명) | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Template | Sec/Step | Load(s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 39.52 | 0.3952 | 0.5082 | 0.4101 | 33.0% | 82.2% | chat_template | 0.081 | 97.1 |
| 2 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 38.52 | 0.3852 | 0.4988 | 0.4056 | 32.7% | 83.2% | chat_template | 0.080 | 130.1 |
| 3 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData` | 38.26 | 0.3826 | 0.4620 | 0.3905 | 28.4% | 64.4% | chat_template | 0.293 | 377.3 |
| 4 | `Qwen/Qwen3.5-9B` | 38.10 | 0.3810 | 0.4921 | 0.3527 | 20.8% | 78.2% | chat_template | 0.268 | 123.8 |
| 5 | `nvidia/Nemotron-Terminal-32B` | 38.09 | 0.3809 | 0.5058 | 0.3827 | 40.3% | 58.7% | chat_template | 0.819 | 154.9 |
| 6 | `nvidia/Nemotron-Terminal-14B` | 37.70 | 0.3770 | 0.4688 | 0.3849 | 40.6% | 57.1% | chat_template | 0.360 | 98.8 |
| 7 | `Qwen/Qwen3.5-35B-A3B-FP8` | 36.44 | 0.3644 | 0.5086 | 0.3317 | 23.1% | 77.6% | chat_template | 0.200 | 222.8 |
| 8 | `Qwen/Qwen3.5-35B-A3B` | 36.41 | 0.3641 | 0.5068 | 0.3330 | 22.1% | 78.2% | chat_template | 0.228 | 363.1 |
| 9 | `Qwen/Qwen3.5-27B` | 36.30 | 0.3630 | 0.4985 | 0.3343 | 22.1% | 74.9% | chat_template | 0.893 | 102.6 |
| 10 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData` | 36.25 | 0.3625 | 0.4797 | 0.3723 | 26.1% | 61.7% | chat_template | 0.205 | 207.3 |
| 11 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData` | 36.05 | 0.3605 | 0.4601 | 0.3690 | 28.1% | 59.4% | chat_template | 0.206 | 158.5 |
| 12 | `nvidia/Nemotron-Terminal-8B` | 35.80 | 0.3580 | 0.4649 | 0.3592 | 35.3% | 54.5% | chat_template | 0.273 | 95.7 |
| 13 | `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | 35.68 | 0.3568 | 0.4822 | 0.3313 | 22.8% | 74.6% | chat_template | 0.904 | 203.7 |
| 14 | `LLM-OS-Models/Ouro-2.6B-Thinking-Terminal-SFT` | 35.61 | 0.3561 | 0.4586 | 0.3647 | 25.1% | 61.1% | chat_template | 3.358 | 135.3 |
| 15 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-1Epoch-FullFT-2BData` | 35.60 | 0.3560 | 0.4537 | 0.3752 | 26.7% | 62.7% | chat_template | 0.291 | 179.0 |
| 16 | `Qwen/Qwen3.5-4B` | 35.42 | 0.3542 | 0.4836 | 0.3292 | 17.8% | 75.6% | chat_template | 0.185 | 120.0 |
| 17 | `Qwen/Qwen3.5-2B` | 35.10 | 0.3510 | 0.4944 | 0.3220 | 18.2% | 81.8% | chat_template | 0.077 | 112.8 |
| 18 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-TemplateMasked` | 33.46 | 0.3346 | 0.4363 | 0.3549 | 31.7% | 65.0% | chat_template | 0.177 | 220.0 |
| 19 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-TemplateMasked` | 33.35 | 0.3335 | 0.4242 | 0.3590 | 26.1% | 66.3% | chat_template | 0.180 | 198.6 |
| 20 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout` | 32.85 | 0.3285 | 0.4194 | 0.3336 | 31.0% | 55.4% | chat_template | 0.150 | 32.9 |
| 21 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 32.77 | 0.3277 | 0.3953 | 0.3541 | 18.2% | 24.8% | chat_template | 0.348 | 300.9 |
| 22 | `Qwen/Qwen3.6-27B` | 32.56 | 0.3256 | 0.4346 | 0.3149 | 15.8% | 73.9% | chat_template | 0.889 | 178.2 |
| 23 | `LLM-OS-Models/LFM2-8B-A1B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout` | 32.41 | 0.3241 | 0.4327 | 0.3220 | 29.4% | 56.8% | chat_template | 0.126 | 36.6 |
| 24 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout` | 31.86 | 0.3186 | 0.4102 | 0.3321 | 30.0% | 55.8% | chat_template | 0.151 | 33.0 |
| 25 | `LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT` | 31.74 | 0.3174 | 0.4062 | 0.3410 | 24.8% | 63.7% | chat_template | 1.698 | 92.4 |
| 26 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout-7GPU` | 31.02 | 0.3102 | 0.4063 | 0.3201 | 29.0% | 54.8% | chat_template | 0.128 | 131.5 |
| 27 | `LLM-OS-Models/LFM2-8B-A1B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout` | 31.02 | 0.3102 | 0.4063 | 0.3201 | 29.0% | 54.8% | chat_template | 0.126 | 35.7 |
| 28 | `Qwen/Qwen3.6-35B-A3B-FP8` | 30.57 | 0.3057 | 0.4248 | 0.2873 | 14.5% | 75.2% | chat_template | 0.203 | 181.9 |
| 29 | `Qwen/Qwen3.6-35B-A3B` | 30.28 | 0.3028 | 0.4093 | 0.2879 | 14.2% | 73.3% | chat_template | 0.234 | 360.2 |
| 30 | `LLM-OS-Models/Ouro-2.6B-Terminal-SFT` | 29.58 | 0.2958 | 0.3624 | 0.3156 | 22.8% | 29.4% | chat_template | 5.154 | 332.6 |
| 31 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout-8GPU` | 28.64 | 0.2864 | 0.3775 | 0.2904 | 29.0% | 50.5% | chat_template | 0.085 | 49.9 |
| 32 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout` | 28.64 | 0.2864 | 0.3775 | 0.2904 | 29.0% | 50.5% | chat_template | 0.086 | 25.4 |
| 33 | `google/gemma-4-26B-A4B-it` | 28.51 | 0.2851 | 0.4057 | 0.2643 | 14.2% | 71.9% | chat_template | 0.277 | 747.8 |
| 34 | `LLM-OS-Models/Ouro-1.4B-terminal-sft` | 28.30 | 0.2830 | 0.3520 | 0.3141 | 22.4% | 27.1% | chat_template | 2.344 | 83.1 |
| 35 | `gyung/LFM2-8B-Terminal-SFT-Unsloth` | 28.23 | 0.2823 | 0.3817 | 0.2851 | 27.1% | 53.8% | chat_template | 0.124 | 36.0 |
| 36 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout` | 28.10 | 0.2810 | 0.3615 | 0.2941 | 29.4% | 50.5% | chat_template | 0.085 | 25.4 |
| 37 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 27.33 | 0.2733 | 0.3526 | 0.2872 | 24.1% | 62.0% | chat_template | 0.124 | 267.1 |
| 38 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 27.31 | 0.2731 | 0.3643 | 0.2804 | 21.8% | 62.0% | chat_template | 0.147 | 69.7 |
| 39 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 27.28 | 0.2728 | 0.3389 | 0.3062 | 10.2% | 13.9% | chat_template | 0.379 | 269.7 |
| 40 | `google/gemma-4-31B-it` | 26.33 | 0.2633 | 0.3513 | 0.2571 | 10.9% | 67.3% | chat_template | 1.362 | 845.5 |
| 41 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 26.27 | 0.2627 | 0.3581 | 0.2681 | 16.8% | 58.1% | chat_template | 0.179 | 227.6 |
| 42 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 24.74 | 0.2474 | 0.3390 | 0.2456 | 12.5% | 56.1% | chat_template | 0.178 | 228.6 |
| 43 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 22.45 | 0.2245 | 0.3097 | 0.2314 | 18.8% | 47.2% | chat_template | 0.083 | 57.2 |
| 44 | `google/gemma-4-E4B-it` | 19.43 | 0.1943 | 0.3150 | 0.1843 | 10.9% | 53.8% | chat_template | 0.204 | 300.3 |
| 45 | `google/gemma-4-E2B-it` | 17.23 | 0.1723 | 0.2854 | 0.1618 | 7.3% | 59.4% | chat_template | 0.156 | 234.8 |
| 46 | `LiquidAI/LFM2-2.6B` | 17.06 | 0.1706 | 0.2229 | 0.2160 | 12.9% | 29.4% | chat_template | 0.152 | 55.0 |
| 47 | `ByteDance/Ouro-1.4B` | 15.06 | 0.1506 | 0.1988 | 0.1625 | 8.9% | 37.3% | chat_template | 1.946 | 74.8 |
| 48 | `LiquidAI/LFM2.5-1.2B-Instruct` | 14.46 | 0.1446 | 0.2374 | 0.1526 | 10.6% | 60.1% | chat_template | 0.056 | 39.8 |
| 49 | `ByteDance/Ouro-1.4B-Thinking` | 12.69 | 0.1269 | 0.2026 | 0.1299 | 9.2% | 26.7% | chat_template | 2.115 | 65.9 |
| 50 | `LiquidAI/LFM2-24B-A2B` | 10.87 | 0.1087 | 0.1466 | 0.1163 | 5.3% | 54.5% | chat_template | 0.165 | 236.2 |
| 51 | `LiquidAI/LFM2-8B-A1B` | 10.04 | 0.1004 | 0.1405 | 0.1223 | 5.9% | 27.4% | chat_template | 0.124 | 61.9 |
| 52 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 9.34 | 0.0934 | 0.1377 | 0.0931 | 2.6% | 4.6% | chat_template | 0.183 | 72.7 |
| 53 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 9.15 | 0.0915 | 0.1259 | 0.1017 | 3.0% | 6.9% | chat_template | 0.184 | 76.7 |
| 54 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | chat_template | 1.774 | 300.1 |
| 55 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | chat_template | 1.770 | 300.1 |

## LFM vs Qwen 분석

- 현재 성공 결과 기준 최고 Qwen은 `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`이고 Score `39.52`다.
- 현재 성공 결과 기준 최고 LFM은 `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-TemplateMasked`이고 Score `33.46`다. 최고 Qwen 대비 `6.06`점 낮다.
- Liquid-CLI 방식 LFM 중 최고는 `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout`이며 Score `32.85`다. 최고 Qwen 대비 `6.67`점 낮다.
- 평균 Precision은 Qwen `0.4748`, LFM `0.3452`이고, 평균 Recall은 Qwen `0.3467`, LFM `0.2711`다. LFM은 정답 명령 일부를 맞히는 능력보다 불필요하거나 틀린 명령을 섞지 않는 쪽에서 더 크게 밀린다.
- 평균 Valid JSON은 Qwen `73.6%`, LFM `53.5%`다. 이 벤치는 JSON command 형식 안정성이 곧 점수로 이어지므로, LFM의 포맷 안정성 부족이 점수 하락의 직접 원인이다.
- LFM2-8B-A1B Liquid-CLI는 `1Epoch`가 `2Epoch`보다 높고, LFM2-24B TemplateMasked도 `1Epoch`가 `2Epoch`보다 근소하게 높다. 현재 데이터에서는 LFM 계열이 2epoch에서 명령 선택이 더 좋아지기보다 JSON/명령 precision이 흔들리는 경향이 있다.
- Qwen 계열은 base 모델 자체의 terminal command priors와 ChatML 포맷 적합성이 강하고, FullFT 결과도 Valid JSON과 Precision을 유지한다. 반면 LFM은 base 점수가 낮은 상태에서 SFT 상승폭은 크지만, Qwen 상위권의 포맷 안정성과 command precision까지는 아직 못 따라간다.
- 이 차이를 단순히 `모델 지능 차이`라고만 보기는 어렵다. TB2-lite는 일반 추론 지능보다 `터미널 명령을 JSON으로 안정적으로 내는 능력`을 강하게 재므로, 현재 결과는 지능 차이와 포맷/토크나이저/학습 경로 차이가 섞인 값이다.
- 따라서 현재 낮은 LFM 점수는 단순 모델 크기 문제가 아니라 `base prior`, `JSON 형식 안정성`, `assistant command precision`, `epoch별 과학습/포맷 흔들림`이 합쳐진 결과로 보는 게 맞다.

## 해석 기준

- `rank_eligible=false` 또는 `Template=raw_fallback`인 결과는 정상 chat template 평가가 아니므로 순위에서 제외한다.
- 같은 모델의 epoch/checkpoint 비교는 `Score`를 우선 보고, 거의 같으면 `Recall`, `Precision`, `Valid JSON`을 함께 본다.
- README legacy 386-step 표와 직접 비교하지 않는다.
