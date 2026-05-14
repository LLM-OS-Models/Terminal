# README 모델 corrected TB2-lite vLLM 재평가

생성 시각: `2026-05-08T02:10:34.673274+00:00`

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
| 1 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch` | 39.56 | 0.3956 | 0.4702 | 0.4808 | 40.6% | 17.2% | gemma4_native | 6.820 | 43.8 |
| 2 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 39.52 | 0.3952 | 0.5082 | 0.4101 | 33.0% | 82.2% | chat_template | 0.081 | 97.1 |
| 3 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 38.52 | 0.3852 | 0.4988 | 0.4056 | 32.7% | 83.2% | chat_template | 0.080 | 130.1 |
| 4 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData` | 38.26 | 0.3826 | 0.4620 | 0.3905 | 28.4% | 64.4% | chat_template | 0.293 | 377.3 |
| 5 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-1Epoch` | 38.12 | 0.3812 | 0.4405 | 0.4787 | 42.6% | 13.5% | gemma4_native | 6.854 | 38.8 |
| 6 | `Qwen/Qwen3.5-9B` | 38.10 | 0.3810 | 0.4921 | 0.3527 | 20.8% | 78.2% | chat_template | 0.268 | 123.8 |
| 7 | `nvidia/Nemotron-Terminal-32B` | 38.09 | 0.3809 | 0.5058 | 0.3827 | 40.3% | 58.7% | chat_template | 0.819 | 154.9 |
| 8 | `nvidia/Nemotron-Terminal-14B` | 37.70 | 0.3770 | 0.4688 | 0.3849 | 40.6% | 57.1% | chat_template | 0.360 | 98.8 |
| 9 | `Qwen/Qwen3.5-35B-A3B-FP8` | 36.44 | 0.3644 | 0.5086 | 0.3317 | 23.1% | 77.6% | chat_template | 0.200 | 222.8 |
| 10 | `Qwen/Qwen3.5-35B-A3B` | 36.41 | 0.3641 | 0.5068 | 0.3330 | 22.1% | 78.2% | chat_template | 0.228 | 363.1 |
| 11 | `Qwen/Qwen3.5-27B` | 36.30 | 0.3630 | 0.4985 | 0.3343 | 22.1% | 74.9% | chat_template | 0.893 | 102.6 |
| 12 | `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` | 36.29 | 0.3629 | 0.4813 | 0.3460 | 17.5% | 80.5% | chat_template | 5.577 | - |
| 13 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData` | 36.25 | 0.3625 | 0.4797 | 0.3723 | 26.1% | 61.7% | chat_template | 0.205 | 207.3 |
| 14 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData` | 36.05 | 0.3605 | 0.4601 | 0.3690 | 28.1% | 59.4% | chat_template | 0.206 | 158.5 |
| 15 | `nvidia/Nemotron-Terminal-8B` | 35.80 | 0.3580 | 0.4649 | 0.3592 | 35.3% | 54.5% | chat_template | 0.273 | 95.7 |
| 16 | `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | 35.68 | 0.3568 | 0.4822 | 0.3313 | 22.8% | 74.6% | chat_template | 0.904 | 203.7 |
| 17 | `LLM-OS-Models/Ouro-2.6B-Thinking-Terminal-SFT` | 35.61 | 0.3561 | 0.4586 | 0.3647 | 25.1% | 61.1% | chat_template | 3.358 | 135.3 |
| 18 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-1Epoch-FullFT-2BData` | 35.60 | 0.3560 | 0.4537 | 0.3752 | 26.7% | 62.7% | chat_template | 0.291 | 179.0 |
| 19 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-1Epoch` | 35.55 | 0.3555 | 0.4650 | 0.3776 | 27.4% | 65.3% | gemma4_native | 3.924 | 41.4 |
| 20 | `DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF:Q4_K_M` | 35.47 | 0.3547 | 0.4756 | 0.3397 | 17.8% | 78.9% | chat_template | 9.964 | 6.6 |
| 21 | `Jackrong/Qwopus3.6-35B-A3B-v1-GGUF:Q4_K_M` | 35.45 | 0.3545 | 0.4929 | 0.3287 | 18.5% | 81.2% | chat_template | 4.835 | 5.7 |
| 22 | `Qwen/Qwen3.5-4B` | 35.42 | 0.3542 | 0.4836 | 0.3292 | 17.8% | 75.6% | chat_template | 0.185 | 120.0 |
| 23 | `Qwen/Qwen3.5-2B` | 35.10 | 0.3510 | 0.4944 | 0.3220 | 18.2% | 81.8% | chat_template | 0.077 | 112.8 |
| 24 | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-2Epoch` | 34.98 | 0.3498 | 0.4737 | 0.3576 | 30.4% | 35.0% | gemma4_native | 0.277 | 53.2 |
| 25 | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-1Epoch` | 34.42 | 0.3442 | 0.4823 | 0.3397 | 27.1% | 45.5% | gemma4_native | 0.360 | 93.6 |
| 26 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-TemplateMasked` | 33.46 | 0.3346 | 0.4363 | 0.3549 | 31.7% | 65.0% | chat_template | 0.177 | 220.0 |
| 27 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-TemplateMasked` | 33.35 | 0.3335 | 0.4242 | 0.3590 | 26.1% | 66.3% | chat_template | 0.180 | 198.6 |
| 28 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout` | 32.85 | 0.3285 | 0.4194 | 0.3336 | 31.0% | 55.4% | chat_template | 0.150 | 32.9 |
| 29 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 32.77 | 0.3277 | 0.3953 | 0.3541 | 18.2% | 24.8% | chat_template | 0.348 | 300.9 |
| 30 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch` | 32.57 | 0.3257 | 0.4417 | 0.3396 | 30.7% | 61.7% | gemma4_native | 4.036 | 43.2 |
| 31 | `Qwen/Qwen3.6-27B` | 32.56 | 0.3256 | 0.4346 | 0.3149 | 15.8% | 73.9% | chat_template | 0.889 | 178.2 |
| 32 | `LLM-OS-Models/LFM2-8B-A1B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout` | 32.41 | 0.3241 | 0.4327 | 0.3220 | 29.4% | 56.8% | chat_template | 0.126 | 36.6 |
| 33 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout` | 31.86 | 0.3186 | 0.4102 | 0.3321 | 30.0% | 55.8% | chat_template | 0.151 | 33.0 |
| 34 | `LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT` | 31.74 | 0.3174 | 0.4062 | 0.3410 | 24.8% | 63.7% | chat_template | 1.698 | 92.4 |
| 35 | `LLM-OS-Models/LFM2-8B-A1B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout` | 31.02 | 0.3102 | 0.4063 | 0.3201 | 29.0% | 54.8% | chat_template | 0.126 | 35.7 |
| 36 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout-7GPU` | 31.02 | 0.3102 | 0.4063 | 0.3201 | 29.0% | 54.8% | chat_template | 0.128 | 131.5 |
| 37 | `Qwen/Qwen3.6-35B-A3B-FP8` | 30.57 | 0.3057 | 0.4248 | 0.2873 | 14.5% | 75.2% | chat_template | 0.203 | 181.9 |
| 38 | `Qwen/Qwen3.6-35B-A3B` | 30.28 | 0.3028 | 0.4093 | 0.2879 | 14.2% | 73.3% | chat_template | 0.234 | 360.2 |
| 39 | `LLM-OS-Models/Ouro-2.6B-Terminal-SFT` | 29.58 | 0.2958 | 0.3624 | 0.3156 | 22.8% | 29.4% | chat_template | 5.154 | 332.6 |
| 40 | `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch` | 28.96 | 0.2896 | 0.3829 | 0.3254 | 24.4% | 26.4% | gemma4_native | 5.950 | 43.2 |
| 41 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout-8GPU` | 28.64 | 0.2864 | 0.3775 | 0.2904 | 29.0% | 50.5% | chat_template | 0.085 | 49.9 |
| 42 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout` | 28.64 | 0.2864 | 0.3775 | 0.2904 | 29.0% | 50.5% | chat_template | 0.086 | 25.4 |
| 43 | `google/gemma-4-26B-A4B-it` | 28.51 | 0.2851 | 0.4057 | 0.2643 | 14.2% | 71.9% | chat_template | 0.277 | 747.8 |
| 44 | `LLM-OS-Models/Ouro-1.4B-terminal-sft` | 28.30 | 0.2830 | 0.3520 | 0.3141 | 22.4% | 27.1% | chat_template | 2.344 | 83.1 |
| 45 | `gyung/LFM2-8B-Terminal-SFT-Unsloth` | 28.23 | 0.2823 | 0.3817 | 0.2851 | 27.1% | 53.8% | chat_template | 0.124 | 36.0 |
| 46 | `Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M` | 28.21 | 0.2821 | 0.4135 | 0.2506 | 14.5% | 53.8% | gemma4_native | 5.688 | 7.5 |
| 47 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout` | 28.10 | 0.2810 | 0.3615 | 0.2941 | 29.4% | 50.5% | chat_template | 0.085 | 25.4 |
| 48 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 27.33 | 0.2733 | 0.3526 | 0.2872 | 24.1% | 62.0% | chat_template | 0.124 | 267.1 |
| 49 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 27.31 | 0.2731 | 0.3643 | 0.2804 | 21.8% | 62.0% | chat_template | 0.147 | 69.7 |
| 50 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 27.28 | 0.2728 | 0.3389 | 0.3062 | 10.2% | 13.9% | chat_template | 0.379 | 269.7 |
| 51 | `google/gemma-4-31B-it` | 26.33 | 0.2633 | 0.3513 | 0.2571 | 10.9% | 67.3% | chat_template | 1.362 | 845.5 |
| 52 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 26.27 | 0.2627 | 0.3581 | 0.2681 | 16.8% | 58.1% | chat_template | 0.179 | 227.6 |
| 53 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-1Epoch` | 25.70 | 0.2570 | 0.3615 | 0.2717 | 15.2% | 34.3% | gemma4_native | 0.325 | 51.8 |
| 54 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-2Epoch` | 24.92 | 0.2492 | 0.3667 | 0.2447 | 11.6% | 34.0% | gemma4_native | 0.317 | 51.8 |
| 55 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 24.74 | 0.2474 | 0.3390 | 0.2456 | 12.5% | 56.1% | chat_template | 0.178 | 228.6 |
| 56 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 22.45 | 0.2245 | 0.3097 | 0.2314 | 18.8% | 47.2% | chat_template | 0.083 | 57.2 |
| 57 | `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-1Epoch` | 21.08 | 0.2108 | 0.2886 | 0.2422 | 11.2% | 21.1% | gemma4_native | 5.650 | 44.7 |
| 58 | `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch` | 19.52 | 0.1952 | 0.2626 | 0.2091 | 15.5% | 25.7% | gemma4_native | 7.116 | 36.0 |
| 59 | `google/gemma-4-E4B-it` | 19.36 | 0.1936 | 0.3184 | 0.1822 | 11.6% | 54.8% | chat_template | 0.205 | 175.8 |
| 60 | `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-1Epoch` | 18.66 | 0.1866 | 0.2370 | 0.2152 | 14.5% | 25.1% | gemma4_native | 7.020 | 45.3 |
| 61 | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-2Epoch` | 18.47 | 0.1847 | 0.2514 | 0.1980 | 16.8% | 17.2% | gemma4_native | 0.302 | 52.6 |
| 62 | `google/gemma-4-E2B-it` | 17.40 | 0.1740 | 0.2918 | 0.1613 | 7.3% | 57.1% | chat_template | 0.148 | 139.6 |
| 63 | `LiquidAI/LFM2-2.6B` | 17.06 | 0.1706 | 0.2229 | 0.2160 | 12.9% | 29.4% | chat_template | 0.152 | 55.0 |
| 64 | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-2Epoch` | 16.22 | 0.1622 | 0.2747 | 0.1678 | 15.2% | 16.2% | gemma4_native | 0.289 | 73.4 |
| 65 | `ByteDance/Ouro-1.4B` | 15.06 | 0.1506 | 0.1988 | 0.1625 | 8.9% | 37.3% | chat_template | 1.946 | 74.8 |
| 66 | `LiquidAI/LFM2.5-1.2B-Instruct` | 14.46 | 0.1446 | 0.2374 | 0.1526 | 10.6% | 60.1% | chat_template | 0.056 | 39.8 |
| 67 | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-1Epoch` | 12.80 | 0.1280 | 0.1792 | 0.1364 | 10.2% | 14.2% | gemma4_native | 0.383 | 93.8 |
| 68 | `ByteDance/Ouro-1.4B-Thinking` | 12.69 | 0.1269 | 0.2026 | 0.1299 | 9.2% | 26.7% | chat_template | 2.115 | 65.9 |
| 69 | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-1Epoch` | 11.35 | 0.1135 | 0.1767 | 0.1191 | 6.6% | 7.3% | gemma4_native | 0.219 | 45.3 |
| 70 | `LiquidAI/LFM2-24B-A2B` | 10.87 | 0.1087 | 0.1466 | 0.1163 | 5.3% | 54.5% | chat_template | 0.165 | 236.2 |
| 71 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 10.67 | 0.1067 | 0.1507 | 0.1067 | 4.0% | 5.9% | chat_template | 0.185 | 65.7 |
| 72 | `LiquidAI/LFM2-8B-A1B` | 10.04 | 0.1004 | 0.1405 | 0.1223 | 5.9% | 27.4% | chat_template | 0.124 | 61.9 |
| 73 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 9.22 | 0.0922 | 0.1249 | 0.1023 | 3.3% | 5.9% | chat_template | 0.184 | 67.5 |
| 74 | `ByteDance/Ouro-2.6B` | 6.46 | 0.0646 | 0.0976 | 0.0692 | 5.0% | 16.5% | chat_template | 4.607 | 99.6 |
| 75 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | chat_template | 1.774 | 300.1 |
| 76 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | chat_template | 1.770 | 300.1 |

전체 순위 반영 메모:

- 완료된 Native Gemma 4 결과 16개와 요청 외부 모델 완료분 4개를 이 표에 직접 반영했다.
- 현재 전체 1위는 `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch`, Score `39.56`이다. 기존 1위 Qwen3.5-2B SFT 2Epoch `39.52`보다 `+0.04` 높다.
- 26B-A4B-it native 1epoch도 Score `38.12`로 전체 5위에 들어갔다. 31B-it native 1epoch는 `35.55`로 전체 19위, 31B-it native 2epoch는 `32.57`로 전체 30위다.
- 요청 외부 모델 완료분은 `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` 12위, `DavidAU/Qwen3.6-27B-Heretic...:Q4_K_M` 20위, `Jackrong/Qwopus3.6-35B-A3B-v1-GGUF:Q4_K_M` 21위, `Jiunsong/supergemma4-26b...:Q4_K_M` 46위다.
- 아직 실행 중인 `Qwen3.6-40B Deckard GGUF`, `ZAYA1-74B-preview`, `DeepSeek-V4-Flash`는 결과 JSON이 생기는 즉시 같은 표에 재정렬 반영한다.

### 점수 해석: 잘 된 것과 안 된 것

핵심 기준은 단순 JSON 유효율이 아니라 `Cmd F1`이다. 이번 태스크는 다음 행동 명령을 얼마나 많이 맞히는지가 중요하므로, `Valid JSON`이 높아도 recall이 낮으면 순위가 내려가고, JSON 유효율이 낮아도 command coverage와 first command가 강하면 점수가 오른다.

잘 된 모델:

- `gemma-4-26B-A4B-it` native SFT 2epoch가 Score `39.56`으로 전체 1위다. Precision `0.4702`, Recall `0.4808`, First Cmd `40.6%`라 명령 집합을 넓게 맞힌다. Valid JSON은 `17.2%`로 낮지만, 실제 command F1이 가장 높아서 평가 기준상 1위가 맞다.
- 같은 26B-A4B-it 1epoch도 Score `38.12`, Recall `0.4787`, First Cmd `42.6%`로 강하다. 2epoch가 `+1.44` 오른 이유는 precision이 `0.4405 -> 0.4702`로 올라가면서 command 과다/오답이 줄었기 때문이다.
- `gemma-4-31B-it` native 1epoch는 Score `35.55`, Valid JSON `65.3%`다. 26B-A4B-it보다 format 안정성은 훨씬 좋지만 Recall `0.3776`이라 command coverage가 낮아 전체 19위에 머문다.
- 요청 외부 모델 중 `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL`은 Score `36.29`, Valid JSON `80.5%`, Precision `0.4813`으로 안정적이다. 기존 `Qwen/Qwen3.5-27B` Score `36.30`과 사실상 동급이며, GGUF/MTP 경로로도 상위권에 들어왔다.
- `Jackrong/Qwopus3.6-35B-A3B-v1-GGUF:Q4_K_M`과 `DavidAU/Qwen3.6-27B-Heretic...:Q4_K_M`은 각각 Score `35.45`, `35.47`이다. 둘 다 Valid JSON `80%` 안팎이고 precision은 높지만 recall이 `0.3287~0.3397`이라 26B-A4B-it native처럼 많은 명령을 잡지는 못한다.

잘 안 된 모델:

- 31B-it native 2epoch는 Score `32.57`로 1epoch `35.55`보다 `-2.98` 낮다. Valid JSON은 `61.7%`로 여전히 높지만 Precision `0.4417`, Recall `0.3396` 모두 1epoch보다 낮아졌다. 큰 dense it 모델은 2epoch에서 format은 유지해도 action recall이 떨어진 것으로 본다.
- 31B base native는 2epoch `28.96`, 1epoch `21.08`이다. 2epoch가 `+7.88` 개선됐지만 it 모델과 격차가 크다. base 모델은 terminal assistant prior가 약해 SFT를 해도 JSON/명령 정책을 충분히 못 따라온다.
- 26B-A4B base native는 1epoch `18.66`, 2epoch `19.52`로 낮다. MoE 크기 자체보다 instruction prior 부재가 더 큰 병목이다. 같은 26B-A4B라도 it 계열은 `38~39점대`, base는 `18~19점대`라 차이가 `약 20점` 난다.
- 작은 Gemma base도 마찬가지다. E4B-it native 2epoch는 `34.98`인데 E4B base native 2epoch는 `18.47`, E2B-it native 1epoch는 `25.70`인데 E2B base native 2epoch는 `16.22`다. base 모델은 규모와 무관하게 terminal JSON command 형식 습득이 약하다.
- `Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M`은 Score `28.21`, Valid JSON `53.8%`로 Qwen GGUF 계열보다 낮다. Gemma 계열 prompt/template을 맞췄는데도 Recall `0.2506`이라 행동 명령을 넓게 복원하지 못한다.

이번 결과에서 보이는 패턴:

- `it` 사전정렬 + native SFT가 가장 중요하다. 26B-A4B-it native는 전체 1위지만 26B-A4B base native는 하위권이다.
- 큰 모델이라고 자동으로 오르지 않는다. 31B-it는 format 안정성은 좋지만 action recall이 낮아 26B-A4B-it보다 약하다.
- GGUF 인기 모델들은 JSON 안정성은 높다. 다만 TB2-lite에서는 shell command recall이 부족하면 35~36점대에서 멈춘다.
- Valid JSON과 Score는 같은 방향이 아니다. 26B-A4B-it native 2epoch는 Valid JSON `17.2%`로 낮아도 Score `39.56`이고, Qwen GGUF 계열은 Valid JSON `78~81%`라도 Score `35~36점대`다.

### 실제 최고 모델 판단

현재 이 벤치마크 기준 최고 모델은 `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch`로 본다. Score `39.56`으로 1위이고, 2위 `Qwen3.5-2B SFT 2Epoch`의 `39.52`보다 `+0.04` 높다. 차이는 작지만 우연으로만 보기 어려운 부분은 Recall `0.4808`과 First Cmd `40.6%`다. 이 모델은 다음 행동 후보를 넓게 맞히는 능력이 강하다.

다만 실전 배치 기준으로는 1위와 2위의 선택이 갈린다.

- 품질 우선이면 `gemma-4-26B-A4B-it native 2epoch`다. 전체 Score 1위이고, 1epoch도 Score `38.12`로 5위라 학습 경로가 재현성 있게 먹혔다.
- 비용/속도/안정성까지 보면 `Qwen3.5-2B SFT 2Epoch`가 여전히 매우 강하다. Score `39.52`로 사실상 공동 1위인데 Sec/Step `0.081`이다. Gemma 26B-A4B-it native 2epoch는 Sec/Step `6.820`이라 약 `84배` 느리다.
- 공개 비학습 모델 중 최고는 `Qwen/Qwen3.5-9B`다. Score `38.10`으로 전체 6위이고, `nvidia/Nemotron-Terminal-32B` `38.09`, `nvidia/Nemotron-Terminal-14B` `37.70`보다 높다. 즉 이 태스크에서는 유명 terminal-tuned 모델보다도 raw Qwen 9B의 기본 command prior가 더 잘 맞았다.

이 결과가 특이한 이유:

- 1위는 MoE인 `Gemma 26B-A4B-it` SFT이고, 2위는 순수 2B dense SFT다. 크기만 보면 26B-A4B가 압도적으로 커 보이지만 active parameter 관점에서는 A4B라 실제 실행 성격은 작고 빠른 MoE에 가깝다.
- 2B SFT가 26B/31B 계열과 경쟁하는 이유는 데이터/포맷 적합도가 모델 크기보다 더 중요했기 때문이다. `Qwen3.5-2B SFT`는 Valid JSON `82.2%`, Precision `0.5082`라 포맷과 명령 선택이 안정적이다.
- Gemma 26B-A4B-it native는 Valid JSON은 낮지만 Recall이 매우 높다. 즉 JSON wrapper는 자주 깨져도, 실제 shell action 단위는 많이 맞힌다. 이 벤치마크의 score가 command F1이므로 1위가 된다.
- 학습하지 않은 `Qwen/Qwen3.5-9B`가 Score `38.10`인 것은 중요하다. 이 모델은 별도 terminal SFT 없이도 Precision `0.4921`, Recall `0.3527`, Valid JSON `78.2%`다. 기본 instruction/code prior가 이 데이터셋에 매우 잘 맞는다.

유명 모델과 외부 GGUF 모델이 기대보다 낮은 이유:

- `Nemotron-Terminal-32B`는 First Cmd `40.3%`로 첫 행동은 강하지만 Score `38.09`로 Qwen 9B와 거의 동률이다. First Cmd가 높아도 전체 command set recall/precision 평균에서 확실히 앞서지는 못했다.
- `Nemotron-Terminal-14B`는 Score `37.70`으로 강하지만, Qwen 9B `38.10`보다 낮다. terminal-tuning이 되어 있어도 TB2-lite의 replay command 분포와 완전히 같은 것은 아니다.
- `Qwopus3.6-35B-A3B GGUF`와 `Qwen3.6-27B Heretic GGUF`는 Score `35.45~35.47`이다. Valid JSON은 `78.9~81.2%`로 좋지만 Recall이 `0.3287~0.3397`에 머문다. 즉 답 형식은 잘 지키지만 필요한 명령을 충분히 많이 복원하지 못한다.
- `unsloth/Qwen3.6-27B-MTP-GGUF`는 Score `36.29`로 외부 요청 모델 중 현재 최고지만, 기존 `Qwen/Qwen3.5-27B` `36.30`과 거의 같다. MTP/추론 속도 이점은 있을 수 있지만, 이 점수만 보면 command 품질이 크게 뛰지는 않는다.
- `supergemma4-26b-uncensored GGUF`는 Score `28.21`이다. Gemma 계열 prompt를 맞춰도 Recall `0.2506`, Valid JSON `53.8%`라 command-following과 포맷 안정성이 둘 다 부족했다.

실패/저조 결과의 원인 판단:

- 기존 HF-FSDP 31B SFT 1/2epoch는 둘 다 Score `0.00`이다. 같은 31B 계열이라도 native SFT 31B-it는 `35.55`까지 나오므로, 31B 모델 자체 한계가 아니라 기존 HF-FSDP checkpoint/export 또는 학습 포맷이 깨진 실패로 본다.
- Gemma base 계열은 전반적으로 약하다. 26B-A4B base 2epoch `19.52`, 31B base 2epoch `28.96`이고, 같은 크기의 it 계열보다 크게 낮다. 이 태스크는 terminal assistant 형식과 JSON command target이 중요해서 base 모델에는 SFT만으로 부족했다.
- 31B-it native 2epoch가 1epoch보다 낮은 것은 overfit 또는 action diversity 손실 가능성이 있다. Valid JSON은 유지되지만 Recall이 `0.3776 -> 0.3396`으로 떨어져, 더 많이 학습하면서 안전하고 짧은 행동으로 수렴했을 가능성이 있다.
- Gemma 26B-A4B-it 2epoch는 반대로 Precision이 `0.4405 -> 0.4702`로 개선됐다. MoE it 모델에는 2epoch가 command 선택 정밀도를 올리는 쪽으로 작용했다.

벤치마크 한계:

- 이 표는 TB2-lite corrected 303-step replay 기준이다. 일반 대화, 수학, 긴 코딩, 검색형 작업의 전체 능력을 대표하지 않는다.
- Score는 `100 * avg_command_f1`이다. JSON 문법, 설명 품질, 안전성, reasoning 품질은 보조 지표일 뿐이다.
- Valid JSON이 낮아도 command parser가 명령을 뽑아낼 수 있으면 점수가 높게 나온다. 따라서 1위 Gemma 26B-A4B-it native는 실제 제품에 넣기 전에 JSON 형식 안정화 후처리나 추가 SFT가 필요하다.
- GGUF 평가는 llama.cpp/llama-server 경로라 vLLM BF16 평가와 완전히 같은 런타임 조건이 아니다. 다만 모두 같은 303-step 데이터와 같은 scoring code로 계산했기 때문에 순위 비교의 큰 방향은 유효하다.
- 속도와 비용은 별도 판단해야 한다. 2B SFT는 품질이 거의 1위인데 압도적으로 빠르고, Gemma 26B-A4B-it native는 품질은 1위지만 비용이 크다.

## Gemma 4 재평가 및 재학습 메모

진행 시각: `2026-05-08~2026-05-09 KST`

참조:

- Hugging Face collection: `https://huggingface.co/collections/google/gemma-4`
- E2B-it model card: `https://huggingface.co/google/gemma-4-E2B-it`
- 현재 재평가 출력: `/home/work/.data/tb2_lite_eval/gemma4_retemplate_full_20260508`

### full 303 재평가

아래 값은 이 문서와 같은 기준인 `Score = 100 * avg_command_f1`로 계산했다.
`2026-05-08` renderer correction 이후 E2B/E4B 계열 값은 그대로 유효하고, 26B-A4B/31B 계열 값은 previous assistant history의 empty thought channel 과삽입을 제거한 설정으로 재측정 예정이다.

| 모델 | 설정 | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | 기존 Score | 변화 | 기존표 대체 시 순위 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google/gemma-4-E2B-it` | strip thinking history, no empty thought channel | 17.42 | 0.1742 | 0.2832 | 0.1698 | 6.9% | 59.1% | 17.40 | +0.02 | 45 |
| `google/gemma-4-E2B-it` | strip thinking history, forced empty thought channel | 17.33 | 0.1733 | 0.3001 | 0.1673 | 9.6% | 52.8% | 17.40 | -0.07 | 45 |
| `google/gemma-4-E4B-it` | strip thinking history, no empty thought channel | 19.95 | 0.1995 | 0.3167 | 0.1997 | 12.2% | 58.4% | 19.36 | +0.59 | 44 |
| `google/gemma-4-E4B-it` | strip thinking history, forced empty thought channel | 20.31 | 0.2031 | 0.3173 | 0.2028 | 10.2% | 42.2% | 19.36 | +0.95 | 44 |
| `google/gemma-4-26B-A4B-it` | strip thinking history, native empty thought channel | 29.78 | 0.2978 | 0.4129 | 0.2833 | 16.2% | 70.6% | 28.51 | +1.27 | 31 |
| `google/gemma-4-31B-it` | strip thinking history, native empty thought channel | 29.21 | 0.2921 | 0.3872 | 0.2882 | 12.5% | 69.0% | 26.33 | +2.88 | 33 |
| `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | strip thinking history, native empty thought channel | 29.83 | 0.2983 | 0.3716 | 0.3329 | 10.2% | 18.2% | 27.28 | +2.55 | 30 |
| `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | strip thinking history, native empty thought channel | 32.31 | 0.3231 | 0.3960 | 0.3561 | 17.5% | 31.4% | 32.77 | -0.46 | 23 |
| `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | strip thinking history, no empty thought channel | 11.53 | 0.1153 | 0.1545 | 0.1167 | 4.0% | 11.2% | 10.67 | +0.86 | 50 |
| `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | strip thinking history, no empty thought channel | 11.50 | 0.1150 | 0.1438 | 0.1310 | 4.3% | 10.9% | 9.22 | +2.28 | 51 |
| `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | strip thinking history, native empty thought channel | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | 0.00 | +0.00 | 55 |
| `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | strip thinking history, native empty thought channel | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | 0.00 | +0.00 | 56 |

해석:

- E2B/E4B는 `<think>...</think>` 히스토리 제거 후에도 점수가 크게 회복되지 않았다. 따라서 낮은 점수의 핵심 원인은 단순 chat template 한 가지가 아니라 `JSON command 포맷 순응`, `터미널 명령 prior`, `학습 데이터 포맷 충돌`이 섞인 문제로 본다.
- E4B는 약 `+1`점 개선되지만 기존표 기준 순위는 여전히 `44위`다. E2B는 `17.40 -> 17.42`로 사실상 동일하며 `45위`다.
- 26B-A4B/31B base는 템플릿/히스토리 패치로 의미 있게 오른다. 특히 31B base는 `26.33 -> 29.21`로 올라, 이전 31B base 점수에는 history/template 손실이 섞여 있었다.
- 기존 26B-A4B SFT e2는 여전히 강하지만 `32.77 -> 32.31`로 근소 하락했다. 템플릿 패치보다 기존 SFT 데이터의 command recall 효과가 점수에 더 크게 남아 있는 것으로 보인다.
- 기존 E2B SFT는 재평가 후에도 `11.5`점 수준이라 base `17.4`보다 낮다. 작은 Gemma의 구 SFT는 포맷을 망가뜨린 실패 케이스로 본다.
- E4B 기존 SFT e1/e2는 vLLM 로딩 실패: checkpoint에 `model.layers.24~41.self_attn.k_norm.weight`가 없어서 초기화 실패한다. 이는 단순 점수 문제가 아니라 checkpoint/export가 현재 Gemma 4 구조와 맞지 않는 문제다.
- 기존 31B SFT e1/e2는 재평가해도 `0.00` 그대로다. base 31B는 `29.21`까지 올라가므로, 31B 자체 문제가 아니라 기존 31B SFT checkpoint/export 또는 text-only weight extraction 경로가 망가진 것으로 본다.
- 31B SFT 실제 출력은 JSON이 아니라 `la la de la...` 반복이다. aggregate도 `avg_pred_cmds=0.0`, `valid_json_pct=0.0`이다. 학습 로그의 loss는 NaN으로 터진 형태가 아니므로, 단순 학습 중단보다 저장/로드/weight extraction 또는 잘못된 학습 포맷으로 출력 분포가 붕괴한 케이스로 본다.

### native SFT full 303 평가 결과

`2026-05-09 09:56 KST` 기준 결과다. 출력 디렉터리: `/home/work/.data/tb2_lite_eval/gemma4_native_sft_20260509`

| 모델 | HF repo | Checkpoint | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Sec/Step | 기존 대비 | 전체표 대입 순위 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `google/gemma-4-E4B-it` | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-1Epoch` | `checkpoint-1021` | 34.42 | 0.3442 | 0.4823 | 0.3397 | 27.1% | 45.5% | 0.360 | E4B-it retemplate best `20.31` 대비 `+14.11` | 18 |
| `google/gemma-4-E2B-it` | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-1Epoch` | `checkpoint-1021` | 25.70 | 0.2570 | 0.3615 | 0.2717 | 15.2% | 34.3% | 0.325 | E2B-it retemplate `17.42` 대비 `+8.28`, 구 SFT e1 `10.67` 대비 `+15.03` | 43 |
| `google/gemma-4-E2B-it` | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-2Epoch` | `checkpoint-2042` | 24.92 | 0.2492 | 0.3667 | 0.2447 | 11.6% | 34.0% | 0.317 | native 1epoch 대비 `-0.78`, 구 SFT e2 `9.22` 대비 `+15.70` | 44 |
| `google/gemma-4-E2B` | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-2Epoch` | `checkpoint-2042` | 16.22 | 0.1622 | 0.2747 | 0.1678 | 15.2% | 16.2% | 0.289 | native base 1epoch 대비 `+4.87`, 그래도 E2B-it 1epoch 대비 `-9.48` | 50 |
| `google/gemma-4-E4B` | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-1Epoch` | `checkpoint-1021` | 12.80 | 0.1280 | 0.1792 | 0.1364 | 10.2% | 14.2% | 0.383 | base+template 주입만으로는 instruction prior 부족 | 53 |
| `google/gemma-4-E2B` | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-1Epoch` | `checkpoint-1021` | 11.35 | 0.1135 | 0.1767 | 0.1191 | 6.6% | 7.3% | 0.219 | base+template 주입만으로는 instruction prior 부족 | 55 |

### 26B/31B native SFT full 303 평가 결과

`2026-05-14 08:43 UTC` 기준 결과다. 출력 디렉터리: `/home/work/.data/tb2_lite_eval/gemma4_native_sft_20260509`

공통 실행 설정:

- evaluator: `tb2_lite/scripts/replay_eval.py`
- launcher: `gemma4_native_sft/scripts/run_large_native_eval_8gpu.py`
- vLLM: `.vllm-0_19_1`, text-only, `max_model_len=49152`, `max_tokens=1024`, `temperature=0.0`
- runtime: 8개 모델 1GPU씩 병렬, `gpu_memory_utilization=0.90`, `max_num_seqs=8`, `max_num_batched_tokens=16384`, `--enforce-eager`, `--disable-custom-all-reduce`

| 모델 | HF repo | Checkpoint | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Sec/Step | Load(s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `google/gemma-4-26B-A4B-it` | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch` | `checkpoint-2040` | 39.56 | 0.3956 | 0.4702 | 0.4808 | 40.6% | 17.2% | 6.820 | 43.8 |
| `google/gemma-4-26B-A4B-it` | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-1Epoch` | `checkpoint-1020` | 38.12 | 0.3812 | 0.4405 | 0.4787 | 42.6% | 13.5% | 6.854 | 38.8 |
| `google/gemma-4-31B-it` | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-1Epoch` | `checkpoint-510` | 35.55 | 0.3555 | 0.4650 | 0.3776 | 27.4% | 65.3% | 3.924 | 41.4 |
| `google/gemma-4-31B-it` | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch` | `checkpoint-1020` | 32.57 | 0.3257 | 0.4417 | 0.3396 | 30.7% | 61.7% | 4.036 | 43.2 |
| `google/gemma-4-31B` | `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch` | `checkpoint-1020` | 28.96 | 0.2896 | 0.3829 | 0.3254 | 24.4% | 26.4% | 5.950 | 43.2 |
| `google/gemma-4-31B` | `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-1Epoch` | `checkpoint-510` | 21.08 | 0.2108 | 0.2886 | 0.2422 | 11.2% | 21.1% | 5.650 | 44.7 |
| `google/gemma-4-26B-A4B` | `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch` | `checkpoint-1020` | 19.52 | 0.1952 | 0.2626 | 0.2091 | 15.5% | 25.7% | 7.116 | 36.0 |
| `google/gemma-4-26B-A4B` | `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-1Epoch` | `checkpoint-510` | 18.66 | 0.1866 | 0.2370 | 0.2152 | 14.5% | 25.1% | 7.020 | 45.3 |

해석:

- 26B-A4B-it native 2epoch가 `39.56`으로 현재 Gemma 계열 최고다. 이 표 기준 전체 1위권 Qwen SFT `39.52`와 같은 레벨이며, 1epoch `38.12`보다 `+1.44` 높다.
- 31B-it native는 1epoch `35.55`, 2epoch `32.57`로 2epoch가 낮다. 31B-it는 JSON valid가 `60%+`로 높지만, command F1은 26B-A4B-it보다 낮다.
- base 계열은 31B base 2epoch `28.96`을 제외하면 낮다. 26B-A4B base는 `18~19점대`라 instruction prior 없는 base SFT만으로는 주력 후보가 아니다.
- 이번 31B native 결과는 이전 HF-FSDP 31B SFT `0.00` 문제와 다르다. native 경로에서는 31B-it가 정상 JSON을 생성하고 30점대 점수를 낸다.

해석:

- 4B-it는 확실히 낫다. `34.42`는 현재 전체표에 대입하면 `18위`이며, 최고 LFM `33.46`보다 `+0.96`, LFM2-2.6B SFT `32.85`보다 `+1.57` 높다.
- 다만 Qwen 상위권과는 아직 차이가 있다. `Qwen3.5-2B SFT 2Epoch` `39.52` 대비 `-5.10`, `Qwen3.5-2B base` `35.10` 대비 `-0.68`, `Ouro-2.6B-Thinking SFT` `35.61` 대비 `-1.19`다.
- E2B-it는 native 전처리로 구 SFT 실패를 크게 회복했다. 하지만 `25점대`라 LFM/Ouro/Qwen 주력 후보와는 아직 거리가 있다.
- E2B-it는 2epoch가 1epoch보다 낮다. 현재 Gemma 작은 모델은 epoch를 더 먹인다고 자동으로 좋아지지 않고, JSON/recall 균형이 흔들릴 수 있다.
- base 모델 SFT(`E2B`, `E4B`)는 낮다. `E2B base`는 2epoch에서 `11.35 -> 16.22`로 회복했지만, instruction-following prior가 있는 `E2B-it`에는 `-9.48` 낮다. 작은 base는 RL 후보에서 제외하는 쪽이 맞다.
- 현재 Gemma RL 후보는 `E4B-it native 1epoch` 1개가 조건부로 살아났다. 2epoch가 유지되거나 26B native가 이보다 높게 나오면 Gemma 후보를 1~2개까지 넣는다.

### 요청 외부 모델 full 303 평가 결과

`2026-05-14 09:58 UTC` 기준 완료분이다. 출력 디렉터리: `/home/work/.data/tb2_lite_eval/requested_models_20260514`

| 모델 | Backend | Score | Next Action | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Sec/Step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unsloth/Qwen3.6-27B-MTP-GGUF` `UD-Q4_K_XL` | `llama-server` MTP HTTP | 36.29 | 30.65 | 0.3629 | 0.4813 | 0.3460 | 17.5% | 80.5% | 5.577 |
| `DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF` `Q4_K_M` | `llama.cpp` CUDA | 35.47 | 30.17 | 0.3547 | 0.4756 | 0.3397 | 17.8% | 78.9% | 9.964 |
| `Jackrong/Qwopus3.6-35B-A3B-v1-GGUF` `Q4_K_M` | `llama.cpp` CUDA | 35.45 | 30.36 | 0.3545 | 0.4929 | 0.3287 | 18.5% | 81.2% | 4.835 |
| `Jiunsong/supergemma4-26b-uncensored-gguf-v2` `Q4_K_M` | `llama.cpp` CUDA | 28.21 | 24.10 | 0.2821 | 0.4135 | 0.2506 | 14.5% | 53.8% | 5.688 |

진행 중:

- `DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF` `Q4_K_M`
- `Zyphra/ZAYA1-74B-preview` 4GPU `tp=4` 수동 실행

### full 303 재평가 실행 결과

동일 evaluator와 vLLM `0.19.1` 환경에서 아래 모델들을 다시 돌렸다.

- 완료: `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it`, 기존 E2B SFT e1/e2, 기존 26B-A4B SFT e1/e2, 기존 31B SFT e1/e2
- 실패: 기존 E4B SFT e1/e2는 `k_norm.weight` 누락으로 vLLM load 실패
- 공통 평가 옵션: `--strip-thinking-history on`, `--language-model-only`, `max_model_len=49152`, `temperature=0.0`, `max_tokens=1024`
- 26B-A4B/31B 계열은 non-thinking일 때 Gemma 4 템플릿의 empty thought channel을 유지한다.
- E2B/E4B 계열은 기본적으로 empty thought channel이 없는 템플릿을 사용하되, 강제 삽입 ablation도 함께 확인했다.

### Gemma 4 tokenizer/template 확인

컬렉션 기준 대상 모델:

- `google/gemma-4-E2B`, `google/gemma-4-E2B-it`
- `google/gemma-4-E4B`, `google/gemma-4-E4B-it`
- `google/gemma-4-26B-A4B`, `google/gemma-4-26B-A4B-it`
- `google/gemma-4-31B`, `google/gemma-4-31B-it`

확인 결과:

- 8개 모두 tokenizer vocab/len은 `262144`, `bos=<bos>`, `eos=<eos>`, `pad=<pad>`로 맞다.
- base 모델(`E2B`, `E4B`, `26B-A4B`, `31B`)은 tokenizer에 `chat_template`가 없다. base까지 SFT하려면 `it` 계열과 같은 Gemma 4 turn template을 명시적으로 주입해야 한다.
- `E2B-it`와 `E4B-it`는 같은 chat template hash(`33204f1acb5b`)를 쓴다. `enable_thinking=False`에서 final prompt가 `<|turn>model\n`로 끝나며 empty thought channel을 자동 삽입하지 않는다.
- `26B-A4B-it`와 `31B-it`는 같은 chat template hash(`94899c0f917d`)를 쓴다. `enable_thinking=False`에서 final prompt 뒤에 `<|channel>thought\n<channel|>` empty thought channel이 붙는다.
- renderer correction `2026-05-08`: empty thought channel은 완료된 이전 assistant history에는 넣지 않고, 마지막 generation prompt에만 붙인다. HF tokenizer로 `system/user/assistant/user + generation_prompt`를 대조해 확인했다. 학습 전처리는 이미 HF `apply_chat_template` 기반이라 영향이 없고, 26B/31B baseline 재평가는 이 correction으로 다시 돌려 갱신한다.
- 학습 full sequence는 assistant target 뒤에 `<turn|>`로 닫힌다. 예: `<bos><|turn>system\n...<turn|>\n<|turn>user\n...<turn|>\n<|turn>model\n{...}<turn|>\n`.
- thinking mode를 켜면 system turn 앞쪽에 `<|think|>`가 들어간다. 이번 터미널 JSON SFT와 평가는 모두 `enable_thinking=False`가 기본이다.
- Gemma 4 model card는 multi-turn에서 과거 assistant thought를 다음 turn history에 넣지 말라고 명시한다. TB2-lite replay 데이터는 기존 assistant history에 `<think>...</think>`가 많이 포함되어 있어 Gemma 4 전처리에서는 반드시 제거해야 한다.

전처리/cache 정책:

- raw/filter 단계는 모델 공통으로 공유 가능하다.
- text/tokenized cache는 하나로 공유하지 않는다. 최소 `E2B/E4B template`, `26B-A4B/31B template`, `base 모델용 주입 template`로 분리한다. 실제 학습에서는 충돌을 피하려고 모델별 cache root를 둔다.
- 학습 label은 final assistant response만 loss를 주고, system/user/history/prompt 토큰은 `-100`으로 masking한다.
- supervised text는 `generation_prompt + assistant JSON + <turn|>`로 만든다. 26B-A4B/31B는 non-thinking generation prompt에 empty thought channel이 들어가므로, completed assistant turn 렌더링만 쓰면 평가 prompt와 target 위치가 어긋날 수 있다.
- target response는 evaluator가 기대하는 JSON command object만 남긴다. 과거 `<think>` 블록, transcript continuation, terminal output 복사는 학습 target에서 제거한다.
- Qwen/ChatML 토큰(`<|im_start|>`, `<|im_end|>`)이 들어간 기존 processed text는 Gemma 4 학습에 재사용하지 않는다.

### 작은 모델 native 전처리 완료

새 파이프라인: `gemma4_native_sft/scripts/prepare_native_dataset.py`

원본: `/home/work/.data/liquid_cli_sft/datasets/sft_data`

공통 설정:

- `max_seq_length=8192`
- `strip_thinking_history=true`
- `target_json_only=true`
- `keep_empty_commands=false`
- holdout: `eval/eval_dataset.jsonl`, `holdout_hash=116362516728454d`
- loss mask: prompt/history는 `-100`, assistant JSON response만 label

| 모델 | 캐시 경로 | Template model | Template injected | Rows | Candidate turns | Non-JSON/no-command skip | Too-long skip | Prompt p50/p95 | Label p50/p95 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `google/gemma-4-E2B-it` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E2B-it__liquid_raw_json_masked_8192` | `google/gemma-4-E2B-it` | false | 16,322 | 27,520 | 5,210 | 5,988 | 3,757 / 7,159 | 390 / 925 |
| `google/gemma-4-E4B-it` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E4B-it__liquid_raw_json_masked_8192` | `google/gemma-4-E4B-it` | false | 16,322 | 27,520 | 5,210 | 5,988 | 3,757 / 7,159 | 390 / 925 |
| `google/gemma-4-E2B` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E2B__liquid_raw_json_masked_8192` | `google/gemma-4-E2B-it` | true | 16,322 | 27,520 | 5,210 | 5,988 | 3,757 / 7,159 | 390 / 925 |
| `google/gemma-4-E4B` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E4B__liquid_raw_json_masked_8192` | `google/gemma-4-E4B-it` | true | 16,322 | 27,520 | 5,210 | 5,988 | 3,757 / 7,159 | 390 / 925 |

해석:

- 작은 모델 4개는 tokenizer vocab은 같지만, base에는 chat template이 없으므로 base 캐시는 `*-it` 템플릿을 명시 주입했다.
- `Too-long skip=5,988`은 멀티턴 전체 히스토리를 prompt로 유지하기 때문에 생긴다. 직전 user만 쓰면 row 수는 늘지만 평가와 다른 입력 분포가 되므로, 현재는 평가와 같은 conversation-prefix 방식이 맞다.
- 캐시 크기는 각 약 `892M`이다.

### 재학습 계획

평가와 원인 분석이 끝나면 `gemma4_sft`를 고쳐 쓰거나 새 폴더 `gemma4_native_sft`를 만들어 Gemma 4 전용 파이프라인으로 다시 학습한다. 우선순위는 작은 모델부터다.

1. `google/gemma-4-E2B-it`: 가장 빠른 ablation. template/masking/target JSON이 제대로 먹는지 확인한다.
2. `google/gemma-4-E4B-it`: 작은 모델 중 실제 점수 기대가 더 높다. E2B보다 우선 성능 검증 가치가 크다.
3. `google/gemma-4-E2B`: base 모델 SFT. tokenizer에는 template이 없으므로 E2B-it template을 명시 주입한다.
4. `google/gemma-4-E4B`: base 모델 SFT. E4B-it template을 명시 주입한다.
5. `google/gemma-4-26B-A4B-it`: full 후보. 기존표에서 base `28.51`, 구 SFT e2 `32.77`이라 가장 현실적인 상위권 후보.
6. `google/gemma-4-26B-A4B`: base MoE 후보. 26B-A4B-it template 계열을 주입하되 empty thought channel 정책을 유지한다.
7. `google/gemma-4-31B-it`: 기존 SFT가 `0.00`이라 먼저 dense 구조 weight/export 손상 여부를 확인한 뒤 학습한다.
8. `google/gemma-4-31B`: base dense 후보. 31B-it template 계열을 주입하고 31B 전용 로딩/저장 코드를 따로 검증한다.

성공 기준:

- small model 재학습은 최소 base 대비 하락하지 않아야 한다. E2B는 `17.4+`, E4B는 `20.3+`가 1차 하한이다.
- 26B-A4B는 기존 SFT e2 `32.77`을 넘기는 것이 목표다.
- 31B는 먼저 `0.00` 재현 원인을 제거하고, base `26.33` 이상 회복을 1차 목표로 둔다.

### 학습 실행 상태

`2026-05-08` 기준:

- smoke: `google/gemma-4-E2B-it`, 1GPU, 83-row smoke cache, 3 steps 완료. loss `14.9 -> 8.834 -> 8.207`, checkpoint 저장 성공.
- 본학습 시작: 작은 모델 4개를 2GPU씩 병렬 실행.
- 진행 snapshot `2026-05-08T15:35Z`: `E2B-it 1300/2042`, `E2B 1133/2042`, `E4B-it 587/2042`, `E4B 586/2042`. 네 run 모두 loss가 내려가고 있으며 학습 로그 기준 NaN/OOM/Traceback 없음.
- checkpoint 상태: `E2B-it checkpoint-1021`, `E2B checkpoint-1021` 저장 완료. 이는 1epoch checkpoint이며, full 2epoch 학습 완료는 아니다. `E4B-it`, `E4B`는 아직 epoch1 checkpoint 전이다.
- `2026-05-09 09:30 KST` snapshot: `E2B-it` 2epoch 완료/평가 완료, `E2B base 1995/2042`, `E4B-it 1029/2042`, `E4B base 1025/2042`.
- 현재 페이스 기준 완료 예상: `E2B base 2026-05-09 09:52 KST`, `E4B-it 2026-05-10 01:56 KST`, `E4B base 2026-05-10 03:39 KST`. E4B 계열 epoch1 checkpoint는 저장/평가가 완료됐고, 2epoch checkpoint는 각 full 완료 시 저장된다.
- VRAM snapshot: GPU0 `92902/143771MB`, GPU1 `91736/143771MB`, GPU2 `111990/143771MB`, GPU3 `125930/143771MB`, GPU4 `98098/143771MB`, GPU5 `143140/143771MB`, GPU6 `129410/143771MB`, GPU7 `121112/143771MB`. 8장 모두 학습 프로세스가 점유 중이며, FSDP 특성상 순간 util은 rank별로 출렁인다.
- HF upload 상태: `.env`의 `export HF_TOKEN=...` 형식을 upload helper가 못 읽어서 `E2B-it` 1epoch 업로드가 반복 실패했으나, 토큰 값은 출력되지 않았다. `upload_model_repo.py`를 수정해 `export` prefix를 처리하게 했고, `E2B-it` 1epoch와 `E2B` 1epoch는 HF 업로드 완료됐다.
- 26B/31B용 native 전처리 완료: `26B-A4B-it`, `26B-A4B`, `31B-it`, `31B` 모두 `processed_rows=16311`, `skipped_too_long=5999`, prompt token `p50/p95=3760/7156`, assistant token `p50/p95=390/925`.

| 모델 | GPU | Dataset | Effective batch | Output |
| --- | --- | --- | ---: | --- |
| `google/gemma-4-E2B-it` | 0,1 | `google__gemma-4-E2B-it__liquid_raw_json_masked_8192` | 16 | `/home/work/.data/gemma4_native_sft/models/google__gemma-4-E2B-it__terminal_sft_native_liquid_2epoch` |
| `google/gemma-4-E4B-it` | 2,3 | `google__gemma-4-E4B-it__liquid_raw_json_masked_8192` | 16 | `/home/work/.data/gemma4_native_sft/models/google__gemma-4-E4B-it__terminal_sft_native_liquid_2epoch` |
| `google/gemma-4-E2B` | 4,5 | `google__gemma-4-E2B__liquid_raw_json_masked_8192` | 16 | `/home/work/.data/gemma4_native_sft/models/google__gemma-4-E2B__terminal_sft_native_liquid_2epoch` |
| `google/gemma-4-E4B` | 6,7 | `google__gemma-4-E4B__liquid_raw_json_masked_8192` | 16 | `/home/work/.data/gemma4_native_sft/models/google__gemma-4-E4B__terminal_sft_native_liquid_2epoch` |

배치 기준:

- 현재 설정은 `per_device_train_batch_size=1`, `nproc_per_node=2`, `gradient_accumulation_steps=8`이므로 effective batch는 `16`이다.
- 8K sequence Gemma 4는 activation memory가 커서 micro batch를 크게 잡는 것이 항상 유리하지 않다. 우선 OOM 없이 안정적으로 2epoch를 완료시키고, E2B에서만 `per_device=2` ablation을 추가해 throughput/score를 비교한다.
- HF 업로드 스크립트는 `.env`를 조용히 로드하되 토큰 값은 출력하지 않는다. 공용 컴퓨터 기준으로 토큰/대화 히스토리는 로그와 리포트에 남기지 않는다.
- checkpoint publish monitor 실행 중: PID `3479746`, log `/home/work/.data/gemma4_native_sft/logs/native_checkpoint_monitor_20260508.log`, state `/home/work/.data/gemma4_native_sft/monitor_state.json`. 180초마다 본학습 output을 스캔하고, checkpoint 파일이 120초 이상 안정되면 HF에 즉시 올린다.
- native checkpoint 평가는 `gemma4_native_sft/scripts/eval_native_checkpoint.sh`로 고정한다. 옵션은 `thinking-mode off`, `strip-thinking-history on`, `gemma4-empty-thought-channel auto`, `language-model-only`이며, corrected 303-step TB2-lite를 사용한다.
- 큰 모델 config 준비 완료: `sft_gemma4_26b_a4b_it_native_8gpu.env`, `sft_gemma4_26b_a4b_base_native_8gpu.env`, `smoke_gemma4_31b_it_native_8gpu.env`, `sft_gemma4_31b_it_native_8gpu.env`, `sft_gemma4_31b_base_native_8gpu.env`. 31B는 full 학습 전에 `MAX_STEPS=3` smoke checkpoint를 vLLM로 먼저 검증한다.
- 작은 모델 checkpoint가 저장될 때마다 즉시 `staging -> HF upload`를 먼저 처리한다. TB2-lite full 303 평가는 GPU가 비는 즉시 붙이고, 평가가 끝나면 `MD 기록 -> HF README 점수 갱신 -> best 후보 선정` 순서로 처리한다. 2epoch 완료 후에는 best checkpoint와 final symlink 기준을 비교한다.
- 작은 4개 다음 학습 순서: `26B-A4B-it` full 8GPU, `26B-A4B` base full 8GPU, `31B-it` smoke 8GPU, smoke 통과 시 `31B-it` full 8GPU, 마지막 `31B` base full 8GPU. 31B smoke가 JSON sanity에 실패하면 31B full은 보류하고 저장/로드 경로를 먼저 고친다.

HF 업로드 이름:

| 모델 | 1epoch repo | 2epoch/final repo |
| --- | --- | --- |
| `google/gemma-4-E2B-it` | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-1Epoch` | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-2Epoch` |
| `google/gemma-4-E4B-it` | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-1Epoch` | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-2Epoch` |
| `google/gemma-4-E2B` | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-1Epoch` | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-2Epoch` |
| `google/gemma-4-E4B` | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-1Epoch` | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-2Epoch` |
| `google/gemma-4-26B-A4B-it` | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-1Epoch` | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch` |
| `google/gemma-4-26B-A4B` | `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-1Epoch` | `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch` |
| `google/gemma-4-31B-it` | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-1Epoch` | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch` |
| `google/gemma-4-31B` | `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-1Epoch` | `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch` |

예상 완료 시간표:

| 단계 | GPU 계획 | 예상 시점(KST) | 처리 |
| --- | --- | --- | --- |
| `E2B-it` epoch1/2epoch 평가 | 0,1 | 완료: `2026-05-09 09:20` 전후 | 1epoch `25.70`, 2epoch `24.92`. best는 1epoch. |
| `E4B-it` epoch1 평가 | 0 | 완료: `2026-05-09 09:29` | 1epoch `34.42`. 현재 Gemma small best. |
| `E4B` base epoch1 평가 | 1 | 완료: `2026-05-09 09:29` | 1epoch `12.80`. base는 낮음. |
| `E2B` base 2epoch 완료 | 4,5 해제 | 완료: `2026-05-09 09:49` | checkpoint-2042 저장. |
| `E2B` base 2epoch 평가/카드 갱신 | 4 | 완료: `2026-05-09 09:56` | 2epoch `16.22`. 1epoch `11.35`보다 올랐지만 small base는 제외. |
| `E4B-it` 2epoch 완료 | 2,3 해제 | `2026-05-10 01:56` 전후 | E4B-it 2epoch 평가, best 선정, HF README 점수 갱신. |
| `E4B` 2epoch 완료 | 6,7 해제 | `2026-05-10 03:39` 전후 | E4B base 2epoch 평가, best 선정, HF README 점수 갱신. |
| 작은 4개 평가/업로드 정리 | 8GPU 병렬 평가 가능 | `2026-05-10 05:00~06:30` | 남은 checkpoint 평가와 HF card 점수 갱신을 끝낸다. |
| `26B-A4B-it` native full | 8GPU | 작은 모델 정리 직후 | 첫 30 step으로 ETA 재산정. 8K native라 현재 추정 `24~48h`. |
| `26B-A4B` native full | 8GPU | `26B-A4B-it` 직후 | 첫 30 step으로 ETA 재산정. 현재 추정 `24~48h`. |
| `31B-it` smoke | 8GPU | 26B 두 개 후 | `MAX_STEPS=3`, checkpoint 저장/로드/vLLM JSON sanity. 예상 `0.5~2h`. |
| `31B-it` native full | 8GPU | smoke 통과 직후 | 첫 30 step으로 ETA 재산정. dense 31B 8K라 현재 추정 `36~72h`. |
| `31B` native full | 8GPU | `31B-it` 성공 후 | 첫 30 step으로 ETA 재산정. 현재 추정 `36~72h`. |

전체 기준:

- 작은 모델 4개 학습/평가/업로드 완료 목표는 `2026-05-10 05:00~06:30 KST` 전후다.
- 큰 모델까지 모두 끝나는 전체 일정은 26B/31B의 실제 8K step time에 따라 크게 달라진다. 현재 보수 범위는 `2026-05-13~2026-05-17` 사이이며, 각 큰 모델 시작 후 30 step 시점에 정확도를 높여 다시 기록한다.
- GPU 정책은 `8GPU training 우선`, 8GPU가 필요 없는 중간 구간은 `평가/업로드/재평가/ablation`으로 빈 GPU를 채우는 방식이다.

한국 시간(KST, UTC+9) 기준 운영 메모:

- `2026-05-09 09:30 KST` 기준, GPU 0/1은 E4B epoch1 평가 직후 비었고, GPU 2/3은 E4B-it 학습, GPU 4/5는 E2B base 학습, GPU 6/7은 E4B base 학습에 점유되어 있다.
- GPU 4/5는 `2026-05-09 09:52 KST` 전후 E2B base 완료 후 평가/업로드 작업에 돌린다.
- 26B-A4B-it는 8GPU가 필요하므로, E4B-it/E4B base가 끝나고 작은 모델 평가가 정리되는 `2026-05-10 05:00~06:30 KST` 전후가 가장 이른 시작 시점이다.
- 따라서 8GPU 전체를 큰 학습에 다시 묶는 시점은 `2026-05-10 06:00 KST` 안팎으로 본다. 그 전까지는 남는 GPU를 checkpoint 평가, HF 업로드 검증, 필요 시 baseline 재평가로 채운다.

### 26B-A4B-it 4GPU batch tuning update

`2026-05-09 16:16 KST` 기준 실제 운영 결과다. E4B-it/E4B base 2epoch가 아직 GPU 2/3, 6/7에서 진행 중이므로, 비어 있던 GPU 0/1/4/5를 사용해 26B-A4B-it native full을 먼저 시작했다.

데이터/길이 조건:

- dataset: `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-26B-A4B-it__liquid_raw_json_masked_8192`
- rows: `16,311`
- input length: mean `4,389`, p50 `4,234`, p75 `5,915`, p90 `7,162`, p95 `7,651`, p99 `8,075`
- `4096` 초과 샘플: `8,505/16,311 = 52.1%`
- `6144` 초과 샘플: `3,557/16,311 = 21.8%`
- 전처리는 `too_long` 샘플을 truncate하지 않고 drop한다. 따라서 4096 재전처리는 속도는 빨라지지만 학습 샘플 절반 이상을 버리므로, 현재 본학습은 8192를 유지한다.

OOM 원인:

- Gemma 4 26B-A4B-it는 MoE이고, config 기준 `num_experts=128`, `top_k_experts=8`, `moe_intermediate_size=704`, `hidden_size=2816`, `num_hidden_layers=30`이다.
- 공식 vLLM recipe도 26B-A4B-it를 `26B total / 4B active`, `128` experts, top-8 routing MoE로 설명한다: https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it
- vLLM Gemma 4 guide 기준 26B-A4B-it BF16 추론은 `1x 80GB`로 가능하지만, 이는 optimizer state/gradient/activation까지 드는 SFT memory와 다르다: https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- Hugging Face Transformers Gemma4 문서는 `logits_to_keep` forward argument를 제공한다. native trainer는 full vocab logits 전체를 남기지 않도록 active label position만 골라 loss를 계산하는 custom loss를 사용한다: https://huggingface.co/docs/transformers/model_doc/gemma4
- `Gemma4TextExperts`는 expert weights를 `gate_up_proj`, `down_proj` 두 큰 3D Parameter로 들고 있어 expert block 하나의 FSDP unshard가 bf16 기준 약 `1.42GiB`를 요구한다.
- 실제 OOM 로그도 반복적으로 `Tried to allocate 1.42 GiB` 또는 grouped-mm expert/SDPA activation 할당에서 실패했다. 따라서 active parameter가 4B라도, training memory는 inference active size처럼 작게 보지 않는다.
- `flash-attn`/`flash_attn_2_cuda`는 현재 env에 설치되어 있지 않다. Gemma4 code path는 `flash_attention_2`를 지원하지만 바로 켤 수 없다.

Tuning 결과:

| 설정 | effective batch | 결과 | 메모 |
| --- | ---: | --- | --- |
| `bsz16/14/12/10/8/6 acc1`, activation FSDP | 64~24 | OOM | 4GPU 8192에서 너무 큼 |
| `bsz8/7/6/5/4/3 acc1`, no-prefetch activation FSDP | 32~12 | OOM | expert unshard 또는 grouped-mm/attention 피크 |
| `bsz2/acc2`, no-prefetch activation FSDP | 16 | OOM | `158MiB` 부족, 거의 통과했지만 full에는 불안정 |
| `bsz2/acc1`, no-prefetch activation FSDP | 8 | OOM | expert unshard `1.42GiB` 부족 |
| `bsz1/acc4`, no-prefetch activation FSDP | 16 | OOM | SDPA attention `3.99GiB` 부족 |
| `bsz1/acc4`, qwen sizewrap FSDP + `gradient_checkpointing=1` | 16 | PASS | 4-step smoke 평균 `80.24 sec/step` |

채택 설정:

- config: `gemma4_native_sft/configs/sft_gemma4_26b_a4b_it_native_4gpu.env`
- run name: `gemma4_26b_a4b_it_native_4gpu_bsz1_acc4_gc_qwenfsdp`
- GPUs: `0,1,4,5`
- per-device batch: `1`
- gradient accumulation: `4`
- effective batch: `16`
- FSDP: `full_shard auto_wrap`
- FSDP config: `qwen_sft/configs/fsdp_qwen_full_shard_sizewrap.json`
- gradient checkpointing: `1`
- BF16 flag: `0`, but the native loader casts the model weights to `torch.bfloat16`
- full training start: `2026-05-09 16:16:37 KST`
- launcher log: `/home/work/.data/gemma4_native_sft/logs/launcher_gemma4_26b_a4b_it_full_bsz1_acc4_gc_qwenfsdp_20260509T071637Z.log`
- train log: `/home/work/.data/gemma4_native_sft/logs/gemma4_26b_a4b_it_native_4gpu_bsz1_acc4_gc_qwenfsdp_20260509T071638Z.log`

ETA:

- total optimizer steps: `ceil(16311 / 16) * 2 = 2040`
- smoke step time: `80.24 sec/step`
- first full-run step: `1/2040`, `85.89 sec/step`, loss `6.388` at `2026-05-09 16:24 KST`
- estimated train time: `45.5~48.7h`
- 1epoch checkpoint 예상: `2026-05-10 15:00~17:00 KST`
- 2epoch 완료 예상: `2026-05-11 13:40~17:00 KST`

운영 결론:

- 4GPU에서 가장 빠른 실제 통과 설정은 `bsz1/acc4`다. 큰 batch를 억지로 올리면 expert unshard/attention 피크 때문에 OOM이 난다.
- 8GPU가 완전히 비면 `bsz1/acc2` 또는 `bsz2/acc1`을 다시 테스트할 가치가 있다. 8GPU는 expert unshard 크기 자체를 줄이지는 않지만 shard/gradient/optimizer 여유를 늘려 activation 피크를 낮출 가능성이 있다.
- 현재 full run은 4GPU에서 계속 진행하고, checkpoint가 저장되면 HF 업로드 -> vLLM 평가 -> 이 보고서와 model card 갱신 순서로 처리한다.

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

## 평가 코드 공정성 점검

Qwen 3.5 점수가 매우 높아서 평가 코드가 Qwen에 특화됐는지 확인했다.

확인 범위:

- `tb2_lite/scripts/replay_eval.py`
- `tb2_lite/scripts/prompt_builder.py`
- `tb2_lite/scripts/replay_metrics.py`
- `tb2_lite/scripts/summarize_corrected_results.py`
- 실제 결과 JSON: `/home/work/.data/tb2_lite_eval/corrected_readme_models_vllm/*.json`

결론:

- 점수 계산 자체는 모델명을 보지 않는다. `score_commands()`와 `aggregate_scores()`는 `pred_command_units`와 `ref_command_units`만 비교한다.
- 순위 기준은 `Score = 100 * avg_command_f1`이고, 이 계산에도 Qwen/LFM/Gemma/Ouro별 가중치는 없다.
- Qwen 결과는 `template_status_counts={'chat_template': 303}`으로 기록되어 있다. 즉 Qwen 전용 fallback이 아니라 tokenizer의 `apply_chat_template`가 303개 전부 정상 적용됐다.
- `prompt_builder.py`에는 Qwen 문자열 분기가 있지만, 이는 `chat_template` 실패 시 ChatML fallback을 쓰기 위한 경로다. 현재 Qwen 성공 결과에는 이 fallback이 걸리지 않았다.
- 따라서 Qwen 상위권은 평가 코드 특혜보다는 `터미널 command prior`, `JSON 포맷 안정성`, `ChatML/assistant 응답 포맷 적합성`이 TB2-lite 태스크와 잘 맞아서 나온 결과로 본다.
- 다만 TB2-lite 자체가 일반 지능 벤치가 아니라 `터미널 next-action JSON command 재현` 벤치이므로, Qwen에 유리한 domain bias는 존재한다고 보는 것이 맞다.

## 모델군별 해석

현재 모델군별 성격:

| 모델군 | Base prior | SFT 반응 | 속도/효율 | 현재 해석 |
| --- | --- | --- | --- | --- |
| Qwen 3.5 | 매우 강함 | 좋지만 상승 여지는 상대적으로 작음 | 매우 빠름 | 최고점 후보. Base부터 TB2-lite 포맷과 잘 맞는다. |
| LFM | 낮거나 중간 | 매우 좋음 | 매우 빠름 | 학습 효율과 운영 효율이 가장 좋다. RL 반복 실험에 유리하다. |
| Ouro | 낮은 base도 있으나 SFT 상승폭 큼 | 좋음 | 매우 느림 | 학습은 잘 먹지만 RL/serving 효율은 불리하다. 고점 확인용 후보. |
| Gemma 4 | 작은 모델 base는 낮음, 26B/31B base는 중간 | 기존 SFT는 실패 사례, native 결과 대기 | 모델별 편차 큼 | native Gemma template으로 다시 확인해야 한다. LFM보다 못 나오면 RL 제외. |

Qwen:

- `Qwen3.5-2B base`가 이미 Score `35.10`이고, `Qwen3.5-9B base`는 `38.10`이다.
- `Qwen3.5-2B SFT 2Epoch`는 Score `39.52`로 현재 전체 1위다.
- SFT가 평가를 속였다기보다 base 모델 자체가 터미널 명령, JSON, ChatML 응답 형식에 강하다.
- 이 벤치에서는 큰 모델이라고 자동으로 이기지 않는다. `gemma-4-E4B-it` native SFT가 `34.42`까지 올랐는데도 `Qwen3.5-2B base` `35.10`보다 `0.68` 낮고, `Qwen3.5-2B SFT 2Epoch` `39.52`보다는 `5.10` 낮다.
- 학습하지 않은 `Qwen3.5-9B base`가 `38.10`인 점까지 보면, Qwen 3.5 계열은 TB2-lite의 `terminal next-action JSON command` 분포와 base prior가 유난히 잘 맞는 모델군이다.
- RL을 한다면 최고점 갱신 후보는 Qwen이 가장 안정적이다. 단, 이미 상위권이라 추가 상승 여지는 LFM보다 작을 수 있다.

LFM:

- `LFM2-2.6B base` Score `17.06`에서 `LFM2-2.6B SFT 2Epoch` Score `32.85`까지 올라 상승폭이 약 `+15.79`다.
- `LFM2-24B-A2B base` Score `10.87`에서 TemplateMasked SFT Score `33.46`까지 올라 상승폭이 약 `+22.59`다.
- 속도도 좋다. `LFM2-2.6B SFT`는 `0.150 sec/step`, `LFM2-24B-A2B TemplateMasked`는 `0.177 sec/step`, `LFM2-8B-A1B SFT`는 `0.126 sec/step` 수준이다.
- 따라서 LFM은 `학습이 잘 먹고`, `빠르고`, `비용 대비 점수가 잘 오르는` 모델군으로 본다.
- 약점은 Qwen 대비 Valid JSON과 command precision이다. RL에서는 이 두 항목을 reward로 직접 밀어야 한다.

Ouro:

| 모델 | Base Score | SFT Score | 상승폭 | Sec/Step | 해석 |
| --- | ---: | ---: | ---: | ---: | --- |
| `Ouro-1.4B` | 15.06 | 28.30 | +13.24 | 2.344 | 학습 반응은 좋지만 속도 불리. |
| `Ouro-1.4B-Thinking` | 12.69 | 31.74 | +19.05 | 1.698 | Thinking SFT가 더 잘 먹는다. |
| `Ouro-2.6B` | 6.46 | 29.58 | +23.12 | 5.154 | base는 낮지만 SFT 상승폭은 큼. |
| `Ouro-2.6B-Thinking` | 미측정/별도 | 35.61 | 큼 | 3.358 | 점수는 좋지만 RL loop에는 느리다. |

- Ouro는 학습이 안 먹는 모델이 아니다. 오히려 base 대비 SFT 상승폭만 보면 꽤 잘 먹는다.
- 다만 LFM 대비 속도가 너무 느리다. `LFM2-2.6B SFT`가 `0.150 sec/step`인데 `Ouro-2.6B SFT`는 `5.154 sec/step`이라 약 34배 느리다.
- Ouro config 기준 `model_type=ouro`, `Ouro-2.6B`는 48 layers, hidden size `2048`, attention heads `16`, kv heads `16`, max position `65536`, sliding window 없음이다. 긴 prompt에서 attention/KV 비용이 크고, Qwen만큼 vLLM fast path 최적화를 타지 못하는 것으로 본다.
- 따라서 Ouro는 `점수는 좋고 학습도 먹지만`, RL처럼 rollout을 많이 돌리는 작업에서는 LFM/Qwen보다 후순위다.
- Ouro를 넣는다면 `Ouro-2.6B-Thinking-Terminal-SFT` 하나를 고점 확인용으로 소량 RL 테스트하는 정도가 맞다.

Gemma 4:

- 기존 Gemma 작은 모델 SFT는 base보다 낮아서 실패 사례로 본다. Native 재학습 후에는 E4B-it가 `34.42`까지 올라 실패 원인은 일부 제거됐지만, Qwen 3.5 계열의 base prior를 넘지는 못했다.
- 현재 native Gemma 4 rerun은 `Gemma 전용 chat template`, `thinking history 제거`, `assistant JSON target only`, `base 모델 template 주입`을 적용한 새 경로다.
- `E4B-it native 1epoch`는 최고 LFM `33.46`보다 높아 RL 후보 1개로 볼 수 있다. 반면 E2B-it `25.70/24.92`, E2B/E4B base `11.35/12.80`은 제외한다.
- Gemma가 `32점 이상`이면 1개를 RL 후보에 넣고, `33~35점` 근처면 2개까지 후보에 넣는다. 현재 small 기준으로는 `E4B-it native 1epoch`만 조건을 만족한다.
- 26B-A4B native SFT가 기존 SFT e2 `32.77`을 넘기면 Gemma도 본격 후보군으로 다시 본다.
- 큰 모델 기대치는 남아 있다. baseline에서 `26B-A4B-it`는 `28.51`, `31B-it`는 `26.33`이지만, E4B-it처럼 native SFT가 `+10` 이상 먹히면 35점대 이상 가능성이 있다. 다만 Qwen 2B SFT `39.52`를 넘길지는 별도 검증이 필요하다.

## 무중단 운영 계획

우선순위는 `학습`, `HF 업로드`, `vLLM 평가`, `문서/모델카드 갱신`, `다음 학습 투입` 순서다. GPU가 비면 평가나 smoke를 먼저 넣고, 8GPU가 모두 비는 순간에는 큰 모델 본학습을 시작한다.

현재 자동화:

- Checkpoint publish monitor: 새 `checkpoint-*`가 안정화되면 HF staging 후 업로드한다. state는 `/home/work/.data/gemma4_native_sft/monitor_state.json`에 저장한다.
- HF card supervisor: 1시간마다 LLM-OS-Models model card를 갱신한다. Native-Liquid repo는 pending 카드로 덮지 않도록 막았다.
- 3시간 status supervisor: `KST` 기준 진행률, GPU 사용량, ETA, 업로드/평가 상태를 `/home/work/.data/gemma4_native_sft/logs/native_pipeline_status_latest.md`에 쓴다.
- native 평가 고정 명령: `gemma4_native_sft/scripts/eval_native_checkpoint.sh`, vLLM `0.19.1`, `thinking-mode off`, `strip-thinking-history on`, `gemma4-empty-thought-channel auto`.
- 자동 평가 watcher: `gemma4_native_sft/scripts/auto_eval_native_checkpoints.py --watch`. `tb2_eval=pending` checkpoint를 찾아 빈 GPU에서 vLLM 평가를 실행하고, 결과 JSON/monitor state/HF card를 갱신한다.

진행 중/즉시 작업:

1. `E2B base` 2epoch 완료/평가 완료. `2026-05-09 09:56 KST` 결과는 Score `16.22`, 전체 `50위`다.
2. `E2B base` 2epoch 결과는 전체 순위와 native 결과 표에 반영했다. HF repo/model card는 native 결과 기준으로 갱신 대상이다.
3. `26B-A4B-it` 1GPU smoke는 OOM으로 실패했다. 원인: `world_size=1`에서 FSDP가 `NO_SHARD`로 바뀌어 한 GPU에 `137.88GB`까지 올라가고 추가 `2.84GB`에서 실패.
4. `31B-it` 1GPU smoke도 OOM으로 실패했다. 원인: 같은 `NO_SHARD` 경로, 한 GPU에 `139.58GB`까지 올라가고 추가 `442MB`에서 실패.
5. `26B-A4B-it` 2GPU smoke도 optimizer step에서 OOM으로 실패했다. 각 GPU가 약 `139GB`까지 차서 추가 `1GB` 안팎 allocation에 실패했다.
6. `31B-it` 2GPU smoke도 optimizer step에서 OOM으로 실패했다. 각 GPU가 약 `139.8GB`까지 차고 추가 `222MB` allocation에서 실패했다.
7. `26B-A4B` base 2GPU smoke도 optimizer step에서 OOM으로 실패했다. rank0은 추가 `1.42GiB`, rank1은 추가 `968MiB` allocation에서 실패했다.
8. `26B-A4B-it` 4GPU batch tuning 완료. 4GPU/8192에서 큰 micro batch는 모두 OOM이고, 통과한 최속 설정은 `per_device=1`, `grad_acc=4`, effective batch `16`, qwen sizewrap FSDP + gradient checkpointing이다.
9. `26B-A4B-it` full native SFT 진행 중. GPUs `0,1,4,5`, run `gemma4_26b_a4b_it_native_4gpu_bsz1_acc4_gc_qwenfsdp`, 시작 `2026-05-09 16:16:37 KST`.
10. `E4B-it`와 `E4B base` 2epoch는 GPUs `2,3`, `6,7`에서 계속 진행 중이다. 저장되면 HF 업로드 -> vLLM 평가 -> 보고서/model card 갱신 순서로 바로 처리한다.

작은 모델 남은 일정:

| 단계 | 예상 시점(KST) | 처리 |
| --- | --- | --- |
| `E2B base` 2epoch 저장/평가 | 완료: `2026-05-09 09:56` | Score `16.22`, 전체 `50위`, report 반영 |
| `E4B-it` 2epoch 저장 | `2026-05-10 02:00~03:00` 현재 추정 | HF 업로드, `gemma4_e4b_it_native_e2` 평가, 1epoch `34.42`와 비교 |
| `E4B base` 2epoch 저장 | `2026-05-10 02:00~03:00` 현재 추정 | HF 업로드, `gemma4_e4b_base_native_e2` 평가 |
| 작은 모델 정리 | `2026-05-10 03:00~05:00` | best checkpoint 선정, 전체 순위/모델카드 갱신 |
| `26B-A4B-it` 1epoch 저장 | `2026-05-10 15:00~15:20` 현재 추정 | HF 업로드, vLLM 평가, E4B-it `34.42`와 비교 |
| `26B-A4B-it` 2epoch 완료 | `2026-05-11 13:40~14:20` 현재 추정 | HF 업로드, vLLM 평가, report/model card 갱신 |

큰 모델 학습 순서:

1. `26B-A4B-it` 4GPU full native SFT: 현재 진행 중. 1epoch checkpoint가 저장되면 바로 업로드/평가해서 계속 돌릴지, 8GPU 재시도 가치가 있는지 판단한다.
2. `26B-A4B` base: small base가 낮으므로 `it` 결과를 먼저 본 뒤 4GPU/8GPU smoke부터 다시 판단한다.
3. `31B-it`: 기존 31B SFT `0.00` 실패와 1/2GPU smoke OOM 이력이 있으므로, 26B 결과와 GPU 해제 상황을 보고 4GPU 또는 8GPU smoke부터 시작한다.
4. `31B` base: 마지막. base template 주입 경로 검증용이며, small base 결과상 기대치는 낮다.

결과 처리 규칙:

- checkpoint 저장 즉시 HF 업로드를 먼저 한다.
- vLLM 평가가 끝나면 `MODEL_EVALUATION_REPORT.md` 전체 순위 표, native 결과 섹션, 모델군 해석을 즉시 갱신한다.
- HF model card는 native 결과 디렉터리 기준으로 해당 repo만 force update한다.
- 점수는 항상 `Score = 100 * avg_command_f1`로만 말한다. 로그의 legacy `next_action_score`와 섞지 않는다.
- GPU가 2장만 비면 smoke/eval을 넣고, 8장이 비면 큰 모델 본학습을 넣는다.

## RL 후보 계획

RL은 한 번에 너무 많은 모델을 벌리지 않고, Qwen 2개와 LFM 2개를 먼저 잡고 Gemma는 결과 조건부로 추가한다.

| 그룹 | 후보 | 목적 |
| --- | --- | --- |
| Qwen | `Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 현재 최고점 `39.52`. 최고점 갱신 가능성 확인. |
| Qwen | `Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData` | 더 큰 Qwen 계열 상한 확인. |
| LFM | `LFM2-2.6B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout` | 빠른 RL 파이프라인 검증. 비용 대비 상승폭 확인. |
| LFM | `LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-TemplateMasked` | 효율 좋은 MoE 본선 후보. |
| Gemma | native 결과 best 0~2개 | LFM보다 잘 나오거나 최소 비슷할 때만 추가. |
| Ouro | `Ouro-2.6B-Thinking-Terminal-SFT` 선택적 소량 | 점수 고점 후보지만 속도가 느려 후순위. |

실행 순서:

1. `LFM2-2.6B`로 reward/RL 파이프라인을 먼저 검증한다.
2. `Qwen3.5-2B`로 최고점 갱신 가능성을 확인한다.
3. `LFM2-24B-A2B`로 효율 좋은 MoE 본선 실험을 한다.
4. `Qwen3.5-9B`로 Qwen 큰 모델 상한을 확인한다.
5. Gemma native 결과가 괜찮으면 `Gemma best 1~2개`를 추가한다.
6. 시간이 남고 속도 비용을 감수할 만하면 Ouro thinking SFT를 소량 RL ablation으로만 넣는다.

Reward 설계 방향:

- `valid_json` 강제: JSON 파싱 실패는 큰 penalty.
- command F1 직접 최적화: `commands[].keystrokes` token F1을 reward 핵심으로 둔다.
- first command exact 보조: 실제 터미널 에이전트에서는 첫 명령이 중요하므로 보조 reward로 둔다.
- precision penalty: 불필요하거나 위험한 명령을 많이 내는 경우 감점한다.
- premature complete penalty: reference가 완료가 아닌데 `task_complete=true`를 내면 감점한다.

## 해석 기준

- `rank_eligible=false` 또는 `Template=raw_fallback`인 결과는 정상 chat template 평가가 아니므로 순위에서 제외한다.
- 같은 모델의 epoch/checkpoint 비교는 `Score`를 우선 보고, 거의 같으면 `Recall`, `Precision`, `Valid JSON`을 함께 본다.
- README legacy 386-step 표와 직접 비교하지 않는다.

<!-- GEMMA4_NATIVE_AUTO_RESULTS_START -->
## Gemma 4 Native 자동 평가 현황

업데이트: `2026-05-14 14:21:18 KST`

점수 기준: `Score = 100 * avg_command_f1`. 이 섹션은 `monitor_state.json`과 TB2-lite 결과 JSON에서 자동 생성된다.

| Native 순위 | HF repo | Epoch | Checkpoint | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Sec/Step |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 34.98 | 0.3498 | 0.4737 | 0.3576 | 30.4% | 35.0% | 0.277 |
| 2 | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 34.42 | 0.3442 | 0.4823 | 0.3397 | 27.1% | 45.5% | 0.360 |
| 3 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 25.70 | 0.2570 | 0.3615 | 0.2717 | 15.2% | 34.3% | 0.325 |
| 4 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 24.92 | 0.2492 | 0.3667 | 0.2447 | 11.6% | 34.0% | 0.317 |
| 5 | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 18.47 | 0.1847 | 0.2514 | 0.1980 | 16.8% | 17.2% | 0.302 |
| 6 | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 16.22 | 0.1622 | 0.2747 | 0.1678 | 15.2% | 16.2% | 0.289 |
| 7 | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 12.80 | 0.1280 | 0.1792 | 0.1364 | 10.2% | 14.2% | 0.383 |
| 8 | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 11.35 | 0.1135 | 0.1767 | 0.1191 | 6.6% | 7.3% | 0.219 |

평가 대기:

- `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-1Epoch`
- `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch`
- `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-1Epoch`
- `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch`
- `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-1Epoch`
- `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch`
- `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-1Epoch`
- `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch`
<!-- GEMMA4_NATIVE_AUTO_RESULTS_END -->
