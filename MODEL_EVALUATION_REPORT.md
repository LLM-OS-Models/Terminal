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
| 44 | `google/gemma-4-E4B-it` | 19.36 | 0.1936 | 0.3184 | 0.1822 | 11.6% | 54.8% | chat_template | 0.205 | 175.8 |
| 45 | `google/gemma-4-E2B-it` | 17.40 | 0.1740 | 0.2918 | 0.1613 | 7.3% | 57.1% | chat_template | 0.148 | 139.6 |
| 46 | `LiquidAI/LFM2-2.6B` | 17.06 | 0.1706 | 0.2229 | 0.2160 | 12.9% | 29.4% | chat_template | 0.152 | 55.0 |
| 47 | `ByteDance/Ouro-1.4B` | 15.06 | 0.1506 | 0.1988 | 0.1625 | 8.9% | 37.3% | chat_template | 1.946 | 74.8 |
| 48 | `LiquidAI/LFM2.5-1.2B-Instruct` | 14.46 | 0.1446 | 0.2374 | 0.1526 | 10.6% | 60.1% | chat_template | 0.056 | 39.8 |
| 49 | `ByteDance/Ouro-1.4B-Thinking` | 12.69 | 0.1269 | 0.2026 | 0.1299 | 9.2% | 26.7% | chat_template | 2.115 | 65.9 |
| 50 | `LiquidAI/LFM2-24B-A2B` | 10.87 | 0.1087 | 0.1466 | 0.1163 | 5.3% | 54.5% | chat_template | 0.165 | 236.2 |
| 51 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 10.67 | 0.1067 | 0.1507 | 0.1067 | 4.0% | 5.9% | chat_template | 0.185 | 65.7 |
| 52 | `LiquidAI/LFM2-8B-A1B` | 10.04 | 0.1004 | 0.1405 | 0.1223 | 5.9% | 27.4% | chat_template | 0.124 | 61.9 |
| 53 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 9.22 | 0.0922 | 0.1249 | 0.1023 | 3.3% | 5.9% | chat_template | 0.184 | 67.5 |
| 54 | `ByteDance/Ouro-2.6B` | 6.46 | 0.0646 | 0.0976 | 0.0692 | 5.0% | 16.5% | chat_template | 4.607 | 99.6 |
| 55 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | chat_template | 1.774 | 300.1 |
| 56 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | chat_template | 1.770 | 300.1 |

## Gemma 4 재평가 및 재학습 메모

진행 시각: `2026-05-08`

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
- 현재 페이스 기준 완료 예상: `E2B-it 2026-05-08 ~22:35Z`, `E2B 2026-05-08 ~23:55Z`, `E4B-it 2026-05-09 ~19:05Z`, `E4B 2026-05-09 ~19:25Z`. E4B 계열 epoch1 checkpoint는 `2026-05-08 ~23:45~24:00Z` 전후로 본다.
- VRAM snapshot: GPU0 `92902/143771MB`, GPU1 `91736/143771MB`, GPU2 `111990/143771MB`, GPU3 `125930/143771MB`, GPU4 `98098/143771MB`, GPU5 `143140/143771MB`, GPU6 `129410/143771MB`, GPU7 `121112/143771MB`. 8장 모두 학습 프로세스가 점유 중이며, FSDP 특성상 순간 util은 rank별로 출렁인다.
- HF upload 상태: `.env`의 `export HF_TOKEN=...` 형식을 upload helper가 못 읽어서 `E2B-it` 1epoch 업로드가 반복 실패했으나, 토큰 값은 출력되지 않았다. `upload_model_repo.py`를 수정해 `export` prefix를 처리하게 했고, 현재 `E2B-it` 1epoch upload가 재시도 중이다.
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
- checkpoint publish monitor 실행 중: PID `3103778`, log `/home/work/.data/gemma4_native_sft/logs/native_checkpoint_monitor_20260508.log`, state `/home/work/.data/gemma4_native_sft/monitor_state.json`. 180초마다 본학습 output을 스캔하고, checkpoint 파일이 120초 이상 안정되면 HF에 즉시 올린다.
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

| 단계 | GPU 계획 | 예상 시점(UTC) | 처리 |
| --- | --- | --- | --- |
| `E2B-it` epoch1 checkpoint | 0,1 계속 학습 중 | 완료: `2026-05-08`, `checkpoint-1021` | 저장 완료. HF 1epoch 업로드 재시도 중. GPU 여유 없어서 평가는 대기. |
| `E2B` epoch1 checkpoint | 4,5 계속 학습 중 | 완료: `2026-05-08`, `checkpoint-1021` | 저장 완료. `E2B-it` 업로드 후 HF 1epoch 업로드 큐. GPU 여유 없어서 평가는 대기. |
| `E4B-it` epoch1 checkpoint | 2,3 계속 학습 중 | `2026-05-08 23:45~24:00` | 저장 즉시 보존/HF 업로드 큐 등록. |
| `E4B` epoch1 checkpoint | 6,7 계속 학습 중 | `2026-05-08 23:45~24:05` | 저장 즉시 보존/HF 업로드 큐 등록. |
| `E2B-it` 2epoch 완료 | 0,1 해제 | `2026-05-08 22:30~22:50` | GPU 0/1로 `E2B-it` checkpoint 평가, best 선정, HF README 점수 갱신. |
| `E2B` 2epoch 완료 | 4,5 해제 | `2026-05-08 23:45~2026-05-09 00:10` | GPU 4/5로 `E2B` checkpoint 평가, best 선정, HF README 점수 갱신. |
| E2B 후 E4B 대기 구간 | 0,1,4,5 활용 | `2026-05-08 21:00` 이후 | E2B 평가가 끝나면 renderer correction 후 26B/31B baseline 재평가 또는 E2B microbatch ablation으로 빈 GPU를 채운다. |
| `E4B-it` 2epoch 완료 | 2,3 해제 | `2026-05-09 18:50~19:20` | E4B-it checkpoint 평가, best 선정, HF README 점수 갱신. |
| `E4B` 2epoch 완료 | 6,7 해제 | `2026-05-09 19:10~19:40` | E4B checkpoint 평가, best 선정, HF README 점수 갱신. |
| 작은 4개 평가/업로드 정리 | 8GPU 병렬 평가 가능 | `2026-05-09 20:00~23:00` | 남은 checkpoint 평가와 HF card 점수 갱신을 끝낸다. |
| `26B-A4B-it` native full | 8GPU | 작은 모델 정리 직후 | 첫 30 step으로 ETA 재산정. 8K native라 현재 추정 `24~48h`. |
| `26B-A4B` native full | 8GPU | `26B-A4B-it` 직후 | 첫 30 step으로 ETA 재산정. 현재 추정 `24~48h`. |
| `31B-it` smoke | 8GPU | 26B 두 개 후 | `MAX_STEPS=3`, checkpoint 저장/로드/vLLM JSON sanity. 예상 `0.5~2h`. |
| `31B-it` native full | 8GPU | smoke 통과 직후 | 첫 30 step으로 ETA 재산정. dense 31B 8K라 현재 추정 `36~72h`. |
| `31B` native full | 8GPU | `31B-it` 성공 후 | 첫 30 step으로 ETA 재산정. 현재 추정 `36~72h`. |

전체 기준:

- 작은 모델 4개 학습/평가/업로드 완료 목표는 `2026-05-09 18:00Z` 전후다.
- 큰 모델까지 모두 끝나는 전체 일정은 26B/31B의 실제 8K step time에 따라 크게 달라진다. 현재 보수 범위는 `2026-05-13~2026-05-17` 사이이며, 각 큰 모델 시작 후 30 step 시점에 정확도를 높여 다시 기록한다.
- GPU 정책은 `8GPU training 우선`, 8GPU가 필요 없는 중간 구간은 `평가/업로드/재평가/ablation`으로 빈 GPU를 채우는 방식이다.

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
