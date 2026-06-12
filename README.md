# Terminal 모델 평가 리포트

생성 시각: `2026-05-15T04:39:36+00:00` (Step-3.5-Flash, LLaDA2.1-flash 결과 반영: `2026-05-16`, HRM-Text-1B 결과 반영: `2026-05-23`, LFM2.5-8B-A1B 결과 반영: `2026-06-05`, KoHRM stage4d LoRA/full SFT 결과 반영: `2026-06-06`, Qwen3.5-2B fast-continue fullconv 결과 반영: `2026-06-06`, Gemma4-12B/Mellum2-12B 평가 반영: `2026-06-06`, KoHRM LFM25 Epoch2 결과 반영: `2026-06-06 15:28 KST`, KoHRM LFM25 Epoch3 결과 반영: `2026-06-06 20:15 KST`, NVIDIA Nemotron-3 Ultra 550B GGUF 결과 반영: `2026-06-07 05:45 KST`, LFM2.5 ECHO RLVR checkpoint-610 결과 반영: `2026-06-12 15:12 KST`, raw LFM2.5 clean-start ECHO RLVR checkpoint-25~1225 상세 문서 반영: `2026-06-13 08:06 KST`)

이 문서는 corrected 303-step TB2-lite 평가 JSON을 다시 읽어서 정리한 루트 평가 리포트다.
기존 프로젝트 개요 README는 `PROJECT_OVERVIEW_2026-05-02.md`로 이동했다.

## 빠른 점수 보기

- 순위 기준: `Score = 100 * avg_command_f1`
- 보조 지표: `First Cmd`, `Valid JSON`, `Sec/Step`
- raw `LiquidAI/LFM2.5-8B-A1B` 기준점은 README에서 Score `36.53`으로 고정한다.
- 현재 최고점은 `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch + ECHO RLVR LoRA checkpoint-610`, Score `54.05`이다.
- 결과 디렉터리: `/home/work/.data/tb2_lite_eval/corrected_readme_models_vllm`
- GLM-5.1 API 결과: `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/glm51_api/`

주의: 이 README의 점수는 실제 Docker 기반 TerminalBench-2.0 pass@1이 아니라, 빠른 반복 실험을 위한 `TB2-lite corrected 303-step replay` proxy 점수다. 도커를 쓰지 못하는 현재 환경에서 실제 터미널 task를 매번 완전 격리 실행하면 너무 느리고 불안정하므로, command F1/first command/JSON validity 중심으로 빠르게 비교한다. 따라서 이 표는 모델 간 방향성과 checkpoint sweep용으로 읽어야 하며, 최종 공개 성능은 Docker/Harbor/Terminus 계열의 실제 실행 평가로 다시 검증해야 한다.

상세 RLVR 실행/평가/원인 분석은 README 상단에서 제외하고 문서로 분리했다.

- LFM2.5 ECHO RLVR 현재 상태(KO): [`docs/LFM25_ECHO_RLVR_CURRENT_STATUS_KO_20260612.md`](docs/LFM25_ECHO_RLVR_CURRENT_STATUS_KO_20260612.md)
- LFM2.5 raw ECHO RLVR 재시작 기록(KO): [`docs/LFM25_RAW_ECHO_RLVR_RESTART_20260612.ko.md`](docs/LFM25_RAW_ECHO_RLVR_RESTART_20260612.ko.md)
- LFM2.5 ECHO RLVR GPU6 평가 기록: [`docs/ECHO_RLVR_GPU6_EVAL_20260612.md`](docs/ECHO_RLVR_GPU6_EVAL_20260612.md)
- LFM2.5 ECHO RLVR runbook(KO): [`docs/LFM25_ECHO_RLVR_RUNBOOK_KO_20260612.md`](docs/LFM25_ECHO_RLVR_RUNBOOK_KO_20260612.md)
- LFM2.5 SFT/RLVR 비교 분석(KO): [`docs/LFM25_SFT_RLVR_COMPARATIVE_ANALYSIS_20260613.ko.md`](docs/LFM25_SFT_RLVR_COMPARATIVE_ANALYSIS_20260613.ko.md)
- 전체 연구 노트와 시각화는 README 하단 참고 섹션에 둔다.

## 진행 중: LFM2.5 ECHO RLVR GPU6 평가

2026-06-12 기준 GPU 6번에서 ECHO RLVR LoRA checkpoint TB2-lite 평가를 계속 진행한다.

- 상세 기록: [`docs/ECHO_RLVR_GPU6_EVAL_20260612.md`](docs/ECHO_RLVR_GPU6_EVAL_20260612.md)
- 결과 디렉터리: `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612`
- 현재 비교 최고점: `lfm25-echo-rlvr-parentrun-checkpoint-610` Score `54.05`
- RLVR 평가 완료 개수: `285`

## 전체 순위

| 순위 | 모델(HF 저장소명) | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Template | Sec/Step | Load(s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch + ECHO RLVR LoRA checkpoint-610` | 54.05 | 0.5405 | 0.6021 | 0.5571 | 54.5% | 77.9% | chat_template+LoRA | 0.112 | 63.0 |
| 2 | `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch` | 52.30 | 0.5230 | 0.5854 | 0.5431 | 49.5% | 76.9% | chat_template | 0.087 | 44.7 |
| 3 | `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-2Epoch` | 50.48 | 0.5048 | 0.5695 | 0.5296 | 49.2% | 74.9% | chat_template | 0.092 | 76.7 |
| 4 | `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF:UD-Q4_K_M` | 49.97 | 0.4997 | 0.6191 | 0.4785 | 49.2% | 81.5% | chat_template | 26.688 | 542.9 |
| ~4 | `deepseek-ai/DeepSeek-V4-Pro (Valid-only 기능적 동등 보정, 162/303)` | 48.19* | 0.4819* | — | — | — | — | valid step 기능적 동등 분석 | — | — |
| 4 | `Zyphra/ZAYA1-74B-preview` | 48.15 | 0.4815 | 0.6196 | 0.5017 | 51.8% | 74.6% | chat_template | 4.151 | 1192.6 |
| 5 | `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch2` | 45.90 | 0.4590 | 0.5031 | 0.5098 | 44.9% | 68.3% | kohrm-local-prefixlm-export | 10.842 | 10.9 |
| 6 | `LLM-OS-Models/Qwen3.5-2B-Terminal-ToolCall-FullConv-FastContinue-1Epoch` | 44.79 | 0.4479 | 0.5266 | 0.4701 | 34.3% | 83.2% | chat_template | 0.079 | 15.5 |
| 7 | `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch3` | 43.57 | 0.4357 | 0.4703 | 0.5003 | 45.5% | 61.7% | kohrm-local-prefixlm-export-nocompile | 11.156 | 2.9 |
| ~4 | `deepseek-ai/DeepSeek-V4-Pro (기능적 동등 보정, 162/303)` | 42.46* | 0.4246* | — | — | — | — | 기능적 동등 분석 | — | — |
| 4 | **`GLM-5.1 (z.ai API)`** | **41.68** | **0.4168** | **0.5377** | **0.4007** | **24.1%** | **90.1%** | **anthropic-api** | **3.298** | **-** |
| ~5 | `deepseek-ai/DeepSeek-V4-Pro (chat t=0.0, m=4096, 공식, 162/303)` | 40.29* | 0.4029* | 0.5752 | 0.4112 | 45.1% | 44.7% | deepseek_official_mp8_chat_t0_m4096 | 689.3 | 15.0 |
| 5 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch` | 39.56 | 0.3956 | 0.4702 | 0.4808 | 40.6% | 17.2% | gemma4_native | 6.820 | 43.8 |
| 6 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 39.52 | 0.3952 | 0.5082 | 0.4101 | 33.0% | 82.2% | chat_template | 0.081 | 97.1 |
| 7 | `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch1` | 38.56 | 0.3856 | 0.4262 | 0.4341 | 37.0% | 55.1% | kohrm-local-prefixlm-export | 8.314 | 7.0 |
| 7 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 38.52 | 0.3852 | 0.4988 | 0.4056 | 32.7% | 83.2% | chat_template | 0.080 | 130.1 |
| 8 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData` | 38.26 | 0.3826 | 0.4620 | 0.3905 | 28.4% | 64.4% | chat_template | 0.293 | 377.3 |
| 9 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-1Epoch` | 38.12 | 0.3812 | 0.4405 | 0.4787 | 42.6% | 13.5% | gemma4_native | 6.854 | 38.8 |
| 10 | `Qwen/Qwen3.5-9B` | 38.10 | 0.3810 | 0.4921 | 0.3527 | 20.8% | 78.2% | chat_template | 0.268 | 123.8 |
| 11 | `nvidia/Nemotron-Terminal-32B` | 38.09 | 0.3809 | 0.5058 | 0.3827 | 40.3% | 58.7% | chat_template | 0.819 | 154.9 |
| 12 | `Qwen/Qwen3.5-397B-A17B-FP8` | 37.81 | 0.3781 | 0.5107 | 0.3443 | 21.5% | 86.1% | chat_template | 0.860 | 826.5 |
| 13 | `nvidia/Nemotron-Terminal-14B` | 37.70 | 0.3770 | 0.4688 | 0.3849 | 40.6% | 57.1% | chat_template | 0.360 | 98.8 |
| 14 | `Qwen/Qwen3.5-122B-A10B-FP8` | 37.28 | 0.3728 | 0.5155 | 0.3408 | 20.5% | 84.2% | chat_template | 0.655 | 746.9 |
| 15 | `DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF:Q4_K_M` | 37.09 | 0.3709 | 0.5010 | 0.3558 | 20.8% | 83.8% | chat_template | 14.422 | 141.5 |
| 16 | `LiquidAI/LFM2.5-8B-A1B` | 36.53 | 0.3653 | 0.4812 | 0.3685 | 39.9% | 59.1% | chat_template | 0.097 | 103.4 |
| 17 | `Qwen/Qwen3.5-35B-A3B-FP8` | 36.44 | 0.3644 | 0.5086 | 0.3317 | 23.1% | 77.6% | chat_template | 0.200 | 222.8 |
| 18 | `Qwen/Qwen3.5-35B-A3B` | 36.41 | 0.3641 | 0.5068 | 0.3330 | 22.1% | 78.2% | chat_template | 0.228 | 363.1 |
| 19 | `Qwen/Qwen3.5-27B` | 36.30 | 0.3630 | 0.4985 | 0.3343 | 22.1% | 74.9% | chat_template | 0.893 | 102.6 |
| 20 | `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` | 36.29 | 0.3629 | 0.4813 | 0.3460 | 17.5% | 80.5% | chat_template | 5.577 | - |
| 21 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData` | 36.25 | 0.3625 | 0.4797 | 0.3723 | 26.1% | 61.7% | chat_template | 0.205 | 207.3 |
| 22 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData` | 36.05 | 0.3605 | 0.4601 | 0.3690 | 28.1% | 59.4% | chat_template | 0.206 | 158.5 |
| 23 | `nvidia/Nemotron-Terminal-8B` | 35.80 | 0.3580 | 0.4649 | 0.3592 | 35.3% | 54.5% | chat_template | 0.273 | 95.7 |
| 24 | `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | 35.68 | 0.3568 | 0.4822 | 0.3313 | 22.8% | 74.6% | chat_template | 0.904 | 203.7 |
| 25 | `LLM-OS-Models/Ouro-2.6B-Thinking-Terminal-SFT` | 35.61 | 0.3561 | 0.4586 | 0.3647 | 25.1% | 61.1% | chat_template | 3.358 | 135.3 |
| 26 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-1Epoch-FullFT-2BData` | 35.60 | 0.3560 | 0.4537 | 0.3752 | 26.7% | 62.7% | chat_template | 0.291 | 179.0 |
| 27 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-1Epoch` | 35.55 | 0.3555 | 0.4650 | 0.3776 | 27.4% | 65.3% | gemma4_native | 3.924 | 41.4 |
| 28 | `DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF:Q4_K_M` | 35.47 | 0.3547 | 0.4756 | 0.3397 | 17.8% | 78.9% | chat_template | 9.964 | 6.6 |
| 29 | `Jackrong/Qwopus3.6-35B-A3B-v1-GGUF:Q4_K_M` | 35.45 | 0.3545 | 0.4929 | 0.3287 | 18.5% | 81.2% | chat_template | 4.835 | 5.7 |
| 30 | `Qwen/Qwen3.5-4B` | 35.42 | 0.3542 | 0.4836 | 0.3292 | 17.8% | 75.6% | chat_template | 0.185 | 120.0 |
| 31 | `deepseek-ai/DeepSeek-V4-Pro (chat t=0.0, m=1024, 175/303)` | 35.40 | 0.3540 | 0.4872 | 0.3336 | 29.7% | 52.6% | deepseek_official_mp8_chat_t0 | 376.0 | 15.0 |
| 32 | `Qwen/Qwen3.5-2B` | 35.10 | 0.3510 | 0.4944 | 0.3220 | 18.2% | 81.8% | chat_template | 0.077 | 112.8 |
| 33 | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-2Epoch` | 34.98 | 0.3498 | 0.4737 | 0.3576 | 30.4% | 35.0% | gemma4_native | 0.277 | 53.2 |
| 34 | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-1Epoch` | 34.42 | 0.3442 | 0.4823 | 0.3397 | 27.1% | 45.5% | gemma4_native | 0.360 | 93.6 |
| 35 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-TemplateMasked` | 33.46 | 0.3346 | 0.4363 | 0.3549 | 31.7% | 65.0% | chat_template | 0.177 | 220.0 |
| 36 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-TemplateMasked` | 33.35 | 0.3335 | 0.4242 | 0.3590 | 26.1% | 66.3% | chat_template | 0.180 | 198.6 |
| ~37 | `deepseek-ai/DeepSeek-V4-Pro (chat t=0.3, m=4096, 86/303)` | ~33* | 0.33* | — | — | — | 59.3% | deepseek_official_mp8_chat_t03_m4096 | 814.0 | 15.0 |
| 37 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout` | 32.85 | 0.3285 | 0.4194 | 0.3336 | 31.0% | 55.4% | chat_template | 0.150 | 32.9 |
| 38 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 32.77 | 0.3277 | 0.3953 | 0.3541 | 18.2% | 24.8% | chat_template | 0.348 | 300.9 |
| 39 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch` | 32.57 | 0.3257 | 0.4417 | 0.3396 | 30.7% | 61.7% | gemma4_native | 4.036 | 43.2 |
| 40 | `Qwen/Qwen3.6-27B` | 32.56 | 0.3256 | 0.4346 | 0.3149 | 15.8% | 73.9% | chat_template | 0.889 | 178.2 |
| 41 | `LLM-OS-Models/LFM2-8B-A1B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout` | 32.41 | 0.3241 | 0.4327 | 0.3220 | 29.4% | 56.8% | chat_template | 0.126 | 36.6 |
| 42 | `MiniMaxAI/MiniMax-M2.7` | 32.29 | 0.3229 | 0.4492 | 0.2940 | 13.5% | 70.6% | chat_template | 0.445 | 642.9 |
| 43 | `deepseek-ai/DeepSeek-V4-Flash` | 32.22 | 0.3222 | 0.4511 | 0.3037 | 24.4% | 44.2% | deepseek_official_mp4 | 178.033 | 15.6 |
| 44 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout` | 31.86 | 0.3186 | 0.4102 | 0.3321 | 30.0% | 55.8% | chat_template | 0.151 | 33.0 |
| 45 | `LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT` | 31.74 | 0.3174 | 0.4062 | 0.3410 | 24.8% | 63.7% | chat_template | 1.698 | 92.4 |
| 46 | `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-Top2-Terminal-Tool-Merge-Epoch1` | 31.59 | 0.3159 | 0.3859 | 0.3415 | 24.8% | 73.3% | kohrm-local-prefixlm-export | 7.5 | 7.3 |
| 47 | `LLM-OS-Models/LFM2-8B-A1B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout` | 31.02 | 0.3102 | 0.4063 | 0.3201 | 29.0% | 54.8% | chat_template | 0.126 | 35.7 |
| 48 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout-7GPU` | 31.02 | 0.3102 | 0.4063 | 0.3201 | 29.0% | 54.8% | chat_template | 0.128 | 131.5 |
| 49 | `Qwen/Qwen3.6-35B-A3B-FP8` | 30.57 | 0.3057 | 0.4248 | 0.2873 | 14.5% | 75.2% | chat_template | 0.203 | 181.9 |
| 50 | `Qwen/Qwen3.6-35B-A3B` | 30.28 | 0.3028 | 0.4093 | 0.2879 | 14.2% | 73.3% | chat_template | 0.234 | 360.2 |
| 51 | `LLM-OS-Models/Ouro-2.6B-Terminal-SFT` | 29.58 | 0.2958 | 0.3624 | 0.3156 | 22.8% | 29.4% | chat_template | 5.154 | 332.6 |
| 52 | `KoHRM-Text-1.4B-stage4d + terminal-tool-core-r64 LoRA` | 29.11 | 0.2911 | 0.3988 | 0.2768 | 22.1% | 63.4% | kohrm-local-lora-prefixlm | 17.217 | 23.2 |
| 53 | `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch` | 28.96 | 0.2896 | 0.3829 | 0.3254 | 24.4% | 26.4% | gemma4_native | 5.950 | 43.2 |
| 54 | `KoHRM-Text-1.4B-stage4d + terminal-comp-jsonfix-r64 LoRA` | 28.80 | 0.2880 | 0.3834 | 0.2878 | 24.4% | 66.7% | kohrm-local-lora-prefixlm | 17.564 | 23.4 |
| 55 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout-8GPU` | 28.64 | 0.2864 | 0.3775 | 0.2904 | 29.0% | 50.5% | chat_template | 0.085 | 49.9 |
| 56 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-LiquidCLI-TemplateHoldout` | 28.64 | 0.2864 | 0.3775 | 0.2904 | 29.0% | 50.5% | chat_template | 0.086 | 25.4 |
| 57 | `google/gemma-4-12B-it` | 28.58 | 0.2858 | 0.3512 | 0.3324 | 17.5% | 23.4% | gemma4text_fallback | 9.163 | 166.1 |
| 57 | `google/gemma-4-26B-A4B-it` | 28.51 | 0.2851 | 0.4057 | 0.2643 | 14.2% | 71.9% | chat_template | 0.277 | 747.8 |
| 58 | `KoHRM-Text-1.4B-stage4d + comp-terminal-80m LoRA` | 28.44 | 0.2844 | 0.3718 | 0.2803 | 22.8% | 67.3% | kohrm-local-lora-prefixlm | 17.004 | 20.1 |
| 59 | `LLM-OS-Models/Ouro-1.4B-terminal-sft` | 28.30 | 0.2830 | 0.3520 | 0.3141 | 22.4% | 27.1% | chat_template | 2.344 | 83.1 |
| 60 | `gyung/LFM2-8B-Terminal-SFT-Unsloth` | 28.23 | 0.2823 | 0.3817 | 0.2851 | 27.1% | 53.8% | chat_template | 0.124 | 36.0 |
| 61 | `Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M` | 28.21 | 0.2821 | 0.4135 | 0.2506 | 14.5% | 53.8% | gemma4_native | 5.688 | 7.5 |
| 62 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-1Epoch-LiquidCLI-TemplateHoldout` | 28.10 | 0.2810 | 0.3615 | 0.2941 | 29.4% | 50.5% | chat_template | 0.085 | 25.4 |
| 63 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 27.33 | 0.2733 | 0.3526 | 0.2872 | 24.1% | 62.0% | chat_template | 0.124 | 267.1 |
| 64 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 27.31 | 0.2731 | 0.3643 | 0.2804 | 21.8% | 62.0% | chat_template | 0.147 | 69.7 |
| 65 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 27.28 | 0.2728 | 0.3389 | 0.3062 | 10.2% | 13.9% | chat_template | 0.379 | 269.7 |
| 66 | `KoHRM-Text-1.4B-stage4d + terminal-tool-jsonfix-r32 LoRA` | 27.15 | 0.2715 | 0.3831 | 0.2574 | 20.8% | 64.4% | kohrm-local-lora-prefixlm | 16.990 | 23.1 |
| 67 | `deepseek-ai/DeepSeek-V4-Pro (thinking t=1.0, 301/303)` | 26.66 | 0.2666 | 0.3733 | 0.2366 | 23.5% | 31.0% | deepseek_official_mp8 | 441.6 | 15.0 |
| 68 | `google/gemma-4-31B-it` | 26.33 | 0.2633 | 0.3513 | 0.2571 | 10.9% | 67.3% | chat_template | 1.362 | 845.5 |
| 69 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 26.27 | 0.2627 | 0.3581 | 0.2681 | 16.8% | 58.1% | chat_template | 0.179 | 227.6 |
| 70 | `KoHRM-Text-1.4B-stage4d + behavior-jsonfix-r32 LoRA` | 26.23 | 0.2623 | 0.3807 | 0.2507 | 21.1% | 71.0% | kohrm-local-lora-prefixlm | 15.954 | 22.3 |
| 71 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-1Epoch` | 25.70 | 0.2570 | 0.3615 | 0.2717 | 15.2% | 34.3% | gemma4_native | 0.325 | 51.8 |
| 72 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-2Epoch` | 24.92 | 0.2492 | 0.3667 | 0.2447 | 11.6% | 34.0% | gemma4_native | 0.317 | 51.8 |
| 73 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 24.74 | 0.2474 | 0.3390 | 0.2456 | 12.5% | 56.1% | chat_template | 0.178 | 228.6 |
| 74 | `KoHRM-Text-1.4B-stage4d + behavior-core-r64 LoRA` | 24.68 | 0.2468 | 0.3409 | 0.2405 | 21.8% | 64.4% | kohrm-local-lora-prefixlm | 16.974 | 23.0 |
| 75 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 22.45 | 0.2245 | 0.3097 | 0.2314 | 18.8% | 47.2% | chat_template | 0.083 | 57.2 |
| 76 | `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-1Epoch` | 21.08 | 0.2108 | 0.2886 | 0.2422 | 11.2% | 21.1% | gemma4_native | 5.650 | 44.7 |
| 77 | `Zyphra/ZAYA1-8B` | 20.61 | 0.2061 | 0.2844 | 0.2117 | 17.5% | 35.3% | chat_template | 2.339 | 598.3 |
| 78 | `inclusionAI/LLaDA2.1-flash` | 20.07 | 0.2007 | 0.3150 | 0.1819 | 11.2% | 28.7% | sglang_llada_suffix | 2.502 | - |
| 79 | `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch` | 19.52 | 0.1952 | 0.2626 | 0.2091 | 15.5% | 25.7% | gemma4_native | 7.116 | 36.0 |
| 80 | `google/gemma-4-E4B-it` | 19.36 | 0.1936 | 0.3184 | 0.1822 | 11.6% | 54.8% | chat_template | 0.205 | 175.8 |
| 81 | `JetBrains/Mellum2-12B-A2.5B-Thinking` | 18.96 | 0.1896 | 0.2682 | 0.2023 | 15.5% | 35.3% | chat_template | 4.536 | 138.4 |
| 82 | `stepfun-ai/Step-3.5-Flash` | 18.80 | 0.1880 | 0.2710 | 0.1790 | 13.9% | 27.4% | step3p5_vllm_bf16 | 5.368 | - |
| 83 | `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-1Epoch` | 18.66 | 0.1866 | 0.2370 | 0.2152 | 14.5% | 25.1% | gemma4_native | 7.020 | 45.3 |
| 84 | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-2Epoch` | 18.47 | 0.1847 | 0.2514 | 0.1980 | 16.8% | 17.2% | gemma4_native | 0.302 | 52.6 |
| 85 | `google/gemma-4-E2B-it` | 17.40 | 0.1740 | 0.2918 | 0.1613 | 7.3% | 57.1% | chat_template | 0.148 | 139.6 |
| 86 | `LiquidAI/LFM2-2.6B` | 17.06 | 0.1706 | 0.2229 | 0.2160 | 12.9% | 29.4% | chat_template | 0.152 | 55.0 |
| 87 | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-2Epoch` | 16.22 | 0.1622 | 0.2747 | 0.1678 | 15.2% | 16.2% | gemma4_native | 0.289 | 73.4 |
| 88 | `ByteDance/Ouro-1.4B` | 15.06 | 0.1506 | 0.1988 | 0.1625 | 8.9% | 37.3% | chat_template | 1.946 | 74.8 |
| 89 | `LiquidAI/LFM2.5-1.2B-Instruct` | 14.46 | 0.1446 | 0.2374 | 0.1526 | 10.6% | 60.1% | chat_template | 0.056 | 39.8 |
| 90 | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-1Epoch` | 12.80 | 0.1280 | 0.1792 | 0.1364 | 10.2% | 14.2% | gemma4_native | 0.383 | 93.8 |
| 91 | `ByteDance/Ouro-1.4B-Thinking` | 12.69 | 0.1269 | 0.2026 | 0.1299 | 9.2% | 26.7% | chat_template | 2.115 | 65.9 |
| 92 | `KoHRM-Text-1.4B-stage4d direct` | 11.48 | 0.1148 | 0.1995 | 0.0961 | 5.9% | 38.9% | kohrm-local-prefixlm | 14.001 | 13.0 |
| 93 | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-1Epoch` | 11.35 | 0.1135 | 0.1767 | 0.1191 | 6.6% | 7.3% | gemma4_native | 0.219 | 45.3 |
| 94 | `LiquidAI/LFM2-24B-A2B` | 10.87 | 0.1087 | 0.1466 | 0.1163 | 5.3% | 54.5% | chat_template | 0.165 | 236.2 |
| 95 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 10.67 | 0.1067 | 0.1507 | 0.1067 | 4.0% | 5.9% | chat_template | 0.185 | 65.7 |
| 96 | `KoHRM-Text-1.4B-stage4d synth,cot` | 10.36 | 0.1036 | 0.1696 | 0.0878 | 4.0% | 36.0% | kohrm-local-prefixlm | 14.406 | 12.0 |
| 97 | `LiquidAI/LFM2-8B-A1B` | 10.04 | 0.1004 | 0.1405 | 0.1223 | 5.9% | 27.4% | chat_template | 0.124 | 61.9 |
| 98 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 9.22 | 0.0922 | 0.1249 | 0.1023 | 3.3% | 5.9% | chat_template | 0.184 | 67.5 |
| 99 | `ByteDance/Ouro-2.6B` | 6.46 | 0.0646 | 0.0976 | 0.0692 | 5.0% | 16.5% | chat_template | 4.607 | 99.6 |
| 100 | `sapientinc/HRM-Text-1B` | 0.40 | 0.0040 | 0.0057 | 0.0040 | 1.3% | 4.3% | hrm_text_prefixlm | 3.976 | 4.4 |
| 101 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | chat_template | 1.774 | 300.1 |
| 102 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0% | 0.0% | chat_template | 1.770 | 300.1 |



전체 순위 반영 메모:

- 완료된 Native Gemma 4 결과 16개, 요청 외부 모델 완료분 18개, Qwen3.5 대형 FP8 완료분 2개, Qwen3.5-2B fast-continue fullconv 평가 1개, GLM-5.1 API 평가 1개, LFM2.5-8B-A1B ToolBench Full SFT 1epoch/2epoch 평가 2개, KoHRM-Text stage4d base/LoRA full 303-step 평가 8개, KoHRM-Text full SFT 4개를 이 표에 반영했다.
- 현재 전체 1위는 `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch + ECHO RLVR LoRA checkpoint-610`, Score `54.05`이다. SFT 1Epoch 단독 Score `52.30`보다 `+1.75`, 2epoch Score `50.48`보다 `+3.57`, `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF:UD-Q4_K_M` Score `49.97`보다 `+4.08` 높다. raw clean-start ECHO RLVR은 checkpoint-1225 Score `45.32`까지 상승해 raw LFM2.5 README 기준점 `36.53`보다 `+8.79` 높지만, 아직 SFT 1Epoch 단독보다는 `-6.98` 낮다.
- `KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch2`는 Score `45.90`으로 KoHRM 계열 최고점으로 유지한다. Epoch1 Score `38.56`보다 `+7.34`, Qwen fast-continue fullconv Score `44.79`보다 `+1.11` 높고, LFM2.5 ToolBench SFT 2epoch Score `50.48`보다는 `-4.58` 낮다. Epoch3는 Score `43.57`로 Epoch2보다 `-2.33` 낮아 current best로 올리지 않는다. 8-shard Epoch2 평가 wall time은 가장 느린 shard 기준 `3285.1s`, 약 `54분 45초`다.
- `KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch3`는 Score `43.57`로 완료됐다. Cmd F1 `0.4357`, Precision `0.4703`, Recall `0.5003`, First Cmd `45.5%`, Valid JSON `61.7%`, 303/303 step이다. Epoch2 Score `45.90`보다 `-2.33` 낮아 KoHRM current best는 Epoch2로 유지한다.
- `Qwen3.5-2B-Terminal-ToolCall-FullConv-FastContinue-1Epoch`는 Score `44.79`로 Qwen 계열 최고점이다. 기존 `Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` Score `39.52`보다 `+5.27` 높고, GLM-5.1 API Score `41.68`보다도 `+3.11` 높다. 같은 결과 JSON의 legacy `next_action_score`는 `41.64`이며, 전체 순위 기준에는 쓰지 않는다.
- 26B-A4B-it native 2epoch는 이제 Score `39.56`, Qwen3.5-2B SameCount SFT 2epoch는 Score `39.52`다. 26B-A4B-it native 1epoch는 Score `38.12`, 31B-it native 1epoch는 `35.55`, 31B-it native 2epoch는 `32.57`이다.
- Qwen3.5 대형 FP8 완료분은 `Qwen/Qwen3.5-397B-A17B-FP8` Score `37.81`로 전체 12위, `Qwen/Qwen3.5-122B-A10B-FP8` Score `37.28`로 전체 14위다. 둘 다 vLLM `tp=8`, expert parallel, prefix caching, Qwen3 reasoning parser, `max_model_len=32768`로 평가했다.
- `LiquidAI/LFM2.5-8B-A1B`는 Score `36.53`으로 신규 반영했다. 기존 `LiquidAI/LFM2-8B-A1B` base Score `10.04`보다 `+26.49` 높고, `Qwen/Qwen3.5-35B-A3B-FP8` Score `36.44`를 근소하게 앞선다. First Cmd `39.9%`, Recall `0.3685`, Sec/Step `0.097`이라 8B급 빠른 base 모델 중 매우 강한 결과다.
- KoHRM-Text stage4d 계열은 base direct 2개와 LoRA 6개를 full 303-step으로 평가했다. 최고 LoRA는 `terminal-tool-core-r64` Score `29.11`이고, top2 terminal/tool merge full SFT 1epoch는 Score `31.59`, LFM2.5/ToolBench full SFT는 Epoch1 Score `38.56`, Epoch2 Score `45.90`이다. Epoch2는 LoRA 최고보다 `+16.79`, top2 full SFT보다 `+14.31`, Ouro-1.4B-Thinking-Terminal-SFT보다 `+14.16` 높아 현재 KoHRM 최고 결과다. HRM PrefixLM 구조는 현재 로컬 전용 evaluator로 평가했으며 vLLM chat model 경로는 쓰지 않았다.
- 요청 외부 모델 + API 평가 완료분 중 최고는 `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF:UD-Q4_K_M` Score `49.97`다. 전체표에서는 LFM2.5 ToolBench SFT 1epoch/2epoch 다음 3위이며, 비학습/외부 모델 최고 결과다. 그 다음 주요 완료분은 `Zyphra/ZAYA1-74B-preview` Score `48.15`, `GLM-5.1 (z.ai API)` Score `41.68`, `DavidAU/Qwen3.6-40B Deckard...:Q4_K_M` Score `37.09`, `LiquidAI/LFM2.5-8B-A1B` Score `36.53`, `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL` Score `36.29`다. `google/gemma-4-12B-it`은 text-only fallback으로 Score `28.58`, `JetBrains/Mellum2-12B-A2.5B-Thinking`은 Score `18.96`으로 완료했다. 둘 다 상위 terminal-action 후보로 보기에는 낮다.
- `DeepSeek-V4-Flash`는 공식 converted MP4 inference 경로로 `2 x tp=4`, 8GPU full replay 303-step을 완료했고 Score `32.22`로 전체 43위다. 이전 vLLM/FP8 경로의 BOS 반복 0점 실패는 최종 결과에서 대체했다. `google/gemma-4-31B-it-assistant`는 아직 결과 JSON 없음.
- `Step-3.5-Flash`는 공식 vLLM Step-3.5 recipe 계열 설정을 기준으로 FP8 route를 먼저 시도했지만 로컬 CUDA/vLLM 조합에서 illegal address와 MTP layer config 문제가 반복되어, 최종 반영은 BF16 원본 모델 `tp=8`, expert parallel, `max_model_len=49152`로 완료한 303-step 결과다. Score는 `18.80`으로 전체 80위다.

### LFM2.5-8B-A1B ToolBench Full SFT 분석

`LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`는 corrected TB2-lite 303-step full replay에서 Score `52.30`을 기록했다. 같은 `score = 100 * avg_command_f1` 기준이며, 결과 JSON은 `tb2_lite/results/20260605T_live_hrm_lora_lfm_epoch1/LFM2.5-8B-A1B-terminal-toolbench-full-1epoch-checkpoint-1542.json`이다. 같은 run의 2epoch 최종 checkpoint는 Score `50.48`이며, 결과 JSON은 `tb2_lite/results/20260605T_all_idle_eval/LFM2.5-8B-A1B-terminal-toolbench-full-2epoch-final.json`이다.

핵심 지표:
- 1epoch: Score `52.30`, Cmd F1 `0.5230`, Precision `0.5854`, Recall `0.5431`
- 1epoch: First Cmd `49.5%`, Valid JSON `76.9%`, Next Action score `51.46`
- 1epoch: 평균 정답 command `38.42`개, 평균 예측 command `21.20`개
- 1epoch: vLLM 평가 속도 `0.087 sec/step`, load `44.7s`, 303/303 step 완료
- 2epoch: Score `50.48`, Cmd F1 `0.5048`, Precision `0.5695`, Recall `0.5296`
- 2epoch: First Cmd `49.2%`, Valid JSON `74.9%`, Next Action score `50.1`
- 2epoch: 평균 정답 command `38.42`개, 평균 예측 command `22.50`개
- 2epoch: vLLM 평가 속도 `0.092 sec/step`, load `76.7s`, 303/303 step 완료

왜 점수가 높게 나왔는가:
- base `LiquidAI/LFM2.5-8B-A1B`가 이미 Score `36.53`으로 강했고, full terminal + ToolBench conversation SFT가 1epoch에서 Recall을 `0.3685 -> 0.5431`로 크게 올렸다.
- Precision `0.5854`도 높아 command를 많이 맞히면서도 불필요한 command 폭증이 제한됐다. 평균 예측 command는 정답보다 적은 `21.20`개라 보수적이지만, 이전 LFM 계열처럼 지나치게 비거나 산만하지 않았다.
- 1epoch는 early step F1 `0.6210`이 매우 높고, `scientific_computing` `0.6394`, `security` `0.6139`, `software_engineering` `0.6023`, `data_querying` `0.5823`, `data_science` `0.5489`에서 강했다. 파일 확인, 데이터 검사, 스크립트 실행, 보안 수정, 과학 계산처럼 명령 패턴이 분명한 태스크에서 특히 잘 맞았다.
- 2epoch는 `model_training` `0.5931`, `data_science` `0.6020`, `code` `0.2385`처럼 일부 영역이 개선됐지만, 전체적으로 평균 예측 command가 `21.20 -> 22.50`으로 늘고 Valid JSON이 `76.9% -> 74.9%`로 내려가면서 Precision/Recall이 모두 약간 하락했다. 이 run에서는 1epoch가 더 좋은 체크포인트다.

약점과 실패 케이스:
- 1epoch late step F1은 `0.4455`로 early `0.6210`보다 낮다. 긴 대화 후반으로 갈수록 이미 끝났다고 판단하거나 다음 검증 command를 생략하는 경향이 있다.
- 1epoch 전체 303개 중 invalid JSON이 `70`개, zero-F1이 `23`개였다. 실패는 단순 지식 부족보다 `<think>` 텍스트가 JSON 앞에 붙거나, `task_complete=true`를 너무 빨리 내는 형식 문제가 많다.
- 가장 약한 영역은 `code` F1 `0.1613`이며 zero-F1 8개, invalid JSON 9개가 몰렸다. 알고리즘 풀이형 task-042/task-043에서 긴 사고 텍스트를 먼저 내고 commands JSON을 깨뜨리거나, 실제로는 `ls`, `cat`, `python3` 검증이 필요한데 설명만 길게 출력했다.
- `math` F1 `0.4146`, `swe` `0.4378`, `dependency_management` `0.4723`도 상대적으로 약하다. 특히 후반 SWE/수학 단계에서는 정답이 `sed`, `grep`, `cat`, `python3`로 구체 파일을 재확인하는 흐름인데, 모델은 이미 해결됐다고 보고 completion을 앞당기는 경우가 있었다.
- 1epoch에서 `pred_task_complete=true`가 reference보다 빠른 step은 5개였다. completion 판단은 아직 과신이 남아 있다.

### KoHRM-Text-1.4B stage4d LoRA 평가 분석

KoHRM-Text stage4d 계열은 corrected TB2-lite 303-step full replay에서 base direct 2개와 LoRA 6개를 같은 `score = 100 * avg_command_f1` 기준으로 다시 평가했다. 결과 디렉터리는 `tb2_lite/results/20260605T_live_hrm_lora_lfm_epoch1/`이며, HRM PrefixLM 구조에 맞춘 로컬 evaluator를 사용했다. 아직 vLLM의 일반 HF causal/chat model 경로로 안정 실행되는 구조가 아니라서, 표의 `Sec/Step`은 vLLM 수치가 아니다.

| 모델 | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Sec/Step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `terminal-tool-core-r64 LoRA` | 29.11 | 0.2911 | 0.3988 | 0.2768 | 22.1% | 63.4% | 17.217 |
| `terminal-comp-jsonfix-r64 LoRA` | 28.80 | 0.2880 | 0.3834 | 0.2878 | 24.4% | 66.7% | 17.564 |
| `comp-terminal-80m LoRA` | 28.44 | 0.2844 | 0.3718 | 0.2803 | 22.8% | 67.3% | 17.004 |
| `terminal-tool-jsonfix-r32 LoRA` | 27.15 | 0.2715 | 0.3831 | 0.2574 | 20.8% | 64.4% | 16.990 |
| `behavior-jsonfix-r32 LoRA` | 26.23 | 0.2623 | 0.3807 | 0.2507 | 21.1% | 71.0% | 15.954 |
| `behavior-core-r64 LoRA` | 24.68 | 0.2468 | 0.3409 | 0.2405 | 21.8% | 64.4% | 16.974 |
| `stage4d direct` | 11.48 | 0.1148 | 0.1995 | 0.0961 | 5.9% | 38.9% | 14.001 |
| `stage4d synth,cot` | 10.36 | 0.1036 | 0.1696 | 0.0878 | 4.0% | 36.0% | 14.406 |

해석:
- LoRA는 필수에 가깝다. 최고 `terminal-tool-core-r64`는 base direct Score `11.48`에서 `29.11`로 `+17.63` 올랐다. First Cmd도 `5.9% -> 22.1%`, Valid JSON도 `38.9% -> 63.4%`로 같이 개선됐다.
- `terminal-tool-core-r64`, `terminal-comp-jsonfix-r64`, `comp-terminal-80m` 세 개가 28점대 후반에 붙어 있다. ToolBench/terminal action을 같이 본 데이터가 command recall을 직접 끌어올린 것으로 본다.
- JSON-fix 계열은 형식 안정성에는 도움이 있지만 점수 1위 조건은 아니었다. `behavior-jsonfix-r32`가 Valid JSON `71.0%`로 가장 높지만 Score는 `26.23`이다. 이 벤치마크에서는 JSON 유효율보다 실제 command coverage가 더 크게 작동한다.
- `synth,cot`는 base direct보다 낮았다. HRM PrefixLM에 사고/합성 조건을 얹는 것만으로는 terminal JSON action을 잘 만들지 못했고, 오히려 불필요한 텍스트가 출력 계약을 흐렸다.
- LoRA가 성능을 크게 올렸지만 LFM2.5 full SFT 1epoch Score `52.30`, Qwen3.5-2B full SFT 2epoch Score `39.52`와는 아직 차이가 크다. 그래서 현재 후속 작업은 adapter가 아니라 full SFT로 진행한다.

왜 `LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT`보다 낮은가:
- KoHRM LoRA 최고점은 `29.11`이고, Ouro-1.4B-Thinking-Terminal-SFT는 `31.74`다. 차이는 `-2.63`점으로 크지는 않지만, 현재 TB2-lite 기준에서는 Ouro thinking SFT가 더 낫다.
- 가장 큰 차이는 첫 행동과 command coverage다. KoHRM 최고 LoRA는 First Cmd `22.1%`, Recall `0.2768`이고, Ouro-1.4B-Thinking-Terminal-SFT는 First Cmd `24.8%`, Recall `0.3410`이다. KoHRM은 JSON을 꽤 만들지만 필요한 명령 묶음을 덜 넓게 복원한다.
- KoHRM stage4d base는 PrefixLM pretraining/continued training 쪽 성격이 강하다. terminal replay가 요구하는 출력은 `{"commands":[{"keystrokes":...}]}` 형태의 next-action JSON인데, base direct는 이 계약을 거의 내재화하지 못해 Score `11.48`, First Cmd `5.9%`에 머물렀다.
- LoRA는 출력 형식을 크게 고쳤지만 adapter 용량과 학습 데이터 폭의 한계가 남았다. Valid JSON은 `63~71%`까지 올라갔지만, 실제 채점은 JSON 유효성보다 command F1을 본다. `behavior-jsonfix-r32`처럼 Valid JSON `71.0%`인 run도 Score는 `26.23`으로 낮다.
- Ouro thinking SFT는 느리지만, 이미 terminal thinking/output 분포에 더 잘 맞는다. 평균 속도는 Ouro-1.4B-Thinking-Terminal-SFT `1.698 sec/step`, KoHRM LoRA `15~17 sec/step`이라 KoHRM 로컬 PrefixLM evaluator가 훨씬 느리고, 점수도 낮아 현재 LoRA만으로는 채택 우선순위가 낮다.
- 결론은 adapter를 더 미세하게 흔드는 것보다 full SFT가 맞다. LoRA가 base 대비 `+17.63`을 만든 것은 방향이 맞다는 증거지만, Ouro/Qwen/LFM 상위권을 넘으려면 KoHRM 자체 가중치를 terminal/tool JSON action 분포로 직접 이동시켜야 한다.

완료된 full SFT 및 평가:
- `KoHRM-Text-1.4B-fullsft-top2-terminal-tool-merge`: 현재 1, 2위 LFM2.5 계열의 terminal/tool raw를 합친 full SFT 실험이다. 준비 데이터는 `kohrm_sft_top2_terminal_tool_raw8192_v1`, context `8192`, 약 `245M` tokens, `GBS=90112`, LR `2e-5`, 1epoch로 학습했고 `fsdp2_epoch_1`을 HF safetensors export로 변환했다.
- export 경로는 `/home/work/.data/hrm_text_exports/KoHRM-Text-1.4B-fullsft-top2-terminal-tool-merge-epoch1`이고, Hugging Face repo는 `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-Top2-Terminal-Tool-Merge-Epoch1`이다. 모델카드는 `base_model: LLM-OS-Models/KoHRM-Text-1.4B`, `base_model_relation: finetune`로 표기했다.
- full 303-step replay 최종 Score는 `31.59`다. Cmd F1 `0.3159`, Precision `0.3859`, Recall `0.3415`, First Cmd `24.8%`, Valid JSON `73.3%`다. KoHRM LoRA 최고 Score `29.11`보다 `+2.48` 높고, `LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT` Score `31.74`보다 `-0.15` 낮다.
- 처음 시도한 `GBS=180224`는 4GPU local token budget이 GPU당 `45056` tokens까지 커져 CUDA OOM이 났다. 안정 학습은 8GPU pretraining 때의 GPU당 token budget과 맞춘 `22528` tokens/GPU 설정에서 이뤄졌다.

최근 full SFT 및 평가:
- `KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench`: LFM2.5 성공 데이터와 ToolBench terminal turn을 KoHRM PrefixLM target으로 재전처리한 전체 데이터다. 준비 데이터는 `kohrm_sft_lfm25_terminal_toolbench_full_v1`, context `8192`, 약 `1.51B` tokens, 4GPU에서 `GBS=90112`, LR `2e-5`로 학습했고 `fsdp2_epoch_1`을 HF safetensors export로 변환했다. export 경로는 `/home/work/.data/hrm_text_exports/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch1`이고, Hugging Face repo는 `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch1`이다.
- full 303-step replay 최종 Score는 `38.56`이다. Cmd F1 `0.3856`, Precision `0.4262`, Recall `0.4341`, First Cmd `37.0%`, Valid JSON `55.1%`, 평균 예측 command `27.33`개다. 결과 JSON은 `tb2_lite/results/20260606T_kohrm_lfm25_fullsft_eval_sdpa4_b16_p5/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch1-sdpa4-b16-merged.json`이다.
- 이 결과는 KoHRM LoRA 최고 Score `29.11`보다 `+9.45`, top2 full SFT Score `31.59`보다 `+6.97`, Ouro-1.4B-Thinking-Terminal-SFT Score `31.74`보다 `+6.82` 높다. `Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` Score `38.52`보다도 `+0.04` 높고, `Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` Score `39.52`와는 `-0.96` 차이다. 다만 Qwen fast-continue fullconv Score `44.79`, LFM2.5 full SFT 1epoch Score `52.30`에는 아직 못 미친다.
- `KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch2`: Epoch1에서 이어서 같은 LFM25/ToolBench 데이터 1 pass를 더 돌린 full SFT다. 8GPU, `GBS=180224`, LR `2e-5`, checkpoint `8754` step, Epoch2 학습 wall time은 약 `3시간 16분`이다. export 경로는 `/home/work/.data/hrm_text_exports/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch2`이고, Hugging Face repo는 `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch2`다.
- Epoch2 full 303-step replay 최종 Score는 `45.90`이다. Cmd F1 `0.4590`, Precision `0.5031`, Recall `0.5098`, First Cmd `44.9%`, Valid JSON `68.3%`, 평균 예측 command `25.16`개, Next Action `45.60`이다. 결과 JSON은 `tb2_lite/results/20260606T_kohrm_lfm25_epoch2_eval_sdpa8_b16/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch2-sdpa8-b16-merged.json`이다.
- Epoch2는 Epoch1 대비 Score `+7.34`, Precision `+0.0769`, Recall `+0.0757`, First Cmd `+7.9%p`, Valid JSON `+13.2%p`다. 특히 Valid JSON과 command recall이 같이 오른 점이 크다. 보통 형식 안정성을 올리면 command 폭이 줄어 Recall이 떨어질 수 있는데, 이번에는 둘 다 개선되어 단순 JSON repair가 아니라 terminal next-action 분포 자체가 더 맞아졌다.
- Epoch2는 Qwen fast-continue fullconv Score `44.79`보다 `+1.11`, Qwen3.5-2B SameCount 2epoch Score `39.52`보다 `+6.38`, Ouro-1.4B-Thinking-Terminal-SFT Score `31.74`보다 `+14.16` 높다. 다만 LFM2.5-8B-A1B ToolBench SFT 2epoch Score `50.48`보다 `-4.58`, 1epoch Score `52.30`보다 `-6.40` 낮아 아직 최상위 LFM2.5의 coverage/first-action 수준에는 못 미친다.
- `Qwen3.5-2B-Terminal-ToolCall-FullConv-FromSameCount`: 기존 `Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`에서 이어서 LFM2.5 full-conversation terminal/toolcall 데이터로 full SFT를 진행했다. 4096 context, 4GPU, full fine-tuning이다. `b18/b8/b7/b6`는 Qwen3.5 Gated DeltaNet torch fallback OOM 또는 gradient checkpointing metadata mismatch로 실패했고, 안정 런은 `b4_nogc_4096`, global batch `16`이었다. 1epoch 평가 최종 Score는 `44.79`이며 Qwen 계열 최고점으로 전체표에 반영했다.

운영 스냅샷, 2026-06-06 20:15 KST 업데이트:
- `KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch2` 평가는 완료됐다. 최종 Score `45.90`, Next Action `45.60`, 303/303 step 완료다. 현재 KoHRM 최고 결과이며 전체표에서는 ZAYA `48.15` 아래, Qwen fast-continue `44.79` 위다.
- Epoch2 평가는 8-shard SDPA batch 16으로 돌렸고, 가장 느린 shard 기준 generation time `3285.1s`, 약 `54분 45초`가 걸렸다. load는 `10.9s`, 평균 `10.842 sec/step`이다. Epoch2 학습 자체는 8GPU에서 약 `3시간 16분` 걸렸다.
- Epoch2 Hugging Face 업로드는 `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch2`로 완료했다. 모델 카드에는 `base_model: LLM-OS-Models/KoHRM-Text-1.4B`, `base_model_relation: finetune`, Epoch1 parent, TB2-lite 점수와 사용법을 적는다.
- Epoch3는 `2026-06-06 15:42:27 KST`에 Epoch2 checkpoint에서 이어서 8GPU로 시작했고, 학습은 약 `3시간 16분 30초`에 완료됐다. checkpoint root는 `/home/work/.data/hrm_text_checkpoints/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch3-from-epoch2-gbs180k-8gpu`, export는 `/home/work/.data/hrm_text_exports/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch3`, Hugging Face repo는 `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch3`다.
- Epoch3 full 303-step replay 최종 Score는 `43.57`다. Cmd F1 `0.4357`, Precision `0.4703`, Recall `0.5003`, First Cmd `45.5%`, Valid JSON `61.7%`, 평균 예측 command `25.82`개, Next Action `44.15`다. 결과 JSON은 `tb2_lite/results/20260606T_kohrm_lfm25_epoch3_eval_sdpa8_b16/KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch3-sdpa8-b16-nocompile-merged.json`이다.
- Epoch3는 Epoch2 대비 Score `-2.33`, Precision `-0.0328`, Recall `-0.0095`, First Cmd `+0.6%p`, Valid JSON `-6.6%p`다. First Cmd는 소폭 올랐지만 JSON 안정성과 command precision이 내려가 전체 점수는 하락했다. 따라서 배포/대표 checkpoint는 Epoch2를 유지한다.
- Epoch3 이후 장기 평가였던 `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF` `UD-Q4_K_M` 8GPU llama.cpp 평가는 완료됐다. 최종 Score `49.97`, Next Action `49.74`, Cmd F1 `0.4997`, Precision `0.6191`, Recall `0.4785`, First Cmd `49.2%`, Valid JSON `81.5%`, 평균 `26.688 sec/step`, load `542.9s`, generation `8086.4s`, 303/303 step이다. LFM2.5 ToolBench SFT 1epoch/2epoch 바로 아래 전체 3위이며, 비학습/외부 모델 중 최고점이다.
- Harness-1 -> LFM2.5 작업은 `LiquidAI/LFM2.5-8B-A1B`에 논문식 LoRA SFT warm-start를 붙이는 방향으로 준비했다. 논문은 SFT와 RL 모두 LoRA rank `32`이며, SFT는 BC+/Web/Patents/SEC teacher trajectories `899`개를 turn별 약 `26K` examples로 확장하고, RL은 SEC train query `3,453`개에서 약 `82K` rollout을 돌리는 구조다. GPU 모델/학습 시간/비용은 논문에 공개되어 있지 않고 Tinker managed service로 low-level worker가 추상화됐다고 되어 있다. 로컬 구현은 `Liquid-CLI/scripts/build_lfm_harness1_dataset.py`, `Liquid-CLI/train_unsloth_processed_lora.py`, `Liquid-CLI/scripts/run_lfm_harness1_lora_sft.sh`에 추가했다. BrowseComp+ public source와 decrypted 830-query file은 `/home/work/.data/harness1/external/BrowseComp-Plus`에 준비했다. 다만 이 파일은 문제/정답/qrel이고 Harness SFT teacher trajectory가 아니므로, 실제 SFT 시작에는 Harness `.env.local`, Chroma index, API credentials 또는 별도 확보한 trajectory JSON/JSONL이 필요하다.

운영 스냅샷, 2026-06-06 09:52 KST 업데이트:
- `KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench-epoch1` 평가는 완료됐다. 최종 Score `38.56`, Next Action `38.09`, 303/303 step 완료다. 현재 KoHRM 최고 결과이며, 모델 카드와 HF 업로드 대상으로 확정했다.
- `google/gemma-4-12B-it` 평가는 text-only fallback 4-shard로 완료됐다. 최종 Score `28.58`, Next Action `25.26`, First Cmd `17.5%`, Valid JSON `23.4%`, Sec/Step `9.163`이다. vLLM Gemma4 경로는 local shape mismatch로 실패했고, 완료 점수는 `tb2_lite/scripts/replay_eval_gemma4_text.py` fallback 경로 기준이다.
- `JetBrains/Mellum2-12B-A2.5B-Thinking` vLLM `tp=2` 평가는 완료됐다. 최종 Score `18.96`, Next Action `17.92`, First Cmd `15.5%`, Valid JSON `35.3%`, Sec/Step `4.536`이다. `model_training`, `security`, `debugging`은 상대적으로 낫지만 `swe`, `code`가 낮아 전체 점수가 낮다.
- 09:52 KST 기준 `nvidia-smi`에서 GPU 0~7은 모두 비어 있었다. `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF` `UD-Q4_K_M` 9-shard 다운로드는 완료됐고, 다음 장기 평가는 모든 GPU를 쓰는 Nemotron GGUF 평가다. 긴 run은 `--save-every` partial 저장으로 중간 점수를 남긴다.

운영 스냅샷, 2026-06-05 14:03 KST 업데이트:
- GPU `1,2,3,7`: `KoHRM-Text-1.4B-fullsft-lfm25-terminal-toolbench` 학습 중이다. 최신 저장 checkpoint는 `step_9000`이며, `step_8000 -> step_9000` 구간 속도는 약 22분 49초/1000 steps다. 이 속도 기준 예상 완료는 2026-06-05 `16:56 KST` 전후다.
- GPU `0,4,5,6`: `Qwen3.5-2B-Terminal-ToolCall-FullConv-FromSameCount` full SFT가 `b4_nogc_4096`으로 진행 중이다. 2026-06-05 14:03 KST 기준 `43/40760` steps, loss는 `0.48~0.86` 범위에서 정상 출력 중이고, 안정 구간 step time은 약 `7.8~8.9s/step`이다. 1epoch 완료 예상은 2026-06-07 `12:00~15:00 KST`, 2epoch 전체 완료 예상은 2026-06-09 `11:00~15:00 KST`다. 아직 평가 점수는 없다.
- KoHRM LoRA repo 8개는 `base_model_relation: adapter`, `library_name: peft`로 업데이트했고 public 상태를 확인했다.


### KoHRM LFM25 Epoch3 결과 분석

Epoch3는 Epoch2 checkpoint에서 같은 LFM25/ToolBench full SFT 데이터를 한 pass 더 이어 학습한 결과다. 학습 loss는 W&B summary 기준 `0.5271`, train accuracy는 `0.86567`까지 내려갔지만, TB2-lite full replay에서는 Score `43.57`로 Epoch2 `45.90`보다 `-2.33` 낮았다.

좋아진 점:
- First Cmd는 `44.9% -> 45.5%`로 `+0.6%p` 올랐다. 첫 행동 선택 감각은 유지되거나 소폭 좋아졌다.
- `model_training`은 Epoch3에서 F1 `0.4910`로 강한 축에 들어왔다. 일부 학습/실험형 명령 패턴은 추가 pass가 도움이 됐다.
- early bucket은 `0.5184`로 여전히 가장 강하다. 초반 파일 파악/조회형 단계는 유지됐다.

나빠진 점:
- Cmd F1은 `0.4590 -> 0.4357`, Precision은 `0.5031 -> 0.4703`, Recall은 `0.5098 -> 0.5003`로 모두 하락했다.
- Valid JSON은 `68.3% -> 61.7%`로 `-6.6%p` 내려갔다. 첫 명령은 맞히더라도 최종 JSON contract와 command set 복원이 흔들린다.
- 약점 source group은 `math` `0.3150`, `dependency_management` `0.3580`, `security` `0.3580`, `data_processing` `0.3621`, `swe` `0.3815`다. 후반 수정/검증이나 dependency/security 계열은 추가 epoch가 오히려 안정성을 깎았다.

JSON 안정성:
- JSON을 잘 맞추는 능력은 Epoch3에서 오르지 않았다. 흐름은 Epoch1 `55.1%` -> Epoch2 `68.3%` -> Epoch3 `61.7%`다. Epoch2에서는 command coverage와 JSON validity가 같이 올랐지만, Epoch3에서는 First Cmd만 소폭 오르고 JSON contract가 내려갔다.
- LoRA 실험에서도 비슷한 신호가 있었다. `behavior-jsonfix-r32`는 Valid JSON `71.0%`로 높았지만 Score는 `26.23`에 머물렀다. 즉 JSON-fix 데이터는 형식에는 도움을 주지만, terminal next-action 점수를 올리려면 `commands[].keystrokes` 자체를 더 맞혀야 한다.
- 이번 Epoch3 하락은 단순히 "JSON 데이터가 부족했다"기보다, 같은 target을 한 번 더 반복하면서 모델이 출력해야 할 command 후보와 완료 판단을 더 좁게 잡고, 긴 출력에서 field/array contract를 안정적으로 닫는 능력은 약해진 쪽에 가깝다.

해석:
- Epoch2까지는 같은 데이터 반복이 JSON 안정성과 command coverage를 같이 올렸지만, Epoch3에서는 반복 학습이 출력 분포를 더 정교하게 만들기보다 일부 command를 과하게 좁히거나 JSON formatting을 흔든 것으로 보인다.
- 과적합이라고 부를 수 있는 부분은 있다. train loss `0.5271`과 train accuracy `0.86567`은 좋아졌는데 held-out replay Score와 Valid JSON이 내려갔기 때문이다. 다만 전형적인 암기형 과적합만으로 보기는 어렵고, PrefixLM 출력 contract가 얇은 상태에서 같은 분포를 더 돌리며 command distribution과 `task_complete`/JSON closing 습관이 틀어진 "contract overfit" 쪽으로 보는 게 맞다.
- 다음 액션은 Epoch3를 계속 미는 것보다 Epoch2를 기준으로 유지하고, 더 다양한 terminal/tool replay나 DPO/RL-style preference, 혹은 late-step/error-recovery 중심 데이터로 보강하는 쪽이 낫다. LFM2.5나 Qwen 2B는 이미 vLLM/chat-template 경로와 평가 속도가 안정적이므로, RL을 빨리 돌려 비교하기 좋은 축이고, HRM은 별도로 JSON-only decoding/stop handling/PrefixLM 전용 serving을 연구해야 한다.

### KoHRM 성능 원인과 다음 액션

현재 확인된 결론은 `KoHRM-Text-1.4B-stage4d`가 terminal/tool action을 강하게 배울 수 있지만, 같은 LFM25/ToolBench 데이터를 3번째 pass까지 계속 돌리는 것은 성능을 더 밀어 올리지 못했다는 것이다. base direct Score `11.48`에서 최고 LoRA Score `29.11`, top2 full SFT Score `31.59`, LFM25/ToolBench full SFT Epoch1 Score `38.56`, Epoch2 Score `45.90`까지 올랐지만 Epoch3는 Score `43.57`로 내려갔다. 따라서 현재 KoHRM current best는 Epoch2다. Epoch3는 First Cmd를 `45.5%`로 소폭 올렸으나 Precision `0.4703`와 Valid JSON `61.7%`가 내려가 전체 command F1이 하락했다.

잘하는 것:
- LoRA와 full SFT 반응은 확실하다. 최고 LoRA는 base 대비 `+17.63`점이고, LFM25 Epoch2 full SFT는 최고 LoRA보다 다시 `+16.79`점 올랐다. HRM이 terminal JSON action을 못 배우는 모델은 아니다.
- LFM25 Epoch2는 First Cmd `44.9%`, Recall `0.5098`, Precision `0.5031`, Valid JSON `68.3%`까지 올라갔다. Epoch1의 First Cmd `37.0%`, Recall `0.4341`, Precision `0.4262`, Valid JSON `55.1%`에서 모두 개선됐고, 1.4B급 모델로 Qwen fast-continue fullconv Score `44.79`까지 넘어섰다.
- source group 기준 `data_querying` `0.6881`, `data_science` `0.4901`, `debugging` `0.4857`, `math` `0.4845`, `software_engineering` `0.4770`, `file_operations` `0.4710`이 상대적으로 강하다. 데이터 조회, 디버깅, 파일 확인, 수학/과학 계산처럼 다음 명령 패턴이 분명한 태스크에서 command set 복원이 좋아졌다.
- `terminal-tool-core-r64`, `terminal-comp-jsonfix-r64`, `comp-terminal-80m`이 모두 Score `28.44~29.11`에 몰려 있다. terminal 데이터와 tool/JSON 데이터가 함께 들어가면 특정 seed나 단일 데이터 운에만 기대지 않고 28점대 후반까지는 반복적으로 올라간다.
- JSON-fix 데이터는 형식 안정성에 도움을 준다. `behavior-jsonfix-r32`는 Valid JSON `71.0%`로 최고였고, 다른 LoRA들도 base보다 JSON 유효율이 크게 높았다.
- KoHRM은 작은 1.4B급 모델임에도 full SFT 후 Score `31.59`로 `LFM2.5-1.2B` SFT 2epoch `28.64`와 KoHRM LoRA들을 넘었다. 작은 모델 후보로는 가능성이 남아 있다.

못하는 것:
- command recall은 많이 좋아졌지만 최상위 대비 아직 낮다. Epoch2 Recall은 `0.5098`로 Qwen fast-continue `0.4701`보다 높지만, LFM2.5 1epoch Recall `0.5431`과는 차이가 있다. 필요한 `ls`, `cat`, `sed`, `python`, `pytest`, `grep/find` 흐름을 더 정확하고 넓게 복원해야 한다.
- First Cmd는 Epoch2에서 `44.9%`까지 올랐지만, LFM2.5 SFT 1epoch `49.5%`, ZAYA `51.8%`와는 아직 차이가 난다. 터미널 에이전트에서는 첫 `ls`, 파일 확인, 테스트 실행 방향이 틀리면 뒤 command F1도 무너진다.
- late bucket이 아직 병목이다. Epoch2는 early bucket `0.5458`, mid `0.4533`, late `0.3910`이다. Epoch1의 early `0.5016`, mid `0.3453`, late `0.3277`보다 전부 올랐지만, 후반 수정/검증/마무리 명령 생략은 아직 남아 있다.
- source group 약점은 `swe` `0.3590`, `data_processing` `0.4017`, `dependency_management` `0.4025`, `security` `0.4220`, `model_training` `0.4283`이다. 특히 repo 수정형 SWE, dependency 설치/검증, 학습 스크립트형 작업에서 command sequence를 덜 복원한다.
- JSON 유효율과 점수가 정비례하지 않는다. `behavior-jsonfix-r32`는 Valid JSON `71.0%`로 최고지만 Score는 `26.23`이다. 형식만 고쳐서는 부족하고, 정답 command set을 더 많이 맞혀야 한다.
- `synth,cot` 조건은 현재 direct 평가에서 오히려 손해였다. base `11.48`보다 synth,cot `10.36`이 낮아, 사고문을 덧붙이는 방식이 PrefixLM 출력 계약을 흐렸을 가능성이 크다.
- 속도는 아직 약점이다. KoHRM LoRA 로컬 evaluator는 `15~17 sec/step`이었고 full SFT export 4-shard wall 기준도 vLLM 모델들보다 느리다. 현재 HRM 구조는 vLLM 일반 causal/chat model fast path를 그대로 타지 못해 평가/반복 실험 비용이 크다.

왜 Ouro-1.4B-Thinking-Terminal-SFT보다 낮은가:
- Epoch2 이전에는 Ouro thinking SFT가 이미 terminal reasoning/output 분포에 더 잘 맞아 KoHRM top2 full SFT보다 근소하게 높았다. 하지만 LFM25 Epoch2 이후에는 KoHRM Score `45.90`으로 Ouro-1.4B-Thinking-Terminal-SFT Score `31.74`보다 `+14.16` 높다. 이제 Ouro 대비 약점은 점수가 아니라 평가/생성 속도와 vLLM fast path 미지원이다.
- KoHRM stage4d는 pretraining/continued training 쪽의 PrefixLM 모델이라 terminal next-action JSON을 기본 출력 습관으로 갖고 있지 않다. LoRA는 그 습관을 일부 보정하지만, base의 생성 분포 전체를 바꾸기에는 adapter 용량과 데이터 신호가 부족하다.
- TB2-lite는 `task_complete` 설명력이 아니라 `commands[].keystrokes` F1을 본다. KoHRM은 완성 판단과 설명 텍스트가 앞서거나, 짧은 command만 내는 경우가 있어 precision은 그럭저럭 나오지만 recall과 first command가 막힌다.
- 그래서 LoRA는 “가능성 확인”으로는 성공이고, top2 full SFT는 “Ouro 근접”까지는 성공이었다. LFM25 Epoch2는 “Qwen fast 초과”까지 성공했다. 남은 목표는 LFM2.5 50점대와의 `4~6점` 차이를 줄이는 것이다.

더 할 수 있는 것:
- top2 full SFT는 LoRA `29.11`을 넘어 Score `31.59`까지 올라갔다. 따라서 adapter 한계는 확인됐고, 다음 병목은 command precision, first action, 긴 후반 step 복원, HRM generation 속도다.
- `lfm25-terminal-toolbench` full SFT 2epoch는 Score `45.90`으로 성공했고, 3epoch는 Score `43.57`로 하락했다. 같은 데이터 추가 pass만으로는 더 밀어 올리지 말고, Epoch2를 기준으로 late-step/error-recovery 데이터, JSON-only target 강화, preference/DPO식 command selection 보강을 추가하는 쪽이 맞다.
- 데이터는 JSON-only assistant target을 더 강하게 해야 한다. `<think>`, 설명문, premature `task_complete=true`를 줄이고, 각 step의 첫 command와 검증 command를 더 많이 보존해야 한다.
- RL/DPO를 먼저 돌린다면 LFM2.5와 Qwen 2B가 우선순위다. 이유는 평가/serving이 vLLM fast path를 잘 타고, 이미 SFT 점수가 높아 reward signal을 빠르게 확인할 수 있기 때문이다. HRM은 바로 RL로 밀기보다 Epoch2 checkpoint를 고정한 뒤 JSON schema constrained decoding, `task_complete=false`/command array 보존 보상, late-step replay 보상, stop token/EOA 조기 종료를 먼저 안정화해야 한다.
- 평가 속도는 별도 개선 대상이다. vLLM이 바로 안 맞으면 HF export + 전용 batched PrefixLM generation을 더 최적화하고, stop token/EOA 조기 종료를 정확히 잡아 `max_tokens=1024` 전량 생성 낭비를 줄여야 한다.
- full SFT 결과가 좋아도 weak area 분석은 계속 필요하다. 현재 배포 기준은 Epoch2이고, 약점은 `swe`, `dependency_management`, `model_training`, late bucket, First Cmd의 LFM/ZAYA 대비 격차다. Epoch3에서 Valid JSON/Precision이 꺾였으므로 추가 단순 epoch보다 데이터 구성과 출력 contract 보강을 우선한다.

### 점수 해석: 잘 된 것과 안 된 것

핵심 기준은 단순 JSON 유효율이 아니라 `Cmd F1`이다. 이번 태스크는 다음 행동 명령을 얼마나 많이 맞히는지가 중요하므로, `Valid JSON`이 높아도 recall이 낮으면 순위가 내려가고, JSON 유효율이 낮아도 command coverage와 first command가 강하면 점수가 오른다.

잘 된 모델:

- `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`가 Score `52.30`으로 전체 1위다. Precision `0.5854`, Recall `0.5431`, First Cmd `49.5%`, Valid JSON `76.9%`라 command coverage와 형식 안정성을 동시에 끌어올렸다.
- 같은 run의 `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-2Epoch`는 Score `50.48`로 전체 2위다. 1epoch보다 약간 낮지만 `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF:UD-Q4_K_M` Score `49.97`보다 `+0.51`, `Zyphra/ZAYA1-74B-preview` Score `48.15`보다 `+2.33` 높다.
- `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF:UD-Q4_K_M`는 Score `49.97`로 전체 3위이자 비학습/외부 모델 최고다. Precision `0.6191`, Recall `0.4785`, First Cmd `49.2%`, Valid JSON `81.5%`라 command precision과 format 안정성이 동시에 높다. 다만 550B MoE GGUF라 평균 `26.688 sec/step`으로 매우 느리다.
- `Zyphra/ZAYA1-74B-preview`는 Score `48.15`로 전체 4위다. Precision `0.6196`, Recall `0.5017`, First Cmd `51.8%`, Valid JSON `74.6%`라 여전히 SFT 없이 매우 강하고, Nemotron보다 훨씬 빠르다.
- `LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-Top2-Terminal-Tool-Merge-Epoch1`은 Score `31.59`로 KoHRM 계열 최고다. base direct `11.48` 대비 `+20.11`, 최고 LoRA `29.11` 대비 `+2.48`이고, First Cmd `24.8%`, Valid JSON `73.3%`라 full SFT가 LoRA보다 출력 형식과 첫 행동을 더 안정화했다.
- `KoHRM-Text-1.4B-stage4d + terminal-tool-core-r64 LoRA`는 Score `29.11`로 adapter 계열 최고다. base direct `11.48` 대비 `+17.63`이고, First Cmd `22.1%`, Valid JSON `63.4%`라 LoRA SFT만으로도 HRM PrefixLM을 terminal JSON action 형식으로 실제로 이동시킨 것이 확인됐다.
- `gemma-4-26B-A4B-it` native SFT 2epoch는 Score `39.56`으로 Gemma 계열 최고다. Precision `0.4702`, Recall `0.4808`, First Cmd `40.6%`라 명령 집합을 넓게 맞힌다. Valid JSON은 `17.2%`로 낮지만, 실제 command F1이 강하다.
- 같은 26B-A4B-it 1epoch도 Score `38.12`, Recall `0.4787`, First Cmd `42.6%`로 강하다. 2epoch가 `+1.44` 오른 이유는 precision이 `0.4405 -> 0.4702`로 올라가면서 command 과다/오답이 줄었기 때문이다.
- `gemma-4-31B-it` native 1epoch는 Score `35.55`, Valid JSON `65.3%`다. 26B-A4B-it보다 format 안정성은 훨씬 좋지만 Recall `0.3776`이라 command coverage가 낮아 상위권에는 못 들어갔다.
- `Qwen/Qwen3.5-397B-A17B-FP8`은 Score `37.81`로 전체 12위, `Qwen/Qwen3.5-122B-A10B-FP8`은 `37.28`로 전체 14위다. 둘 다 Valid JSON `84~86%`, Precision `0.51`대로 형식과 정밀도는 좋다.
- 기존 요청 외부 모델 중 `Zyphra/ZAYA1-74B-preview`는 최고이고, GGUF/vLLM 완료분 다음 상위는 `DavidAU/Qwen3.6-40B Deckard...:Q4_K_M` Score `37.09`다. `unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL`은 Score `36.29`, Valid JSON `80.5%`, Precision `0.4813`으로 안정적이며 기존 `Qwen/Qwen3.5-27B` Score `36.30`과 사실상 동급이다.
- `Jackrong/Qwopus3.6-35B-A3B-v1-GGUF:Q4_K_M`과 `DavidAU/Qwen3.6-27B-Heretic...:Q4_K_M`은 각각 Score `35.45`, `35.47`이다. 둘 다 Valid JSON `80%` 안팎이고 precision은 높지만 recall이 `0.3287~0.3397`이라 26B-A4B-it native처럼 많은 명령을 잡지는 못한다.
- `MiniMaxAI/MiniMax-M2.7`은 Score `32.29`다. vLLM `tp=8`, expert parallel, FP8로 303-step full run을 정상 완료했고, Valid JSON `70.6%`, Precision `0.4492`라 형식과 정밀도는 중간 이상이다. 다만 Recall `0.2940`, First Cmd `13.5%`, 평균 예측 command `4.08`개로 너무 짧고 보수적으로 답해 상위권에는 못 들어갔다.

잘 안 된 모델:

- 31B-it native 2epoch는 Score `32.57`로 1epoch `35.55`보다 `-2.98` 낮다. Valid JSON은 `61.7%`로 여전히 높지만 Precision `0.4417`, Recall `0.3396` 모두 1epoch보다 낮아졌다. 큰 dense it 모델은 2epoch에서 format은 유지해도 action recall이 떨어진 것으로 본다.
- 31B base native는 2epoch `28.96`, 1epoch `21.08`이다. 2epoch가 `+7.88` 개선됐지만 it 모델과 격차가 크다. base 모델은 terminal assistant prior가 약해 SFT를 해도 JSON/명령 정책을 충분히 못 따라온다.
- 26B-A4B base native는 1epoch `18.66`, 2epoch `19.52`로 낮다. MoE 크기 자체보다 instruction prior 부재가 더 큰 병목이다. 같은 26B-A4B라도 it 계열은 `38~39점대`, base는 `18~19점대`라 차이가 `약 20점` 난다.
- 작은 Gemma base도 마찬가지다. E4B-it native 2epoch는 `34.98`인데 E4B base native 2epoch는 `18.47`, E2B-it native 1epoch는 `25.70`인데 E2B base native 2epoch는 `16.22`다. base 모델은 규모와 무관하게 terminal JSON command 형식 습득이 약하다.
- `Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M`은 Score `28.21`, Valid JSON `53.8%`로 Qwen GGUF 계열보다 낮다. Gemma 계열 prompt/template을 맞췄는데도 Recall `0.2506`이라 행동 명령을 넓게 복원하지 못한다.
- `MiniMaxAI/MiniMax-M2.7`은 229B급 MoE이고 H200 8장 vLLM에서 VRAM을 GPU당 약 `134GB`까지 잘 썼지만, 점수는 `32.29`에 그쳤다. 약점은 명확하다. `code` F1 `0.0908`, `swe` `0.2185`, `data_processing` `0.2623`, `data_querying` `0.2686`이 낮고, First Cmd도 `13.5%`라 첫 행동 선택이 약하다. 강한 쪽은 `system_administration` `0.4771`, `dependency_management` `0.4695`, `file_operations` `0.4147`이다. 즉 운영/패키지/파일 명령은 괜찮지만 코드/SWE/데이터 처리 next-action에는 약하다.
- `KoHRM-Text-1.4B-FullSFT-Top2-Terminal-Tool-Merge-Epoch1`은 Score `31.59`까지 올랐지만 상위권과의 차이가 명확하다. Precision `0.3859`는 Ouro-1.4B-Thinking-Terminal-SFT `0.4062`보다 낮고, Recall `0.3415`는 LFM2.5 1epoch `0.5431`, Qwen3.5-2B SFT 2epoch `0.4101`보다 낮다. 즉 JSON을 만들고 첫 명령을 맞히는 비율은 개선됐지만, reference가 요구하는 command set을 충분히 넓고 정확하게 복원하지 못한다. 실패 패턴은 후반 step에서 검증/수정 명령을 줄이거나, 설명/완료 판단이 앞서고, `ls/cat/sed/python/pytest` 흐름 중 일부를 생략하는 쪽이다.
- `sapientinc/HRM-Text-1B`는 Score `0.40`으로 전체 98위다. 실행은 8GPU data parallel shard로 정상 완료했지만, Valid JSON `4.3%`, 평균 예측 command `0.41`개, First Cmd `1.3%`라 TB2-lite가 요구하는 `commands[].keystrokes` 형식을 거의 생성하지 못했다. 이 결과는 새 KoHRM stage4d LoRA가 아니라 pre-alignment 공개 HRM baseline을 terminal next-action JSON agent로 바로 쓰기 어렵다는 신호로 본다.
- `DeepSeek-V4-Pro`는 1.6T params, 49B activated, 384 experts MoE 모델이다. **총 4회 평가**를 진행했다:

  **평가 이력:**

  | Run | Mode | Temp | max_tokens | Steps | Score | Valid JSON | Prec | Recall | Exact | Sec/Step | 비고 |
  |-----|------|------|-----------|-------|-------|-----------|------|--------|-------|----------|------|
  | 1 | thinking | 1.0 | 1024 | 301/303 | 26.66 | 31.0% | 0.3770 | 0.2404 | 23.3% | 442 | 크래시 종료 |
  | 2 | chat | 0.0 | 1024 | 175/303 | 35.40 | 52.6% | 0.4872 | 0.3336 | 29.7% | 376 | 완료 |
  | 3 | chat | 0.3 | 4096 | 86/303 | 36.50 | 59.3% | 0.4961 | 0.3995 | 37.2% | 814 | 속도 2.2배 |
  | **4** | **chat** | **0.0** | **4096** | **162/303** | **40.29*** | **44.7%** | **0.5752** | **0.4112** | **45.1%** | **689** | **진행 중, 최고** |

  *Run 4는 162/303 step 기준 (2026-05-23 07:09, 53.1% 완료, ETA ~27시간)

  **파라미터 분석 결과 (4회 평가 확정):**
  - **t=0.3은 Pro에게 역효과**: Run 2(t=0.0) 동일 71 step Score 42.86 → Run 3(t=0.3) 39.72 (-3.14). 온도를 올리면 Valid JSON은 비슷(71.8→70.4%)하지만 Precision 하락(0.6193→0.5389)으로 명령 품질이 저하. Pro는 384 expert MoE라 greedy(t=0.0)가 더 안정적.
  - **m=4096 + t=0.0이 최적 조합**: Run 4가 동일 71 step 기준으로 Score 40.87, Valid JSON 67.6%, Precision 0.5752, Recall 0.4112로 **종합 최고**. Run 2(t=0.0, m=1024) 대비 Score -1.99지만 Valid JSON -4.2%p, Recall +0.0299로 m=4096의 invalid step 손실을 줄이는 효과가 확인됨.
  - **m=4096의 효과**: Invalid step의 F1이 Run 2 35.96 → Run 4 34.17로 비슷하지만, 전체 Valid JSON% 향상으로 invalid step 비율 감소. 아직 71 step으로 mid/late bucket 분석 부족.
  - **thinking mode는 최악**: t=1.0+thinking에서 Valid JSON 31%. thinking token이 JSON을 깨먹음. 단 Valid JSON step의 F1은 0.4750으로 높음.

  **동일 71 step 교차 비교:**

  | Run | Score | Valid | Prec | Recall | Exact | Valid F1 | Invalid F1 |
  |-----|-------|-------|------|--------|-------|----------|------------|
  | Run1 (think) | 30.55 | 39.4% | 0.4373 | 0.2626 | 31.0% | 47.50 | 19.52 |
  | Run2 (t=0/m1k) | **42.86** | **71.8%** | **0.6193** | 0.3813 | 45.1% | 45.56 | 35.96 |
  | Run3 (t=0.3/m4k) | 39.72 | 70.4% | 0.5389 | 0.4115 | 39.4% | 46.62 | 23.30 |
  | Run4 (t=0/m4k) | 40.87 | 67.6% | 0.5752 | **0.4112** | **45.1%** | 44.08 | **34.17** |

  Run 4는 Run 2 다음으로 높은 Score지만, Precision 0.5752, Recall 0.4112로 균형 잡힌 성능. Invalid step F1 34.17은 4개 Run 중 최고라 m=4096이 truncated JSON 손실을 줄이는 효과가 확인됨.

  **최적 파라미터**: chat mode, t=0.0, m=4096 (Run 4). 근거:
  1. t=0.0이 Precision/Exact 최고 (Run 2, 4에서 증명)
  2. m=4096이 Invalid step F1 향상 (34.17 vs 19.52 thinking, 23.30 t=0.3)
  3. 전체 Recall 최고 (0.4112, t=0.3 Run 3의 0.4115와 사실상 동급)
  4. 다만 속도는 Run 2(376 s/step) 대비 1.83배 느림 (689 s/step)

  **Pro 강점 (Run 4, 71 step 기준):**
  - `security` F1 `0.6769` — 파일 탐색/설정 분석, Valid JSON 80%
  - `dependency_management` F1 `0.5187` — 패키지 설치/버전 관리
  - `debugging` F1 `0.5184` — 환경 분석→원인 파악→수정 흐름
  - `system_administration` F1 `0.5160` — 시스템 상태 확인/설정
  - `data_science` F1 `0.4870` — 데이터 분석 도구 활용
  - `file_operations` F1 `0.4611` — Valid JSON 100%, 형식 안정성 최고

  **Pro 약점:**
  - `model_training` F1 `0.1447`, `math` `0.1865`, `data_processing` `0.2803`
  - **JSON truncation**: mid bucket(61-70)에서 Valid JSON 10%로 급락. analysis가 대부분의 토큰을 소모해 commands가 잘림.
  - **과도한 분석**: Pro는 "문제를 풀이"하려는 경향이 강해 환경 탐색(ls, cat)보다 직접 코드 작성을 우선. reference와 형태가 달라 F1 하락.

  **Run 4 Score 예측:**
  - 71 step 기준 Score `40.87` → 162 step 기준 Score `40.29` (early+mid+late-1 혼합)
  - 버킷별: early(1-60) 42.62, mid(61-120) 35.01, late-1(121-162) 43.78
  - late-1 bucket에서 Score가 43.78로 회복. mid bucket의 하락이 일시적이었을 가능성.
  - **최종 예상 Score: 40~43** (late bucket 회복세 지속 시 Run2 최종 35.40 크게 상회 확실)

  **이 점수가 고평가인지 저평가인지:**
  - **크게 저평가되었다.** Pro의 실제 명령 품질(F1=0.4408 on valid steps)은 Flash 전체 평균(0.32)보다 37% 높고 GLM-5.1(0.42)과 동급.
  - Valid JSON step F1 기준 이론 상한 Score = `44.08` (GLM-5.1 41.68을 넘어 2위).
  - 기능적 동등 분석 결과, 16개 step에서 모델이 충분한 정보를 얻는 명령을 생성했으나 reference와 형식 차이로 F1 < 0.7. 실제 터미널 능력은 **~49점**으로 추정.

  **Run 4 기능적 동등 분석 (2026-05-23, 162/303 step 기준):**

  공식 Score(문자열 F1) 외에, 예측 명령이 참조 명령과 "문자열은 달라도 같은 목표를 달성"하는지 분석했다. GPU 없이 결과 JSONL 파일만으로 평가 가능하다. 162개 step 중 valid JSON 72개를 대상으로 수동 분석했고, invalid 90개는 truncation으로 인한 것이므로 기존 F1 그대로 유지했다.

  **3가지 Score 비교:**

  | 평가 방식 | Score | 의미 | 순위 |
  |-----------|------:|------|------|
  | 공식 (string F1) | **40.29** | reference와 문자열이 정확히 일치해야 점수 인정 | ~30위 |
  | 기능적 동등 보정 | **42.46** | 같은 목표를 달성하는 다른 명령도 인정 | **2위** (GLM-5.1 41.68 돌파) |
  | Valid-only 보정 | **48.19** | Valid JSON step만 기능적 동등 보정 적용 | **1위** (ZAYA1-74B 48.15 동급) |

  **평가 방법:**

  1. 결과 파일 `DeepSeek-V4-Pro.incr.shard00.jsonl`의 각 step에서 `ref_command_units`(참조 명령)와 `pred_command_units`(예측 명령)을 비교
  2. 문자열이 다르더라도 다음 기준으로 기능적 동등 판단:
     - **같은 파일 읽기**: `cat /app/file.py` vs `head -30 /app/file.py` → 같은 파일 내용 확인 → 동등
     - **경로 결합**: `cd /app && ls -la` vs `pwd`, `ls -la`, `cd /app` (분리) → 같은 작업 디렉토리 확인 → 동등
     - **직접 접근**: `cat /app/src/file.py` vs `cd /app && find . -name file.py && cat src/file.py` → pred가 더 효율적 → 동등 이상
     - **방어적 탐색**: `ls -la /app/output/ 2>/dev/null || echo "not found"` vs `ls -la /app/output/` → 같은 정보 + 에러 처리 → 동등
     - **결과 파일 생성**: `cat > /app/solution.txt << 'EOF' ...` vs `echo '...' > /app/solution.txt` → 같은 출력 파일 → 동등
  3. 판단 결과 원래 F1보다 높으면 보정 F1 적용, 낮으면 원래 F1 유지 (하향 조절 없음)
  4. **제한사항**: 303 step 중 162 step만 완료되어 전체 결과가 아님. 완료 시 Score 변동 가능.

  **주요 보정 사례:**

  | Step | Task | Group | 공식 F1 | 보정 F1 | 이유 |
  |------|------|-------|--------|---------|------|
  | 1 | task-049 | math | 0.011 | 0.550 | pred가 탐색 없이 직접 solution.py 작성. ref는 환경 확인만 했으나 pred가 더 앞선 행동 |
  | 1 | task-048 | math | 0.189 | 0.550 | `cat > solution.txt << 'EOF'` vs `echo '...' > solution.txt`. 같은 파일, 다른 작성법 |
  | 1 | task-047 | swe | 0.200 | 0.650 | pred가 `cat /app/src/file.py`로 직접 접근. ref는 `find`로 탐색 후 접근. 더 효율적 |
  | 1 | task-017 | sci_comp | 0.250 | 0.650 | pred가 `cat nbody_config.json`으로 핵심 설정 직독. ref는 ls 후 cat |
  | 1 | task-031 | dep_mgmt | 0.315 | 0.550 | pred가 `df -h && python3 --version && pip3 --version`으로 한 번에 env 체크 |
  | 1 | task-027 | sys_admin | 0.429 | 0.650 | pred가 services.conf, nginx-template.conf, health-status.json 모두 직독. ref와 동일 파일 |
  | 1 | task-005 | swe | 0.505 | 0.700 | pred가 ls+ls input+ls modules+cat project_config로 ref와 동일 구조 탐색 |
  | 1 | task-002 | swe | 0.667 | 0.800 | pred가 `cat legacy_makefile.txt` 직접 읽기. ref는 ls 후 cat. 동등+효율 |
  | 1 | task-028 | data_query | 0.667 | 0.850 | pred가 `head -100 movies.ttl`. ref는 `head -30`. 더 많은 정보 확인 |
  | 1 | task-003 | swe | 0.817 | 0.850 | pred가 spec 읽기+src/build 구조까지 추가 탐색. ref보다 더 많은 정보 수집 |
  | 1 | task-009 | debugging | 0.697 | 0.750 | pred가 runtime_logs, data_processor, input_data 모두 읽음. 핵심 파일 커버 |

  **구조적 저평가 원인 (162 step 분석으로 재확인):**

  1. **명령 경로 결합 (가장 빈번)**: Pro가 `cd /app && ls -la`처럼 명령을 합치면 reference의 개별 명령 3개(`pwd`, `ls -la`, `cd /app`)와 token overlap이 줄어든다. 기능은 동일하거나 더 효율적이지만 F1은 하락.
  2. **직접 파일 접근**: Pro가 파일 경로를 알면 `cat /app/src/file.py`로 바로 읽는다. Reference는 `find` → `cat` 순으로 탐색하지만, 실제 터미널에서는 Pro의 방식이 더 빠르다.
  3. **과도한 분석 → truncation**: 162 step 중 90개(55.6%)가 invalid JSON. analysis에 토큰을 많이 써서 commands가 잘리는 구조적 한계. Valid step F1 43.29 vs Invalid step F1 23.40로 20점 차이.
  4. **방어적 명령**: `ls -la /app/output/ 2>/dev/null || echo "not found"`는 실전에서 더 안전하지만, 문자열이 길어져 reference와의 overlap이 줄어든다.

  **버킷별 기능적 동등 분석:**

  | 구간 | 공식 Score | Valid% | 비고 |
  |------|-----------|--------|------|
  | early(1-60) | 42.62 | 78% | 탐색 명령이 ref와 유사. truncation 적음 |
  | mid(61-120) | 35.01 | 27% | truncation이 73%를 무효화. 능력은 좋으나 출력 공간 부족 |
  | late-1(121-162) | 43.78 | 21% | Score 회복. truncation은 여전하나 valid step 품질 높음 |

  mid 구간에서 공식 Score 35.01이지만, valid step만 보면 **43.4**로 early와 비슷한 수준이다. 즉 mid 하락은 모델 능력 저하가 아니라 **출력 길이 한계** 때문이다.

  **공정성 한계 및 확장 가능성:**

  이 기능적 동등 분석은 **DeepSeek-V4-Pro Run 4에만 적용**했다. 리더보드의 다른 모든 모델(ZAYA1-74B, GLM-5.1, Qwen3.5, Gemma SFT 등)은 공식 Score(string F1)만으로 순위가 매겨져 있다. 따라서:

  1. **Pro만 유리한 기준을 적용한 것**이다. 다른 모델도 같은 방식으로 분석하면 점수가 올라갈 수 있다. 특히 ZAYA1-74B, GLM-5.1 등 zero-shot 대형 모델은 reference와 다른 경로를 자주 선택하므로 기능적 동등 보정 시 상승폭이 클 가능성이 높다.
  2. **SFT 모델은 보정 효과가 작을 것**으로 예상된다. SFT 모델은 reference 형식을 학습했으므로 이미 reference와 비슷한 명령을 낸다. 기능적 동등 보정의 상향 폭은 zero-shot 모델에 비해 작을 것이다. 이는 SFT 모델의 공식 Score가 실력에 더 근접한다는 기존 분석과 일치한다.
  3. **순위 변동 가능성**: ZAYA1-74B에 같은 분석을 적용하면 48.15에서 더 올라갈 수 있으므로, Pro의 "~1위" 주장은 Pro에만 유리한 단일 모델 분석 결과로 해석해야 한다.
  4. **모든 모델에 동일 분석을 적용해야 공정한 비교**가 가능하다. 현실적으로 시간이 제약이 되어 Pro 1개 모델만 분석했다. 향후 전체 모델에 기능적 동등 분석을 일괄 적용하면 "기능적 동등 기준 리더보드"를 만들 수 있다.
  5. **162/303 step만 분석**했다. 나머지 141 step이 완료되면 점수가 변할 수 있으며, 특히 late bucket의 난이도가 높아지면 공식/보정 Score 모두 하락할 가능성이 있다.
- `DeepSeek-V4-Flash`는 최종 공식 MP4 경로에서 Score `32.22`로 전체 43위다. Precision `0.4511`, First Cmd `24.4%`는 중위권이지만 Valid JSON `44.2%`, Recall `0.3037`이라 필요한 명령을 충분히 넓게 복원하지 못했다. `dependency_management` F1 `0.5048`은 강하지만 `math` `0.1302`, `data_processing` `0.1496`, `code` `0.1688`이 낮다. 가장 큰 운영상 약점은 속도다. 평균 `178.033s/step`이라 정상 점수는 나왔지만 대량 평가/실서비스에는 현재 공식 MP4 경로가 너무 느리다.
- `stepfun-ai/Step-3.5-Flash`는 vLLM `tp=8`, expert parallel, BF16 원본 모델로 full 303-step을 정상 완료했지만 Score `18.80`으로 낮다. 가장 큰 문제는 출력 형식과 짧은 command set이다. Valid JSON은 `27.4%`, invalid JSON은 `220/303`이고, 평균 예측 command는 `2.60`개로 reference 평균 `38.42`개보다 훨씬 적다. Precision `0.2710`, Recall `0.1790`, First Cmd `13.9%`라 첫 명령 선택과 command coverage가 모두 약하다. `software_engineering` F1 `0.2919`, `data_science` `0.2637`, `dependency_management` `0.2397`은 상대적으로 낫지만 `code` `0.0846`, `model_training` `0.1143`, `math` `0.1245`, `swe` `0.1274`가 매우 낮다.

이번 결과에서 보이는 패턴:

- ZAYA1-74B-preview는 `early` bucket F1 `0.5909`가 특히 높다. 초반 탐색/환경 파악/첫 명령 선택에서 기존 모델들을 크게 앞선다.
- 그래도 약점은 있다. source group 기준 `code` F1 `0.2490`이 가장 낮고, `math` `0.3995`, `swe` `0.4164`, `data_processing` `0.4183`도 상대적으로 약하다. 전체 303개 중 JSON invalid가 77개, command F1 0점 step이 23개라 완벽한 모델은 아니다.
- ZAYA 74B를 제외하면 `it` 사전정렬 + native SFT가 가장 중요하다. 26B-A4B-it native는 학습 모델 중 1위지만 26B-A4B base native는 하위권이다.
- 큰 모델이라고 자동으로 오르지 않는다. 31B-it는 format 안정성은 좋지만 action recall이 낮아 26B-A4B-it보다 약하다.
- GGUF 인기 모델들은 JSON 안정성은 높다. 다만 TB2-lite에서는 shell command recall이 부족하면 35~36점대에서 멈춘다.
- Valid JSON과 Score는 같은 방향이 아니다. ZAYA1-74B-preview는 Valid JSON `74.6%`와 Score `48.15`가 같이 높지만, 26B-A4B-it native 2epoch는 Valid JSON `17.2%`로 낮아도 Score `39.56`이고, Qwen GGUF 계열은 Valid JSON `78~83.8%`라도 Score `35~37점대`다.
- **DeepSeek-V4-Pro의 생성 길이 문제**는 대형 모델의 공통 과제다. Pro는 analysis에 ~800토큰을 쓰고 commands에 도달하기 전에 1024토큰 한계에 걸린다. 1.6T MoE가 "생각"은 잘하지만 "행동"을 출력할 공간이 부족한 구조적 한계. max_new_tokens 4096으로 완화했지만 mid bucket(61-70)에서 Valid JSON이 10%로 급락하는 것으로 보아 여전히 긴 분석이 commands를 밀어내는 현상이 발생.

  **Run 4 10-step 트렌드 (Score / Valid JSON%):**
  ```
  Step  1-10: 37.05 / 90.0%
  Step 11-20: 44.81 / 90.0%
  Step 21-30: 38.02 / 100.0%
  Step 31-40: 49.25 / 100.0%  ← 최고 구간
  Step 41-50: 41.05 / 70.0%
  Step 51-60: 45.52 / 20.0%   ← Valid 급락, Score는 유지
  Step 61-70: 29.52 / 10.0%   ← Score 하락
  Step 71:    49.46 / 0.0%
  ```
  1-40 step까지 Valid JSON 95%, Score 42.28로 매우 강력. 41-60 step에서 Valid JSON 급락하지만 Score는 유지 (invalid F1이 높아서). 61-70 step에서 Score까지 하락. 이 패턴은 Run 2, 3에서도 동일하게 관찰됨.
- Step-3.5-Flash는 그 반대쪽 실패 패턴이다. vLLM이 정상 생성하고 일부 step에서는 F1 `0.8~1.0`도 나오지만, 전체적으로 터미널 transcript를 그대로 이어 쓰거나 요약문을 내는 비율이 높아 strict JSON이 깨지고, 명령 수가 너무 적어 Recall이 낮다. 이 모델은 raw reasoning/대화 능력보다 루트 `README.md` 방식의 structured terminal replay 적합성이 낮게 나온 케이스로 보는 것이 맞다.

ZAYA1-74B-preview가 크게 앞선 구체적 이유:

- 단순히 Next Action 보조 점수만 높은 것이 아니다. 공식 Score 기준인 Cmd F1이 `0.4815`라 직전 1위 Gemma 26B-A4B-it native 2epoch `0.3956`보다 `+0.0859` 높다. 100점 환산으로 `+8.59`점이다.
- Precision이 `0.6196`으로 기존 최고권보다 확실히 높다. Gemma 26B-A4B-it native 2epoch의 Precision `0.4702`보다 `+0.1494`라, 필요한 명령을 많이 찍기만 한 것이 아니라 틀린 명령을 덜 섞는다.
- Recall도 `0.5017`로 높다. Gemma 26B-A4B-it native 2epoch `0.4808`보다 `+0.0209`라 coverage도 더 넓다. Precision과 Recall이 동시에 오른 것이 이번 점수 급등의 핵심이다.
- First Cmd `51.8%`가 특히 크다. 기존 최고권 Gemma 26B-A4B-it native 2epoch `40.6%`, Qwen3.5-2B SFT 2epoch `33.0%`보다 높아서 첫 터미널 행동 선택이 매우 강하다.
- bucket별로는 early `0.5909`, mid `0.4399`, late `0.4307`이다. 초반 탐색 단계에서 압도적이고, 중후반으로 갈수록 이점이 줄어든다. 즉 “처음 뭘 해야 하는지”를 매우 잘 잡는 모델이다.
- 잘하는 source group은 `dependency_management` F1 `0.5697`, `file_operations` `0.5527`, `data_querying` `0.5392`, `security` `0.5285`, `scientific_computing` `0.5252`, `model_training` `0.5164`다. 패키지/파일/쿼리/보안/과학 계산/학습 루틴 명령에서 강하다.
- 틀리는 부분도 명확하다. `code` F1 `0.2490`이 가장 낮고, `math` `0.3995`, `swe` `0.4164`, `data_processing` `0.4183`이 약하다. 코드 편집형 작업, 수학 풀이형 셸 작업, SWE command sequence, 데이터 전처리 명령에서는 아직 흔들린다.
- JSON 안정성은 강하지만 완벽하지 않다. Valid JSON `74.6%`라 상위권이지만 303 step 중 77 step은 JSON이 깨졌고, command F1 0점 step도 23개 있다. 실사용 최고 후보지만 output format guard나 후처리는 여전히 필요하다.

ZAYA1-74B-preview의 점수는 오히려 저평가됐을 가능성이 있다:

- 이 벤치마크는 reference command를 얼마나 비슷하게 재현했는지를 보는 replay 평가다. 모델이 더 좋은 대체 명령, 더 짧은 해결 경로, 더 안전한 사전 확인 명령을 냈어도 reference와 다르면 command F1에서 깎인다.
- ZAYA1-74B-preview는 First Cmd `51.8%`라 첫 행동 선택이 매우 강하다. 첫 명령은 agent가 실제로 작업을 시작할 때 체감 품질에 크게 영향을 주는데, 전체 Score는 이후 command set overlap까지 평균내므로 이 장점이 완전히 반영되지는 않는다.
- Precision `0.6196`은 모델이 낸 명령 중 맞는 비율이 매우 높다는 뜻이다. Recall `0.5017`도 높지만, reference에 있는 모든 명령을 다 복원하지 못하면 점수가 깎인다. 실제 환경에서는 reference의 모든 명령을 따라 할 필요 없이 더 짧게 해결할 수 있으므로 이 부분은 보수적으로 평가된 값일 수 있다.
- JSON invalid 77개가 점수를 깎지만, command parser가 뽑아낸 명령 품질은 여전히 압도적이다. 실사용에서는 structured output repair, retry, schema forcing, tool-call wrapper를 붙이면 이 손실은 줄일 수 있다.
- 특히 early bucket F1 `0.5909`는 실전 agent 품질을 강하게 시사한다. 초반 탐색/환경 파악/첫 방향 설정은 이후 모든 행동을 좌우하는데, 이 구간에서 ZAYA가 기존 모델들을 크게 앞선다.
- 따라서 `48.15`는 “TB2-lite reference replay 기준의 하한에 가까운 강한 점수”로 보는 편이 맞다. 일반적인 터미널 에이전트 운용에서는 대체 경로 인정, 실행 결과 기반 채점, output repair를 넣으면 체감 성능은 표 점수보다 더 좋을 가능성이 크다.

### 점수 저평가 분석: TB2-lite Score vs 실제 터미널 능력 (2026-05-21 추가)

TB2-lite Score는 "reference 명령을 얼마나 비슷하게 재현했는가"를 측정하지, "실제로 터미널에서 얼마나 유용한가"를 직접 측정하지 않는다. 이 차이가 모든 모델의 Score를 실제 터미널 능력보다 낮게 만든다. 하지만 **모델마다 저평가 정도가 다르며**, 이 차이를 이해하면 순위를 올바르게 해석할 수 있다.

#### 1. 저평가의 구조적 원인

**원인 A: 다른 경로로 같은 목표**

모델이 reference와 다르지만 기능적으로 동등하거나 더 나은 명령을 내도 F1이 깎인다.

| 예측 명령 | 참조 명령 | 실제 효과 | F1 손실 |
|-----------|-----------|-----------|---------|
| `ls -la /app/webapp/` | `cd /app && ls -la webapp/` | 동일한 파일 목록 | -0.40 |
| `cat > /app/solution.txt << 'EOF' ...` | `echo '...' > /app/solution.txt` | 동일한 파일 생성 | -0.81 |
| `pip install psycopg2==2.9.0` | `apt-get install libpq-dev && pip install psycopg2` | 더 효율적 | -0.76 |
| `cd /app && ls -la` | `pwd && ls -la && cd /app` | 동일 정보 + 더 짧음 | -0.71 |
| `python3 -c "import yaml"` | `pip3 list \| grep -i yaml` | 동일 확인 | -0.98 |

DeepSeek-V4-Pro Run 4 기준, F1 < 0.3인 27개 step 중 10개가 "더 짧은 경로"로 인한 손실이다. 모델이 `cd /app && ls -la`로 합쳐서 내면 reference의 `pwd`, `ls -la`, `cd /app` 세 명령과 token overlap이 줄어든다.

**원인 B: JSON 생성 한계로 인한 명령 누락**

analysis가 대부분의 토큰 예산을 소모해 commands 배열이 잘리면 JSON이 invalid가 된다.

```
Valid JSON steps (49):  avg F1 = 44.48
Invalid JSON steps (27): avg F1 = 35.02
차이: 9.46점
```

Run 4에서 F1 < 0.3인 27개 step 중 14개(51.9%)가 invalid JSON이다. 모델이 명령을 "알고" 있어도 출력 공간이 부족하면 0점 처리된다. 이것이 대형 reasoning 모델(Pro, ZAYA1)에게 특히 불리하다.

**원인 C: 명령 수 불일치**

Run 4 평균 예측 명령 6.1개 vs 평균 참조 명령 8.2개 (비율 0.74). 50개 step 중 18개(36%)가 참조의 절반도 안 되는 명령을 생성한다. 더 적은 명령으로 충분한 정보를 얻을 수 있어도, reference의 모든 중간 단계를 재현하지 않으면 Recall이 깎인다.

#### 2. SFT가 Score를 올리는 진짜 이유

같은 기반 모델의 Base vs SFT 비교:

| 모델 | Base Score | Best SFT Score | 차이 | 증가율 |
|------|-----------|---------------|------|--------|
| gemma-4-26B-A4B-it | 28.51 | 39.56 (SFT 2ep) | +11.05 | +39% |
| gemma-4-31B-it | 26.33 | 35.55 (SFT 1ep) | +9.22 | +35% |
| gemma-4-E4B-it | 19.36 | 34.98 (SFT 2ep) | +15.62 | +81% |
| gemma-4-E2B-it | 17.40 | 25.70 (SFT 1ep) | +8.30 | +48% |

SFT 개선의 핵심은 Precision이 아니라 **Recall** 향상에 있다:

```
gemma-4-26B-A4B-it:
  Base:     Prec 0.4057, Recall 0.2643
  SFT 2ep:  Prec 0.4702, Recall 0.4808
  → Recall 82% 향상, Precision 16% 향상
```

이것은 모델이 "터미널을 더 잘하게 된 것"이 아니라 **"reference의 풀이 방식을 학습한 것"**이다. SFT는 모델에게 "이 상황에서는 이 순서로 이 명령을 써라"는 패턴을 가르친다. 따라서 SFT 모델의 Score 향상 중 상당 부분은 **실력 향상이 아니라 format alignment**다.

증거: 과도한 SFT는 Score를 하락시킨다. gemma-4-31B-it은 SFT 1ep에서 35.55, 2ep에서 32.57로 -2.98 하락. Reference 패턴에 overfitting하면 novel 상황에서 유연성이 떨어진다.

#### 3. 모델별 저평가 정도 추정

| 모델 유형 | 저평가 정도 | 근거 |
|-----------|-----------|------|
| **Terminal SFT 모델** (gemma-4-26B SFT 등) | 낮음 (~0~3점) | Reference 형식을 학습하여 Score와 실력이 근접 |
| **Zero-shot 대형 모델** (ZAYA1-74B, GLM-5.1) | 중간 (~3~5점) | 높은 추론 능력이지만 reference와 다른 경로 선택 |
| **대형 Reasoning MoE** (DeepSeek-V4-Pro) | 높음 (~8~10점) | 과도한 분석→truncation, 다른 해결 경로, combined commands |
| **보수적 모델** (MiniMax-M2.7) | 낮음 (~1~2점) | 적은 명령으로 정밀하게 답, Recall 손실은 실제 약점과 일치 |
| **Base 비instruct 모델** | 낮음 (~0점) | Format 이해 부족이 실제 약점과 일치 |
| **비-chat PrefixLM/pre-alignment 모델** (HRM-Text-1B) | 낮음 (~0~1점) | `commands[].keystrokes` 출력 계약을 거의 따르지 않아 후처리로 살릴 명령이 적음 |

DeepSeek-V4-Pro의 경우 (162/303 step, 2026-05-23 업데이트):
- 공식 Score 40.29 → 기능적 동등 보정 42.46 (+2.18) → Valid-only 보정 48.19 (+7.90)
- **공식 Score 대비 ~8~10점 저평가**. Truncation(55.6% invalid) + 문자열 매칭 한계가 주원인.
- 20개 valid step에서 기능적 동등 인정. 특히 직접 파일 접근, 명령 결합, 방어적 탐색이 빈번.
- **실제 터미널 능력 추정: 42~48점** (GLM-5.1과 ZAYA1 사이 수준. 공식 2위~1위)

#### 4. 순위 해석 가이드

1. **순위 자체는 유효하다**: 모든 모델이 같은 기준으로 평가되므로 상대 비교는 의미 있다.
2. **Score 절대값은 하한선**: 실제 터미널 능력은 Score보다 항상 높거나 같다.
3. **SFT 모델 vs Zero-shot 비교 시 주의**: SFT 모델은 "reference 재현"에 특화되어 있어, zero-shot 대형 모델과의 Score 차이가 실제 능력 차이를 과대평가할 수 있다.
4. **가장 공정한 비교는 zero-shot끼리**: ZAYA1(48.15), GLM-5.1(41.68), DeepSeek-V4-Pro(41.53), Qwen3.5-397B(37.81)가 SFT 없이 낸 Score는 실력에 더 가깝다.
5. **Valid JSON%가 Score를 결정하는 구조**: Valid JSON step의 F1이 모델의 실제 명령 품질을 더 잘 반영한다. Pro의 Valid-only 보정 Score 48.19은 공식 Score 40.29보다 실력에 가깝다.

### GLM-5.1 API 평가 결과 (2026-05-15 추가)

GLM-5.1을 Anthropic 호환 API(z.ai) 경유로 TB2-lite full 303-step replay 평가를 실행했다. 별도 로컬 GPU 로딩 없이 API 호출만으로 평가를 완료했다.

평가 조건:
- 백엔드: `anthropic-api` (z.ai GLM-5.1 endpoint, Messages API)
- `temperature=0.0`, `max_tokens=2048`, `concurrency=4`
- 프롬프트: replay dataset의 `messages` 필드를 그대로 전달 (chat template 없이 API가 처리)
- 평가 시간: 999.2초 (3.298s/step)
- 결과 파일: `tb2_lite/results/glm51_api/GLM-5.1-API-full.json`

Score `41.68`로 현재 전체표에서는 LFM2.5 ToolBench SFT 1epoch/2epoch와 ZAYA1-74B-preview 다음 상위권에 올랐다. 기존 Gemma 최고인 `gemma-4-26B-A4B-it SFT 2epoch` `39.56`보다 `+2.12` 높다.

잘하는 것:
- Valid JSON `90.1%`로 85개 모델 중 JSON 안정성 1위. ZAYA1-74B `74.6%`, Qwen3.5-2B SFT `82.2%`보다 높다.
- Precision `0.5377`로 ZAYA1-74B `0.6196` 다음으로 높다. 낸 명령 중 맞는 비율이 매우 좋다.
- Recall `0.4007`로 기존 3위권 `0.4808~0.4101`과 비견한다.
- bucket별로 early `0.4425`, mid `0.4117`, late `0.4002`로 후반까지 안정적이다.
- debugging F1 `0.5222`, model_training `0.5172`, dependency_management `0.5165`, system_administration `0.5017`이 강하다.
- API 호출만으로 평가가 가능해 로컬 GPU 불필요, Sec/Step `3.298`으로 ZAYA1 `4.151`보다 빠르다.

못하는 것:
- First Cmd `24.1%`로 ZAYA1 `51.8%`, gemma-26B SFT `40.6%`보다 낮다. 첫 행동 선택이 상대적으로 약하다.
- math F1 `0.2556`, swe `0.3030`, scientific_computing `0.3213`이 약하다. 수학/SWE/과학 계산 영역에서 reference 재현율이 낮다.
- code F1 `0.3533`로 경쟁 모델 대비 중간 수준이다.
- data_science F1 `0.3358`, data_processing F1 `0.4098`로 데이터 분석/처리 영역도 개선 여지가 있다.

이 점수가 고평가인지 저평가인지:
- 이 점수는 **공정한 평가에 가깝지만, 약간 저평가될 수 있다.** 이유는 다음과 같다.
- 첫째, 로컬 모델들은 `apply_chat_template`으로 프롬프트를 구성하지만, GLM-5.1은 Messages API가 자체적으로 포맷팅을 처리한다. 프롬프트 포맷이 다르면 모델의 출력 분포가 달라질 수 있다. 특히 GLM 시리즈는 Qwen 계열 chat template과 다를 수 있어, replay dataset이 Qwen 기준으로 작성된 점은 GLM-5.1에 불리하게 작동할 수 있다.
- 둘째, API 호출의 `temperature=0.0`은 greedy 디코딩이지만, 서버 측 추가 샘플링/필터링이 있을 수 있다. 로컬 vLLM/transformers의 `do_sample=False`와 완전히 동일하지 않을 수 있다.
- 셋째, `max_tokens=2048`인데 일부 긴 스크립트(heredoc) 스텝에서 잘릴 수 있다. 로컬 평가는 보통 `max_tokens=1024`를 썼지만, API는 2048로 설정해 이 부분은 오히려 유리했다.
- 넷째, 이 평가는 **Terminal SFT를 전혀 하지 않은 zero-shot**이다. ZAYA1-74B를 제외하면 3위~84위 대부분의 상위 모델이 Terminal SFT를 받았다. SFT 없이 API만으로 2위에 오른 것은 GLM-5.1의 기본 instruction-following과 code prior가 매우 강하다는 의미다.
- 반대로 고평가 근거도 있다. API 평가는 서버 측에서 모델 버전/가중치가 변경될 수 있고, 로컬 평가와 완전히 같은 체크포인트임을 보장할 수 없다. 또한 `concurrency=4`로 병렬 호출했을 때 서버 부하로 인한 응답 품질 저하가 있을 수 있다.
- 종합하면 **약간 저평가~공정한 평가**로 본다. SFT 없이 2위, Valid JSON 최고, Precision 2위는 실력이 맞고, First Cmd 24.1%와 math/swe 약점도 실제 한계다. Terminal SFT를 적용하면 Score가 추가로 오를 가능성이 크다.

### 실제 최고 모델 판단

현재 이 벤치마크 기준 최고 모델은 `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`다. Score `52.30`으로 전체 1위이고, 같은 run의 2epoch `50.48`이 전체 2위다. 비학습/외부 모델 최고는 `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF:UD-Q4_K_M` Score `49.97`이며, LFM 1epoch가 Nemotron보다 `+2.33` 높다.

다만 실전 배치 기준으로는 품질, API 운용, 속도/비용의 선택이 갈린다.

- 품질 우선이면 현재는 LFM2.5 ToolBench SFT 1epoch다. Score `52.30`, Recall `0.5431`, Sec/Step `0.087`로 품질과 속도가 동시에 좋다.
- 비학습/외부 모델 품질 우선이면 `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF:UD-Q4_K_M`다. Score `49.97`, Valid JSON `81.5%`로 가장 높다. 속도까지 같이 보면 `Zyphra/ZAYA1-74B-preview`도 유력하다. Score `48.15`, First Cmd `51.8%`, Sec/Step `4.151`이라 Nemotron보다 훨씬 빠르다.
- API 운용 우선이면 `GLM-5.1`이 대안이다. Score `41.68`, Valid JSON `90.1%`로 포맷 안정성 최고, API 호출만으로 로컬 GPU 불필요다.
- 비용/속도/안정성까지 보면 `Qwen3.5-2B SFT 2Epoch`도 여전히 강하다. Score `39.52`, Sec/Step `0.081`이라 매우 빠르다.
- Nemotron-3 Ultra, ZAYA, GLM-5.1을 제외한 공개 비학습 모델 중 최고는 `Qwen/Qwen3.5-9B`다. Score `38.10`이고, `nvidia/Nemotron-Terminal-32B` `38.09`, `nvidia/Nemotron-Terminal-14B` `37.70`보다 높다.

이 결과가 특이한 이유:

- LFM2.5 ToolBench SFT 1epoch/2epoch가 Score `52.30`/`50.48`로 기존 최고권을 크게 넘은 것이 최신 주요 변화다.
- `GLM-5.1 API`는 Score `41.68`, Valid JSON `90.1%`로 SFT 없이 기존 Gemma 최고 `39.56`를 제쳤다.
- 그 아래 학습 모델 중에서는 MoE인 `Gemma 26B-A4B-it` SFT와 순수 2B dense인 `Qwen3.5-2B SFT`가 강하다. 크기만 보면 26B-A4B가 압도적으로 커 보이지만 active parameter 관점에서는 A4B라 실제 실행 성격은 작고 빠른 MoE에 가깝다.
- 2B SFT가 26B/31B 계열과 경쟁하는 이유는 데이터/포맷 적합도가 모델 크기보다 더 중요했기 때문이다. `Qwen3.5-2B SFT`는 Valid JSON `82.2%`, Precision `0.5082`라 포맷과 명령 선택이 안정적이다.
- Gemma 26B-A4B-it native는 Valid JSON은 낮지만 Recall이 매우 높다. 즉 JSON wrapper는 자주 깨져도, 실제 shell action 단위는 많이 맞힌다. 이 벤치마크의 score가 command F1이므로 LFM을 제외한 Gemma 계열에서는 최고다.
- 학습하지 않은 모델 중에서는 `GLM-5.1` Score `41.68`이 압도적이고, 그 다음은 `Qwen/Qwen3.5-9B` Score `38.10`이다. 둘 다 SFT 없이 기본 instruction/code prior만으로 상위권에 올랐다.

유명 모델과 외부 GGUF 모델이 기대보다 낮은 이유:

- `Nemotron-Terminal-32B`는 First Cmd `40.3%`로 첫 행동은 강하지만 Score `38.09`로 Qwen 9B와 거의 동률이다. First Cmd가 높아도 전체 command set recall/precision 평균에서 확실히 앞서지는 못했다.
- `Nemotron-Terminal-14B`는 Score `37.70`으로 강하지만, Qwen 9B `38.10`보다 낮다. terminal-tuning이 되어 있어도 TB2-lite의 replay command 분포와 완전히 같은 것은 아니다.
- `Qwopus3.6-35B-A3B GGUF`와 `Qwen3.6-27B Heretic GGUF`는 Score `35.45~35.47`이다. Valid JSON은 `78.9~81.2%`로 좋지만 Recall이 `0.3287~0.3397`에 머문다. 즉 답 형식은 잘 지키지만 필요한 명령을 충분히 많이 복원하지 못한다.
- `unsloth/Qwen3.6-27B-MTP-GGUF`는 Score `36.29`로 외부 요청 모델 중 현재 최고지만, 기존 `Qwen/Qwen3.5-27B` `36.30`과 거의 같다. MTP/추론 속도 이점은 있을 수 있지만, 이 점수만 보면 command 품질이 크게 뛰지는 않는다.
- `supergemma4-26b-uncensored GGUF`는 Score `28.21`이다. Gemma 계열 prompt를 맞춰도 Recall `0.2506`, Valid JSON `53.8%`라 command-following과 포맷 안정성이 둘 다 부족했다.

실패/저조 결과의 원인 판단:

- 기존 HF-FSDP 31B SFT 1/2epoch는 둘 다 Score `0.00`이다. 같은 31B 계열이라도 native SFT 31B-it는 `35.55`까지 나오므로, 31B 모델 자체 한계가 아니라 기존 HF-FSDP checkpoint/export 또는 학습 포맷이 깨진 실패로 본다.
- `DeepSeek-V4-Flash`의 최종 Score는 `32.22`다. vLLM/FP8 경로에서는 MARLIN shared memory, backend kernel 호환, BOS token 반복 문제가 있었지만, 공식 converted MP4 inference 경로는 `2 x tp=4` shard로 full 303 replay를 끝까지 완료했다.
- DeepSeek 공식 MP4 완료 run은 `2 x tp=4` shard로 GPU 0-3/4-7을 나눠 사용했고, 각 GPU는 마지막 구간에서 약 `127~133GB` VRAM을 점유했다. Load `15.6s`, generation `53944.1s`, 평균 `178.033s/step`으로 매우 느렸지만, vLLM/FP8 실패와 달리 실제 JSON/command 출력이 생성됐다. 병목은 GPU 부족이 아니라 공식 MP4 경로의 generation 속도와 낮은 JSON 안정성/recall이다.
- 따라서 DeepSeek V4 Flash는 더 이상 0점 실패로 보지 않는다. 다만 평균 `178.033s/step`이라 속도는 압도적으로 느리고, Valid JSON `44.2%`, Recall `0.3037`이 낮아 점수는 MiniMax와 거의 같은 중위권에 머문다. 품질 자체는 보수적으로는 정상 완료 결과, 운영 관점에서는 공식 MP4 경로가 너무 느려 실전 평가/배포 비용이 큰 결과로 해석한다.
- Gemma base 계열은 전반적으로 약하다. 26B-A4B base 2epoch `19.52`, 31B base 2epoch `28.96`이고, 같은 크기의 it 계열보다 크게 낮다. 이 태스크는 terminal assistant 형식과 JSON command target이 중요해서 base 모델에는 SFT만으로 부족했다.
- 31B-it native 2epoch가 1epoch보다 낮은 것은 overfit 또는 action diversity 손실 가능성이 있다. Valid JSON은 유지되지만 Recall이 `0.3776 -> 0.3396`으로 떨어져, 더 많이 학습하면서 안전하고 짧은 행동으로 수렴했을 가능성이 있다.
- Gemma 26B-A4B-it 2epoch는 반대로 Precision이 `0.4405 -> 0.4702`로 개선됐다. MoE it 모델에는 2epoch가 command 선택 정밀도를 올리는 쪽으로 작용했다.

큰 Qwen/Gemma 모델의 약점:

- Qwen은 2B SFT에서는 크게 올랐지만 큰 Qwen에서는 상승폭이 작다. `Qwen3.5-2B`는 base `35.10`에서 SFT 2epoch `39.52`로 `+4.42` 올랐다. 반면 `Qwen3.5-9B`는 base `38.10`에서 SFT 2epoch `38.26`으로 `+0.16`만 올랐고, 1epoch는 `35.60`으로 오히려 `-2.50` 낮다.
- Qwen 9B SFT가 크게 못 오른 핵심은 format 안정성 손실이다. 9B base는 Valid JSON `78.2%`인데, 9B SFT 2epoch는 `64.4%`, 1epoch는 `62.7%`다. Recall은 base `0.3527`에서 e2 `0.3905`로 오르지만, Precision은 `0.4921 -> 0.4620`으로 내려간다. 즉 더 많이 시도하지만 더 지저분해진다.
- Qwen 9B SFT의 약한 source group은 `swe`, `code`, `data_processing`이다. 9B SFT 2epoch 기준 `swe` F1 `20.25`, `code` `30.91`, `data_processing` `33.82`다. 반면 `data_science` `48.68`, `system_administration` `48.30`, `security` `45.25`는 강하다. 큰 Qwen SFT는 시스템/보안/데이터 과학 쪽 routine command는 잘 잡지만, repo 수정형 SWE와 순수 code 작업에서 약하다.
- Qwen 27B/35B base도 같은 패턴이다. `Qwen3.5-27B`는 Score `36.30`, `Qwen3.5-35B-A3B`는 `36.41`인데, 둘 다 `code`가 각각 F1 `13.09`, `16.10`으로 낮다. 큰 모델이 더 일반적으로 똑똑해도, 이 평가의 command replay에서는 code/SWE command exactness가 병목이다.
- Qwen3.5 초대형 MoE도 크기 대비 점수 상승은 제한적이다. `397B-A17B-FP8`은 Score `37.81`, `122B-A10B-FP8`은 `37.28`로 `Qwen/Qwen3.5-9B` `38.10`보다 낮고, 2B SFT 2epoch `39.52`와도 차이가 난다. Precision은 각각 `0.5107`, `0.5155`로 높지만 Recall이 `0.3443`, `0.3408`에 머물러 command coverage가 부족하다.
- 두 Qwen3.5 대형 FP8은 JSON을 못 지킨 실패가 아니다. Valid JSON은 `397B` `86.1%`, `122B` `84.2%`로 높다. 낮은 쪽은 First Cmd `21.5%`, `20.5%`와 Recall `0.34대`다. 즉 답 형식은 잘 맞추지만, TB2-lite reference가 요구하는 첫 터미널 행동과 이후 명령 집합을 충분히 넓게 복원하지 못한다.
- Qwen3.5 대형 FP8은 vLLM 실행 자체는 성공했다. 공식 Qwen3.5 recipe 방향에 맞춰 `tp=8`, expert parallel, prefix caching, Qwen3 reasoning parser, chunked multimodal input 유지, `max_model_len=32768`로 돌렸고 `397B`는 load `826.5s`, generation `260.7s`, `122B`는 load `746.9s`, generation `198.6s`였다. 성능 병목은 런타임 실패가 아니라 평가 태스크 적합도다.

Qwen3.5 대형 FP8 상세 판단:

- 장점은 명확하다. 두 모델 모두 Precision이 `0.51`대라 낸 명령 중 맞는 비율은 높고, Valid JSON도 `84~86%`로 상위권이다. `has_analysis`/`has_plan`도 `84~86%`라 구조화된 답변 습관은 강하다. 8GPU vLLM FP8 MoE 실행도 안정화 후에는 303 step을 끝까지 완료했다.
- 단점은 더 중요하다. 평균 reference command 수가 `38.42`인데, 두 모델의 평균 예측 command 수는 `7.68~7.69`뿐이다. 이 평가에서는 “정답에 포함된 필요한 명령을 얼마나 넓게 복원했는가”가 점수의 핵심이므로, Precision이 높아도 Recall이 `0.34`대에 묶이면 Score가 37점대에서 막힌다.
- 첫 명령 선택도 약하다. `397B` First Cmd `21.5%`, `122B` `20.5%`는 `ZAYA1-74B-preview` `51.8%`, Gemma 26B-A4B-it native e2 `40.6%`, Nemotron-Terminal-14B `40.6%`보다 크게 낮다. 터미널 에이전트에서는 첫 `ls`, `find`, `cat`, `pytest`, `python` 같은 방향 설정이 중요해서 이 지표가 낮으면 체감 성능도 흔들릴 수 있다.
- `397B`가 `122B`보다 크게 앞서지 않는다. Score 차이는 `+0.53`, Recall 차이는 `+0.0035`, First Cmd 차이는 `+1.0%p`뿐이다. active parameter가 A17B로 더 크지만 이 replay 태스크에서는 크기 이점이 거의 점수로 전환되지 않았다.
- 이유는 모델 지능 부족이라기보다 태스크/출력 정책 mismatch에 가깝다. Qwen3.5 대형은 instruction-following과 JSON wrapper는 잘 맞추지만, TB2-lite가 요구하는 “다음 터미널 명령 후보를 많이, reference와 비슷하게, 첫 행동까지 정확히” 내는 성향이 약하다. 일반 대화/장문 reasoning에서는 큰 MoE가 유리해도, 이 평가는 짧고 구체적인 shell action recall이 더 중요하다.
- 또 하나는 보수적 출력이다. 평균 예측 command 수가 8개 미만이라는 것은 모델이 틀릴 만한 명령을 덜 내고 안전한 소수 명령만 제안하는 쪽으로 수렴했다는 뜻이다. 그 결과 Precision은 높지만 Recall과 Cmd F1이 눌린다.
- 이 점수는 “대형 Qwen이 나쁘다”보다는 “Terminal replay용으로는 별도 SFT/format forcing 없이 크기만 키워서는 부족하다”가 더 정확하다. 같은 계열의 `Qwen3.5-2B`는 SFT 후 Score `39.52`까지 갔고, raw `Qwen3.5-9B`도 `38.10`이다. 데이터/출력 정책 적합도가 파라미터 수보다 더 컸다.
- 개선하려면 대형 Qwen에 terminal-specific SFT를 다시 하거나, 평가 prompt에서 command를 더 넓게 열거하게 만들고, 첫 명령을 별도 필드로 강제하는 편이 맞다. 단순히 FP8 대신 BF16/Int4로 바꾸는 것은 런타임/속도 문제에는 영향이 있어도 지금 보이는 Recall/First Cmd 약점을 근본적으로 해결하지는 못한다.

ZAYA1-74B-preview와 Qwen3.5 대형의 차이:

- 이번 결과에서 `ZAYA1-74B-preview`는 괴랄한 예외 케이스로 보는 게 맞다. Score `48.15`는 Qwen3.5-397B `37.81`보다 `+10.34`, Qwen3.5-122B `37.28`보다 `+10.87` 높다. 같은 대형 MoE/대형 공개 모델 범주로 묶기 어려울 정도의 차이다.
- 가장 큰 차이는 첫 행동과 coverage다. ZAYA 74B는 First Cmd `51.8%`, Recall `0.5017`인데, Qwen3.5-397B는 First Cmd `21.5%`, Recall `0.3443`이다. 즉 ZAYA는 “처음 무엇을 칠지”와 “필요 명령을 얼마나 넓게 포함할지”를 동시에 잘 맞히고, Qwen3.5 대형은 형식은 지키지만 실제 행동 폭이 좁다.
- Precision도 ZAYA가 더 높다. ZAYA Precision `0.6196`은 Qwen3.5-397B `0.5107`, 122B `0.5155`보다 높다. 보통 Recall을 높이면 Precision이 내려가는데, ZAYA는 둘 다 높다. 그래서 점수 차이가 단순한 스타일 차이가 아니라 실제 command overlap 차이로 크게 난다.
- JSON 안정성만 보면 Qwen3.5 대형이 더 좋아 보인다. Qwen3.5-397B Valid JSON `86.1%`, 122B `84.2%`, ZAYA `74.6%`다. 하지만 이 벤치의 Score는 JSON 자체가 아니라 command F1이라, JSON을 더 잘 지켜도 명령 recall/첫 명령이 약하면 순위가 내려간다.
- ZAYA도 완벽하지는 않다. `code`, `math`, `swe`, `data_processing` 쪽은 상대적으로 약하고 JSON invalid도 77 step 있다. 그래도 약점이 있는 상태에서 전체 command F1이 `0.4815`라, 현재 표에서는 명백한 최고 후보다.
- 결론적으로 74B ZAYA는 “대형이라서 높은” 것이 아니라 “터미널 행동 replay 분포에 우연히 또는 설계상 잘 맞는” 모델이다. 반대로 Qwen3.5 122B/397B는 모델 크기와 일반 능력은 커도, terminal command replay에서는 보수적이고 좁게 답해서 기대보다 약하게 나온다.
- 실제 운영 관점에서는 ZAYA 74B가 품질 1순위지만 비용/속도 부담이 크고, Qwen3.5 대형은 안정적인 JSON/정밀도 장점이 있어 schema-bound assistant에는 쓸 수 있다. 그러나 “다음 터미널 행동을 공격적으로 넓게 잡는 에이전트”로는 현재 점수상 ZAYA, Gemma 26B-A4B-it SFT, Qwen 2B SFT, raw Qwen 9B가 더 낫다.

Qwen3.5 대형 점수의 저평가 가능성:

- 저평가 가능성은 꽤 있다. 이 평가는 실제 실행 성공률이 아니라 reference replay command overlap이다. Qwen3.5 대형이 더 짧고 보수적인 해결 경로를 제안했거나, reference와 다른 순서의 동등한 명령을 냈다면 Cmd F1에서 깎인다.
- 특히 평균 예측 command 수가 `7.68~7.69`로 적다. reference 평균 `38.42`와 비교하면 모델이 명령을 훨씬 덜 열거한다. 실제 agent loop에서는 한 번에 38개 명령을 다 맞힐 필요가 없고, 첫 몇 개를 실행하며 관찰을 갱신하면 되므로, 전체 command set recall 기반 Score는 보수적 모델을 과소평가할 수 있다.
- JSON 안정성과 Precision은 실제 장점이다. Qwen3.5-397B Valid JSON `86.1%`, Precision `0.5107`; 122B Valid JSON `84.2%`, Precision `0.5155`다. 구조화 출력이 중요한 제품형 agent에서는 이 장점이 표의 Score보다 더 크게 느껴질 수 있다.
- 다만 “저평가만으로 설명된다”라고 보기는 어렵다. First Cmd가 `20~21%`라 첫 행동 자체가 reference와 자주 어긋난다. 첫 명령은 대체 경로 가능성이 있지만, 이 정도 차이면 단순 채점 편향뿐 아니라 terminal replay prior 부족도 함께 있다고 봐야 한다.
- ZAYA1-74B-preview와의 차이가 이를 보여준다. ZAYA도 같은 replay 평가의 불리함을 받지만 First Cmd `51.8%`, Recall `0.5017`, Precision `0.6196`을 동시에 냈다. 따라서 Qwen3.5 대형의 실제 능력이 표보다 높을 수는 있어도, 현재 TB2-lite 다음 행동 채점에서는 ZAYA급으로 숨은 성능이 묻혔다고 보기는 어렵다.
- 결론은 “절대 능력은 저평가됐을 수 있으나, terminal next-action 모델로서의 상대 순위는 어느 정도 신뢰 가능”이다. 정확한 실전 판단을 하려면 replay F1 외에 실제 sandbox 실행 성공률, 첫 명령만 실행 후 재평가하는 agent-loop 평가, 동등 명령 canonicalization 평가를 추가해야 한다.
- Gemma 31B-it native 1epoch는 Valid JSON `65.3%`로 26B-A4B-it보다 안정적이지만 Score는 `35.55`다. 약점은 early bucket이다. 31B-it e1의 early bucket F1은 `31.23`이고, 26B-A4B-it e2는 `47.59`다. 초반 계획/첫 환경 파악 단계에서 31B가 필요한 명령을 덜 넓게 맞힌다.
- Gemma 31B-it 2epoch는 early bucket이 더 크게 무너진다. e2 early bucket F1은 `18.23`이고, late는 `38.76`, mid는 `38.50`이다. 즉 후반 반복/정리 단계는 괜찮지만, 초반 탐색 명령과 첫 방향 설정이 약하다.
- Gemma 31B base 2epoch도 같은 조짐이다. early bucket F1 `12.89`, late `38.74`다. base/31B 계열은 뒤쪽 단계보다 초반 command selection이 특히 약하다.
- Gemma 31B 계열의 약한 source group은 `data_processing`, `code`, `scientific_computing`, `file_operations`다. 31B-it e1은 `data_processing` F1 `19.98`, `code` `23.39`, `scientific_computing` `25.02`이고, e2도 `data_processing` `13.51`, `code` `19.99`로 낮다. 반대로 `dependency_management`, `math`, `system_administration`은 상대적으로 높다.
- 결론적으로 큰 모델들이 약한 부분은 "복잡한 코딩을 못한다"라기보다, 이 벤치마크에서 요구하는 `정확한 터미널 다음 행동`, 특히 초반 탐색/파일 조작/데이터 처리/SWE command sequence를 짧고 정확하게 복원하는 부분이다. 큰 모델은 설명적/보수적 출력을 하거나 JSON wrapper는 지키지만, command recall이 부족하면 Score가 안 오른다.

벤치마크 한계:

- 이 표는 TB2-lite corrected 303-step replay 기준이다. 일반 대화, 수학, 긴 코딩, 검색형 작업의 전체 능력을 대표하지 않는다.
- Score는 `100 * avg_command_f1`이다. JSON 문법, 설명 품질, 안전성, reasoning 품질은 보조 지표일 뿐이다.
- Valid JSON이 낮아도 command parser가 명령을 뽑아낼 수 있으면 점수가 높게 나온다. 따라서 학습 모델 1위 Gemma 26B-A4B-it native는 실제 제품에 넣기 전에 JSON 형식 안정화 후처리나 추가 SFT가 필요하다.
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

- 26B-A4B-it native 2epoch가 `39.56`으로 현재 Gemma 계열 최고다. 최신 전체표에서는 LFM2.5 ToolBench SFT 1epoch/2epoch가 `52.30`/`50.48`로 1, 2위이고, Nemotron-3 Ultra GGUF는 `49.97`로 비학습/외부 모델 최고다. Gemma 26B-A4B-it 2epoch는 Qwen SFT `39.52`와 같은 39점대 학습 모델 최고권이며 1epoch `38.12`보다 `+1.44` 높다.
- 31B-it native는 1epoch `35.55`, 2epoch `32.57`로 2epoch가 낮다. 31B-it는 JSON valid가 `60%+`로 높지만, command F1은 26B-A4B-it보다 낮다.
- base 계열은 31B base 2epoch `28.96`을 제외하면 낮다. 26B-A4B base는 `18~19점대`라 instruction prior 없는 base SFT만으로는 주력 후보가 아니다.
- 이번 31B native 결과는 이전 HF-FSDP 31B SFT `0.00` 문제와 다르다. native 경로에서는 31B-it가 정상 JSON을 생성하고 30점대 점수를 낸다.

해석:

- 4B-it는 확실히 낫다. `34.42`는 현재 전체표 기준 중상위권이며, 기존 LFM2-24B SFT `33.46`보다 `+0.96`, LFM2-2.6B SFT `32.85`보다 `+1.57` 높다. 다만 신규 LFM2.5-8B ToolBench SFT `52.30`과는 큰 차이가 있다.
- 다만 Qwen 상위권과는 아직 차이가 있다. `Qwen3.5-2B SFT 2Epoch` `39.52` 대비 `-5.10`, `Qwen3.5-2B base` `35.10` 대비 `-0.68`, `Ouro-2.6B-Thinking SFT` `35.61` 대비 `-1.19`다.
- E2B-it는 native 전처리로 구 SFT 실패를 크게 회복했다. 하지만 `25점대`라 LFM/Ouro/Qwen 주력 후보와는 아직 거리가 있다.
- E2B-it는 2epoch가 1epoch보다 낮다. 현재 Gemma 작은 모델은 epoch를 더 먹인다고 자동으로 좋아지지 않고, JSON/recall 균형이 흔들릴 수 있다.
- base 모델 SFT(`E2B`, `E4B`)는 낮다. `E2B base`는 2epoch에서 `11.35 -> 16.22`로 회복했지만, instruction-following prior가 있는 `E2B-it`에는 `-9.48` 낮다. 작은 base는 RL 후보에서 제외하는 쪽이 맞다.
- 현재 Gemma RL 후보는 `E4B-it native 1epoch` 1개가 조건부로 살아났다. 2epoch가 유지되거나 26B native가 이보다 높게 나오면 Gemma 후보를 1~2개까지 넣는다.

### 요청 외부 모델 full 303 평가 결과

`2026-06-07` 기준 완료분이다. 출력 디렉터리: `/home/work/.data/tb2_lite_eval/requested_models_20260514`, `/home/work/.data/tb2_lite_eval/minimax_m27_20260515T042432Z_minimax_m27_full_tp8_ep`, `tb2_lite/results/20260516T100702Z_step_bf16_vllm0191_tp8_ep_len49152_eval`, `tb2_lite/results/20260516T122826Z_llada21_flash_sglang_full_suffix_len256`, `tb2_lite/results/20260523T_hrm_text_8gpu_direct_m1024_b32`, `tb2_lite/results/20260606T_nemotron3_ultra_550b_a55b_gguf_q4km_eval`

| 모델 | Backend | Score | Next Action | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Sec/Step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF` `UD-Q4_K_M` | `llama.cpp` CUDA 8GPU | 49.97 | 49.74 | 0.4997 | 0.6191 | 0.4785 | 49.2% | 81.5% | 26.688 |
| `Zyphra/ZAYA1-74B-preview` | `vLLM` `tp=4` | 48.15 | 49.24 | 0.4815 | 0.6196 | 0.5017 | 51.8% | 74.6% | 4.151 |
| `Qwen/Qwen3.5-397B-A17B-FP8` | `vLLM` `tp=8` EP FP8 | 37.81 | 32.92 | 0.3781 | 0.5107 | 0.3443 | 21.5% | 86.1% | 0.860 |
| `Qwen/Qwen3.5-122B-A10B-FP8` | `vLLM` `tp=8` EP FP8 | 37.28 | 32.25 | 0.3728 | 0.5155 | 0.3408 | 20.5% | 84.2% | 0.655 |
| `DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF` `Q4_K_M` | `llama.cpp` CUDA | 37.09 | 32.20 | 0.3709 | 0.5010 | 0.3558 | 20.8% | 83.8% | 14.422 |
| `unsloth/Qwen3.6-27B-MTP-GGUF` `UD-Q4_K_XL` | `llama-server` MTP HTTP | 36.29 | 30.65 | 0.3629 | 0.4813 | 0.3460 | 17.5% | 80.5% | 5.577 |
| `DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF` `Q4_K_M` | `llama.cpp` CUDA | 35.47 | 30.17 | 0.3547 | 0.4756 | 0.3397 | 17.8% | 78.9% | 9.964 |
| `Jackrong/Qwopus3.6-35B-A3B-v1-GGUF` `Q4_K_M` | `llama.cpp` CUDA | 35.45 | 30.36 | 0.3545 | 0.4929 | 0.3287 | 18.5% | 81.2% | 4.835 |
| `MiniMaxAI/MiniMax-M2.7` | `vLLM` `tp=8` EP FP8 | 32.29 | 26.65 | 0.3229 | 0.4492 | 0.2940 | 13.5% | 70.6% | 0.445 |
| `deepseek-ai/DeepSeek-V4-Flash` | official MP4 `2 x tp=4` full8 | 32.22 | 29.87 | 0.3222 | 0.4511 | 0.3037 | 24.4% | 44.2% | 178.033 |
| `Jiunsong/supergemma4-26b-uncensored-gguf-v2` `Q4_K_M` | `llama.cpp` CUDA | 28.21 | 24.10 | 0.2821 | 0.4135 | 0.2506 | 14.5% | 53.8% | 5.688 |
| `Zyphra/ZAYA1-8B` | `vLLM` | 20.61 | 19.68 | 0.2061 | 0.2844 | 0.2117 | 17.5% | 35.3% | 2.339 |
| `inclusionAI/LLaDA2.1-flash` | `SGLang` `tp=8` dLLM suffix | 20.07 | 17.41 | 0.2007 | 0.3150 | 0.1819 | 11.2% | 28.7% | 2.502 |
| `stepfun-ai/Step-3.5-Flash` | `vLLM` `tp=8` EP BF16 | 18.80 | 17.33 | 0.1880 | 0.2710 | 0.1790 | 13.9% | 27.4% | 5.368 |
| `sapientinc/HRM-Text-1B` | `transformers` PrefixLM 8GPU shard | 0.40 | 0.67 | 0.0040 | 0.0057 | 0.0040 | 1.3% | 4.3% | 3.976 |

미완료/유효 점수 없음:

- `google/gemma-4-31B-it-assistant`: 아직 결과 JSON 없음.

MiniMax-M2.7 상세:

- `MiniMaxAI/MiniMax-M2.7`은 vLLM `0.20.2`, `tp=8`, expert parallel, FP8, `max_model_len=49152`, `max_tokens=1024`로 303-step full 평가를 완료했다. 로드는 `642.9s`, 생성은 `134.8s`, 평균 `0.445s/step`이었다.
- Score는 `32.29`다. JSON 자체는 Valid `70.6%`로 무난하고 Precision `0.4492`도 나쁘지 않지만, Recall `0.2940`, First Cmd `13.5%`, 평균 예측 command `4.08`개라 필요한 명령을 너무 적게 낸다.
- bucket별로는 early `0.4003`, mid `0.2895`, late `0.2909`다. 초반은 그럭저럭 버티지만, 중후반 command coverage가 빠르게 약해진다.
- source group 기준 강점은 `system_administration` F1 `0.4771`, `dependency_management` `0.4695`, `file_operations` `0.4147`이다. 약점은 `code` `0.0908`, `swe` `0.2185`, `data_processing` `0.2623`, `data_querying` `0.2686`이다.
- 결론은 모델이 작동하지 않은 실패가 아니라, TB2-lite terminal replay에 대해 너무 보수적인 출력 정책을 가진 정상 완료 결과다. 일반 reasoning 능력보다 이 벤치가 요구하는 “첫 터미널 명령과 넓은 command set recall”이 약하게 나온 케이스다.

DeepSeek V4 Flash 상세:

- 최종 반영 결과는 공식 `deepseek-ai/DeepSeek-V4-Flash` converted MP4 inference 경로의 303-step full replay다. GPU 0-3과 4-7을 `2 x tp=4` shard로 나눴고, `max_model_len=49152`, `max_new_tokens=1024`, batch size `8`, thinking mode `thinking`으로 실행했다. merged JSON은 `tb2_lite/results/20260515T114312Z_deepseek_flash_official_mp4_full8_len49152_bs8_retry/DeepSeek-V4-Flash.json`이다.
- Score는 `32.22`, Next Action `29.87`, Cmd F1 `0.3222`, Precision `0.4511`, Recall `0.3037`, First Cmd `24.4%`, Valid JSON `44.2%`다. 이전 vLLM/FP8 경로의 BOS 반복 0점 실패와 달리, 이번 공식 MP4 경로는 실제 JSON/command를 생성했고 평가 가능한 정상 결과다.
- 강점은 `dependency_management` F1 `0.5048`, `system_administration` `0.3912`, `model_training` `0.3757`, `data_science` `0.3731`, `scientific_computing` `0.3701`, `software_engineering` `0.3696`이다. early bucket F1도 `0.4085`라 초반 탐색/환경 파악은 중위권 이상이다.
- 약점은 `math` F1 `0.1302`, `data_processing` `0.1496`, `code` `0.1688`, `swe` `0.2587`이다. 평균 예측 command는 `4.48`개인데 ref 평균은 `38.42`개라, 필요한 command set recall이 낮다. Valid JSON도 `44.2%`라 형식 안정성이 MiniMax `70.6%`, Qwen3.5 대형 `84~86%`보다 낮다.
- 속도는 가장 큰 문제다. load는 `15.6s`지만 generation은 `53944.1s`, 평균 `178.033s/step`이다. 마지막 shard 기준 전체 run이 15시간 가까이 걸렸고, 같은 8GPU 대형 평가인 MiniMax `0.445s/step`, Qwen3.5-397B `0.860s/step`과 비교하면 운영성이 매우 나쁘다.
- 결론적으로 DeepSeek V4 Flash는 실패 모델이 아니라 정상 완료 중위권 모델로 정정한다. 다만 TB2-lite terminal replay에서는 JSON 안정성/recall/속도 때문에 기대한 초상위권은 아니고, 현재 기준으로는 MiniMax-M2.7 바로 아래인 전체 43위다.

Step-3.5-Flash 상세:

- 최종 반영 결과는 `stepfun-ai/Step-3.5-Flash` BF16 원본 모델을 vLLM `0.19.1`, `tp=8`, expert parallel, `max_model_len=49152`, `max_tokens=1024`, concurrency `16`으로 실행한 full 303-step replay다. 결과 파일은 `tb2_lite/results/20260516T100702Z_step_bf16_vllm0191_tp8_ep_len49152_eval/step_3_5_flash_bf16_vllm_tp8_ep_len49152.json`이다.
- Score는 `18.80`, Next Action `17.33`, Cmd F1 `0.1880`, Precision `0.2710`, Recall `0.1790`, First Cmd `13.9%`, Valid JSON `27.4%`다. GPU 8장을 모두 사용했고 VRAM은 GPU당 약 `140.47GB / 143.77GB`까지 찼다. 생성 시간은 `1626.5s`, 평균 `5.368s/step`이다.
- 이 결과는 평가 실행 실패가 아니라 모델 출력 품질 문제다. vLLM 서버는 `303/303` 요청을 모두 `200 OK`로 처리했고 evaluator 오류도 없었다. 다만 strict JSON invalid가 `220/303`이라 `analysis`, `plan`, `commands` 구조를 안정적으로 지키지 못했다.
- 가장 큰 약점은 command recall이다. reference 평균 command가 `38.42`개인데 Step-3.5-Flash의 평균 예측 command는 `2.60`개다. 그래서 틀린 명령을 많이 내는 모델이라기보다, 필요한 셸 명령을 충분히 넓게 복원하지 못하는 모델로 본다.
- bucket별 F1은 early `0.2686`, mid `0.1603`, late `0.1476`이다. 초반 탐색은 조금 낫지만 중후반으로 갈수록 계획 유지와 command coverage가 빠르게 약해진다.
- 상대적으로 나은 source group은 `software_engineering` F1 `0.2919`, `data_science` `0.2637`, `dependency_management` `0.2397`, `file_operations` `0.2292`, `system_administration` `0.2289`다. 약한 영역은 `code` `0.0846`, `model_training` `0.1143`, `math` `0.1245`, `swe` `0.1274`, `scientific_computing` `0.1385`다.
- 고평가/저평가 관점에서는 약간 저평가 여지는 있다. invalid JSON preview를 보면 모델이 터미널 transcript나 자연어 요약을 이어 쓰는 경우가 많아, 사람이 보면 일부 유용한 행동 의도가 남아 있어도 strict parser가 버린다. 하지만 Cmd F1 기준 zero step이 `167/303`이고 First Cmd도 `13.9%`라 순위가 크게 뒤집힐 정도의 저평가는 아니다. 현재 벤치마크 방식에서는 하위권 정상 결과로 보는 것이 맞다.

LLaDA2.1-flash 상세:

- 최종 반영 결과는 `inclusionAI/LLaDA2.1-flash`를 SGLang `0.5.9`, `tp=8`, `--dllm-algorithm JointThreshold`, `flashinfer`, BF16, `max_new_tokens=256`, `block_length=32`, `threshold=0.5`, `editing_threshold=0.0`, `max_post_steps=16`으로 실행한 303-step full replay다. 결과 파일은 `tb2_lite/results/20260516T122826Z_llada21_flash_sglang_full_suffix_len256/llada21_flash_sglang_full_suffix_len256.json`이다.
- Score는 `20.07`, Next Action `17.41`, Cmd F1 `0.2007`, Precision `0.3150`, Recall `0.1819`, First Cmd `11.2%`, Valid JSON `28.7%`다. 전체 순위는 66위이며 `Zyphra/ZAYA1-8B` `20.61` 바로 아래, `Step-3.5-Flash` `18.80` 위다.
- 실행 자체는 정상이다. vLLM은 `LLaDA2MoeModelLM` 미지원으로 불가했고, SGLang clean env에서 TP8 서버가 올라갔다. GPU 8장은 각 약 `115.6GB / 143.8GB` VRAM을 잡고 생성 중 util `84~86%`를 유지했다. 전체 생성 시간은 `758.0s`, 평균 `2.502s/step`이다.
- 주의할 점은 이 결과가 `prompt-suffix rescue` 조건이라는 것이다. 기본 프롬프트만 쓰면 모델이 task 내부 `validation.json`/deliverable schema를 따라가며 `commands` JSON을 안정적으로 내지 못했다. 그래서 마지막 assistant turn 직전에 `analysis, plan, commands, task_complete`만 출력하라는 suffix를 붙였다. 이 suffix는 평가 포맷 준수를 돕지만, 다른 모델의 순수 chat-template 결과와는 완전히 동일 조건이 아니므로 순위 해석에는 보수적으로 봐야 한다.
- 강점은 일부 초반 탐색과 시스템/데이터 처리류 명령이다. bucket별 F1은 early `0.2559`, mid `0.2354`, late `0.1187`이고, source group 기준 `system_administration` `0.3250`, `data_processing` `0.2840`, `scientific_computing` `0.2698`, `software_engineering` `0.2533`이 상대적으로 낫다.
- 약점은 command set이 너무 짧고 포맷 안정성이 낮다는 점이다. reference 평균 command는 `38.42`개인데 LLaDA 평균 예측 command는 `1.47`개뿐이다. `108/303` step은 추출된 command가 0개였고, Valid JSON도 `87/303`에 그쳤다. source group 기준 `code` `0.0890`, `swe` `0.1096`, `data_querying` `0.1361`, `dependency_management` `0.1480`, `security` `0.1562`가 약하다.
- 고평가/저평가 관점에서는 양쪽이 섞인다. suffix 없이 strict하게 보면 더 낮게 나올 가능성이 커서 현재 `20.07`은 포맷 힌트 덕분에 약간 고평가일 수 있다. 반대로 dLLM 구조상 한 번에 짧은 답을 내는 쪽으로 튜닝되어 reference의 긴 command list를 덜 복원하는 특성이 있어, 실제 agent loop에서 한두 명령씩 실행하며 재질문하는 방식이면 replay F1보다 체감 성능이 조금 나을 수 있다. 그래도 First Cmd `11.2%`, avg_pred_cmds `1.47`, late F1 `0.1187`을 보면 상위권 후보는 아니고, 현재 벤치에서는 하위권 정상 완료 결과로 보는 것이 맞다.

HRM-Text-1B 상세:

- 최종 반영 결과는 `sapientinc/HRM-Text-1B`를 Hugging Face `transformers 5.9.0`의 `hrm_text` 구현으로 실행한 full 303-step replay다. 로컬 기본 `transformers 5.5.4`는 `hrm_text` 아키텍처를 인식하지 못해 `/tmp/hrm-eval-env` 격리 환경에 `transformers>=5.9.0`을 설치했다. 결과 파일은 `tb2_lite/results/20260523T_hrm_text_8gpu_direct_m1024_b32/HRM-Text-1B-direct-m1024-b32-full.json`이다.
- 실행 방식은 HRM-Text의 PrefixLM 구조에 맞춰 일반 chat template 대신 `<|im_start|>{condition_prefix}...<|im_end|>` 입력을 만들고, prompt token 전체에 `token_type_ids=1`을 넣는 전용 evaluator를 사용했다. condition은 `direct`이며 실제 prefix는 `<|object_ref_start|>`다. 설정은 BF16, `max_model_len=4096`, `max_tokens=1024`, `batch_size=32`, `num_shards=8`, 8GPU data parallel shard다. GPU당 VRAM은 약 `122.8~124.6GB / 143.8GB`까지 사용했고, 전체 wall time은 `1204.7s`, 평균 `3.976s/step`이다.
- Score는 `0.40`, Next Action `0.67`, Cmd F1 `0.0040`, Precision `0.0057`, Recall `0.0040`, First Cmd `1.3%`, Valid JSON `4.3%`다. 전체 순위는 98위이며 `ByteDance/Ouro-2.6B` Score `6.46` 아래, 기존 Gemma 31B HF-FSDP 실패 0점 두 개 위다. 즉 0점 실패는 아니지만, TB2-lite 기준으로는 사실상 사용 불가 수준의 하위권 결과다.
- 가장 큰 문제는 model intelligence 자체보다 **출력 계약 불일치**다. TB2-lite는 `analysis`, `plan`, `commands: [{keystrokes, duration}]`, `task_complete` 구조를 요구하는데, HRM-Text-1B는 이 구조를 거의 따르지 않았다. 평균 예측 command는 `0.41`개뿐이고, reference 평균 command는 `38.42`개다. 즉 명령을 많이 틀렸다기보다, 평가 가능한 shell action을 거의 제출하지 않았다.
- 출력 샘플을 보면 모델은 종종 task 결과물 JSON, 설정 요약, 터미널 transcript 비슷한 텍스트, 데이터 구조 예시를 생성한다. 예를 들어 `commands[].keystrokes` 배열 대신 `"project_summary"`, `"dependency_graph"`, `"library_relationships"` 같은 deliverable schema를 이어 쓰거나, 이전 터미널 로그를 리스트처럼 복사한다. 이 출력은 사람이 보면 문제 맥락을 일부 따라간 것으로 보일 수 있지만, agent가 다음에 실행할 명령으로는 바로 사용할 수 없다.
- bucket별 F1도 early `0.0135`, mid `0.0000`, late `0.0000`이라 초반 몇 개 step을 제외하면 command replay가 거의 성립하지 않는다. source group에서도 의미 있는 강점은 없다. 상대적으로 점수가 조금 남은 곳은 `security` F1 `0.0236`, `data_processing` `0.0157`, `scientific_computing` `0.0111` 정도지만, 절대값이 너무 낮아 실제 강점으로 해석할 수 없다.
- 후처리 가능성도 낮다. 기존 보수적 후처리 원칙은 invalid JSON repair와 `"keystrokes"` 필드 추출까지만 허용하고, `analysis` 본문이나 복사된 터미널 로그에서 `ls`, `cat`, `python` 같은 문자열을 command로 채굴하지 않는다. HRM 결과에서 valid JSON이면서 command가 있는 step은 `1/303`, literal `"keystrokes"` 문자열이 보이는 step은 `6/303`, shell-like 문자열이 보이는 preview는 `40/303`이다. `40/303`을 억지로 command-mining하면 점수는 조금 오를 수 있지만, 복사된 로그/설명/데이터를 실행 의도라고 오인하는 고평가가 된다.
- 따라서 이 결과는 DeepSeek-V4-Pro처럼 "좋은 명령을 냈지만 reference 문자열과 달라 깎인" 저평가 케이스가 아니다. HRM-Text-1B의 일반 추론/텍스트 생성 능력은 이 점수만으로 판단하면 안 되지만, **TB2-lite terminal next-action JSON agent 평가로는 공정한 낮은 점수**에 가깝다. 제대로 된 점수를 올리려면 후처리가 아니라 terminal JSON command 형식에 대한 SFT, constrained decoding, 또는 별도 agent wrapper가 필요하다. 그런 wrapper가 명령을 번역해 주면 그때는 모델 단독 점수가 아니라 wrapper 포함 시스템 점수로 별도 표기해야 한다.

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
- 현재 성공 결과 기준 최고 LFM은 `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`이고 Score `52.30`다. 최고 Qwen 대비 `+12.78`점 높다.
- 이번 LFM2.5-8B-A1B SFT 1epoch는 기존 최고 LFM이던 `LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-TemplateMasked` Score `33.46`보다 `+18.84` 높다. 모델 크기보다 데이터 구성과 LFM2.5 base prior가 더 크게 작용한 결과로 본다.
- Qwen 상위권은 Valid JSON이 `80%`대까지 올라가는 반면, 이번 LFM2.5 SFT 1epoch는 Valid JSON `76.9%`다. 그래도 Recall `0.5431`이 Qwen2B SFT `0.4101`보다 높고 Precision도 `0.5854`로 높아 최종 Cmd F1이 앞섰다.
- 핵심 차이는 command coverage다. Qwen2B SFT는 형식 안정성은 매우 좋지만 일부 단계에서 짧고 보수적인 command set을 낸다. LFM2.5 SFT는 정답 command set을 더 넓게 복원하면서 precision 손실을 억제했다.
- 다만 LFM2.5 SFT도 JSON 안정성은 아직 완벽하지 않다. 1epoch 기준 invalid JSON `70/303`, zero-F1 `23/303`, empty command `19/303`이 남아 있어, 후속 학습에서는 `<think>` 제거, JSON-only assistant target, 후반 step 검증 command 강화가 필요하다.

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
- `Qwen3.5-2B SFT 2Epoch`는 최신 전체표 기준 Score `39.52`로 Qwen 계열 최고권이다. LFM2.5 ToolBench SFT, ZAYA1-74B-preview, GLM-5.1, Gemma 26B-A4B-it native 2epoch 아래지만, 2B dense 모델로는 여전히 매우 강하다.
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
- vLLM 평가가 끝나면 루트 `README.md` 전체 순위 표, native 결과 섹션, 모델군 해석을 즉시 갱신한다.
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
| 4 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 25.70 | 0.2570 | 0.3615 | 0.2717 | 15.2% | 34.3% | 0.325 |
| 5 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 24.92 | 0.2492 | 0.3667 | 0.2447 | 11.6% | 34.0% | 0.317 |
| 6 | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 18.47 | 0.1847 | 0.2514 | 0.1980 | 16.8% | 17.2% | 0.302 |
| 7 | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 16.22 | 0.1622 | 0.2747 | 0.1678 | 15.2% | 16.2% | 0.289 |
| 8 | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 12.80 | 0.1280 | 0.1792 | 0.1364 | 10.2% | 14.2% | 0.383 |
| 9 | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 11.35 | 0.1135 | 0.1767 | 0.1191 | 6.6% | 7.3% | 0.219 |

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

## DeepSeek-V4-Pro 추론 엔진 호환성 분석

업데이트: `2026-05-17`

DeepSeek-V4-Pro (1.6T params, 49B activated, 384 experts, FP4+FP8 mixed)를 TB2-lite 303-step으로 평가하기 위해 vLLM, SGLang, 공식 inference 세 가지 경로를 시도했다. 결과적으로 **공식 inference만 작동**하며, vLLM과 SGLang은 모두 CUDA/driver 버전 호환성 문제로 실행 불가하다.

### 환경

| 항목 | 사양 |
| --- | --- |
| GPU | 8x NVIDIA H200 (143.8 GB each, 1.15 TB total) |
| Driver | 570.86.10 (CUDA 12.9) |
| 체크포인트 | `/home/work/deepseek_models/DeepSeek-V4-Pro-mp8/` (858 GB, MP8) |
| HF 원본 | `/home/work/.data/huggingface/hub/deepseek-ai_DeepSeek-V4-Pro/` (809 GB, 64 safetensors) |

### vLLM: 불가 (CUDA 13 driver 필요)

| 항목 | 내용 |
| --- | --- |
| 시도 버전 | vLLM 0.20.2 |
| 요구 사양 | torch 2.11.0+cu130 (CUDA 13) |
| 실패 원인 | 시스템 NVIDIA driver 570.86.10이 CUDA 12.9까지만 지원. torch 2.11.0+cu130 실행 시 `"The NVIDIA driver on your system is too old (found version 12090)"` 에러 |
| 대안 시도 | torch 2.10.0+cu129 + vLLM 0.20.2 → C++ ABI 불일치 (`_ZN2at4cuda24getCurrentCUDABlasHandleEv` undefined symbol). vLLM 0.19.1 → `deepseek_v4` tokenizer mode 미지원 |
| 결론 | **Driver 업그레이드 없이는 불가**. 공유 시스템이라 driver 변경 불가 |

### SGLang: 불가 (flash_mla 빌드 실패 + CUDA 그래프 SIGSEGV)

| 항목 | 내용 |
| --- | --- |
| 시도 버전 | SGLang 0.5.12 + torch 2.9.0+cu129 |
| 성공 단계 | 모델 감지 (`DeepseekV4ForCausalLM`), 64 safetensors 샤드 로딩, FP8 디퀀트, 웨이트 GPU 적재 (119.68 GB/GPU) |
| 실패 1 | `--disable-cuda-graph` 없이 실행 시 CUDA 그래프 캡처 단계에서 SIGSEGV (exit code -11). `libucs.so.0` (UCX/NCCL)에서 세그폴트 |
| 실패 2 | `--disable-cuda-graph` 적용 후 서버는 시작됨 (`Uvicorn running on http://127.0.0.1:8000`)하지만, 첫 추론 요청 시 `ModuleNotFoundError: No module named 'flash_mla'` |
| flash_mla 빌드 | `pip install flash-mla` 시 CUDA 커널 빌드 실패 (`flash_mla.h: No such file or directory`). torch 2.9.0에서 빌드 불가 |
| 메모리 이슈 | context_length=16384 + mem-fraction=0.88 → `RuntimeError: Not enough memory`. 웨이트 119.68 GB/GPU + KV 캐시 공간 부족. context_length=8192 + mem-fraction=0.90으로 해결했으나 flash_mla 문제로 실제 추론 불가 |
| 결론 | **flash_mla 없이 DS-V4 attention 백엔드(dsv4) 작동 불가**. flash_mla는 H200 전용 CUDA 커널로 현재 환경에서 빌드 안 됨 |

### 공식 inference: 작동 확인 (느림)

| 항목 | 내용 |
| --- | --- |
| 환경 | `.deepseek-env` (torch 2.12.0.dev+cu128) |
| 상태 | 모델 로딩 성공 (112 GB/GPU), smoke test 통과 |
| 속도 | Flash 기준 ~178s/step. Pro는 5.5x 크기 → 예상 ~500-980s/step, 전체 303-step에 **42-83시간** 예상 |
| 스크립트 | `tb2_lite/scripts/deepseek_replay_eval.py` + `tb2_lite/deepseek_v4/` (커스텀 FP4/F8 커널) |

### 요약

| 엔진 | 상태 | 원인 |
| --- | --- | --- |
| vLLM 0.20.2 | **불가** | CUDA 13 driver 필요 (현재 CUDA 12.9) |
| SGLang 0.5.12 | **불가** | flash_mla 빌드 실패 + CUDA 그래프 SIGSEGV |
| 공식 inference | **가능** | 느리지만 작동 (~42-83시간 예상) |

**핵심 병목**: DeepSeek-V4 시리즈는 FP4+FP8 혼합 양자화와 커스텀 MLA attention을 사용하여, 범용 추론 엔진(vLLM, SGLang)의 지원이 아직 미흡하다. 특히 flash_mla(H200 전용 CUDA 커널)와 CUDA 13 요구사항이 현재 시스템 환경과 맞지 않아 공식 inference 코드만 유일한 실행 경로다.
