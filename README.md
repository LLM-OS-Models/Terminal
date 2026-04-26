# Terminal Agent

Nemotron-Terminal-Corpus 기반 터미널 에이전트 학습과 평가를 위한 작업 폴더입니다.

현재 이 저장소는 크게 세 축으로 운영됩니다.

1. `eval/` 기반 단발 프록시 평가
2. `tb2_lite/` 기반 멀티턴 replay 평가
3. `Liquid-CLI + Unsloth` / `Qwen3.5 + Unsloth` 기반 SFT 준비 및 학습

기준 날짜: `2026-04-26`

## 오늘 기준 핵심 상태

### 1. TB2-lite 평가

`tb2_lite`는 full Terminal-Bench 2.0보다 훨씬 빠르게 모델을 걸러내기 위한 replay 평가 레이어입니다.

주요 문서:

- 루트 성능 요약:
  [PERFORMANCE_SUMMARY_2026-04-26.md](/home/work/.projects/LLM-OS-Models/Terminal/PERFORMANCE_SUMMARY_2026-04-26.md)
- 통합 결과:
  [TB2_LITE_RESULTS_2026-04-26.md](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/docs/TB2_LITE_RESULTS_2026-04-26.md)
- 소형/체크포인트 스윕:
  [TB2_LITE_CHECKPOINT_SWEEP_2026-04-26.md](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/docs/TB2_LITE_CHECKPOINT_SWEEP_2026-04-26.md)
- 미실험/스킵:
  [TB2_LITE_UNTESTED_2026-04-25.md](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/docs/TB2_LITE_UNTESTED_2026-04-25.md)
- `vLLM tp=8` 이슈:
  [TB2_LITE_VLLM_ISSUES_2026-04-26.md](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/docs/TB2_LITE_VLLM_ISSUES_2026-04-26.md)

오늘 핵심 결과:

- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`
  - `Score 29.77`
  - `Cmd F1 0.2912`
  - `First Cmd Exact 31.3%`
  - `0.030 sec/step`
  - `Load 52.6s`
  - 전체 `29개 중 3위`
- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount`
  - `Score 29.66`
  - `Cmd F1 0.2917`
  - `First Cmd Exact 30.8%`
  - `0.029 sec/step`
  - `Load 52.2s`
  - 전체 `29개 중 4위`
- `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData`
  - `Score 28.89`
  - `Cmd F1 0.3185`
  - `First Cmd Exact 22.0%`
  - `0.063 sec/step`
  - `Load 85.2s`
  - 전체 `31개 중 6위`
- `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData`
  - `Score 28.80`
  - `Cmd F1 0.3094`
  - `First Cmd Exact 23.8%`
  - `0.084 sec/step`
  - `Load 73.9s`
  - 전체 `31개 중 7위`
- `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-1Epoch-FullFT-2BData`
  - `Score 27.45`
  - `Cmd F1 0.3021`
  - `First Cmd Exact 21.0%`
  - `0.083 sec/step`
  - `Load 86.5s`
  - 전체 `31개 중 15위`
- `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData`
  - `Score 26.80`
  - `Cmd F1 0.2907`
  - `First Cmd Exact 21.5%`
  - `0.065 sec/step`
  - `Load 60.3s`
  - 전체 `31개 중 17위`
- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
  - `Score 28.51`
  - `Cmd F1 0.2796`
  - `First Cmd Exact 29.8%`
  - `0.025 sec/step`
  - `Load 32.5s`
  - 전체 `29개 중 8위`
- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` (LoRA)
  - `Score 22.84`
  - `Cmd F1 0.2586`
  - `First Cmd Exact 15.8%`
  - `0.028 sec/step`
  - `Load 89.9s`
  - 전체 `27개 중 26위`

현재 `tb2_lite` 점수 확정 모델은 `31개`입니다.

- 이 표에는 `점수가 실제로 나온 모델만` 넣었습니다.
- `Qwen3.5-9B` full FT `1 epoch`와 `2 epoch final`도 이제 포함했습니다.

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
| 18 | `Qwen/Qwen3.5-2B` | 26.52 | 0.2743 | 24.4% | 0.024 | 84.7 |
| 19 | `Qwen/Qwen3.5-4B` | 26.36 | 0.2745 | 23.8% | 0.055 | 77.2 |
| 20 | `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | 26.35 | 0.2744 | 23.8% | 0.282 | 113.4 |
| 21 | `google/gemma-4-26B-A4B-it` | 25.95 | 0.2631 | 25.1% | 0.094 | 123.9 |
| 22 | `Qwen/Qwen3.6-27B` | 25.60 | 0.2702 | 22.3% | 0.282 | 118.5 |
| 23 | `Qwen/Qwen3.6-35B-A3B-FP8` | 25.21 | 0.2658 | 22.0% | 0.082 | 120.5 |
| 24 | `google/gemma-4-31B-it` | 24.70 | 0.2594 | 21.8% | 0.404 | 101.4 |
| 25 | `LiquidAI/LFM2-2.6B` | 24.12 | 0.2379 | 24.9% | 0.032 | 36.4 |
| 26 | `google/gemma-4-E4B-it` | 23.43 | 0.2391 | 22.3% | 0.051 | 129.7 |
| 27 | `LiquidAI/LFM2.5-1.2B-Instruct` | 23.36 | 0.2381 | 22.3% | 0.021 | 31.7 |
| 28 | `LiquidAI/LFM2-8B-A1B` | 23.19 | 0.2336 | 22.8% | 0.025 | 58.2 |
| 29 | `google/gemma-4-E2B-it` | 23.05 | 0.2359 | 21.8% | 0.032 | 116.0 |
| 30 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` | 22.84 | 0.2586 | 15.8% | 0.028 | 89.9 |
| 31 | `LiquidAI/LFM2-24B-A2B` | 22.80 | 0.2323 | 21.8% | 0.050 | 81.6 |

핵심 해석:

- `Qwen3.5-2B`는 LoRA보다 full FT가 압도적으로 좋았습니다.
- `Qwen3.5-4B`는 `1 epoch`가 좋았고, `2 epoch final`은 오히려 내려갔습니다.
- `Qwen3.5-9B`는 반대로 `2 epoch final`이 더 좋았고, base `Qwen/Qwen3.5-9B (27.04)` 대비 `+1.76` 올랐습니다.
- `2 epoch final`은 공개 `gyung` 최고 기록보다 `-0.37`, `Nemotron-Terminal-8B`보다 `-0.25` 낮습니다.
- 반대로 `Nemotron-Terminal-32B`보다 `+0.64`, `LFM 1.2B final`보다 `+0.99`, `LFM 8B final`보다 `+1.26` 높습니다.
- `1 epoch`도 거의 비슷해서, 이번 설정에서는 이미 `1 epoch`에서 대부분의 성능이 올라왔습니다.

### 2. 오늘 확인한 `vLLM` / export 이슈

같은 로컬 full FT 모델을 `vLLM`로 바로 띄우려 했을 때, 원본 `trainer.save_model()` 저장물이 표준 HF `Qwen3.5-VL` key 형식이 아니라서 로딩이 실패했습니다.

원본 key 예:

- `model.language_model.language_model.language_model.layers...`
- `model.language_model.visual...`

`vLLM`이 기대하는 HF key:

- `model.language_model...`
- `model.visual...`

그래서 아래 스크립트로 평가용 정리본을 만들었습니다.

- [fix_fullft_export_prefix.py](/home/work/.projects/LLM-OS-Models/Terminal/qwen_sft/scripts/fix_fullft_export_prefix.py)

실제 평가에 쓴 디렉터리:

- final:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/final-vllmfix3`
- 1 epoch:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/checkpoint-55-vllmfix3`

또한 single-GPU `vLLM` 평가를 위해 새 env도 만들었습니다.

- env:
  `/home/work/.projects/LLM-OS-Models/Terminal/.vllm-uv-env`
- 설치:
  `uv venv` + `uv pip install --torch-backend=auto vllm`
- 주의:
  `unset PYTHONPATH` + `PYTHONNOUSERSITE=1` 필요

### 3. Liquid SFT

원본/준비 코드 경로:

- Liquid-CLI 원본:
  [Liquid-CLI](/home/work/.projects/LLM-OS-Models/Terminal/Liquid-CLI)
- Unsloth 원본:
  `/home/work/.projects/LLM-OS-Models/Terminal/unsloth-src`
- SFT 준비 코드:
  [liquid_sft](/home/work/.projects/LLM-OS-Models/Terminal/liquid_sft)
- 준비/학습 상태 문서:
  [SFT_PREP_STATUS_2026-04-25.md](/home/work/.projects/LLM-OS-Models/Terminal/liquid_sft/docs/SFT_PREP_STATUS_2026-04-25.md)

오늘 기준 완료된 모델:

- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth`

### 4. Qwen SFT

별도 경로:

- [qwen_sft](/home/work/.projects/LLM-OS-Models/Terminal/qwen_sft)
- 상태 문서:
  [QWEN_SFT_STATUS_2026-04-26.md](/home/work/.projects/LLM-OS-Models/Terminal/qwen_sft/docs/QWEN_SFT_STATUS_2026-04-26.md)

오늘 기준 핵심:

- `Qwen3.5-2B` LoRA: 성능 실패
- `Qwen3.5-2B` full FT same-count: 성공
- `Qwen3.5-4B` full FT 2BData: 학습/평가 완료, `1 epoch`가 best
- `Qwen3.5-9B` full FT 2BData: 학습/평가 완료, `2 epoch final`이 best
- HF 업로드 완료:
  `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`
  `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData`

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
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_h200_sft_final_1gpu`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_sft_final_1gpu_vllm`

허깅페이스:

- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`
- `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData`
- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth`
