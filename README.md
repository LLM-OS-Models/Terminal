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
  - 전체 `27개 중 3위`
- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount`
  - `Score 29.66`
  - `Cmd F1 0.2917`
  - `First Cmd Exact 30.8%`
  - `0.029 sec/step`
  - `Load 52.2s`
  - 전체 `27개 중 4위`
- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
  - `Score 28.51`
  - `Cmd F1 0.2796`
  - `First Cmd Exact 29.8%`
  - `0.025 sec/step`
  - `Load 32.5s`
  - 전체 `27개 중 7위`
- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` (LoRA)
  - `Score 22.84`
  - `Cmd F1 0.2586`
  - `First Cmd Exact 15.8%`
  - `0.028 sec/step`
  - `Load 89.9s`
  - 전체 `27개 중 26위`

현재 `tb2_lite` 전체 비교 순위:

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

핵심 해석:

- `Qwen3.5-2B`는 LoRA보다 full FT가 압도적으로 좋았습니다.
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
- HF 업로드 완료:
  `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`

## 저장 경로

학습 결과:

- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_unsloth_lora`
- `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2.5-1.2B-Base__terminal_sft_h200_4gpu`
- `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2-2.6B__terminal_sft_h200_4gpu`
- `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local`

평가 결과:

- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_final_vllmfix3`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_ckpt55_final_vllmfix3`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_h200_sft_final_1gpu`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_sft_final_1gpu_vllm`

허깅페이스:

- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`
- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth`
