# Qwen SFT 상태 (2026-04-26)

## 현재 상태

- `Qwen3.5-2B` 2 epoch LoRA 학습: 완료
- `Qwen3.5-2B` 2 epoch full FT same-count 학습: 완료
- `checkpoint-55`, `checkpoint-110`, `final` 저장: 완료
- single-GPU `tb2_lite + vLLM` 평가: 완료
- 허깅페이스 업로드: 완료
- `Qwen3.5-4B`, `Qwen3.5-9B` config 준비: 완료

## LoRA 런 요약

- 모델:
  `Qwen/Qwen3.5-2B`
- config:
  `qwen_sft/configs/sft_qwen35_2b_lora_8gpu.env`
- output root:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_unsloth_lora`
- final:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_unsloth_lora/final_official`
- log:
  `/home/work/.data/qwen_sft/logs/qwen35_2b_lora_sft_8gpu_20260425T171134Z.log`
- HF target:
  `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth`

학습 결과:

- mode:
  `QLoRA(4bit) + LoRA`
- GPU:
  `8 GPU`
- total step:
  `688`
- epoch:
  `2`
- runtime:
  `3478초` (`57분 58초`)
- train loss:
  `0.8178`

평가 결과:

- Score:
  `22.84`
- Cmd F1:
  `0.2586`
- First Cmd Exact:
  `15.8%`
- Load:
  `89.9초`
- Gen:
  `10.9초`
- 전체 순위:
  `26 / 27`

판단:

- base `Qwen/Qwen3.5-2B (26.52)`보다 `-3.68`
- 이번 설정에서 LoRA 경로는 실패

## Full FT same-count 런

- 모델:
  `Qwen/Qwen3.5-2B`
- config:
  `qwen_sft/configs/sft_qwen35_2b_fullft_samecount_8gpu.env`
- output root:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount`
- log:
  `/home/work/.data/qwen_sft/logs/qwen35_2b_fullft_samecount_8gpu_20260425T235511Z.log`
- HF target:
  `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`

학습 결과:

- mode:
  `full finetuning`
- GPU:
  `8 GPU`
- total step:
  `110`
- epoch:
  `2`
- runtime:
  `1268초` (`21분 08초`)
- train loss:
  `0.3918`
- train samples/sec:
  `5.509`
- train steps/sec:
  `0.087`

산출물:

- `checkpoint-55`
- `checkpoint-110`
- `final`

## 왜 `vllmfix3`가 필요했나

`trainer.save_model()`로 저장된 full FT 원본은 `vLLM`이 바로 읽는 표준 HF `Qwen3.5-VL` 형식이 아니었습니다.

원본 key 예:

- `model.language_model.language_model.language_model.layers...`
- `model.language_model.visual...`

`vLLM`의 `Qwen3-VL` 매퍼가 기대하는 HF key:

- `model.language_model...`
- `model.visual...`

그래서 아래 스크립트로 key prefix를 표준 HF 형식으로 다시 썼습니다.

- [fix_fullft_export_prefix.py](/home/work/.projects/LLM-OS-Models/Terminal/qwen_sft/scripts/fix_fullft_export_prefix.py)

평가에 실제 사용한 디렉터리:

- final:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/final-vllmfix3`
- 1 epoch:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/checkpoint-55-vllmfix3`

## Full FT 평가 결과

결과 JSON:

- [Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount-vllmfix3.json](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_final_vllmfix3/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount-vllmfix3.json)
- [Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount-vllmfix3.json](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_ckpt55_final_vllmfix3/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount-vllmfix3.json)

| 모델 | Score | Cmd F1 | First Cmd Exact | Load(s) | Gen(s) | Sec/Step | 전체 순위 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 29.77 | 0.2912 | 31.3% | 52.6 | 11.6 | 0.030 | 3 |
| `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 29.66 | 0.2917 | 30.8% | 52.2 | 11.2 | 0.029 | 4 |

비교:

- base `Qwen/Qwen3.5-2B (26.52)` 대비
  - `1 epoch`: `+3.14`
  - `2 epoch`: `+3.25`
- LoRA final `22.84` 대비
  - `1 epoch`: `+6.82`
  - `2 epoch`: `+6.93`
- `2 epoch`가 `1 epoch`보다 `+0.11`

판단:

- 이번 `Qwen3.5-2B`에서는 **LoRA보다 full FT가 압도적으로 낫다**
- `1 epoch`만으로도 거의 상위 성능에 도달
- `2 epoch final`은 현재 전체 `27개 중 3위`

## 허깅페이스 업로드

- repo:
  `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`

현재 올라간 구성:

- root: `final` 기준 파일
- `checkpoint-55/`
- `checkpoint-110/`
- `README.md`

## 다음 판단

- `Qwen3.5-2B`는 full FT 경로로 계속 보는 게 맞습니다.
- 이후 `4B`, `9B`도 같은 저장 정리 경로를 재사용하면 됩니다.
- full FT 결과는 이미 `tb2_lite` 상위권이므로, 다음 실험은 `Qwen3.5-4B` 이상으로 넘어갈 가치가 있습니다.
