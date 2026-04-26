# Qwen3.5 SFT 가이드

이 폴더는 기존 `liquid_sft/`를 건드리지 않고, `Qwen3.5` 계열 터미널 SFT를 별도 경로에서 돌리기 위한 레이어입니다.

기준 날짜: `2026-04-26`

## 왜 따로 만들었나

- `Qwen3.5`는 `Unsloth` 문서 기준으로 최신 `transformers v5` 계열이 필요합니다.
- `LFM`과 채팅 템플릿이 달라서 `response-only` 마스킹 구간도 따로 잡아야 합니다.
- `Qwen3.5-2B` 실험 결과가 좋아서 이후 `4B`, `9B`까지 같은 형식으로 확장하려고 폴더를 분리했습니다.
- 지금은 `Qwen3.5-9B` full FT 결과까지 확보했고, 다음 순서는 `27B`입니다.

참고 문서:

- Unsloth Qwen3.5 Fine-tuning Guide  
  https://unsloth.ai/docs/models/qwen3.5/fine-tune

## 저장 경로

- 데이터 루트:
  `/home/work/.data/qwen_sft/datasets`
- 모델 루트:
  `/home/work/.data/qwen_sft/models`
- 로그 루트:
  `/home/work/.data/qwen_sft/logs`

## 현재 완료된 실험

### 1. Qwen3.5-2B LoRA

- output:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_unsloth_lora`
- final:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_unsloth_lora/final_official`
- HF target:
  `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth`
- 평가:
  `Score 22.84`, `Cmd F1 0.2586`, `First 15.8%`

판단:

- base `Qwen/Qwen3.5-2B (26.52)`보다 낮았습니다.
- 즉 이번 설정에서 LoRA는 실패입니다.

### 2. Qwen3.5-2B Full FT same-count

- config:
  `qwen_sft/configs/sft_qwen35_2b_fullft_samecount_8gpu.env`
- output:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount`
- HF repo:
  `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`

학습 결과:

- `8 GPU`
- `2 epochs`
- `110 step`
- `1268초` (`21분 08초`)
- `train_loss 0.3918`

평가 결과:

- `1 epoch`: `Score 29.66`, `Cmd F1 0.2917`, `First 30.8%`
- `2 epoch final`: `Score 29.77`, `Cmd F1 0.2912`, `First 31.3%`

판단:

- `Qwen3.5-2B`는 **LoRA보다 full FT가 훨씬 낫습니다.**
- `2 epoch final`은 현재 전체 `tb2_lite` 기준 `3위`입니다.

### 3. Qwen3.5-4B Full FT 2BData

- config:
  `qwen_sft/configs/sft_qwen35_4b_fullft_samecount_8gpu.env`
- output:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata`
- eval-ready final:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata/final-vllmfix4`
- eval-ready 1 epoch:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata/checkpoint-960-vllmfix4`
- HF target:
  `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData`
- HF upload:
  완료

학습 결과:

- `8 GPU`
- `2 epochs`
- `1920 step`
- `4986초` (`1시간 23분`)
- `train_loss 0.6032`

평가 결과:

- `1 epoch`: `Score 28.89`, `Cmd F1 0.3185`, `First 22.0%`, 전체 `6위`
- `2 epoch final`: `Score 26.80`, `Cmd F1 0.2907`, `First 21.5%`, 전체 `15위`

판단:

- `4B`는 `1 epoch`가 더 좋았습니다.
- 즉 이번 설정에선 `checkpoint-960`가 핵심 체크포인트입니다.

### 4. Qwen3.5-9B Full FT 2BData

- config:
  `qwen_sft/configs/sft_qwen35_9b_fullft_samecount_8gpu.env`
- output:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata`
- eval-ready final:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata/final-vllmfix9`
- eval-ready 1 epoch:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata/checkpoint-2193-vllmfix9`
- HF target:
  `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData`

학습 결과:

- `7 GPU`
- `2 epochs`
- `4386 step`
- `10990초` (`3시간 03분`)
- `train_loss 0.5656`

평가 결과:

- `1 epoch`: `Score 27.45`, `Cmd F1 0.3021`, `First 21.0%`, 전체 `15위`
- `2 epoch final`: `Score 28.80`, `Cmd F1 0.3094`, `First 23.8%`, 전체 `7위`

판단:

- `9B`는 `4B`와 다르게 `2 epoch final`이 더 좋았습니다.
- base `Qwen/Qwen3.5-9B (27.04)` 대비 `1 epoch +0.41`, `2 epoch final +1.76`입니다.
- 점수는 올랐지만 기대한 수준만큼 크게 튀지는 않았습니다.

## 왜 `vllmfix3`가 있나

full FT 원본 저장물은 `trainer.save_model()` 기준으로 key prefix가 한 층 더 감겨서 `vLLM`이 바로 못 읽었습니다.

그래서 아래 스크립트로 표준 HF `Qwen3.5-VL` 형식으로 다시 썼습니다.

- [fix_fullft_export_prefix.py](/home/work/.projects/LLM-OS-Models/Terminal/qwen_sft/scripts/fix_fullft_export_prefix.py)

실제 평가 경로:

- final:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/final-vllmfix3`
- 1 epoch:
  `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount/checkpoint-55-vllmfix3`

## 실행 방식

기본 실행:

```bash
bash qwen_sft/scripts/run_sft_8gpu.sh --config qwen_sft/configs/sft_qwen35_2b_fullft_samecount_8gpu.env
```

드라이런:

```bash
bash qwen_sft/scripts/run_sft_8gpu.sh --config qwen_sft/configs/sft_qwen35_2b_fullft_samecount_8gpu.env --dry-run
```

## 현재 판단

- `Qwen3.5-2B`는 full FT 경로로 계속 보는 게 맞습니다.
- `4B`, `9B`도 같은 저장 정리 경로를 재사용하면 됩니다.
- `9B`까지 완주했고, 다음은 `27B`입니다.
- 평가 엔진은 `/.vllm-uv-env`를 쓰고, 항상 `unset PYTHONPATH` + `PYTHONNOUSERSITE=1`가 필요했습니다.
