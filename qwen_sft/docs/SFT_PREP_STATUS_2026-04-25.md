# Liquid SFT 준비 상태 2026-04-25

## 왜 이 문서를 만들었나

학습이 시작되면 checkpoint가 여러 개 생기기 때문에, 나중에 저장 위치를 바로 찾을 수 있게 현재 준비 상태와 저장 경로를 한 문서에 고정해두기 위해 만들었습니다.

## 현재 상태

- raw dataset download: 완료
- custom non-coding filtering: 완료
- text dataset 생성: 완료
- tokenized dataset 생성: 완료
- base model artifact download: 완료
- Unsloth 학습 환경: 완료
- 8 GPU DDP SFT 코드: 완료
- actual training run: 완료

## 데이터 개수

- raw samples: `139,841`
- filtered samples: `3,510`
- text samples: `3,510`
- tokenized samples: `3,510`

## 저장 경로

- raw dataset:
  `/home/work/.data/liquid_cli_sft/datasets/raw_skill_based`
- filtered dataset:
  `/home/work/.data/liquid_cli_sft/datasets/sft_data`
- training processed cache:
  `/home/work/.data/liquid_cli_sft/datasets/sft_data_unsloth_processed`
- text dataset:
  `/home/work/.data/liquid_cli_sft/datasets/sft_text`
- tokenized dataset:
  `/home/work/.data/liquid_cli_sft/datasets/sft_tokenized`
- training model source:
  `LiquidAI/LFM2-8B-A1B`
- downloaded local artifact:
  `/home/work/.data/liquid_cli_sft/models/unsloth__LFM2-8B-A1B`
- SFT checkpoints:
  `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local`
- additional downloaded local artifact:
  `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2.5-1.2B-Base`
- additional SFT checkpoints:
  `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2.5-1.2B-Base__terminal_sft_h200_4gpu`
- additional downloaded local artifact:
  `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2-2.6B`
- additional SFT checkpoints:
  `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2-2.6B__terminal_sft_h200_4gpu`
- training logs:
  `/home/work/.data/liquid_cli_sft/logs/`

## 체크포인트 정책

- 저장 주기: `1 epoch`
- 자동 삭제: 없음
- 최종 저장 위치:
  `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local/final`

## 실제 학습 결과

- 실행 설정:
  `8 GPU`, `per_device_train_batch_size=16`, `gradient_accumulation_steps=1`
- 총 step:
  `54`
- checkpoint:
  `checkpoint-27`, `checkpoint-54`, `final`
- train runtime:
  `520초` (`8분 39초`)
- train samples/sec:
  `13.25`
- train steps/sec:
  `0.104`
- final output root size:
  `80G`

## HF 목표 이름

- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth`

## 추가 학습 결과

- `LiquidAI/LFM2.5-1.2B-Base`
  - 실행 설정:
    `4 GPU`, `per_device_train_batch_size=32`, `gradient_accumulation_steps=1`
  - 총 step:
    `54`
  - checkpoint:
    `checkpoint-27`, `checkpoint-54`, `final`
  - train runtime:
    `480.2초` (`8분 00초`)
  - train samples/sec:
    `14.35`
  - train steps/sec:
    `0.112`
  - output:
    `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2.5-1.2B-Base__terminal_sft_h200_4gpu`

- `LiquidAI/LFM2-2.6B`
  - 실행 설정:
    `4 GPU`, `per_device_train_batch_size=32`, `gradient_accumulation_steps=1`
  - 총 step:
    `54`
  - checkpoint:
    `checkpoint-27`, `checkpoint-54`, `final`
  - train runtime:
    `909.4초` (`15분 09초`)
  - train samples/sec:
    `7.574`
  - train steps/sec:
    `0.059`
  - output:
    `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2-2.6B__terminal_sft_h200_4gpu`

## 실행 명령

```bash
bash liquid_sft/scripts/run_sft_8gpu.sh
```
