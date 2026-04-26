# Liquid CLI SFT 가이드

이 폴더는 `Liquid-CLI` 원본을 직접 건드리지 않고, 현재 작업 폴더 기준으로 `LFM2-8B-A1B` SFT를 준비하고 학습하기 위한 별도 레이어입니다.

기준 날짜: `2026-04-26`

## 핵심 결론

헷갈리면 안 되는 부분부터 적으면:

- **베이스 모델은 `LiquidAI/LFM2-8B-A1B` 원본 모델**입니다.
- **학습 프레임워크는 Unsloth** 입니다.
- 즉 **“Liquid 원본 모델을 Unsloth 파이프라인으로 학습한 것”** 입니다.
- 최종 산출물은 아래 경로에 저장돼 있습니다.
  `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local/final`

오늘 만든 모델들의 HF 목표 이름:

- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth`

즉 지금 만든 모델은:

- 모델 계보 기준으로는 `LiquidAI/LFM2-8B-A1B` 파생
- 학습 구현 기준으로는 `Unsloth + torchrun DDP`

입니다.

## 현재 경로

- Liquid-CLI 원본:
  `./Liquid-CLI`
- Unsloth 원본:
  `./unsloth-src`
- 준비/학습 코드:
  `./liquid_sft`
- 데이터/모델/출력 루트:
  `/home/work/.data/liquid_cli_sft`

## 데이터 경로

- raw skill-based merge:
  `/home/work/.data/liquid_cli_sft/datasets/raw_skill_based`
- non-coding filtered:
  `/home/work/.data/liquid_cli_sft/datasets/sft_data`
- training processed cache:
  `/home/work/.data/liquid_cli_sft/datasets/sft_data_unsloth_processed`
- text-formatted:
  `/home/work/.data/liquid_cli_sft/datasets/sft_text`
- tokenized:
  `/home/work/.data/liquid_cli_sft/datasets/sft_tokenized`

## 모델 경로

- training model source:
  `LiquidAI/LFM2-8B-A1B`
- downloaded local base artifact:
  `/home/work/.data/liquid_cli_sft/models/unsloth__LFM2-8B-A1B`
- SFT checkpoint/output root:
  `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local`
- final model:
  `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local/final`
- additional downloaded base artifact:
  `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2.5-1.2B-Base`
- additional checkpoint/output root:
  `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2.5-1.2B-Base__terminal_sft_h200_4gpu`
- additional downloaded base artifact:
  `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2-2.6B`
- additional checkpoint/output root:
  `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2-2.6B__terminal_sft_h200_4gpu`

주의:

- 로컬에 `unsloth__LFM2-8B-A1B` 아티팩트를 받아두긴 했지만,
- 실제 학습 설정에서 쓰는 모델명은 `LiquidAI/LFM2-8B-A1B` 입니다.
- 현재 학습 코드가 이 모델명을 가장 안정적으로 인식합니다.

## 현재 상태

- raw dataset download: 완료
- non-coding filtering: 완료
- text dataset 생성: 완료
- tokenized dataset 생성: 완료
- base model artifact download: 완료
- Unsloth 학습 환경: 완료
- 8 GPU DDP 학습 코드: 완료
- actual training run: 완료

데이터 개수:

- raw samples:
  `139,841`
- filtered samples:
  `3,510`
- text samples:
  `3,510`
- tokenized samples:
  `3,510`

## 실제 학습 결과

- base model:
  `LiquidAI/LFM2-8B-A1B`
- framework:
  `Unsloth`
- 실행 방식:
  `torchrun --nproc_per_node 8`
- GPU:
  `8 x H200`
- epochs:
  `2`
- max_seq_length:
  `8192`
- per_device_train_batch_size:
  `16`
- gradient_accumulation_steps:
  `1`
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

대표 로그:

- `/home/work/.data/liquid_cli_sft/logs/sft_8gpu_20260425T145340Z.log`

추가 학습 결과:

- `LiquidAI/LFM2.5-1.2B-Base`
  - framework:
    `Unsloth`
  - 실행 방식:
    `torchrun --nproc_per_node 4`
  - GPU:
    `4 x H200`
  - epochs:
    `2`
  - per_device_train_batch_size:
    `32`
  - gradient_accumulation_steps:
    `1`
  - 총 step:
    `54`
  - runtime:
    `480.2초` (`8분 00초`)
  - train samples/sec:
    `14.35`
  - train steps/sec:
    `0.112`
  - checkpoint:
    `checkpoint-27`, `checkpoint-54`, `final`
  - output:
    `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2.5-1.2B-Base__terminal_sft_h200_4gpu`

- `LiquidAI/LFM2-2.6B`
  - framework:
    `Unsloth`
  - 실행 방식:
    `torchrun --nproc_per_node 4`
  - GPU:
    `4 x H200`
  - epochs:
    `2`
  - per_device_train_batch_size:
    `32`
  - gradient_accumulation_steps:
    `1`
  - 총 step:
    `54`
  - runtime:
    `909.4초` (`15분 09초`)
  - train samples/sec:
    `7.574`
  - train steps/sec:
    `0.059`
  - checkpoint:
    `checkpoint-27`, `checkpoint-54`, `final`
  - output:
    `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2-2.6B__terminal_sft_h200_4gpu`

작은 모델들은 둘 다 아래 설정으로 학습했습니다.

- `2 epochs`
- `save_strategy=epoch`
- `per_device_train_batch_size=32`
- `gradient_accumulation_steps=1`
- model별 global batch:
  `128`

## 체크포인트 정책

- 저장 주기:
  `1 epoch`
- 자동 삭제:
  없음
- checkpoint root:
  `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local`
- final:
  `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local/final`

## 관련 코드

- env 설치:
  `liquid_sft/scripts/setup_env.sh`
- 모델 다운로드:
  `liquid_sft/scripts/download_model.py`
- 데이터 전처리:
  `liquid_sft/scripts/prepare_dataset.py`
- 8GPU 학습 코드:
  `liquid_sft/scripts/train_sft_unsloth_ddp.py`
- 8GPU 실행 런처:
  `liquid_sft/scripts/run_sft_8gpu.sh`
- 설정 파일:
  `liquid_sft/configs/sft_h200_8gpu.env`
  `liquid_sft/configs/sft_h200_4gpu_lfm25_1p2b_base.env`
  `liquid_sft/configs/sft_h200_4gpu_lfm2_2p6b.env`

현재 핵심 설정:

- `MODEL_NAME=LiquidAI/LFM2-8B-A1B`
- `MODEL_PATH=LiquidAI/LFM2-8B-A1B`
- `OUTPUT_DIR=/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local`
- `PER_DEVICE_TRAIN_BATCH_SIZE=16`
- `GRADIENT_ACCUMULATION_STEPS=1`
- `NUM_TRAIN_EPOCHS=2`
- `SAVE_STRATEGY=epoch`

## 실행 예시

### 환경 설치

```bash
bash liquid_sft/scripts/setup_env.sh
```

### 모델 다운로드

```bash
source .liquid-sft-env/bin/activate
python liquid_sft/scripts/download_model.py
```

### 데이터 전처리

```bash
source .liquid-sft-env/bin/activate
python liquid_sft/scripts/prepare_dataset.py
```

### 실제 8GPU SFT 시작

```bash
bash liquid_sft/scripts/run_sft_8gpu.sh
```

## 왜 이렇게 구성했나

- `Liquid-CLI` 원본은 그대로 두고 싶었음
- 학습용 파이프라인은 별도 레이어에서 관리하는 편이 안전함
- Unsloth 쪽이 multi-GPU SFT 준비 속도가 빠름
- H200 8장 환경에서 빠르게 2 epoch를 확인하기에 적합했음

## 참고

- Unsloth multi-GPU:
  `https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth`
- Unsloth DDP:
  `https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp`
