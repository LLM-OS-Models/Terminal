# Gemma4 SFT 가이드

이 폴더는 기존 `liquid_sft/`를 건드리지 않고, `Gemma 4` 계열 터미널 SFT를 별도 경로에서 돌리기 위한 레이어입니다.

기준 날짜: `2026-04-26`

## 왜 따로 만들었나

- `Gemma 4`는 `Qwen`, `LFM`과 채팅 템플릿이 다릅니다.
- `31B-it`, `26B-A4B-it` 같은 text/MoE 계열과
  `E2B`, `E4B` 같은 processor/vision 계열을 분리해서 다뤄야 합니다.
- 그래서 text 경로부터 먼저 독립시켰고, 이후 `E2B`, `E4B`는 vision 경로로 추가할 예정입니다.

참고 문서:

- `Gemma 4 Fine-tuning Guide`
  https://unsloth.ai/docs/models/gemma-4/train

## 현재 준비된 모델

- `google/gemma-4-31B-it`
- `google/gemma-4-26B-A4B-it`

## 저장 경로

- 데이터 루트:
  `/home/work/.data/gemma4_sft/datasets`
- 모델 루트:
  `/home/work/.data/gemma4_sft/models`
- 로그 루트:
  `/home/work/.data/gemma4_sft/logs`

## 설정 파일

- 31B:
  `gemma4_sft/configs/sft_gemma4_31b_8gpu.env`
- 26B-A4B:
  `gemma4_sft/configs/sft_gemma4_26b_a4b_8gpu.env`

## 실행 방식

31B:

```bash
bash gemma4_sft/scripts/run_sft_8gpu.sh --config gemma4_sft/configs/sft_gemma4_31b_8gpu.env
```

26B-A4B:

```bash
bash gemma4_sft/scripts/run_sft_8gpu.sh --config gemma4_sft/configs/sft_gemma4_26b_a4b_8gpu.env
```

## 현재 판단

- `Gemma 4` text 계열은 `AutoTokenizer` / `FastLanguageModel` 기준으로 먼저 붙일 수 있습니다.
- 응답 시작 구간은 `<|turn>model\n` 기준으로 마스킹합니다.
- `E2B`, `E4B`는 `AutoProcessor` + `FastVisionModel` 경로가 필요해서 별도 구현 대상으로 둡니다.
