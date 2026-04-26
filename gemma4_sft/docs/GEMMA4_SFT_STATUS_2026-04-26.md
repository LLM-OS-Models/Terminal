# Gemma4 SFT 상태 (2026-04-26)

## 현재 진행

- 폴더 분리: 완료
- text 계열 학습 코드: 완료
- `31B-it`, `26B-A4B-it` config 준비: 완료
- `E2B`, `E4B` processor/vision 경로 확인: 진행 중

## 준비된 HF target

- `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-2Epoch-Unsloth`

## 현재 확인된 기술 메모

- `Gemma 4`는 일반 chat role을 씁니다.
- text 계열 응답 시작 구간은 `<|turn>model\n` 입니다.
- `26B-A4B-it`, `31B-it`는 text/MoE 경로
- `E2B`, `E4B`는 processor + vision loader 경로
