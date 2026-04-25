# TB2-lite 미실험 / 실패 모델 정리 (2026-04-25)

## 결론

기존 `eval/`에서 점수가 있었던 모델 중, 이번 `tb2_lite`에서 **정상적으로 점수를 못 낸 모델은 1개뿐**입니다.

- `OBLITERATUS/gemma-4-E4B-it-OBLITERATED`

즉, 나머지는 전부 이번 `tb2_lite` replay 평가에 넣었습니다.

## 실패 사유

`OBLITERATUS/gemma-4-E4B-it-OBLITERATED` 는 아래 이유로 최종 스킵했습니다.

- 기본 실행: feature extractor / `preprocessor_config.json` 로딩 실패
- base Gemma tokenizer override 재시도: 동일한 processor 로딩 단계에서 다시 실패

즉 이 모델은 현재 환경 기준으로는 **vLLM replay 평가 경로에 바로 태우기 어려운 상태**입니다.

관련 로그:

- [원본 실패 로그](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T193000Z_remaining/gemma-4-E4B-it-OBLITERATED.log)
- [재시도 실패 로그](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260425T193000Z_remaining/gemma-4-E4B-it-OBLITERATED_retry.log)

## 별도 메모

현재 기준으로는 **기존 eval 대상 모델은 사실상 전부 커버 완료**라고 보면 됩니다.  
추가로 더 돌릴 것은 이제 "미실험 모델 정리"가 아니라 **새로운 후보 발굴** 쪽입니다.
