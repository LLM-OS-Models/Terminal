# DeepSeek V4 전용 평가 진행 기록 (2026-04-25)

## 왜 별도 경로로 전환했는가

이번 DeepSeek 평가는 기존 `tb2_lite`의 공용 vLLM evaluator로 바로 처리하지 않았습니다.

- 현재 환경의 `transformers==5.6.2`에서 `model_type=deepseek_v4`를 인식하지 못함
- 현재 vLLM 경로에서도 `DeepSeek-V4`를 바로 붙이기 어려움
- 공식 모델 카드와 inference README가 **전용 convert + 전용 generate 경로**를 제공함

그래서 이번 런은 **DeepSeek 전용 inference code path**로 전환했습니다.

## 공식 참고 문서

- Pro inference README:
  - https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/README.md
- Flash inference README:
  - https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/README.md
- 공통 chat encoding:
  - https://huggingface.co/unsloth/DeepSeek-V4-Pro

핵심 설정은 공식 README를 따랐습니다.

- `Flash`: `EXPERTS=256`, `MP=4`
- `Pro`: `EXPERTS=384`, `MP=8`
- expert quantization은 기본값 유지
  - 즉 `expert_dtype=fp4` 유지

## 이번에 추가한 코드

- DeepSeek 전용 inference 코드 복사:
  - [tb2_lite/deepseek_v4/encoding/encoding_dsv4.py](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/deepseek_v4/encoding/encoding_dsv4.py)
  - [tb2_lite/deepseek_v4/inference/model.py](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/deepseek_v4/inference/model.py)
  - [tb2_lite/deepseek_v4/inference/convert.py](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/deepseek_v4/inference/convert.py)
  - [tb2_lite/deepseek_v4/inference/generate.py](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/deepseek_v4/inference/generate.py)
  - [tb2_lite/deepseek_v4/inference/kernel.py](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/deepseek_v4/inference/kernel.py)

- DeepSeek 전용 replay evaluator:
  - [tb2_lite/scripts/deepseek_replay_eval.py](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/scripts/deepseek_replay_eval.py)

- 8 GPU 전체 활용용 실행 스크립트:
  - [tb2_lite/run_deepseek_flash_full8.sh](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/run_deepseek_flash_full8.sh)
  - [tb2_lite/run_deepseek_pro_full8.sh](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/run_deepseek_pro_full8.sh)

- Flash shard 병합 스크립트:
  - [tb2_lite/scripts/merge_deepseek_shards.py](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/scripts/merge_deepseek_shards.py)

## 구현 메모

- `fast_hadamard_transform` 패키지는 현재 env에서 wheel/build 문제가 있어 설치 실패
- 그래서 [model.py](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/deepseek_v4/inference/model.py)에 **PyTorch fallback Hadamard transform**을 추가
- `tilelang==0.1.8`는 `.eval-env`에 설치 완료

## 실행 계획

1. `Flash`를 공식 권장 `MP=4`로 convert
2. `Flash`를 `0-3`, `4-7` 두 shard로 나눠 **8 GPU 전체 사용** replay 평가
3. shard 결과 병합
4. 그 후 `Pro`를 공식 권장 `MP=8`로 convert
5. `Pro`를 **8 GPU 전체 사용** replay 평가

## 최종 상태

이번 DeepSeek V4 평가는 **변환까지는 성공**, 하지만 **full replay 평가는 중단**했습니다.

- `Flash` convert 완료
  - source snapshot:
    - `/home/work/.data/huggingface/models--unsloth--DeepSeek-V4-Flash/snapshots/bc486f653513c9179e20a970587dcbe928bf7b96`
  - convert output:
    - `/home/work/deepseek_models/DeepSeek-V4-Flash-mp4`
- `Pro` convert 완료
  - source snapshot:
    - `/home/work/.data/huggingface/models--unsloth--DeepSeek-V4-Pro/snapshots/baeea9247452e63fcbcea672a588b6db36edd378`
  - convert output:
    - `/home/work/deepseek_models/DeepSeek-V4-Pro-mp8`
- `Flash` smoke run은 성공
- `Flash` full replay run은 여러 설정으로 시도했지만, 속도가 비실용적이라 중단
- `Pro` full replay run은 시작하지 않음

중단 사유와 상세 수치는 아래 문서에 정리했습니다.

- [DEEPSEEK_FULL_EVAL_ABORT_2026-04-25.md](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/docs/DEEPSEEK_FULL_EVAL_ABORT_2026-04-25.md)
