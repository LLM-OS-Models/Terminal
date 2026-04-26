# TB2-lite vLLM 이슈 기록 (2026-04-26)

## 왜 이 문서를 만들었나

로컬 SFT final 모델을 `tb2_lite`에서 `vLLM + tp=8`로 바로 평가하려 했지만, 멀티 GPU 초기화에서 실패했기 때문에 원인과 결론을 남깁니다.

## 대상 모델

- 모델:
  `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- 경로:
  `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local/final`

## 시도한 설정

- 엔진:
  `vLLM 0.19.1`
- GPU:
  `0,1,2,3,4,5,6,7`
- tensor parallel:
  `tp=8`
- 데이터:
  `tb2_lite/data/replay_dev_20.jsonl`

## 실패 내용

실패는 inference 시작 전, worker 초기화 단계에서 발생했습니다.

- 실패 위치:
  `vllm/distributed/device_communicators/pynccl.py`
- 핵심 에러:
  `RuntimeError: NCCL error: unhandled cuda error`

직접적으로는 `PyNcclCommunicator` 생성과 `ncclCommInitRank` 초기화 과정에서 worker들이 올라오지 못했습니다.

## 판단

- 로컬 경로라서 실패한 것은 아님
- HF에 업로드하면 자동으로 해결될 문제도 아님
- 이번 이슈의 성격은 **vLLM 멀티 GPU NCCL 초기화 실패**입니다

즉 같은 모델이라도:

- `tp=1`은 정상 동작
- `tp=8`은 현재 환경에서 실패

## 이번 대응

- `tp=8` 시도 중단
- `1 GPU + vLLM`로 full replay 평가 진행
- 정상 평가 결과는 아래 문서에 기록

- [TB2_LITE_RESULTS_2026-04-26.md](/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/docs/TB2_LITE_RESULTS_2026-04-26.md)

## 다음에 볼 것

- `NCCL_DEBUG=INFO`로 상세 재현
- 같은 환경에서 다른 로컬 모델 `tp=8` 재확인
- `vllm` / `torch` / `nccl` 조합 재점검

한 줄 결론:

**이번 멀티 GPU 실패는 모델 품질 문제가 아니라, `vLLM tp=8` 초기화 문제입니다.**
