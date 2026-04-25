# TB2-lite

`tb2_lite/`는 이 저장소에서 **기존 `eval/`보다 한 단계 더 믿을 만한 터미널 평가**를 하기 위해 만든 폴더입니다.

## 왜 만들었나

기존 `eval/` 평가는 빠르다는 장점이 있지만, 실제로는 첫 질문에 대한 단발 응답과 문자열 겹침 위주로 점수를 냅니다.  
이 방식은 모델이 "터미널처럼 보이는 답"을 잘 쓰는지는 빠르게 볼 수 있어도, **중간 출력이 바뀌었을 때 다음 행동을 제대로 고르는지**는 충분히 보지 못합니다.

그래서 `tb2_lite/`는 다음 목적을 위해 따로 분리했습니다.

- `eval/`은 계속 **초고속 1차 스크리닝** 용도로 유지
- `tb2_lite/`는 **멀티턴 상태 추적과 다음 명령 선택 품질**을 더 강하게 보는 평가로 운영
- 나중에 Docker 가능한 환경이 생기면, 이 폴더를 기반으로 **실제 Terminal-Bench 2.0 실행 평가**까지 확장

즉, 이 폴더의 역할은 **"빠르기만 한 평가"와 "너무 느린 실제 벤치" 사이를 메우는 중간 단계**입니다.

## 이 폴더가 보는 것

현재 `tb2_lite/`의 핵심 질문은 하나입니다.

**"모델이 현재 터미널 상태를 보고, 다음에 어떤 명령을 내야 하는지 더 정확하게 판단하는가?"**

이를 위해 기존 데이터셋의 멀티턴 trajectory를 평탄화해서, 각 assistant 턴을 하나의 평가 스텝으로 바꿉니다.  
한마디로 말하면, 기존처럼 첫 턴만 보지 않고 **각 단계의 user 상태를 모두 다시 재생(replay)** 해서 평가합니다.

## 데이터 구성

입력 원본은 `eval/eval_dataset.jsonl` 입니다.

이 데이터는 이미 멀티턴 대화 구조를 가지고 있고, 각 user 메시지에는 보통 아래 정보가 들어 있습니다.

- 현재 해야 할 작업 설명
- 이전에 실행한 명령의 출력
- 현재 터미널 화면 상태

각 assistant 메시지에는 보통 아래 정보가 들어 있습니다.

- `analysis`
- `plan`
- `commands`
- `task_complete`

`tb2_lite/scripts/build_replay_dataset.py` 는 이 구조를 이용해서 데이터를 두 단계로 바꿉니다.

1. 각 task를 여러 개의 step으로 분해
2. 각 step마다
   - 입력: 해당 시점의 user 메시지
   - 정답: 그 다음 assistant JSON 응답

이렇게 만들면 모델이 단순히 첫 답변만 잘하는지가 아니라, **중간 진행 상황이 달라질 때도 계속 맞는 방향으로 가는지** 볼 수 있습니다.

생성되는 주요 파일은 다음과 같습니다.

- `data/replay_full.jsonl`
  - 전체 task의 모든 assistant step을 펼친 full replay 세트
- `data/replay_dev_20.jsonl`
  - 빠른 실험용 dev subset
- `data/task_manifest.json`
  - task 수, step 수, source group 등 메타데이터

## 실험 구성

현재 실험은 vLLM 기반으로 돌아갑니다.

평가 흐름은 다음과 같습니다.

1. replay step마다 user 상태를 프롬프트로 넣음
2. 모델이 assistant JSON 응답을 생성
3. 생성 결과에서 `commands`, `task_complete`, `analysis`, `plan` 등을 파싱
4. 정답 step과 비교해서 점수를 계산

주요 평가지표는 다음과 같습니다.

- `avg_command_f1`
  - 예측한 명령과 정답 명령이 얼마나 비슷한지 보는 핵심 지표
- `first_cmd_exact_pct`
  - 첫 번째 행동을 정확히 맞추는 비율
- `valid_json_pct`
  - 형식을 제대로 지키는 비율
- `complete_true_recall_pct`
  - 실제 완료 시점에 `task_complete=true`를 제대로 내는 비율
- `premature_complete_rate_pct`
  - 아직 안 끝났는데 너무 일찍 완료라고 하는 비율

실무적으로는 `avg_command_f1`과 `first_cmd_exact_pct`를 가장 중요하게 봅니다.  
요약 표에서는 이 둘을 합친 `next_action_score`도 같이 기록합니다.

## 현재 선택한 모델 실험 세트

기본 top-5 실험 세트는 `configs/models_top5_2026-04-25.json` 에 정리돼 있습니다.

현재 기준 선택 이유는 아래와 같습니다.

- `google/gemma-4-E2B-it`
  - 기존 빠른 평가에서 가장 안정적으로 상위권
- `LiquidAI/LFM2-8B-A1B`
  - Liquid 계열 중 가장 유력한 후보
- `nvidia/Nemotron-Terminal-8B`
  - 소형 terminal-specialized baseline
- `nvidia/Nemotron-Terminal-14B`
  - 중형 terminal-specialized baseline
- `google/gemma-4-31B-it`
  - 큰 일반 모델과의 비교 기준

실행 스크립트는 `run_top5.sh` 입니다.

## `eval/` 과의 역할 분리

헷갈리지 않게 역할을 분리하면 아래와 같습니다.

- `eval/`
  - 빠른 프록시 평가
  - 많은 모델을 짧게 훑는 용도
  - 체크포인트 선별용
- `tb2_lite/`
  - 멀티턴 replay 기반 평가
  - 다음 행동 선택 품질 확인용
  - 실제 TB2 전 단계 판단용

즉, 앞으로는 **`eval/`로 후보를 줄이고, `tb2_lite/`로 진짜 괜찮은지 다시 확인**하는 식으로 쓰면 됩니다.

## 현재 한계

현재 머신에는 `docker`가 없어서, 이 폴더는 아직 **진짜 Terminal-Bench 2.0 환경 실행기**는 아닙니다.

그래서 지금 구현은 다음 조건에서의 최선입니다.

- 실제 task environment 실행은 못 함
- 대신 멀티턴 terminal trajectory replay는 가능
- 단발 문자열 overlap보다 훨씬 강한 평가 가능

이 제약은 `ISSUES_2026-04-25.md` 에도 기록했습니다.

## 폴더 구조

- `configs/`
  - 모델 목록과 실행 설정
- `data/`
  - replay 데이터셋과 manifest
- `scripts/`
  - 데이터 빌더, evaluator, 요약기
- `runs/`
  - raw 실행 산출물 보관용
- `results/`
  - 실험 결과 요약 보관용

## 권장 사용 순서

1. `python tb2_lite/scripts/build_replay_dataset.py`
2. `bash tb2_lite/run_top5.sh`
3. `tb2_lite/results/<RUN_ID>/SUMMARY.md` 확인

한 줄로 요약하면, `tb2_lite/`는 **"빠른 프록시보다 더 실제에 가깝고, full TB2보다는 훨씬 가벼운 중간 평가 레이어"** 입니다.
