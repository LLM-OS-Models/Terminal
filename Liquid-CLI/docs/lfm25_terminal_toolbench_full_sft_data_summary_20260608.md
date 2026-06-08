# LFM2.5 Terminal ToolBench Full SFT 데이터 요약

작성일: 2026-06-08

대상 모델:

- `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- 같은 학습 run의 2epoch 모델: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-2Epoch`

## 결론

이 모델의 높은 TB2-lite 점수는 단순히 모델 체급만으로 나온 결과가 아니다.

핵심은 다음 두 종류의 데이터를 크게 합쳐서 학습했다는 점이다.

1. Terminal processed train 전체
2. ToolBench train 전체

최종 merge 데이터는 총 `327,383` rows다.

- Terminal processed rows: `139,841`
- ToolBench train rows: `187,542`
- Total rows: `327,383`

전체 merge 데이터 기준 비율은 다음과 같다.

- Terminal processed: `42.71%`
- ToolBench train: `57.29%`

즉, 이 run은 작은 필터셋만 학습한 것이 아니라, 터미널 코퍼스의 주요 도메인을 폭넓게 포함한 terminal processed 전체와 ToolBench train 전체를 같이 학습한 run이다.

## 사용한 설정

학습 config:

- `Liquid-CLI/configs/sft_h200_4gpu_lfm25_8b_a1b_terminal_toolbench_full.env`

주요 값:

- `MODEL_PATH=LiquidAI/LFM2.5-8B-A1B`
- `DATASET_NAME=gyung/LFM2-Terminal-SFT-Processed`
- `DATASET_SPLIT=train`
- `DATASET_PATH=/home/work/.data/liquid_cli_sft/datasets/lfm25_8b_a1b_terminal_full_toolbench_full_conversations_v1`
- `MAX_SEQ_LENGTH=8192`
- `NUM_TRAIN_EPOCHS=2`
- `HUB_MODEL_ID=LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-2Epoch`

1epoch 모델은 같은 run의 `checkpoint-1542`를 업로드한 모델이고, 2epoch 모델은 `checkpoint-3084` 또는 final 쪽이다.

## 데이터 빌드 경로

데이터 merge 스크립트:

- `Liquid-CLI/scripts/build_lfm_terminal_toolcall_dataset.py`

이 스크립트는 두 데이터를 합친다.

1. `gyung/LFM2-Terminal-SFT-Processed`의 `train` split
2. `HRM-Text/data_toolbench/data/toolllama_G123_dfs_train.json`

빌드 메타데이터:

- `/home/work/.data/liquid_cli_sft/datasets/lfm25_8b_a1b_terminal_full_toolbench_full_conversations_v1/build_meta.json`

확인된 값:

```json
{
  "terminal_dataset_name": "gyung/LFM2-Terminal-SFT-Processed",
  "terminal_split": "train",
  "terminal_rows": 139841,
  "toolbench_path": "HRM-Text/data_toolbench/data/toolllama_G123_dfs_train.json",
  "toolbench_rows": 187542,
  "toolbench_max_rows": 0,
  "total_rows": 327383,
  "format": "LFM conversations with optional assistant tool_calls JSON strings"
}
```

`toolbench_max_rows=0`이므로 ToolBench train 파일은 row cap 없이 전체 사용했다.

단, 여기서 "ToolBench 전체"는 train file 전체를 뜻한다. `toolllama_G123_dfs_eval.json`은 평가용 파일이며 이 SFT train merge에는 넣지 않았다.

## Terminal processed 데이터의 도메인 커버리지

로컬에서 확인한 terminal raw/processed 기준 row count는 `139,841` rows다.

이 row count는 `gyung/LFM2-Terminal-SFT-Processed`로 merge에 들어간 terminal rows 수와 일치한다.

로컬 raw path:

- `/home/work/.data/liquid_cli_sft/datasets/raw_skill_based`

확인된 도메인 prefix 분포:

| Domain prefix | Rows | Terminal 내 비율 | 전체 merge 내 비율 |
| --- | ---: | ---: | ---: |
| `scientific_computing` | 19,482 | 13.93% | 5.95% |
| `data_science` | 17,565 | 12.56% | 5.37% |
| `debugging` | 17,382 | 12.43% | 5.31% |
| `security` | 17,010 | 12.16% | 5.20% |
| `file_operations` | 16,096 | 11.51% | 4.92% |
| `software_engineering` | 15,559 | 11.13% | 4.75% |
| `dependency_management` | 9,935 | 7.10% | 3.03% |
| `data_querying` | 9,535 | 6.82% | 2.91% |
| `data_processing` | 9,287 | 6.64% | 2.84% |
| `system_administration` | 7,776 | 5.56% | 2.38% |
| `model_training` | 214 | 0.15% | 0.07% |

질문에 나온 주요 Nemotron-Terminal-Corpus 도메인 기준으로 보면 모두 포함되어 있다.

- Data Querying: 포함, `9,535` rows
- Model Training: 포함, `214` rows
- Data Processing: 포함, `9,287` rows
- Debugging: 포함, `17,382` rows
- Software Engineering: 포함, `15,559` rows

추가로 security, file operations, dependency management, system administration, data science, scientific computing도 같이 들어 있다.

## 헷갈릴 수 있는 점

repo 안에는 별도의 오래된/실험용 필터링 스크립트가 있다.

- `liquid_sft/scripts/prepare_dataset.py`

이 스크립트는 `nvidia/Nemotron-Terminal-Corpus`에서 일부 split을 받은 뒤 다음처럼 도메인을 제한한다.

- keep: `file_operations`, `data_processing`, `data_querying`, `dependency_management`, `security`
- exclude: `data_science`, `scientific_computing`, `debugging`, `software_engineering`

그 결과 로컬에 `sft_data`라는 `3,510` rows짜리 필터셋도 존재한다.

하지만 `LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`는 이 `3,510` rows filtered dataset만으로 학습한 모델이 아니다.

해당 모델의 full ToolBench SFT config는 `gyung/LFM2-Terminal-SFT-Processed` train 전체 `139,841` rows를 불러오고, 거기에 ToolBench train `187,542` rows를 붙인다.

따라서 이 모델을 설명할 때는 다음처럼 말하는 것이 정확하다.

> Terminal processed train 전체 139,841 rows와 ToolBench train 전체 187,542 rows를 합친 327,383 rows로 full SFT했다.

## 왜 성능이 잘 나왔는가

corrected TB2-lite 303-step full replay 기준:

- Base `LiquidAI/LFM2.5-8B-A1B`: Score `36.53`
- `LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`: Score `52.30`
- `LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-2Epoch`: Score `50.48`

1epoch 핵심 지표:

- Score: `52.30`
- Cmd F1: `0.5230`
- Precision: `0.5854`
- Recall: `0.5431`
- First Cmd: `49.5%`
- Valid JSON: `76.9%`
- Next Action score: `51.46`
- vLLM 평가 속도: `0.087 sec/step`

성능이 오른 이유는 데이터 조합이 TB2-lite의 요구와 잘 맞았기 때문이다.

Terminal corpus는 실제 터미널 작업의 넓은 작업 분포를 제공한다.

- 파일 탐색
- 데이터 확인
- 디버깅
- 의존성 처리
- 보안 수정
- 소프트웨어 엔지니어링
- 과학 계산
- 데이터 처리/쿼리

ToolBench train은 assistant action, tool call, JSON-like structured response 분포를 보강한다.

TB2-lite replay는 결국 "다음 터미널 행동을 얼마나 그럴듯하게 복원하는가"를 강하게 본다. 그래서 terminal corpus만 있거나 ToolBench만 있는 것보다, 두 데이터를 같이 넣은 full conversation SFT가 command recall, valid JSON, first command 선택을 동시에 끌어올린 것으로 볼 수 있다.

실제 README 분석에서도 1epoch는 base 대비 Recall이 크게 올랐다.

- Base Recall: `0.3685`
- 1epoch Recall: `0.5431`

즉, 모델이 더 많은 정답 command set을 복원하게 된 것이 점수 상승의 핵심이다.

## 데이터 비율 관점의 해석

이 SFT 데이터는 터미널 코퍼스만 과하게 넣은 구성이 아니다.

전체 `327,383` rows 중 terminal processed는 `42.71%`, ToolBench train은 `57.29%`다.

이 비율은 TB2-lite에 꽤 잘 맞는다.

- Terminal processed `42.71%`: 실제 쉘 명령, 파일 조작, 디버깅, 데이터 처리 같은 터미널 행동 분포를 학습한다.
- ToolBench train `57.29%`: tool/action 형식, JSON-like 응답, 도구 호출 흐름을 학습한다.

즉, 한쪽으로만 치우친 데이터가 아니라 "무슨 명령을 쳐야 하는가"와 "그 명령을 어떤 형식의 action으로 내야 하는가"를 같이 먹인 구성이다.

TB2-lite 점수가 잘 나온 이유도 여기서 나온다. 터미널 문제를 풀려면 Linux command 지식만 있어서는 부족하고, terminal agent benchmark가 기대하는 tool call 형식도 안정적으로 맞춰야 한다. 이 run은 두 축을 동시에 학습했다.

## 왜 2epoch보다 1epoch가 더 좋았는가

2epoch도 강하지만 1epoch보다 낮다.

- 1epoch Score: `52.30`
- 2epoch Score: `50.48`

2epoch에서는 일부 영역이 개선됐지만, 전체적으로는 평균 예측 command가 늘고 Valid JSON이 내려가면서 precision/recall 균형이 약간 나빠졌다.

- 1epoch Valid JSON: `76.9%`
- 2epoch Valid JSON: `74.9%`
- 1epoch 평균 예측 command: `21.20`
- 2epoch 평균 예측 command: `22.50`

이 run에서는 1epoch가 데이터 적합도와 출력 형식 안정성의 sweet spot이었다.

다시 말하면, "모든 데이터를 많이 먹였기 때문에" 성능이 오른 것은 맞지만, "오래 돌릴수록 무조건 좋아진다"는 결과는 아니었다.

1epoch는 전체 `327,383` rows를 한 번 충분히 본 상태라서 terminal behavior와 tool/action format을 넓게 흡수했다. 반면 2epoch에서는 같은 분포를 한 번 더 반복하면서 출력이 약간 길어지고 형식 안정성이 떨어졌다. 이 때문에 recall 일부 이득보다 precision/format 손실이 더 커졌고, 최종 점수는 1epoch가 더 좋았다.

따라서 이 SFT run의 핵심은 다음처럼 정리하는 것이 정확하다.

> terminal processed 전체와 ToolBench train 전체를 쓰되, 과하게 반복하지 않고 1epoch 지점에서 멈춘 것이 TB2-lite 기준 가장 좋은 균형이었다.

## 한 줄 요약

`LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`는 terminal processed train 전체와 ToolBench train 전체를 합쳐 학습했기 때문에, 터미널 작업 도메인 커버리지와 tool/action 출력 형식을 동시에 배웠고, 그 결과 TB2-lite에서 Score `52.30`으로 강하게 나온 모델이다.
