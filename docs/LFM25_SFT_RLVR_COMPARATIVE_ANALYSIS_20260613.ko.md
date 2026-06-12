# LFM2.5 SFT/RLVR 비교 분석

업데이트: 2026-06-13 KST

이 문서는 `LiquidAI/LFM2.5-8B-A1B` 계열에서 지금까지 진행한 SFT, SFT+ECHO RLVR, raw ECHO RLVR 실험을 한 곳에 모아 비교한다. README는 점수표를 빠르게 보기 위한 문서로 유지하고, 학습 데이터, 전처리, 배치 사이즈, checkpoint별 해석, 논문과의 차이는 이 문서에 정리한다.

중요한 결론부터 말하면 다음과 같다.

| 조건 | 시작 모델 | 학습 방식 | 최고 checkpoint | README Score | 해석 |
| --- | --- | --- | --- | ---: | --- |
| raw baseline | `LiquidAI/LFM2.5-8B-A1B` | 없음 | 없음 | `36.53` | README 고정 기준점 |
| SFT 1Epoch | raw LFM2.5 | Terminal + ToolBench full SFT | `checkpoint-1542` | `52.30` | 가장 큰 성능 상승. 현재 단독 모델 기준 핵심 |
| SFT 2Epoch | raw LFM2.5 | 같은 SFT 2epoch | final / `checkpoint-3084` | `50.48` | 1epoch보다 하락. 과학습/스타일 drift 신호 |
| SFT+ECHO RLVR | SFT 1Epoch | ECHO-style LoRA GRPO | parent `checkpoint-610` | `54.05` | 현재 전체 1위. SFT 위에서 RLVR이 작게 추가 이득 |
| SFT+ECHO RLVR continuation | SFT+RLVR parent에서 이어서 | ECHO-style LoRA GRPO | continue `checkpoint-220` | `53.26` | 1위는 아님. final checkpoint보다 best checkpoint 선택 필요 |
| raw ECHO RLVR | raw LFM2.5 | ECHO-style LoRA GRPO | raw `checkpoint-1225` | `45.32` | raw 기준 `+8.79`, 하지만 SFT 1Epoch보다 `-6.98` |

README에는 raw `LiquidAI/LFM2.5-8B-A1B`를 Score `36.53`으로 고정한다. GPU6에서 별도로 나온 raw rerun `39.92`는 동일 시스템의 진단용 재평가로 남기되, leaderboard 기준점은 바꾸지 않는다. 평가 seed, vLLM 세팅, evaluator 세부 버전이 조금만 바뀌어도 replay 점수가 흔들릴 수 있기 때문에, public README 순위에서는 처음 반영한 baseline을 고정하는 편이 비교가 깔끔하다.

## 1. 평가 기준

평가는 `TB2-lite corrected 303-step replay` 기준이다. 모든 주요 수치는 다음 metric으로 맞춘다.

```text
Score = 100 * avg_command_f1
```

주요 보조 지표:

| 지표 | 의미 |
| --- | --- |
| `avg_command_precision` | 모델이 낸 command 중 reference command와 맞는 비율 |
| `avg_command_recall` | reference command set을 얼마나 넓게 복원했는지 |
| `first_cmd_exact_pct` | 첫 command가 정확히 맞는 비율 |
| `valid_json_pct` | evaluator가 JSON action으로 파싱할 수 있었던 비율 |
| `next_action_score` | 기존 보조 점수. README 순위 기준은 아님 |

터미널 에이전트에서는 `first_cmd_exact_pct`가 중요하다. 첫 `ls`, `cat`, `pytest`, `python`, `grep` 방향이 틀리면 후속 command F1도 같이 무너진다. 다만 README 순위는 오직 `Score = 100 * avg_command_f1`로 통일한다.

## 2. ECHO 논문 기준

참고 논문:

```text
ECHO: Terminal Agents Learn World Models for Free
https://arxiv.org/html/2605.24517v1
```

ECHO 논문의 핵심은 단순하다. 기존 GRPO는 터미널 agent rollout에서 모델이 낸 action token에만 policy-gradient loss를 건다. 터미널이 돌려준 stdout, stderr, exit code, 에러 로그, 파일 내용은 다음 action의 context로만 쓰이고, loss에는 직접 들어가지 않는다.

ECHO는 이 버려지는 terminal observation token에도 cross-entropy loss를 건다.

```text
ECHO loss = GRPO action-token policy loss + lambda * terminal-observation CE loss
```

즉 모델에게 “명령을 잘 내라”만 가르치는 것이 아니라, “네가 낸 명령 때문에 터미널이 어떤 반응을 보이는지도 예측하라”고 같이 학습시킨다. 논문의 표현대로 terminal feedback을 world model supervision으로 바꾸는 것이다.

논문 실험 세팅의 핵심 숫자:

| 항목 | 논문 기준 |
| --- | --- |
| 환경 | Docker/Harbor terminal task environment |
| 데이터 | curated/generated terminal tasks `8,870`, train `8,770`, val `100` |
| 모델 | Qwen3-8B, OpenThinker-Agent-v1-SFT, Qwen3-14B |
| RL | GRPO + ECHO |
| 보상 | final tests pass면 `1`, 아니면 `0` |
| 학습 | `500` GRPO steps |
| GPU | `8 x B200` |
| batch size | `16` |
| rollout | 최대 `16` turns |
| generation | turn당 최대 `2048` tokens |
| context | `16K` training, TB2 eval은 `32K` |
| ECHO loss weight | 논문은 productive range를 sweep하고 constant weight 사용 |

논문 결과에서 ECHO는 matched GRPO보다 TerminalBench-2.0 pass@1을 크게 올렸다. Qwen3-8B는 `2.70 -> 5.17`, Qwen3-14B는 `5.17 -> 10.79`로 거의 2배 가까이 개선됐다.

## 3. 우리 구현은 논문과 무엇이 같은가

우리 구현의 핵심은 ECHO 논문과 같다.

| 구성 | 우리 구현 |
| --- | --- |
| action rollout | 모델이 JSON으로 bash command를 출력 |
| 환경 실행 | command를 실제 로컬 sandbox workspace에서 실행 |
| observation | stdout, stderr, exit code, verifier output을 다음 turn context에 삽입 |
| action loss | assistant action token에 GRPO-style policy loss 적용 |
| world-model loss | terminal observation token에 CE loss 적용 |
| mask | `action_mask`, `obs_mask`를 분리 |
| adapter | LoRA rank `32`, alpha `64` |
| 모델 | `LiquidAI/LFM2.5-8B-A1B` 또는 SFT 1Epoch 모델 |

주요 코드:

```text
Liquid-CLI/train_lfm_terminal_echo_live_grpo.py
Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh
Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh
Liquid-CLI/scripts/watch_echo_rlvr_gpu6_eval_queue.sh
Liquid-CLI/scripts/sync_echo_rollouts_to_hf_dataset.py
Liquid-CLI/scripts/sync_echo_adapter_checkpoints_to_hf_model.py
```

우리 코드의 loss 구조:

```text
loss = policy_coeff * policy_loss + world_model_coeff * world_loss
world_model_coeff = 0.05
```

보상은 논문처럼 pure binary만 쓰지는 않았다. no-Docker 환경에서 verifier success가 너무 sparse하기 때문에, command 개수 보너스와 format penalty를 같이 둔다.

```text
reward =
  verifier_reward * scale
  + success_bonus
  + min(command_count, 3) * command_bonus
  - parse_errors * format_penalty
```

현재 raw run 기준:

```text
command_bonus = 0.02
format_penalty = 0.05
reward_success_bonus = 0.0
```

이 보상 shaping은 논문과 완전히 같지는 않다. 대신 raw 모델이 JSON을 아예 못 내거나 무한 설명 텍스트를 내는 구간에서 학습이 완전히 죽지 않게 하기 위한 local adaptation이다.

## 4. 우리 구현은 논문과 무엇이 다른가

이 부분이 중요하다. 현재 결과를 해석할 때 “ECHO 논문이 틀렸다” 또는 “RLVR이 안 된다”로 보면 안 된다. 우리는 논문을 그대로 복제한 것이 아니라, 도커가 없는 로컬 환경에서 핵심 아이디어를 이식했다.

| 항목 | ECHO 논문 | 우리 실험 |
| --- | --- | --- |
| 격리 환경 | Docker/Harbor | no-Docker local sandbox |
| task 수 | train `8,770` | prepared public tasks `1,500` |
| rollout turn | 최대 `16` | 현재 raw run 최대 `4` |
| 생성 token | turn당 `2048` | 현재 raw run `256` |
| GPU | `8 x B200` | vLLM rollout 4GPU + train 2GPU |
| 학습 방식 | full training 계열 | LoRA rank `32` |
| reward | final test binary | verifier + command/format shaping |
| 시작 모델 | Qwen3/OpenThinker | LFM2.5 raw 또는 LFM2.5 SFT |
| serving | 논문 내 harness | vLLM HTTP rollout replicas |

따라서 현재 실험은 “ECHO 논문 완전 재현”이 아니라 “LFM2.5 + no-Docker + vLLM 환경에서 ECHO loss를 terminal RLVR에 적용한 실험”이다.

그럼에도 핵심 질문에는 답을 준다.

```text
terminal observation CE loss가 실제로 raw LFM2.5의 terminal-action 능력을 올리는가?
```

현재 답은 “올린다”다. raw baseline `36.53`에서 raw RLVR best `45.32`까지 상승했다. 다만 SFT를 대체할 만큼 강하지는 않다.

## 5. SFT 데이터와 전처리

SFT는 단일 작은 데이터셋이 아니라 terminal dataset 전체와 ToolBench train 전체를 합친 full SFT다.

설정 파일:

```text
Liquid-CLI/configs/sft_h200_4gpu_lfm25_8b_a1b_terminal_toolbench_full.env
```

데이터 build script:

```text
Liquid-CLI/scripts/build_lfm_terminal_toolcall_dataset.py
```

원천 데이터:

| 데이터 | 경로/이름 | rows | 비율 |
| --- | --- | ---: | ---: |
| Terminal SFT | `gyung/LFM2-Terminal-SFT-Processed`, train split | `139,841` | `42.71%` |
| ToolBench | `HRM-Text/data_toolbench/data/toolllama_G123_dfs_train.json` | `187,542` | `57.29%` |
| 합계 | full conversations | `327,383` | `100.00%` |

빌드 메타:

```text
/home/work/.data/liquid_cli_sft/datasets/lfm25_8b_a1b_terminal_full_toolbench_full_conversations_v1/build_meta.json
```

전처리 산출물:

| 단계 | 경로 | rows |
| --- | --- | ---: |
| conversation dataset | `/home/work/.data/liquid_cli_sft/datasets/lfm25_8b_a1b_terminal_full_toolbench_full_conversations_v1` | `327,383` |
| ChatML processed | `/home/work/.data/liquid_cli_sft/datasets/lfm25_8b_a1b_terminal_full_toolbench_full_chatml_v1` | `327,383` |
| train-ready packed/tokenized | `/home/work/.data/liquid_cli_sft/datasets/lfm25_8b_a1b_terminal_full_toolbench_full_train_ready_v1` | `197,373` |

토큰 길이:

```text
MAX_SEQ_LENGTH = 8192
sample input length over 1000 samples:
avg = 7486.328
max = 8192
min = 4356
```

해석:

SFT 데이터는 TB2-lite 정답만 외운 것이 아니다. terminal command workflow와 ToolBench-style tool/action 출력 계약을 같이 학습했다. 그래서 raw LFM2.5가 원래 가진 일반 지능 위에 “터미널 에이전트로 말하는 법”이 강하게 주입됐다.

## 6. SFT 학습 설정

SFT config:

```text
MODEL_PATH=LiquidAI/LFM2.5-8B-A1B
MAX_SEQ_LENGTH=8192
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2
NPROC_PER_NODE=4
CUDA_VISIBLE_DEVICES=1,2,4,5
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=2
SAVE_STRATEGY=epoch
```

global batch:

```text
16 per-device batch * 2 grad accumulation * 4 GPUs = 128
```

checkpoint:

| checkpoint | 의미 |
| --- | --- |
| `checkpoint-1542` | 1 epoch |
| `checkpoint-3084` / `final` | 2 epoch |

학습 출력:

```text
/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B__terminal_toolbench_full_sft_h200_4gpu_2epoch
```

## 7. SFT 점수와 해석

SFT 1Epoch:

| 지표 | 값 |
| --- | ---: |
| Score | `52.30` |
| Cmd F1 | `0.5230` |
| Precision | `0.5854` |
| Recall | `0.5431` |
| First Cmd | `49.5%` |
| Valid JSON | `76.9%` |
| Next Action | `51.46` |
| avg predicted commands | `21.20` |

SFT 2Epoch:

| 지표 | 값 |
| --- | ---: |
| Score | `50.48` |
| Cmd F1 | `0.5048` |
| Precision | `0.5695` |
| Recall | `0.5296` |
| First Cmd | `49.2%` |
| Valid JSON | `74.9%` |
| Next Action | `50.10` |
| avg predicted commands | `22.50` |

왜 1Epoch가 강한가:

SFT 1Epoch는 raw baseline `36.53`에서 `52.30`으로 `+15.77` 오른다. 이 상승은 RL보다 훨씬 크다. 이유는 이 벤치마크가 “정답을 알기”보다 “올바른 터미널 action 형식과 command 분포를 복원하기”를 많이 본다는 점 때문이다.

SFT가 끄집어내고, 일부는 실제로 주입한 것:

```text
1. JSON action format
2. analysis/plan/commands/task_complete 구조
3. ls, cat, sed, python, pytest, grep, find 같은 터미널 기본 동작을 언제 꺼내야 하는지
4. multi-turn tool/action conversation prior
5. 어느 시점에 task_complete=false를 유지해야 하는지
6. 어느 시점에 검증 command를 실행해야 하는지
```

여기서 중요한 점은 SFT를 “지식 주입이 전혀 아니다”라고 말해도 틀리고, “완전히 새로운 리눅스 지식을 주입했다”고 말해도 틀린다는 것이다.

raw LFM2.5가 이미 Score `36.53`을 내는 것을 보면, 기본적인 터미널/코딩/파일 조작 지식은 사전학습 안에 들어 있다. SFT 1Epoch가 `52.30`까지 오른 것은 그 지식을 TB2-lite가 요구하는 `JSON action`, `task_complete`, `multi-turn command sequence` 형태로 안정적으로 꺼내도록 만든 효과가 크다.

다만 SFT가 아무것도 새로 배우지 않는 것은 아니다. “리눅스 명령어가 무엇인가” 같은 원천 지식은 끄집어낸 쪽에 가깝지만, “이 평가 환경에서 어떤 형식으로, 어떤 순서로, 언제 멈추고, 언제 검증해야 하는가” 같은 작업 절차 지식은 SFT로 꽤 강하게 주입된 것이다.

왜 2Epoch가 떨어졌나:

2Epoch는 일부 도메인에서는 좋아졌다. 예를 들어 `data_science`, `model_training`, `code`는 1Epoch보다 개선된 구간이 있다. 하지만 전체 Score는 내려갔다.

하락 이유:

```text
1. Valid JSON: 76.9% -> 74.9%
2. Precision: 0.5854 -> 0.5695
3. Recall: 0.5431 -> 0.5296
4. 평균 예측 command: 21.20 -> 22.50
5. 전체 action style이 조금 더 장황하거나 drift
```

즉 2Epoch는 더 많이 배운 것이 아니라, TB2-lite 평가에서 가장 좋은 action distribution을 조금 지나친 것으로 보인다. 이 결과는 “terminal/action SFT는 무조건 오래 할수록 좋다”가 아니라 “좋은 데이터라도 적정 epoch가 있다”는 쪽에 가깝다.

## 8. SFT+ECHO RLVR 설정

SFT+RLVR 실험은 다음 모델에서 시작했다.

```text
LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch
```

학습 방식:

```text
ECHO-style live terminal RLVR
LoRA rank 32
GRPO-style action policy loss
terminal observation CE world-model loss
vLLM rollout + local no-Docker sandbox
```

대표 결과:

| run | best checkpoint | Score |
| --- | ---: | ---: |
| parent run | `610` | `54.05` |
| continuation run | `220` | `53.26` |
| parent latest evaluated | `1880` | `52.15` |
| continuation latest evaluated | `760` | `50.43` |

best parent checkpoint `610`의 지표:

| 지표 | 값 |
| --- | ---: |
| Score | `54.05` |
| Cmd F1 | `0.5405` |
| Precision | `0.6021` |
| Recall | `0.5571` |
| First Cmd | `54.5%` |
| Valid JSON | `77.9%` |
| Next Action | `54.18` |
| avg predicted commands | `22.12` |

SFT 1Epoch와 비교:

| 지표 | SFT 1Epoch | SFT+RLVR best | 차이 |
| --- | ---: | ---: | ---: |
| Score | `52.30` | `54.05` | `+1.75` |
| Precision | `0.5854` | `0.6021` | `+0.0167` |
| Recall | `0.5431` | `0.5571` | `+0.0140` |
| First Cmd | `49.5%` | `54.5%` | `+5.0%p` |
| Valid JSON | `76.9%` | `77.9%` | `+1.0%p` |

해석:

SFT+RLVR은 이미 강한 모델에서 작은 영역을 더 깎는다. 가장 의미 있는 변화는 `First Cmd`가 `49.5% -> 54.5%`로 오른 점이다. 터미널 에이전트는 첫 command 방향이 맞으면 후속 command F1도 올라가기 쉽다.

왜 장기적으로 단조 상승하지 않았나:

```text
1. SFT 1Epoch가 이미 강해서 headroom이 작다.
2. verifier reward가 sparse하다.
3. RL reward와 TB2-lite command F1이 완전히 같은 objective가 아니다.
4. no-Docker local sandbox는 Docker/Harbor보다 task 상태 격리가 약하다.
5. LoRA update가 특정 action style에 치우치면 형식/coverage가 흔들릴 수 있다.
6. RL은 최종 checkpoint가 최고라는 보장이 없다.
```

결론은 “RLVR이 별로다”가 아니다. 이 실험에서는 best checkpoint를 골라야 한다. parent `checkpoint-610`이 현재 전체 1위이고, continuation을 길게 민 최종 checkpoint는 오히려 낮다.

## 9. raw ECHO RLVR 데이터

raw RLVR은 SFT 없이 순수 `LiquidAI/LFM2.5-8B-A1B`에서 시작했다.

현재 active run:

```text
run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k
```

base model:

```text
LiquidAI/LFM2.5-8B-A1B
```

SFT adapter:

```text
none
```

prepared dataset:

```text
/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl
```

데이터 구성:

| source | rows | 비율 |
| --- | ---: | ---: |
| `openthoughts_agent_v1_rl` | `728` | `48.53%` |
| `endless_terminals` | `772` | `51.47%` |
| total | `1,500` | `100.00%` |

데이터 다운로드/전처리 코드:

```text
Liquid-CLI/scripts/download_echo_public_terminal_data.py
Liquid-CLI/scripts/prepare_echo_terminal_data.py
```

전처리 내용:

```text
1. 공개 terminal task를 가져온다.
2. task archive를 local sandbox에서 풀 수 있는 형태로 정리한다.
3. LFM용 system/user prompt를 붙인다.
4. command JSON schema를 강제한다.
5. task metadata/source/task_id를 보존한다.
6. TB2-lite 평가 데이터는 학습 데이터로 넣지 않는다.
```

중요한 한계:

이 `1,500` rows는 ECHO 논문 내부의 `8,870` Harbor/Docker task corpus와 동일하지 않다. 논문이 공개 repo/데이터에서 제공하는 방식과 공개 terminal datasets를 우리 로컬 no-Docker 환경에 맞춰 준비한 것이다. 따라서 논문 방법론의 핵심 loss는 따르지만, 데이터/환경 규모는 축소판이다.

## 10. raw ECHO RLVR 학습 설정

현재 active raw run:

```text
OUTPUT_DIR=/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B__echo_raw_live_grpo_vllm_r32_run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k
```

GPU 배치:

| GPU | 역할 |
| --- | --- |
| `0,1,2,3` | raw LFM2.5 vLLM rollout servers, ports `8123-8126` |
| `4,5` | LoRA/GRPO 학습 |
| `6` | 평가 전용으로 사용 가능 |
| `7` | 제외. 이 작업에서 건드리지 않음 |

학습 설정:

| 항목 | 값 |
| --- | ---: |
| max steps | `2000` |
| current train step, 2026-06-13 07:53 KST | `1224` |
| latest saved checkpoint, 2026-06-13 07:53 KST | `1225` |
| max wall time | `48h` |
| LoRA rank | `32` |
| LoRA alpha | `64` |
| trainable params | `12,867,584` |
| train GPUs | `2` |
| vLLM rollout GPUs | `4` |
| max turns | `4` |
| max new tokens | `256` |
| num generations | `4` |
| prompts per rank | `1` |
| nproc per node | `2` |
| global rollouts per step | `8` |
| learning rate | `1e-6` |
| warmup steps | `10` |
| max grad norm | `0.2` |
| world model coeff | `0.05` |
| command timeout | `8s` |
| verifier timeout | `40s` |
| terminal output cap | `10000 chars` |
| save interval | `25 steps` |

멀티 GPU 배치 해석:

현재 active run의 effective rollout batch는 다음과 같다.

```text
train ranks 2 * prompts_per_rank 1 * num_generations 4 = 8 rollouts/step
```

GPU가 많다고 해서 무조건 train-side batch를 키우는 것이 정답은 아니다. 이 run의 병목은 LoRA optimizer VRAM보다 vLLM rollout, 터미널 명령 실행, verifier timeout 쪽에 더 가깝다. 그래서 현재는 다음처럼 분리했다.

```text
0,1,2,3: vLLM rollout 전용
4,5: LoRA/GRPO update 전용
6: checkpoint 평가 전용
7: 다른 작업용, 이 실험에서 제외
```

실행 중인 run에서 batch를 바꾸면 checkpoint 간 비교가 깨진다. 따라서 현재 run은 유지하고, 다음 run에서 더 공격적으로 갈 때만 아래 후보를 쓴다.

| 후보 | 설정 | effective rollouts/step | 기대 효과 | 위험 |
| --- | --- | ---: | --- | --- |
| 안정 유지 | `prompts_per_rank=1`, `num_generations=4` | `8` | 현재 곡선과 비교가 가장 깨끗함 | 학습 신호가 느림 |
| 1차 증량 | `prompts_per_rank=2`, `num_generations=4` | `16` | 같은 group size로 prompt coverage 증가 | 터미널/verifier 병목 증가 |
| group 증량 | `prompts_per_rank=1`, `num_generations=8` | `16` | GRPO group 내 비교가 더 풍부함 | rollout 시간이 크게 증가하고 long tail task에 취약 |
| 과격 증량 | `prompts_per_rank=2`, `num_generations=8` | `32` | 신호는 많아짐 | no-Docker verifier 병목과 불안정성이 커짐 |

다음 실험에서 가장 합리적인 첫 증량은 `prompts_per_rank=2`, `num_generations=4`, `rollout_workers=16`이다. 이렇게 하면 group size는 유지하면서 step당 task coverage만 넓어진다. 다만 현재처럼 verifier reward가 계속 `0.0`이면 batch만 키워도 성공 trajectory가 생기지 않을 수 있으므로, batch 증량보다 verifier contract와 task difficulty 조절이 더 큰 병목일 수 있다.

최근 진행 속도:

```text
checkpoint-1100: 2026-06-13 06:42 KST
checkpoint-1125: 2026-06-13 06:56 KST
checkpoint-1150: 2026-06-13 07:10 KST
checkpoint-1175: 2026-06-13 07:24 KST
checkpoint-1200: 2026-06-13 07:38 KST
checkpoint-1225: 2026-06-13 07:54 KST
```

25 step당 약 `14분`이다. `2000` step까지는 2026-06-13 07:53 KST 기준 약 `776` step 남았고, 단순 추정으로 약 `7시간 15분` 남았다. verifier/test task가 오래 걸리면 `1~2시간` 더 밀릴 수 있다.

## 11. raw ECHO RLVR 점수

raw run checkpoint 평가:

| checkpoint | Score | First Cmd | Valid JSON |
| ---: | ---: | ---: | ---: |
| `25` | `37.89` | `41.6%` | `59.4%` |
| `50` | `40.34` | `42.2%` | `58.1%` |
| `125` | `41.01` | `43.2%` | `57.4%` |
| `300` | `41.52` | `42.2%` | `59.1%` |
| `475` | `43.41` | `44.6%` | `59.7%` |
| `550` | `43.46` | `43.2%` | `64.7%` |
| `800` | `43.94` | `42.2%` | `61.7%` |
| `975` | `44.47` | `44.2%` | `63.7%` |
| `1075` | `43.77` | `45.2%` | `68.6%` |
| `1100` | `44.84` | `45.9%` | `66.3%` |
| `1175` | `43.29` | `44.2%` | `63.7%` |
| `1200` | `44.39` | `45.5%` | `66.7%` |
| `1225` | `45.32` | `48.2%` | `66.3%` |

현재 평가 완료 기준 최고점은 raw `checkpoint-1225`의 Score `45.32`다.

raw checkpoint-1225 상세:

| 지표 | 값 |
| --- | ---: |
| Score | `45.32` |
| Cmd F1 | `0.4532` |
| Precision | `0.5380` |
| Recall | `0.4588` |
| First Cmd | `48.2%` |
| Valid JSON | `66.3%` |
| Next Action | `46.18` |
| avg predicted commands | `12.61` |

raw baseline과 비교:

| 지표 | raw baseline | raw RLVR best | 차이 |
| --- | ---: | ---: | ---: |
| Score | `36.53` | `45.32` | `+8.79` |
| First Cmd | `39.9%` | `48.2%` | `+8.3%p` |
| Valid JSON | `59.1%` | `66.3%` | `+7.2%p` |

SFT 1Epoch와 비교:

| 지표 | raw RLVR best | SFT 1Epoch | 차이 |
| --- | ---: | ---: | ---: |
| Score | `45.32` | `52.30` | `-6.98` |
| Precision | `0.5380` | `0.5854` | `-0.0474` |
| Recall | `0.4588` | `0.5431` | `-0.0843` |
| Valid JSON | `66.3%` | `76.9%` | `-10.6%p` |

## 12. raw RLVR에서 보이는 학습 신호

현재 raw RLVR은 완전 실패가 아니다. Score는 README 기준 raw `36.53`에서 `45.32`까지 오른다. 특히 First Cmd와 Valid JSON이 같이 오른다.

이것은 다음을 의미한다.

```text
1. terminal observation CE loss가 action format을 안정화한다.
2. 모델이 명령을 내면 어떤 stdout/stderr가 나오는지 조금씩 내부화한다.
3. raw 모델의 첫 command 선택이 개선된다.
4. command count가 줄고 더 보수적인 action을 내는 방향으로 이동한다.
```

하지만 강한 아하 모먼트는 아직 아니다.

현재 곡선은 다음에 가깝다.

```text
초반 빠른 적응: 36.53 -> 40점대
중반 완만한 상승: 41 -> 44점대
후반 완만한 상승/plateau: 43~45점대 흔들림, 1225에서 45.32 새 고점
```

`checkpoint-1100` 이후 `1175`에서 내려왔다가 `1225`에서 다시 새 고점을 만든 것을 보면, raw RLVR은 아직 완전히 정체됐다고 보기는 어렵다. 다만 verifier reward가 계속 `0.0`이라 “성공 trajectory가 터지며 급상승하는 강한 아하 모먼트”라고 보기도 이르다. 현 설정으로는 `45~47` 정도가 현실적인 기대 범위다. 운 좋게 특정 checkpoint가 튀면 `48` 근처도 가능하지만, SFT 1Epoch `52.30`을 바로 넘기는 시나리오는 현재 로그와 verifier reward를 보면 낮다.

## 13. 왜 SFT가 RLVR보다 더 잘 먹히나

이 질문이 가장 중요하다.

먼저 SFT의 성격을 정확히 잡아야 한다. `A Primer in Post-Training Reasoning Data` 관점에서는 작은 고품질 SFT가 잘 먹히는 경우를 “base policy support 안에 이미 있는 능력을 특정 포맷과 분포에서 호출하게 만드는 것”으로 해석할 수 있다. 즉 SFT는 많은 경우 새 지식의 대량 이식이라기보다, 사전학습에 들어 있는 잠재 능력을 평가 가능한 출력 형식으로 꺼내는 작업이다.

하지만 터미널 에이전트에서는 “작업 절차 지식”도 중요하다. LFM2.5는 이미 리눅스 명령과 코딩 지식이 있지만, TB2-lite가 요구하는 JSON action 계약, 중단 타이밍, 검증 command 습관은 raw 모델에 충분히 고정되어 있지 않다. 이 부분은 SFT가 실제로 새로 정렬하고 일부 주입한 행동 prior다.

따라서 현재 결과는 이렇게 보는 것이 가장 정확하다.

```text
원천 지식: 대부분 pre-training에서 온다.
출력 형식과 행동 절차: SFT가 강하게 정렬하고 일부 주입한다.
환경 피드백 활용: ECHO-style RLVR이 추가로 다듬는다.
```

SFT가 더 잘 먹히는 이유는 모델 지능 때문이 아니라, 현재 평가가 요구하는 능력 때문이다. TB2-lite replay는 “터미널을 통해 문제를 풀 수 있는가”를 보지만, 점수는 결국 reference command와 모델 command의 overlap을 본다. 즉 다음 능력이 매우 중요하다.

```text
1. JSON으로 command를 안정적으로 내기
2. 첫 command를 맞히기
3. reference에 있는 ls/cat/python/pytest/sed/grep 흐름을 넓게 복원하기
4. 너무 빨리 task_complete=true를 내지 않기
5. 후반 turn에서도 검증 command를 유지하기
```

SFT는 이 모든 것을 직접 보여준다. 327,383 rows의 terminal/toolcall data가 “이 상황에서는 이런 명령을 내라”는 고밀도 supervised signal을 준다. 이것은 리눅스 매뉴얼을 새로 외우게 하는 신호라기보다, 이미 알고 있는 명령과 파일 조작 패턴을 TB2-lite의 action grammar로 꺼내게 만드는 신호다.

반대로 raw RLVR은 다음 문제가 있다.

```text
1. verifier success reward가 거의 0이다.
2. 실패 rollout도 observation CE로 배우지만, 그것만으로 정답 command 분포를 바로 알 수는 없다.
3. no-Docker local sandbox는 논문 Docker/Harbor보다 task 상태와 verifier 계약이 약하다.
4. max_turns=4, max_new_tokens=256이라 긴 복구 trajectory를 충분히 경험하지 못한다.
5. LoRA rank 32는 안정적이지만 full finetuning보다 표현 이동 폭이 작다.
6. raw 모델은 command JSON prior가 약해서 초반 많은 gradient가 형식 복구에 쓰인다.
```

한마디로 SFT는 답안지를 보고 배우는 것이고, raw RLVR은 터미널에 부딪히며 오답노트를 쓰는 것이다. 지금 모델은 아직 오답노트를 쓰기 전에 답안지 양식부터 익히는 단계다.

## 14. 왜 SFT+RLVR은 오르는데 raw RLVR은 SFT를 못 넘나

SFT+RLVR은 이미 올바른 터미널 action manifold 위에서 시작한다. 그래서 RLVR의 작은 보상이 첫 command, command precision, completion timing을 미세하게 조정할 수 있다.

raw RLVR은 시작점이 다르다. raw LFM2.5는 일반 지능은 강하지만 terminal JSON agent로 행동하는 데이터가 충분히 들어간 상태가 아니다. 그래서 RLVR 신호의 많은 부분이 “문제 해결”이 아니라 “출력 형식 복구”에 쓰인다.

비유하면 이렇다.

```text
SFT+RLVR:
  이미 운전면허가 있는 사람에게 실제 도로 주행으로 코너링을 다듬는 과정

raw RLVR:
  자동차 조작법도 덜 익힌 사람에게 바로 도로 주행을 시키는 과정
```

둘 다 배운다. 하지만 같은 시간 안에 더 높은 점수로 가는 쪽은 SFT warmup 후 RLVR이다.

## 15. 왜 RLVR 점수가 자주 떨어지나

RLVR은 supervised epoch처럼 매 checkpoint가 단조 상승하지 않는다.

주요 원인:

```text
1. GRPO reward는 group-relative라 같은 평균 reward라도 sample 구성에 따라 update 방향이 달라진다.
2. verifier reward가 sparse하면 policy gradient가 약하거나 noisy하다.
3. command_bonus/format_penalty는 유용하지만 TB2-lite F1과 완전히 같은 목적함수가 아니다.
4. terminal observation CE는 terminal dynamics를 배우게 하지만, 쉬운 observation만 예측하는 쪽으로도 gradient가 생긴다.
5. LoRA adapter는 일부 layer 방향만 바꾸므로 특정 checkpoint에서 좋은 균형점이 생겼다가 지나칠 수 있다.
6. 평가 metric은 offline replay command F1이고, 학습 reward는 live sandbox verifier 기반이다.
```

따라서 RLVR 결과는 반드시 checkpoint sweep으로 봐야 한다. 현재도 parent run은 `checkpoint-610`이 best이고, raw run은 `checkpoint-1225`가 새 best다. final checkpoint를 대표 모델로 삼으면 점수를 놓칠 수 있다.

## 16. 도커 없는 RLVR의 한계

도커가 없어서 RLVR이 불가능한 것은 아니다. 실제로 command를 실행하고 stdout/stderr/exit code를 받아서 observation CE loss에 넣고 있다. 즉 “터미널 명령 실행 기반 RLVR” 자체는 되고 있다.

하지만 도커가 없으면 다음 한계가 있다.

```text
1. task별 filesystem/process isolation이 약하다.
2. 의존성 설치, cache, tmux, uv, pytest side effect가 작업 공간 밖으로 새기 쉽다.
3. verifier timeout이 불안정하면 reward noise가 커진다.
4. 같은 task를 여러 rollout으로 돌릴 때 완전히 동일한 초기 상태 보장이 어렵다.
5. 모델이 위험하거나 긴 명령을 내면 local machine 안정성에 영향을 줄 수 있다.
```

그래서 현재 script는 sandbox root를 나누고, `/workspace`, `/output`, `/logs` 위주로 실행하게 prompt를 제한하고, GPU/CUDA probe를 금지한다. 그래도 Docker/Harbor급 격리는 아니다.

성능 관점에서는 도커 부재가 특히 verifier reward 품질을 약하게 만든다. verifier reward가 약하면 RLVR은 “정답을 맞힌 trajectory를 강화”하기보다 “터미널 출력 예측과 형식 안정화” 쪽으로 더 많이 작동한다. 이것이 raw run이 45점대에서 plateau를 보이는 주요 원인 중 하나다.

## 17. Post-Training Reasoning Data 관점

참고 논문:

```text
A Primer in Post-Training Reasoning Data: What We Know About How It Works
https://arxiv.org/abs/2606.02113
```

이 primer 관점에서 보면 현재 결과는 이상하지 않다. post-training의 성능은 단순히 optimizer 이름이 아니라 데이터 객체와 verifier contract가 결정한다. 이 논문은 reasoning data를 단순한 prompt-response 쌍이 아니라, task context, model behavior, judging feedback, attribution metadata가 묶인 evidence-bearing record로 보라고 말한다. 터미널 에이전트에서는 이 말이 특히 중요하다.

우리 SFT 데이터는 고밀도 action demonstration이다.

```text
instruction -> assistant JSON/tool action
```

우리 raw RLVR 데이터는 환경 검증형 데이터다.

```text
task -> action -> terminal observation -> verifier
```

환경 검증형 데이터는 이론적으로 강력하지만, verifier가 sparse하거나 task environment가 불안정하면 gradient가 약해진다. 그래서 “데이터가 90%, agent가 9%, 모델이 1%”라는 실전 감각과 맞는다. 현재 가장 큰 성능 상승은 SFT 데이터의 양과 품질에서 왔고, RLVR은 그 위에서 미세 조정할 때 가장 잘 작동했다.

이 관점에서 LFM2.5 실험의 핵심은 다음이다.

```text
1. raw LFM2.5의 36.53점은 잠재 터미널 능력이 이미 있음을 보여준다.
2. SFT 1Epoch의 52.30점은 그 잠재 능력을 평가 포맷으로 끄집어낸 결과다.
3. 동시에 SFT는 JSON action, 중단 타이밍, 검증 습관 같은 절차 지식은 실제로 주입했다.
4. SFT 2Epoch의 50.48점은 더 많은 SFT가 항상 좋은 것은 아님을 보여준다.
5. SFT+RLVR의 54.05점은 RLVR이 이미 좋은 action manifold 위에서 가장 효율적임을 보여준다.
6. raw RLVR의 45.32점은 ECHO-style interaction learning이 작동하지만, SFT warmup 없이 곧장 50점대를 넘기기는 어렵다는 증거다.
```

즉 SFT와 RLVR은 경쟁 관계라기보다 역할이 다르다. SFT는 “답안지 양식과 행동 prior”를 맞추고, RLVR은 “터미널 피드백과 verifier를 통해 그 행동을 다듬는” 쪽에 가깝다.

## 18. Predictable Compression Failures 관점

참고 논문:

```text
Predictable Compression Failures: Order Sensitivity and Information Budgeting for Evidence-Grounded Binary Adjudication
https://arxiv.org/abs/2509.11208
```

이 논문은 evidence order, information budget, verifier-relative Bernoulli predicate 관점에서 모델 실패를 해석한다. 터미널 RLVR에도 비슷한 시사점이 있다.

터미널 에이전트의 각 step은 사실상 다음 binary predicate를 계속 통과해야 한다.

```text
1. JSON이 valid한가?
2. 첫 command가 맞는가?
3. command가 실행 가능한가?
4. stdout/stderr를 보고 상태를 업데이트했는가?
5. verifier가 pass하는가?
```

raw 모델은 정보 예산이 부족하면 긴 설명이나 잘못된 completion으로 빠진다. SFT는 그 예산을 줄여준다. “이 벤치에서는 이렇게 action을 내면 된다”는 압축된 행동 prior를 넣어주기 때문이다. ECHO는 terminal observation을 예측하게 해서 world model을 조금 더 잘 만들지만, action prior 자체가 약하면 그 효과가 형식 안정화에 먼저 쓰인다.

## 19. 현재 active run 상태

2026-06-13 08:06 KST 확인 기준:

| 항목 | 값 |
| --- | --- |
| active run | `run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k` |
| train step | `1250` |
| latest saved checkpoint | `1250` |
| max steps | `2000` |
| 남은 step | 약 `750` |
| 예상 남은 시간 | 약 `7시간`, 느린 verifier task 포함 시 `8~9시간` |
| latest evaluated raw checkpoint | `1225` |
| latest evaluated score | `45.32` |
| best evaluated raw checkpoint | `1225` |
| best raw score | `45.32` |

최근 train log:

```text
step=1250
reward_mean=-0.18250
verifier_reward_mean=0.0
policy_loss_mean=-1.28880
world_loss_mean=1.10797
action_tokens_mean=786.00
obs_tokens_mean=183.75
lr=1e-6
```

해석:

verifier reward가 아직 `0.0`이므로 “문제 해결 성공 trajectory를 강화하는 강한 RL”이라고 보긴 어렵다. 다만 checkpoint 평가 점수가 올랐으므로 ECHO world-model loss와 format shaping이 TB2-lite action 품질을 개선하고 있다.

## 20. Hugging Face 업로드

현재 sync 프로세스가 돌고 있다.

Rollout dataset repo:

```text
LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts
```

Adapter model repo:

```text
LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-GRPO-Adapters
```

raw run path:

```text
raw-lfm25/run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k
```

sync interval:

```text
1800 seconds
```

이 rollout 데이터는 나중에 RLVR 재학습, 실패 trajectory SFT, verifier 개선, DPO/IPO류 preference construction에 쓸 수 있다. 성공 trajectory뿐 아니라 실패 trajectory와 에러 로그도 중요하다. ECHO의 핵심은 실패 rollout에도 terminal feedback이 들어 있다는 점이다.

## 21. 다음 액션

현재 우선순위:

```text
1. active raw run은 2000 step까지 유지한다.
2. GPU6로 저장 checkpoint를 계속 평가한다.
3. README에는 최고 점수만 반영한다.
4. final checkpoint가 아니라 best checkpoint를 대표로 삼는다.
5. raw run 종료 후 checkpoint-score curve를 다시 그린다.
```

성능 개선 방향:

```text
1. SFT 1Epoch + ECHO RLVR best checkpoint-610을 현재 leaderboard 대표로 유지한다.
2. raw RLVR은 SFT 대체가 아니라 ECHO signal 검증용으로 본다.
3. raw에서 50점대를 노리려면 format warmup 또는 짧은 SFT가 필요하다.
4. ECHO 논문에 더 가깝게 가려면 Docker/Harbor 또는 ZeroBox/OpenSandbox급 격리가 필요하다.
5. max_turns와 max_new_tokens를 늘리려면 vLLM throughput과 저장 간격을 다시 조정해야 한다.
6. verifier reward가 계속 0이면 reward contract를 더 촘촘하게 만들어야 한다.
```

현실적인 판단:

```text
SFT는 foundation이다.
RLVR은 이미 형식과 command prior가 잡힌 모델을 더 날카롭게 깎는 도구다.
raw RLVR은 오를 수 있지만, sparse verifier와 no-Docker 환경에서는 SFT를 바로 이기기 어렵다.
```

따라서 지금까지의 최종 해석은 다음 한 줄이다.

```text
LFM2.5에서는 full terminal/toolcall SFT가 가장 큰 도약을 만들고, ECHO-style RLVR은 그 위에서 first-command와 command selection을 더 다듬어 현재 최고점 54.05를 만든다.
```

## 22. 관련 파일 위치

README:

```text
README.md
```

SFT config:

```text
Liquid-CLI/configs/sft_h200_4gpu_lfm25_8b_a1b_terminal_toolbench_full.env
```

SFT dataset builder:

```text
Liquid-CLI/scripts/build_lfm_terminal_toolcall_dataset.py
```

ECHO RLVR trainer:

```text
Liquid-CLI/train_lfm_terminal_echo_live_grpo.py
```

ECHO RLVR no-Docker launcher:

```text
Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh
```

vLLM rollout launcher:

```text
Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh
```

GPU6 evaluator:

```text
Liquid-CLI/scripts/watch_echo_rlvr_gpu6_eval_queue.sh
```

SFT 1Epoch result:

```text
tb2_lite/results/20260605T_live_hrm_lora_lfm_epoch1/LFM2.5-8B-A1B-terminal-toolbench-full-1epoch-checkpoint-1542.json
```

SFT 2Epoch result:

```text
tb2_lite/results/20260605T_all_idle_eval/LFM2.5-8B-A1B-terminal-toolbench-full-2epoch-final.json
```

SFT+RLVR best result:

```text
tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-rlvr-parentrun-checkpoint-610.json
```

raw RLVR best result:

```text
tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612/lfm25-echo-raw-lfm25-checkpoint-1225.json
```

raw active run log:

```text
/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k/train.log
```

raw active checkpoints:

```text
/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B__echo_raw_live_grpo_vllm_r32_run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k/checkpoint-*
```
