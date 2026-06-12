# LFM2.5 raw ECHO RLVR 재시작 기록

업데이트: 2026-06-12 11:38 UTC / 2026-06-12 20:38 KST

이 문서는 `LiquidAI/LFM2.5-8B-A1B` 순수 raw base에서 ECHO-style terminal RLVR을 다시 시작한 기록이다. 이전 SFT 기반 RLVR 실험과 구분하기 위해 별도 문서로 분리한다.

## 현재 결론

SFT 1Epoch 모델에서 이어서 RLVR을 더 하는 실험은 일단 멈췄다. 지금은 순수 raw `LiquidAI/LFM2.5-8B-A1B`에 새 LoRA adapter를 붙여, ECHO 논문의 핵심 아이디어인 `verifier RL + terminal observation CE loss`가 raw 모델에서 실제로 올라가는지 확인한다.

GPU 배치는 다음처럼 고정했다.

- GPU `0,1,2,3`: raw LFM2.5 vLLM rollout server, ports `8123-8126`
- GPU `4,5`: LoRA/GRPO 학습
- GPU `6`: checkpoint TB2-lite replay 평가 watcher
- GPU `7`: 제외. 이 작업에서 건드리지 않는다.

## 왜 다시 시작했나

이전 active run은 `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`에서 시작했다. 이 모델은 이미 Terminal ToolBench SFT가 강하게 들어간 상태라, RLVR이 score를 올리는지 보기 어렵고, 일부 checkpoint에서는 오히려 score가 내려갔다.

그래서 이번 run의 질문을 바꿨다.

```text
이미 잘 학습된 SFT 모델을 조금 더 깎는 문제가 아니라,
순수 raw LFM2.5가 terminal feedback을 보며 RLVR로 실제로 배울 수 있는가?
```

raw baseline은 같은 TB2-lite replay 기준으로 Score `39.92`다. SFT 1Epoch baseline `52.30`보다 `-12.38` 낮다. 따라서 raw run은 headroom이 크고, RLVR 신호가 보이면 더 명확하게 관찰된다.

## vLLM이 죽었던 정확한 원인

처음에는 GPU가 노는 것처럼 보였지만, 원인은 학습 코드가 아니라 vLLM runtime이었다.

1. `.vllm-lfm-cu12`가 user-site package를 먼저 물었다.

기본 Python path에 `~/.local`이 섞이면서 `torch 2.12.0.dev20260407+cu128`이 로드됐다. 그런데 `.vllm-lfm-cu12`의 vLLM C++ extension은 `torch 2.10.0+cu128` 조합에서 맞는 상태였다. 그 결과 다음 ABI 에러가 났다.

```text
vllm/_C.abi3.so: undefined symbol: _ZN3c1013MessageLoggerC1EPKciib
```

해결: vLLM과 평가/학습 런처에 `PYTHONNOUSERSITE=1`을 추가했다.

2. `.vllm-lfm`은 CUDA13 빌드였다.

`.vllm-lfm`은 `vllm 0.22.1`이라 import는 됐지만, CUDA13 runtime을 요구했다. 현재 드라이버는 CUDA 12.9 계열이라 GPU 초기화에서 실패했다.

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 12090)
```

해결: CUDA13 env가 아니라 `.vllm-lfm-cu12`를 사용한다.

3. background vLLM을 health check 후 셸이 종료되게 띄우면 서버가 같이 정리됐다.

ready까지 뜬 뒤에도 프로세스가 내려간 이유는 모델 문제가 아니라 실행 방식이었다. health check만 하고 command가 끝나면 실행 도구가 같은 process group의 background server를 정리할 수 있다.

해결: `Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh`처럼 마지막에 `wait`하는 long-running launcher 안에서 vLLM을 유지한다.

## 이번에 수정한 코드

다음 스크립트에 `PYTHONNOUSERSITE=1`과 CUDA library path 보강을 넣었다.

- `Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh`
- `Liquid-CLI/scripts/run_lfm25_vllm_server_clean.sh`
- `Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh`
- `Liquid-CLI/scripts/watch_echo_rlvr_gpu6_eval_queue.sh`
- `Liquid-CLI/scripts/evaluate_echo_rlvr_checkpoints_gpu6.sh`

또한 학습 런처에는 `VLLM_SERVED_MODEL` override를 추가했다. raw vLLM server의 served name은 `lfm25-raw`이므로, train script가 OpenAI-compatible API에 `LiquidAI/LFM2.5-8B-A1B` 대신 `lfm25-raw`를 보내도록 맞췄다.

## 데이터

현재 raw run은 다음 prepared dataset을 사용한다.

```text
/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl
```

구성:

- 전체: `1,500` rows
- `openthoughts_agent_v1_rl`: `728` rows, `48.53%`
- `endless_terminals`: `772` rows, `51.47%`
- prompt token min/max: `362 / 1568`
- skipped long/invalid: `0`

중요한 구분:

- TB2-lite replay 평가는 학습 데이터로 넣지 않았다.
- ECHO 논문 내부 parquet/Harbor/Docker 데이터가 그대로 들어간 것은 아니다.
- 현재 run은 공개 terminal interaction 데이터를 local no-Docker sandbox 포맷으로 맞춘 것이다.
- 방법론은 ECHO 핵심을 따른다. 즉 command reward만 쓰지 않고 terminal observation token에도 CE loss를 건다.

## 중단한 raw 첫 run

처음 raw run:

```text
run_20260612T112155Z_echo_raw_lfm25_vllm4_train2_g4_t6_tok512_save50_wm005_2k
```

설정:

- `max_turns=6`
- `max_new_tokens=512`
- `save_steps=50`
- `num_generations=4`
- global rollouts per step: `8`

문제:

- step `0-4`까지만 진행하고 중단했다.
- checkpoint 전이라 저장된 adapter는 없다.
- `action_tokens_mean`이 `1496-3101`까지 튀었다.
- 대략 `2.0-2.5분/step`이라 2000 step은 48시간 안에 어렵다.

해석:

raw 모델은 아직 terminal JSON/tool format을 안정적으로 못 잡는다. 그래서 한 action에서 긴 텍스트를 뱉고, 그 결과 rollout과 terminal execution이 무거워진다. 이 상태로 장기 학습을 돌리면 GPU 시간 대비 checkpoint 획득이 너무 늦다.

## 현재 active raw run

현재 run:

```text
run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k
```

Base model:

```text
LiquidAI/LFM2.5-8B-A1B
```

SFT adapter:

```text
none
```

Output dir:

```text
/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B__echo_raw_live_grpo_vllm_r32_run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k
```

핵심 설정:

- LoRA rank: `32`
- trainable parameters: `12,867,584`
- world-model coefficient: `0.05`
- learning rate: `1e-6`
- warmup steps: `10`
- max steps: `2000`
- max wall time: `48h`
- max turns: `4`
- max new tokens: `256`
- command timeout: `8s`
- verifier timeout: `40s`
- max terminal output chars: `10000`
- save interval: every `25` steps
- global rollouts per step: `8`

## 초기 관측

현재 active raw run은 step `0-5`까지 정상 진행됐다.

초기 로그 요약:

| step | reward_mean | verifier_reward_mean | action_tokens_mean | obs_tokens_mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `-0.1650` | `0.0` | `528.0` | `391.25` |
| 1 | `-0.2000` | `0.0` | `1043.5` | `92.0` |
| 2 | `-0.1662` | `0.0` | `914.75` | `140.75` |
| 3 | `-0.1700` | `0.0` | `1044.0` | `90.5` |
| 4 | `-0.1350` | `0.0` | `634.75` | `288.0` |
| 5 | `-0.1887` | `0.0` | `979.5` | `121.0` |

해석:

- raw라서 verifier reward는 아직 `0`이다.
- 이것만 보고 성능 하락/상승을 판단하면 안 된다.
- 다만 `action_tokens_mean`은 첫 raw run의 `1500-3100`보다 내려갔다.
- 속도는 현재 대략 `30-35초/step` 수준이다.
- 첫 checkpoint `checkpoint-25`는 시작 후 약 `10-15분` 내외, 2000 step은 대략 `17-20시간` 범위로 예상한다.

## 평가와 업로드

HF sync:

- rollout dataset repo: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts`
- adapter model repo: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-GRPO-Adapters`
- path-in-repo prefix: `raw-lfm25/{run_id}`
- sync interval: `1800s`

GPU6 평가:

- evaluator: `Liquid-CLI/scripts/watch_echo_rlvr_gpu6_eval_queue.sh`
- base model: `LiquidAI/LFM2.5-8B-A1B`
- target checkpoints: current raw run output dir
- stride: every `25` steps early
- result dir: `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612`

평가 기준은 README와 동일하다.

```text
Score = 100 * avg_command_f1
```

## checkpoint-25~225 TB2-lite 평가 결과

업데이트: 2026-06-12 13:45 UTC / 2026-06-12 22:45 KST

GPU6 watcher가 current raw clean-start run의 checkpoint-25부터 checkpoint-225까지 README와 같은 TB2-lite replay 기준으로 평가했다.

| checkpoint | README Score | next_action_score | first_cmd_exact | valid_json |
| ---: | ---: | ---: | ---: | ---: |
| 25 | `37.89` | `39.00` | `41.6%` | `59.4%` |
| 50 | `40.34` | `40.90` | `42.2%` | `58.1%` |
| 75 | `39.19` | `39.22` | `39.3%` | `57.1%` |
| 100 | `40.34` | `40.81` | `41.9%` | `58.4%` |
| 125 | `41.01` | `41.67` | `43.2%` | `57.4%` |
| 150 | `40.43` | `40.39` | `40.3%` | `59.4%` |
| 175 | `40.02` | `40.19` | `40.6%` | `58.1%` |
| 200 | `39.67` | `39.95` | `40.6%` | `59.7%` |
| 225 | `41.06` | `40.92` | `40.6%` | `57.4%` |

현재 raw run 최고점은 `checkpoint-225`의 Score `41.06`이다.

해석:

- raw base rerun Score `39.92`와 비교하면 checkpoint-225는 `+1.14` 올랐다.
- 하지만 SFT 1Epoch baseline Score `52.30`과 비교하면 아직 `-11.24` 낮다.
- checkpoint-225가 checkpoint-125를 아주 조금 넘겼지만, 225 step까지는 뚜렷한 아하 모먼트가 아니라 작은 회복/탐색 구간으로 보는 편이 맞다.
- valid JSON은 `57.1%~59.7%` 범위에 머물러 있어, raw 모델의 terminal JSON/tool format prior가 아직 약하다.
- verifier reward가 계속 `0.0`에 가까운 구간이어서, 현재는 sparse verifier 성공보다 ECHO observation loss와 format/penalty shaping이 주로 작동하는 것으로 보인다.

그래도 완전히 실패로 보기는 이르다. raw에서 바로 시작했는데 checkpoint-50/100/125/225가 raw base보다 높게 나왔으므로, terminal feedback 기반 학습 신호가 전혀 없지는 않다. 다만 논문식 아하 모먼트를 보려면 최소 checkpoint-500/1000/2000까지 곡선을 봐야 한다.

## 현재 리스크

1. raw model은 format prior가 약하다.

순수 raw에서 바로 RLVR을 하면 verifier reward가 너무 sparse할 수 있다. 초기 reward가 계속 0이면 GRPO signal보다 format penalty/world loss가 주로 작동한다.

2. ECHO 논문 원본과 인프라가 다르다.

논문은 Harbor/Docker backend를 쓴다. 현재는 no-Docker local sandbox다. path rewrite, timeout, package availability, unsafe command block 차이가 reward noise를 만든다.

3. 평가 metric과 학습 reward가 완전히 같지 않다.

학습은 실제 terminal execution/verifier를 본다. README 평가는 TB2-lite replay의 command F1이다. 실제 task 성공 능력이 조금 좋아져도 command F1이 바로 오르지 않을 수 있다.

4. 너무 오래 돌리는 것만으로는 답이 아니다.

`A Primer in Post-Training Reasoning Data` 관점에서 보면, RL은 모델의 유효 난이도 대역에서 verifier가 선명할 때 가장 잘 작동한다. raw 모델이 너무 못해서 reward가 거의 0이면, 무작정 step을 늘려도 효율이 낮다. 그래서 checkpoint-25/50/75/100의 초반 곡선을 보고 계속 갈지 판단해야 한다.

## 다음 판단 기준

- `checkpoint-25`: raw baseline `39.92` 근처로 유지되는지, format이 망가지는지 확인
- `checkpoint-50`: reward_mean과 valid JSON이 개선되는지 확인
- `checkpoint-100`: TB2-lite Score가 raw baseline 대비 상승하는지 확인
- `checkpoint-200+`: 아하 모먼트가 있는지, 아니면 reward sparse 상태가 지속되는지 판단

만약 `checkpoint-100`까지 verifier reward가 계속 0이고 TB2-lite도 raw baseline 이하라면, 순수 raw에서 바로 RLVR은 너무 sparse하다는 결론에 가까워진다. 그 경우에는 최소한의 format SFT 또는 성공/실패 trajectory SFT warmup을 작게 넣고 다시 RLVR을 거는 편이 맞다.
