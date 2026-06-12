# LFM2.5 ECHO RLVR 실행/평가 기록

업데이트: 2026-06-12 02:34 UTC / 2026-06-12 11:34 KST

## 한 줄 결론

아직 "RLVR이 별 효과 없다"가 결론은 아니다. 현재까지는 continuation `checkpoint-250`이 TB2-lite replay Score `52.88`을 찍어 SFT 1Epoch baseline `52.30`을 `+0.58` 넘겼다.

다만 최신 평가 완료 continuation `checkpoint-630`은 Score `49.85`로 내려갔다. 즉, 지금 관측된 결론은 "RLVR 신호는 있지만 장기 학습이 자동으로 계속 좋아지는 아하 모먼트는 아직 안 보인다"에 가깝다. 최종 checkpoint가 아니라 best checkpoint를 골라야 하는 상황이다.

중요한 축 구분:

- `parentrun checkpoint-N`: 2026-06-09에 시작된 이전 RLVR run의 local step `N`이다. 범위는 `10`부터 `1880`까지다.
- `continue checkpoint-M`: 현재 run에서 이전 `parentrun checkpoint-1880`을 resume adapter로 물고 추가 학습한 local step `M`이다.
- 따라서 continuation `checkpoint-250`은 대략 누적 `1880 + 250 = 2130` step 지점이다.
- 이전 `parentrun checkpoint-1880` 최종만 보면 놓칠 수 있다. 그래서 GPU6에서 parent run의 `checkpoint-10, 20, 30, ... 1880`을 처음부터 끝까지 전수 평가한다.

## 현재 살아 있는 작업

- Base model: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- Active RLVR run: `run_20260611T094438Z_echo_public1500_continue_from_1880_vllm4_train2`
- 학습 출력: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260611T094438Z_echo_public1500_continue_from_1880_vllm4_train2`
- Rollout trace: `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260611T094438Z_echo_public1500_continue_from_1880_vllm4_train2`
- Resume adapter: 이전 ECHO RLVR run의 `checkpoint-1880`
- 체크포인트 저장 주기: 10 train steps
- 확인 시점 최신 학습 step: `670`
- 확인 시점 최신 저장 checkpoint: `checkpoint-670`
- Wall-time stop 기준 예상 종료: 2026-06-13 09:14 UTC / 2026-06-13 18:14 KST 전후

## GPU 배치

- GPU 0-3: vLLM rollout server 4개, ports `8123-8126`
- GPU 4-5: RLVR training rank 2개
- GPU 6: TB2-lite replay checkpoint 평가
- GPU 7: 이 작업에서 제외. 다른 작업용으로 건드리지 않는다.

GPU가 "노는 것처럼" 보이는 이유는 순수 matmul SFT가 아니라 live terminal RLVR이기 때문이다. 한 step 안에 모델 생성, JSON parse, 실제 shell command 실행, stdout/stderr 관찰, verifier 실행, trace 저장, LoRA update가 섞여 있다. 특히 CPU/파일시스템/terminal 실행을 기다리는 구간에서는 GPU 4-5의 VRAM은 잡혀 있어도 util이 0으로 떨어진다. GPU 0-3 vLLM도 rollout 요청이 몰릴 때만 util이 튄다. 이것은 현재 구조의 병목이고, 도커 유무만의 문제는 아니다.

## 데이터

현재 활성 run에서 실제로 쓰는 학습 데이터:

- 파일: `/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl`
- 총 row: `1,500`
- `endless_terminals`: `772`개, `51.47%`
- `openthoughts_agent_v1_rl`: `728`개, `48.53%`

이 데이터는 `Liquid-CLI/scripts/prepare_echo_terminal_data.py`로 만든 LFM용 live terminal task 포맷이다. 각 row에는 `prompt`, `task_id`, `source`, `task_dir` 또는 `task_binary_b64`, `instruction`, `echo_path`, `prompt_tokens`가 들어 있다.

중요한 차이:

- ECHO 논문 원본은 Harbor/Docker task backend를 쓴다.
- 우리는 Docker 없이 local workspace sandbox로 푸는 변형을 쓰고 있다.
- 따라서 논문과 같은 "완전 동일 인프라 재현"은 아니다.
- 대신 논문 방법론의 핵심인 "터미널 출력 token에도 auxiliary CE loss를 걸어 world model을 배우게 한다"는 구조는 구현되어 있다.

현재 활성 run에는 TB2-lite 평가 데이터를 학습에 직접 넣지 않았다. command line에도 `--no-include-tb-dev`, `--no-include-tblite-train`이 들어가 있다. TB2-lite는 지금 checkpoint 선택용 평가로 쓰고 있다. TerminalBench 1/3 계열 데이터를 추가로 섞는 실험은 다음 run에서 별도로 해야 한다. 그래야 TB2 최종 평가 오염과 현재 run의 원인을 분리할 수 있다.

## 학습 방법

실행 스크립트:

- `Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh`
- 핵심 trainer: `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`

학습 루프는 다음 순서다.

1. 모델이 JSON으로 shell command를 낸다.
2. command를 local sandbox에서 실제 실행한다.
3. stdout, stderr, exit code, verifier 결과를 terminal observation으로 만든다.
4. observation을 다음 turn context로 다시 넣는다.
5. trajectory 전체를 token 단위로 다시 forward한다.
6. assistant action token에는 reward 기반 policy loss를 건다.
7. terminal observation token에는 ECHO-style world-model CE loss를 건다.

현재 주요 하이퍼파라미터:

- LoRA rank: `32`
- world model coefficient: `0.03`
- learning rate: `5e-7`
- max turns: `8`
- max new tokens: `512`
- max seq length: `12288`
- command timeout: `20s`
- verifier timeout: `90s`
- prompts per rank: `1`
- generations per prompt: `2`
- rollout workers: `4`
- train GPUs: `2`
- rollout vLLM replicas: `4`

loss 구조는 코드상 다음과 같다.

- `policy_loss`: action token logprob에 GRPO-style advantage를 곱한다.
- `world_loss`: observation token logprob에 cross entropy를 건다.
- 최종 loss: `policy_loss + 0.03 * world_loss`

즉, "터미널 출력을 그냥 다음 행동 참고용으로만 쓰는 GRPO"가 아니라, 실제 terminal output token도 loss에 들어간다. 이 점에서는 ECHO 논문 아이디어를 따른다.

## 지금까지 한 것

- `echo-rl` repo clone 확인.
- `echo-rl/README.md`와 paper 구조 확인.
- Docker/Harbor 대신 local sandbox용 trainer 사용.
- vLLM rollout server 4개를 GPU 0-3에 유지.
- train rank 2개를 GPU 4-5에 유지.
- GPU 6에서 TB2-lite full replay 303-step 평가 watcher 실행.
- checkpoint 평가는 50-step sparse에서 10-step dense로 변경.
- continuation run은 `10` 단위 누락분을 계속 채운다.
- parent run은 `checkpoint-10`부터 `checkpoint-1880`까지 10-step 간격 전체를 처음부터 평가한다. `checkpoint-50` 같은 진짜 초반부도 포함한다.
- rollout trace와 adapter checkpoint를 Hugging Face에 주기적으로 sync.
- 평가 JSON도 HF dataset repo의 eval path로 주기적으로 sync.

## 현재 평가 결과

Baseline:

- SFT 1Epoch: Score `52.30`
- SFT 2Epoch: Score `50.48`
- LiquidAI base: Score `36.53`
- parent ECHO RLVR standalone `checkpoint-1880`: Score `50.05`
- parentrun sweep `checkpoint-10`: Score `51.14`
- parentrun sweep `checkpoint-1830`: Score `51.94`
- parentrun sweep `checkpoint-1880`: Score `51.86`

현재 RLVR 최고:

- continuation `checkpoint-250`: Score `52.88`, Cmd F1 `0.5305`, First Cmd `52.5%`, Valid JSON `74.9%`

GPU6 평가 진행률:

- continuation run: 저장 checkpoint `67`개 중 평가 완료 `38`개, 남은 `29`개
- parent run: 저장 checkpoint `188`개 중 평가 완료 `11`개, 남은 `177`개
- parent run은 현재 `checkpoint-20` 평가 중이며, 이후 `30, 40, 50, ...` 순서로 올라간다.

체크포인트별 TB2-lite replay 결과:

| Checkpoint | Score | Cmd F1 | First Cmd | Valid JSON |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 50.46 | 0.5075 | 49.8% | 76.2% |
| 20 | 51.78 | 0.5276 | 49.5% | 77.6% |
| 30 | 51.36 | 0.5203 | 49.8% | 76.6% |
| 40 | 51.61 | 0.5209 | 50.5% | 77.2% |
| 50 | 51.59 | 0.5163 | 51.5% | 76.2% |
| 60 | 51.44 | 0.5214 | 49.8% | 76.9% |
| 70 | 50.95 | 0.5187 | 48.8% | 77.9% |
| 80 | 51.78 | 0.5276 | 49.5% | 76.6% |
| 90 | 51.49 | 0.5162 | 51.2% | 74.6% |
| 100 | 52.02 | 0.5156 | 53.1% | 77.6% |
| 110 | 51.52 | 0.5238 | 49.5% | 75.6% |
| 120 | 51.21 | 0.5109 | 51.5% | 75.2% |
| 130 | 50.77 | 0.5187 | 48.2% | 78.2% |
| 140 | 51.48 | 0.5177 | 50.8% | 76.9% |
| 150 | 51.16 | 0.5187 | 49.5% | 76.6% |
| 160 | 51.30 | 0.5237 | 48.8% | 76.9% |
| 170 | 50.84 | 0.5111 | 50.2% | 75.2% |
| 180 | 51.86 | 0.5176 | 52.1% | 76.9% |
| 200 | 50.73 | 0.5169 | 48.5% | 75.6% |
| 250 | 52.88 | 0.5305 | 52.5% | 74.9% |
| 300 | 49.87 | 0.5058 | 48.2% | 76.2% |
| 350 | 50.60 | 0.5094 | 49.8% | 76.6% |
| 400 | 51.47 | 0.5218 | 49.8% | 77.6% |
| 450 | 51.13 | 0.5072 | 52.1% | 76.2% |
| 500 | 51.41 | 0.5150 | 51.2% | 75.6% |
| 540 | 51.12 | 0.5152 | 50.2% | 76.9% |
| 550 | 49.38 | 0.4946 | 49.2% | 74.3% |
| 560 | 51.55 | 0.5243 | 49.5% | 76.9% |
| 570 | 50.44 | 0.5097 | 49.2% | 75.9% |
| 580 | 51.12 | 0.5182 | 49.5% | 77.2% |
| 590 | 49.95 | 0.5015 | 49.5% | 74.3% |
| 600 | 51.04 | 0.5085 | 51.5% | 74.9% |
| 610 | 49.93 | 0.5012 | 49.5% | 75.2% |
| 620 | 50.68 | 0.5162 | 48.5% | 75.6% |
| 630 | 49.85 | 0.5086 | 47.5% | 76.6% |
| parent standalone 1880 | 50.05 | 0.5114 | 47.5% | 74.9% |
| parentrun 10 | 51.14 | 0.5154 | 50.2% | 76.6% |
| parentrun 1830 | 51.94 | 0.5269 | 50.2% | 76.9% |
| parentrun 1880 | 51.86 | 0.5215 | 51.2% | 76.9% |

## 해석

긍정적인 신호:

- SFT 1Epoch baseline 52.30보다 높은 checkpoint가 나왔다.
- continuation `checkpoint-250`은 Score 52.88로 현재 전체 README 기준 기존 1위였던 SFT 1Epoch를 넘는다.
- 초기 구간 20, 80, 100, 180도 51점대 후반-52점 초반까지 올라온다.

부정적인 신호:

- `checkpoint-300` 이후에는 49-51점대에서 흔들린다.
- 최신 `checkpoint-630`은 49.85로 baseline보다 낮다.
- parent adapter 1880도 standalone 평가 기준 50.05, parentrun prefix 평가 기준 51.86으로 SFT 1Epoch보다는 낮다. 즉 이전 장기 RLVR adapter가 TB2-lite replay에는 이미 drift를 만든 흔적이 있다.

아직 모르는 것:

- parent run 초중반 `checkpoint-10~1780` 대부분은 아직 평가 중이다.
- 이전 1880-step run의 중간에서 최고점이 나왔을 가능성이 있다.
- 그래서 지금 결론은 "continuation 250이 현재 확인된 최고"이지, "전체 RLVR history의 최고"가 아니다.

현재 결론:

- RLVR 자체가 무의미하다고 보기는 어렵다.
- 그러나 현재 데이터/보상/무도커 sandbox/continuation 설정에서는 장기 학습이 안정적인 단조 개선을 만들지는 못하고 있다.
- 이 run에서는 final checkpoint보다 `checkpoint-250` 같은 early best checkpoint를 선택해야 한다.

## 왜 아하 모먼트가 안 보일 수 있나

1. 이미 강한 SFT에서 시작했다.

SFT 1Epoch가 이미 Score 52.30이다. 남은 headroom이 작아서 reward가 조금만 어긋나도 TB2-lite replay score가 쉽게 내려간다.

2. RL objective와 TB2-lite replay metric이 다르다.

학습은 실제 command 실행/verifier reward를 본다. 평가는 정답 trajectory와 command overlap을 보는 replay metric이다. 실제로 더 잘 푸는 방향이 command-F1에서는 낮게 보일 수 있고, 반대로 command-F1이 높아도 실제 task 성공률과 완전히 같지 않다.

3. Docker/Harbor가 아니다.

논문은 Harbor/Docker backend로 격리된 task container를 쓴다. 우리는 local sandbox를 쓴다. path rewrite, host-sensitive command block, package 차이, timeout 차이, verifier 실행 차이가 reward noise를 만든다.

4. parent adapter에서 이어서 했다.

현재 run은 SFT에서 바로 시작한 clean run이 아니라 이전 RLVR `checkpoint-1880`에서 이어서 시작했다. 이미 policy가 한 번 이동한 상태라 추가 RL이 더 빨리 drift를 만들 수 있다.

5. 데이터가 1,500개로 제한되어 있다.

현재 활성 run은 공개 ECHO 관련 1,500개 task만 사용한다. 논문 수준의 더 큰 Harbor task corpus와 동일하지 않다.

6. verifier reward가 sparse하다.

로그를 보면 `verifier_reward_mean`이 0인 step이 많고, 일부 step에서만 0.25/0.5가 나온다. sparse reward 환경에서는 GRPO advantage가 매우 noisy해진다.

## 다음 개선 방향

바로 할 일:

- GPU6 평가 watcher로 10-step checkpoint를 계속 채운다.
- 최신 checkpoint도 계속 평가한다.
- JSON 결과는 `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612`에 저장한다.
- HF dataset eval path에도 주기적으로 sync한다.
- best checkpoint 기준으로만 모델카드/README claim을 갱신한다.

다음 run 후보:

1. clean start run

SFT 1Epoch에서 직접 시작한다. parent `checkpoint-1880`을 거치지 않는다. 현재 결과상 parent adapter가 이미 TB2-lite에 불리한 drift를 갖고 있을 가능성이 있다.

2. 짧은 early-stop sweep

10-step마다 평가하면서 0-300 step을 더 촘촘히 본다. 현재 최고가 250 근처라 이 구간을 먼저 안정화하는 편이 낫다.

3. world model coefficient ablation

`0.00`, `0.01`, `0.03`, `0.05`를 나눠 본다. 현재 `0.03`은 ECHO signal을 주지만 TB2-lite replay에는 과하거나 약할 수 있다.

4. reward alignment

verifier reward만 보지 말고 command validity, first command, JSON validity, command-F1 proxy를 약하게 섞는 실험을 한다. TB2-lite 점수를 올리는 것이 목적이면 evaluation metric과 더 가까운 auxiliary reward가 필요하다.

5. sandbox 안정화

Docker를 못 쓰는 상황에서는 `zerobox`나 `OpenSandbox` 같은 user-space sandbox를 검토한다. 목표는 host 환경을 망가뜨리지 않으면서도 path/package/verifier 차이를 줄이는 것이다.

6. TerminalBench 1/3 또는 추가 terminal task 투입

현재 활성 run에는 TB1/3 계열 직접 학습 데이터가 들어가지 않았다. 다음 run에서 별도 데이터 split을 만들어 넣을 수 있다. 다만 TB2 최종 평가와 섞이면 leakage 논란이 생기므로, train/eval split을 명확히 나눠야 한다.

## 남은 일

- `checkpoint-190` 이후 dense backfill 평가 계속 진행.
- `checkpoint-640` 이후 최신 checkpoint 평가 계속 진행.
- 평가 결과가 쌓이면 이 문서와 result README를 갱신.
- 유의미한 best checkpoint가 확정되면 HF model repo card에 반영.
- 장기 run이 끝나기 전이라도 `checkpoint-250`을 별도 best candidate로 보존.
