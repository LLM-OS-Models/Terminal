# LFM2.5 ECHO RLVR 현재 상태 노트

업데이트: 2026-06-12 06:12 UTC / 2026-06-12 15:12 KST

이 문서는 현재 진행 중인 `LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch` ECHO-style terminal RLVR 작업의 상태, 데이터, 평가 기준, 남은 리스크를 짧게 정리한다.

## 현재 결론

학습은 계속 진행 중이다. GPU `0-5`는 새 paper-aligned HF on-policy run에 붙어 있고, GPU `6`은 TB2-lite replay 평가 전용으로 계속 사용한다. GPU `7`은 이 작업에서 제외한다.

가장 중요한 변경은 이전 continuation run의 가장 큰 기술 리스크였던 vLLM rollout-policy 동기화 문제를 피했다는 점이다. 새 run은 vLLM 서버가 고정 SFT base를 생성하고 LoRA train rank가 따로 업데이트되는 구조가 아니라, 학습 중인 HF policy가 직접 rollout을 생성한다. 속도는 느리지만 on-policy 성격이 더 강하다.

## 활성 학습 run

- Run ID: `run_20260612T054816Z_echo_paper_aligned_sft1_ddpbarrier_hf6_g4_wm005_2k_fixargs`
- Base model: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- Output dir: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_r32_run_20260612T054816Z_echo_paper_aligned_sft1_ddpbarrier_hf6_g4_wm005_2k_fixargs`
- Trace dir: `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260612T054816Z_echo_paper_aligned_sft1_ddpbarrier_hf6_g4_wm005_2k_fixargs/traces`
- Sandbox root: `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260612T054816Z_echo_paper_aligned_sft1_ddpbarrier_hf6_g4_wm005_2k_fixargs/sandboxes`
- GPUs: train `0,1,2,3,4,5`, eval `6`, excluded `7`
- Backend: `--rollout-backend hf`
- Save interval: every `10` train steps
- Wall-time target: `47.5` hours

## 학습 설정

- LoRA rank: `32`
- Trainable parameters: `12,867,584`
- World-model coefficient: `0.05`
- Learning rate: `1e-7`
- Global rollouts per step: `24` (`6 ranks * 1 prompt/rank * 4 generations`)
- Max turns: `16`
- Max new tokens: `2048`
- Max sequence length: `32768`
- Max terminal output chars: `50000`
- Command timeout: `20s`
- Verifier timeout: `120s`

ECHO 방식의 핵심은 verifier RL loss에 더해 터미널 observation token에도 CE loss를 거는 것이다. 즉 모델이 명령을 맞히는 것만 보지 않고, `stdout/stderr`, `ls`, `cat`, 에러 메시지 같은 터미널 피드백을 내부 world model로 예측하도록 학습시킨다.

## 데이터

현재 run은 다음 prepared JSONL을 사용한다.

`/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl`

구성:

- 전체: `1,500` rows
- `openthoughts_agent_v1_rl`: `728` rows
- `endless_terminals`: `772` rows
- long/invalid skip: `0`
- prompt token min/max: `362 / 1568`

주의: 이 데이터는 공개적으로 확보 가능한 terminal interaction 데이터를 우리 환경에 맞게 포맷한 것이다. Microsoft ECHO repo의 config가 가리키는 내부 parquet 경로(`train8770_sa_q35xml_wip2.parquet` 등)는 로컬 clone에 포함되어 있지 않았다. 따라서 현재 run은 "ECHO objective와 주요 hyperparameter를 따른 paper-aligned run"이지, "논문 내부 데이터와 Harbor/Docker 인프라까지 완전히 동일한 재현"은 아니다.

## 현재 학습 진행

첫 rollout trace가 생성됐다. 현재 확인 시점에는 `rank4`, `rank5` trace가 기록되어 있으며 총 `8`개 rollout이 기록됐다. 모델은 `/workspace`를 대상으로 `ls`, `cat`, `mkdir`, `sort` 등을 호출했고, local sandbox path로 rewrite된 명령의 `stdout/stderr`, exit code, duration, verifier 결과가 저장됐다.

관측된 초기 rollout 예시는 다음과 같다.

- `rank4`: 4 rollout, verifier success 1, fail 3, avg reward 0.2375
- `rank5`: 4 rollout, verifier success 4, fail 0, avg reward 1.1375

실패 예시에는 정답 파일의 공백 포맷 mismatch, `task_complete=true`를 너무 빨리 내서 명령이 실행되지 않은 케이스, unsafe command pattern 차단이 포함된다. 이 실패 궤적이 바로 ECHO-style terminal observation/world-model loss에 넣어야 하는 핵심 신호다.

아직 첫 optimizer step 이전 구간에서는 GPU util이 낮게 보일 수 있다. 이유는 8B LoRA 학습 자체보다 HF generation, 터미널 subprocess 실행, verifier, trace write가 병목이기 때문이다. VRAM을 꽉 채우는 것이 목적이 아니라, 학습 policy가 직접 terminal rollout을 만들고 그 observation을 loss에 넣는 것이 현재 우선순위다.

## GPU6 평가 상태

평가 기준은 README와 동일하다.

```text
Score = 100 * avg_command_f1
```

`next_action_score`는 `avg_command_f1`과 first command exact를 섞은 보조 지표이므로 README 순위 기준으로 쓰지 않는다.

현재 GPU6 sweep에서 관측된 최고점:

| rank | checkpoint | Score | next_action_score | First Cmd | Valid JSON |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `lfm25-echo-rlvr-parentrun-checkpoint-610` | `54.05` | `54.18` | `54.5%` | `77.9%` |
| 2 | `lfm25-echo-rlvr-parentrun-checkpoint-490` | `53.76` | `53.17` | `51.8%` | `77.2%` |
| 3 | `lfm25-echo-rlvr-parentrun-checkpoint-650` | `53.65` | `52.91` | `51.2%` | `76.2%` |
| 4 | `lfm25-echo-rlvr-parentrun-checkpoint-230` | `53.43` | `52.76` | `51.2%` | `77.9%` |
| 5 | `lfm25-echo-rlvr-parentrun-checkpoint-440` | `53.32` | `53.16` | `52.8%` | `75.2%` |

비교 기준:

- SFT 1Epoch baseline: `52.30`
- SFT 2Epoch: `50.48`
- LiquidAI raw base: `36.53`

따라서 현재까지는 RLVR best checkpoint가 SFT 1Epoch 대비 `+1.75`점 높다. 다만 final checkpoint가 아니라 중간 checkpoint 선택이 중요하다.

## GPU6 watcher 수정

`Liquid-CLI/scripts/watch_echo_rlvr_gpu6_eval_queue.sh`에서 두 가지를 수정했다.

1. README 점수 계산을 `next_action_score`가 아니라 `100 * avg_command_f1`로 고쳤다.
2. `EVAL_RUN_SPECS`를 추가해 하나의 GPU6 watcher가 parent run, continuation run, 새 paper-aligned run을 모두 평가할 수 있게 했다.

기존 watcher는 교체했고, 현재 GPU6 watcher는 parent run, continuation run, 새 paper-aligned run checkpoint를 모두 감시한다. 새 run의 첫 checkpoint가 저장되면 같은 TB2-lite replay 기준으로 자동 평가 대상에 들어간다.

## 2026-06-12 06:12 UTC 안정화 메모

이전 새 run `run_20260612T045755Z_echo_paper_aligned_clean_sft1_hf6_g4_wm005`는 첫 optimizer step 전에 NCCL broadcast timeout으로 종료됐다. 원인은 Docker 부재 자체가 아니라, live terminal rollout에서 rank별 소요 시간이 크게 벌어진 상태에서 빠른 rank가 먼저 DDP collective로 들어간 rank skew였다.

이를 막기 위해 `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`에 다음 안정화 패치를 넣었다.

- `--dist-timeout-minutes 360`
- `--no-ddp-broadcast-buffers`
- rollout 수집 이후 optimizer 진입 전 `dist.barrier()`

현재 활성 run은 이 패치를 적용한 `run_20260612T054816Z_echo_paper_aligned_sft1_ddpbarrier_hf6_g4_wm005_2k_fixargs`다. 첫 step에서는 일부 rank가 긴 rollout을 끝내느라 GPU util이 낮게 보일 수 있지만, trace에는 실제 terminal stdout/stderr, blocked command, verifier result가 기록되고 있다.

## A Primer in Post-Training Reasoning Data에서 가져갈 해석

논문 `A Primer in Post-Training Reasoning Data: What We Know About How It Works`는 post-training reasoning data를 단순한 `prompt -> answer` 쌍으로 보지 말고, 다음처럼 검증 계약과 metadata를 포함한 데이터 객체로 봐야 한다고 정리한다.

```text
task/context -> trace/actions -> answer/artifact -> verifier/reward/environment -> attribution metadata
```

이 관점은 현재 ECHO RLVR 실험 해석에 바로 적용된다.

- 긴 RL을 오래 돌리는 것만으로는 부족하다.
- verifier가 sparse/noisy하면 GRPO advantage가 흔들린다.
- no-Docker local sandbox는 environment 재현성을 흔들 수 있다.
- 이미 강하게 SFT된 모델은 reward/eval metric이 조금만 어긋나도 command imitation 능력이 깨질 수 있다.
- 따라서 best checkpoint selection, verifier 안정성, on-policy rollout, metric-aligned validation이 핵심이다.

즉 현재 목표는 단순히 GPU 시간을 태우는 것이 아니라, terminal interaction record를 `trace/actions + terminal observations + verifier result + metadata` 형태로 계속 쌓아 나중에 RLVR 재학습이나 SFT 재사용이 가능한 고품질 데이터 자산으로 만드는 것이다.

## 남은 일

- GPU6 watcher 재시작 후 새 paper-aligned checkpoint도 자동 평가되는지 확인한다.
- 첫 checkpoint가 저장되면 HF adapter repo와 rollout dataset repo 업로드 여부를 확인한다.
- GPU6 평가 결과가 추가될 때마다 최고 Score 기준 README와 평가 문서를 갱신한다.
- 충분히 checkpoint가 쌓이면 score-vs-step 그래프를 다시 생성한다.
- official ECHO 내부 parquet 데이터가 공개되어 있는지 계속 확인하되, 현재는 로컬 공개 데이터 기반으로 명확히 구분해 기록한다.
