# LFM2.5 ECHO RLVR 현재 상태 노트

업데이트: 2026-06-13 00:22 UTC / 2026-06-13 09:22 KST

이 문서는 현재 진행 중인 `LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch` ECHO-style terminal RLVR 작업의 상태, 데이터, 평가 기준, 남은 리스크를 짧게 정리한다.

주의: 이 문서의 중간 본문에는 2026-06-12 초반 SFT 기반 turbo run 설명이 일부 남아 있다. 현재 active 학습은 순수 raw `LiquidAI/LFM2.5-8B-A1B`에서 다시 시작한 run으로 전환됐다. 최신 active run, raw checkpoint 평가, ECHO 논문과의 차이는 [`docs/LFM25_RAW_ECHO_RLVR_RESTART_20260612.ko.md`](LFM25_RAW_ECHO_RLVR_RESTART_20260612.ko.md)와 [`docs/ECHO_RLVR_GPU6_EVAL_20260612.md`](ECHO_RLVR_GPU6_EVAL_20260612.md)를 기준으로 본다.

## 2026-06-13 09:11 KST 즉시 업로드 상태

사용자가 요청한 "일단 된 것들"은 2026-06-13 09:22 KST 기준으로 Hugging Face에 one-shot 수동 sync까지 완료했다. 이후 새 checkpoint와 새 rollout은 loop sync가 주기적으로 다시 반영한다.

업로드 대상:

- Rollout dataset repo: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts`
- SFT 기반 ECHO RLVR adapter model repo: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-GRPO-Adapters`
- Raw clean-start ECHO RLVR adapter model repo: `LLM-OS-Models/LFM2.5-8B-A1B-Raw-ECHO-RLVR-GRPO-Adapters`
- Eval result path: `eval/tb2_lite_gpu6/lfm25_echo_rlvr_gpu6_eval_20260612`

one-shot sync 결과:

| 항목 | 상태 |
| --- | --- |
| Rollout traces | `10,920` rows 업로드 완료 |
| Train steps logged | `1,365`개 로그 반영 |
| Raw adapter checkpoints | `checkpoint-25`부터 `checkpoint-1375`까지 별도 raw adapter repo에 업로드 완료 |
| 이번 one-shot 신규 raw adapter | `checkpoint-25`~`checkpoint-1375` 전체를 새 repo로 분리 업로드 |
| GPU6 eval results | `289`개 결과 업로드 완료 |
| GPU6 eval best so far | `lfm25-echo-rlvr-parentrun-checkpoint-610`, Score `54.05` |

현재 active raw clean-start run:

- Run ID: `run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k`
- Base model: `LiquidAI/LFM2.5-8B-A1B`
- vLLM rollout GPUs: `0,1,2,3`
- Train GPUs: `4,5`
- Eval GPU: `6`
- Excluded GPU: `7`
- 2026-06-13 09:22 KST 기준 latest uploaded raw checkpoint: `checkpoint-1375`
- raw adapter upload: 기존 SFT 기반 adapter repo와 섞지 않고 `LLM-OS-Models/LFM2.5-8B-A1B-Raw-ECHO-RLVR-GRPO-Adapters`로 분리
- 다음 저장 checkpoint: `checkpoint-1400`

Raw clean-start GPU6 평가 최신 구간:

| Checkpoint | Score = 100 * avg_command_f1 | First Cmd | Valid JSON | 비고 |
| ---: | ---: | ---: | ---: | --- |
| 1225 | 45.32 | 48.2% | 66.3% | 최근 구간 고점 후보 |
| 1250 | 43.88 | 45.2% | 65.0% | 하락 |
| 1275 | 43.96 | 43.6% | 67.0% | 횡보 |
| 1300 | 45.69 | 42.9% | 69.0% | 현재 raw clean-start 최고 |
| 1325 | 44.35 | 44.6% | 66.3% | 하락 |
| 1350 | 45.20 | 44.6% | 69.0% | 일부 회복 |

주의할 점:

- README 순위용 Score는 `next_action_score`가 아니라 `100 * aggregate.avg_command_f1`이다.
- 현재 raw clean-start RLVR은 `SFT 1Epoch + RLVR` 최고점인 `54.05`에는 아직 못 미친다.
- 다만 순수 raw base rerun `39.92` 대비로는 raw clean-start RLVR이 `45점대`까지 올라왔으므로, RLVR 자체의 상승 신호는 있다.
- terminal feedback/world-model loss를 포함한 rollout traces는 모두 HF dataset에 쌓고 있으므로, 추후 재-RLVR 또는 SFT용 고품질 interaction 데이터로 재사용 가능하다.

## 현재 결론

학습은 계속 진행 중이다. GPU `0-3`은 vLLM rollout replica, GPU `4-5`는 LoRA/GRPO 학습, GPU `6`은 TB2-lite replay 평가 전용으로 사용한다. GPU `7`은 이 작업에서 제외한다.

가장 중요한 변경은 속도다. 논문/GitHub 기본값에 가까운 `num_generations=16`, `max_turns=16`, `max_new_tokens=2048` 설정은 첫 optimizer step만 15분 이상 걸렸다. 실험 사이클이 너무 느려서 중단했고, ECHO objective는 유지하되 rollout 길이를 줄인 turbo run으로 전환했다.

추가로 `save_steps=5`도 장기 학습에는 과했다. checkpoint 저장, HF adapter sync, rollout dataset sync, GPU6 평가 후보 생성이 너무 자주 끼어들어 학습 throughput을 갉아먹었다. 그래서 `run_20260612T095008Z...`의 `checkpoint-15` 어댑터를 안전 지점으로 잡고, 새 run은 `save_steps=50`으로 다시 시작했다.

현재 run은 엄밀히 말해 ECHO/SkyRL 원본의 weight-sync vLLM 구조와 완전히 동일하지 않다. vLLM replica는 SFT base를 빠르게 생성하고, train rank는 LoRA policy를 업데이트한다. 따라서 paper-identical reproduction이 아니라, no-Docker local 환경에서 ECHO observation loss와 terminal verifier RLVR을 빠르게 실험하기 위한 engineering run이다.

## 활성 학습 run

- Run ID: `run_20260612T101316Z_echo_turbo2_sft1_vllm4_train2_g4_t6_tok512_save50_wm005`
- Base model: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- Resume adapter: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260612T095008Z_echo_turbo_sft1_vllm4_train2_g4_t6_tok512_wm005/checkpoint-15`
- Output dir: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260612T101316Z_echo_turbo2_sft1_vllm4_train2_g4_t6_tok512_save50_wm005`
- Trace dir: `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260612T101316Z_echo_turbo2_sft1_vllm4_train2_g4_t6_tok512_save50_wm005/traces`
- Sandbox root: `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260612T101316Z_echo_turbo2_sft1_vllm4_train2_g4_t6_tok512_save50_wm005/sandboxes`
- GPUs: vLLM rollout `0,1,2,3`, train `4,5`, eval `6`, excluded `7`
- Backend: `--rollout-backend vllm_http`
- Save interval: every `50` train steps
- Wall-time target: `24` hours

## 학습 설정

- LoRA rank: `32`
- Trainable parameters: `12,867,584`
- World-model coefficient: `0.05`
- Learning rate: `1e-6`
- Global rollouts per step: `8` (`2 ranks * 1 prompt/rank * 4 generations`)
- Max turns: `6`
- Max new tokens: `512`
- Max sequence length: `32768`
- Max terminal output chars: `12000`
- Command timeout: `10s`
- Verifier timeout: `45s`

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

현재 turbo run은 모델 로딩, DDP wrap, optimizer 준비를 마치고 rollout trace를 기록하기 시작했다. 모델은 `/workspace`를 대상으로 `ls`, `cat`, `mkdir`, `sort` 등을 호출하고, local sandbox path로 rewrite된 명령의 `stdout/stderr`, exit code, duration, verifier 결과를 저장한다.

관측된 초기 rollout 예시는 다음과 같다.

- 이전 paper-aligned slow run: `num_generations=16`, `max_turns=16`, `max_new_tokens=2048`, 첫 step까지 15분 이상.
- 중간 fast run: `num_generations=8`, `max_turns=8`, `max_new_tokens=768`, 첫 step까지 약 4분대.
- 이전 turbo run: `num_generations=4`, `max_turns=6`, `max_new_tokens=512`, `save_steps=5`, `checkpoint-5/10/15` 저장.
- 현재 turbo2 run: 이전 turbo run `checkpoint-15`에서 어댑터만 이어받고 optimizer state는 새로 시작. `save_steps=50`으로 저장 빈도를 줄여 장기 학습 throughput을 우선한다.

실패 예시에는 정답 파일의 공백 포맷 mismatch, `task_complete=true`를 너무 빨리 내서 명령이 실행되지 않은 케이스, unsafe command pattern 차단이 포함된다. 이 실패 궤적이 바로 ECHO-style terminal observation/world-model loss에 넣어야 하는 핵심 신호다.

rollout 중 GPU util이 낮게 보일 수 있다. 이유는 8B LoRA backward보다 terminal subprocess 실행, verifier, filesystem trace write가 병목이기 때문이다. vLLM은 생성 구간에서 GPU `0-3`을 burst로 쓰고, train GPU `4-5`는 loss/backward 시점에 사용률이 튄다. 따라서 GPU util이 항상 100%가 아니더라도 run이 죽은 것은 아니다. 다만 저장/동기화/평가를 너무 자주 걸면 이 CPU/IO 병목에 추가 오버헤드가 붙기 때문에 장기 run에서는 sparse checkpoint가 맞다.

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
- LiquidAI raw base rerun: `39.92` (`lfm25-raw-base-no-sft-rerun-20260612`)

raw base rerun은 순수 `LiquidAI/LFM2.5-8B-A1B`를 같은 GPU6 TB2-lite replay 조건에서 다시 돌린 값이다. 따라서 현재 비교에서는 다음처럼 보면 된다.

| 모델/조건 | Score | SFT 1Epoch 대비 |
| --- | ---: | ---: |
| pure `LiquidAI/LFM2.5-8B-A1B` raw base rerun | `39.92` | `-12.38` |
| Terminal ToolBench SFT 1Epoch, no RLVR | `52.30` | 기준 |
| 기존 RLVR best parent `checkpoint-610` | `54.05` | `+1.75` |

따라서 현재까지는 RLVR best checkpoint가 SFT 1Epoch 대비 `+1.75`점 높다. 다만 final checkpoint가 아니라 중간 checkpoint 선택이 중요하다.

새 turbo run의 초기 checkpoint는 아직 baseline을 넘지 못했다.

| run/checkpoint | Score | 해석 |
| --- | ---: | --- |
| `run_20260612T095008Z.../checkpoint-5` | `51.89` | SFT 1Epoch `52.30`보다 `-0.41` |
| `run_20260612T095008Z.../checkpoint-10` | `51.11` | SFT 1Epoch보다 `-1.19` |
| `run_20260612T095008Z.../checkpoint-15` | `50.53` | SFT 1Epoch보다 `-1.77` |

이 숫자는 "RLVR이 무조건 실패"라는 결론은 아니다. 5~15 step은 너무 이른 구간이고, 이전 parent run에서도 최고점은 중간 checkpoint 선택에서 나왔다. 다만 초반부터 무작정 오른다는 신호도 아니므로, 현재 run은 50 step 단위로 sparse 평가하면서 score-vs-step 곡선을 다시 봐야 한다.

## GPU6 watcher 수정

`Liquid-CLI/scripts/watch_echo_rlvr_gpu6_eval_queue.sh`에서 두 가지를 수정했다.

1. README 점수 계산을 `next_action_score`가 아니라 `100 * avg_command_f1`로 고쳤다.
2. `EVAL_RUN_SPECS`를 추가해 하나의 GPU6 watcher가 parent run, continuation run, 새 paper-aligned run을 모두 평가할 수 있게 했다.

기존 watcher는 parent run과 continuation run을 계속 평가 중이다. 현재 turbo run의 checkpoint가 저장되면 같은 TB2-lite replay 기준으로 평가 대상에 추가해야 한다.

## 2026-06-12 06:12 UTC 안정화 메모

이전 새 run `run_20260612T045755Z_echo_paper_aligned_clean_sft1_hf6_g4_wm005`는 첫 optimizer step 전에 NCCL broadcast timeout으로 종료됐다. 원인은 Docker 부재 자체가 아니라, live terminal rollout에서 rank별 소요 시간이 크게 벌어진 상태에서 빠른 rank가 먼저 DDP collective로 들어간 rank skew였다.

이를 막기 위해 `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`에 다음 안정화 패치를 넣었다.

- `--dist-timeout-minutes 360`
- `--no-ddp-broadcast-buffers`
- rollout 수집 이후 optimizer 진입 전 `dist.barrier()`

이후 `loss_chunk_tokens` 기반 chunked CE 패치를 넣어 full-vocab `log_softmax` OOM을 피했다. 새 vLLM run은 이 패치를 적용해 `--loss-chunk-tokens 256`으로 실행한다.

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

- 현재 turbo2 run의 `checkpoint-50`이 저장되면 GPU6에서 README 기준으로 평가한다.
- HF adapter repo와 rollout dataset repo 업로드는 유지하되, sync 주기는 rollout `1800s`, adapter `3600s`로 낮춰 학습 병목을 줄인다.
- GPU6 평가 결과가 추가될 때마다 최고 Score 기준 README와 평가 문서를 갱신한다.
- 충분히 checkpoint가 쌓이면 score-vs-step 그래프를 다시 생성한다.
- official ECHO 내부 parquet 데이터가 공개되어 있는지 계속 확인하되, 현재는 로컬 공개 데이터 기반으로 명확히 구분해 기록한다.
