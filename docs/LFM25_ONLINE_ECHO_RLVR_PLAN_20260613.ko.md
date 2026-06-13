# LFM2.5 Online ECHO RLVR 재시작 계획 및 이전 실험 정정

작성 시각: 2026-06-13 KST

## 1. 결론

이제부터의 기준 실험은 `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`에서 시작한다.

이유는 단순하다. 기존 45점대 raw RLVR 체크포인트는 성능 신호는 있었지만, vLLM rollout 서버가 학습 중 업데이트된 LoRA를 사용하지 않았다. 따라서 엄밀한 의미의 on-policy GRPO/RLVR이 아니다. 그 체크포인트에서 이어가면 실험 해석이 더 어려워진다.

새 실험은 다음 구조로 간다.

- GPU 0,1,2,3: vLLM rollout 서버 4개
- GPU 4,5: LoRA RLVR 학습
- GPU 6: 중간 checkpoint 평가
- GPU 7: 사용하지 않음

핵심 변경점은 vLLM 서버를 `--enable-lora`와 runtime LoRA update 모드로 띄우고, trainer가 몇 step마다 최신 LoRA adapter를 저장한 뒤 vLLM의 `/v1/load_lora_adapter` endpoint로 동기화한다는 점이다.

즉, 이제는 “학습된 policy가 다음 rollout을 생성하는 구조”에 가깝다.

2026-06-13 10:55 KST에 실제 online run을 시작했다.

- Run ID: `run_20260613T014646Z_sft1_online_vllm_lora_sync_g4_t4_sync5_wm005`
- 기준 모델: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- 학습 출력: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__online_echo_grpo_vllm_lora_sync_run_20260613T014646Z_sft1_online_vllm_lora_sync_g4_t4_sync5_wm005`
- 런 디렉터리: `/home/work/.data/liquid_cli_sft/live_terminal_echo_online/run_20260613T014646Z_sft1_online_vllm_lora_sync_g4_t4_sync5_wm005`
- vLLM base served model name: `lfm25-sft1-base`
- vLLM LoRA generation model name: `lfm25-sft1-online`
- LoRA sync interval: 5 optimizer updates
- checkpoint 저장 주기: 25 updates
- 계획 길이: 최대 2,000 steps 또는 36시간

## 2. 이전 static-vLLM 실험의 문제

이전 raw LFM2.5 실험은 다음 방식이었다.

- vLLM 4개 서버는 `LiquidAI/LFM2.5-8B-A1B` raw/base 모델만 serving
- trainer는 GPU 4,5에서 LoRA를 업데이트
- rollout 생성은 계속 vLLM의 raw/base 모델이 담당
- 업데이트된 LoRA는 loss 계산에는 사용되지만, 다음 행동 샘플링에는 반영되지 않음

코드상 문제는 명확했다.

`generate_assistant_text_vllm_http()`는 OpenAI-compatible `/completions` 요청에 다음 model 이름만 보냈다.

`lfm25-raw`

LoRA adapter 이름, adapter path, runtime reload, weight sync가 없었다.

따라서 이 실험은 다음처럼 표기해야 한다.

정확한 표현:
`ECHO-style static-vLLM off-policy variant`

부정확한 표현:
`paper-faithful on-policy ECHO RLVR`

이전 최고 결과는 다음과 같이 해석한다.

- raw LFM2.5 baseline: 약 36.53
- static-vLLM raw RLVR 최고 checkpoint: 약 46.06
- 의미: terminal feedback과 ECHO observation CE가 유용한 학습 신호를 준 것은 맞음
- 한계: 현재 policy로 rollout을 다시 생성하지 않았으므로 GRPO의 on-policy 조건을 만족하지 않음

이 때문에 점수가 45점대 근처에서 멈춘 가능성이 있다. 모델이 나아진 행동을 다시 생성해 더 좋은 trajectory를 만들 기회가 없었기 때문이다.

## 3. 논문 방식과의 차이

ECHO 논문 및 `echo-rl` repo는 SkyRL/vLLM 구조를 전제로 한다.

중요한 포인트는 다음과 같다.

- GRPO는 on-policy 성격이 강하다.
- trajectory는 현재 policy에서 생성되어야 한다.
- vLLM을 쓰더라도 trainer의 최신 weight가 inference engine에 동기화되어야 한다.
- ECHO는 여기에 terminal observation token cross-entropy loss를 추가한다.

`echo-rl/README.md`에도 SkyRL/vLLM을 쓰는 이유로 다음 요소가 나온다.

- generated token ids
- logprobs
- attention masks
- ECHO-specific environment-token masks
- batching
- sampling
- weight synchronization
- trajectory construction

따라서 vLLM을 쓰는 것 자체는 맞다. 문제는 vLLM을 static inference server처럼 두면 안 된다는 점이다.

ECHO 논문에서 중요한 문장은 다음과 같이 요약할 수 있다.

- 표준 GRPO는 terminal output을 다음 행동의 context로는 쓰지만, loss는 assistant action token에만 준다.
- ECHO는 같은 rollout과 같은 forward pass를 재사용하면서 terminal observation token에도 cross-entropy loss를 건다.
- observation target은 현재 policy가 실제로 만든 trajectory에서 온다. 그래서 ECHO는 on-policy dense supervision으로 작동한다.
- 논문은 Docker/Harbor 기반 terminal task 환경에서 최대 16턴, 16k context, turn당 2,048 generated tokens, 8 B200 GPU, 500 GRPO steps를 사용한다.
- 학습 corpus는 curated terminal task 2,700개에서 시작해 synthetic task를 추가하고 GPT-5 solve filter를 통과한 8,870개 task 중 8,770개를 train에 쓴다.

우리 환경과 같은 점:

- assistant command와 terminal observation이 교차하는 multi-turn rollout을 만든다.
- sparse reward만 쓰지 않고 terminal observation token에 world-model CE loss를 추가한다.
- vLLM으로 rollout generation을 병렬화한다.
- 최신 policy가 다음 rollout에 반영되도록 LoRA hot-load를 넣었다.

우리 환경과 다른 점:

- Docker/Harbor/Terminus를 쓰지 않는다. 현재는 작업 디렉터리 기반 no-docker sandbox다.
- 논문과 같은 8 B200 full stack/SkyRL weight sync가 아니라, LoRA adapter를 runtime vLLM endpoint로 주기적 hot-load한다.
- 논문은 8,770 train tasks, 우리는 현재 1,500개 prepared mixed tasks다.
- 논문은 최대 16턴/2,048 tokens/turn, 우리는 빠른 반복을 위해 최대 4턴/256 tokens/turn이다.
- 논문 평가는 실제 TerminalBench류 pass@k, 우리는 TB2-lite corrected replay proxy를 중간 신호로 쓴다.

따라서 이 run은 “ECHO 논문의 목적 함수와 online rollout 원칙을 우리 인프라에 맞게 이식한 실험”이지, 논문 환경의 완전 재현은 아니다.

## 4. 새 온라인 방식

새 방식은 다음 흐름이다.

1. SFT 1Epoch 모델을 vLLM base로 띄운다.
2. vLLM 서버는 `--enable-lora`로 시작한다.
3. runtime LoRA update를 허용한다.
4. trainer는 시작 직후 현재 LoRA를 `step_000000`으로 저장한다.
5. 모든 vLLM replica에 `/v1/load_lora_adapter`로 이 adapter를 로드한다.
6. generation 요청은 base model name이 아니라 LoRA adapter name을 사용한다.
7. N step마다 trainer가 최신 LoRA를 저장한다.
8. vLLM은 기존 adapter를 unload하고 최신 adapter를 load한다.
9. 다음 rollout부터 업데이트된 policy가 사용된다.

현재 붙인 핵심 옵션은 다음과 같다.

- `--vllm-lora-name`
- `--vllm-lora-sync-steps`
- `--vllm-lora-sync-dir`
- `--vllm-lora-load-inplace`

vLLM replica 실행 스크립트에는 다음 옵션을 추가했다.

- `ENABLE_LORA=1`
- `MAX_LORA_RANK=32`
- `MAX_LORAS=1`
- `MAX_CPU_LORAS=4`
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`

## 5. 왜 4 vLLM + 2 train + 1 eval인가

이 구성은 현재 장비에서 가장 현실적인 절충이다.

vLLM 4대:
rollout 생성과 terminal interaction이 가장 큰 병목이다. 4개 replica를 두면 여러 generation을 병렬로 처리할 수 있다.

학습 2대:
LoRA rank 32 학습은 full fine-tuning보다 훨씬 가볍다. 8B MoE 모델이라도 2개 GPU DDP로 충분히 step을 돌릴 수 있다.

평가 1대:
GPU 6은 중간 checkpoint를 계속 평가하는 용도다. RLVR은 점수가 단조 증가하지 않는다. 따라서 checkpoint별 평가가 중요하다.

GPU 7:
다른 작업이 있으므로 사용하지 않는다.

이 구성의 장점은 GPU를 놀리지 않으면서도 train/eval/rollout을 분리할 수 있다는 점이다.

단점은 LoRA sync 주기마다 약간의 정지 비용이 있다는 점이다. 그러나 LoRA adapter만 이동하므로 full weight sync보다 훨씬 가볍다.

## 6. SFT 1Epoch에서 시작하는 이유

이번 proper-online RLVR은 raw 45점 checkpoint가 아니라 SFT 1Epoch 모델에서 시작한다.

이유:

- SFT 1Epoch 모델은 README 기준 최고권 성능을 이미 보인 기준 모델이다.
- raw 45점 checkpoint는 static-vLLM off-policy 산물이므로 이어가면 실험 해석이 꼬인다.
- ECHO 논문 계열 실험에서도 RL은 보통 instruction/SFT 능력이 있는 policy에서 더 안정적으로 돈다.
- TerminalBench류 작업은 JSON/tool-call format, command style, recovery behavior가 중요하다. SFT가 이 인터페이스를 이미 잡아준다.

따라서 새 실험의 목표는 다음이다.

`SFT로 터미널 형식과 기본 행동을 잡고, online ECHO RLVR로 실제 terminal feedback에 맞춘 복구/탐색 능력을 개선한다.`

여기서 `A Primer in Post-Training Reasoning Data` 관점도 중요하다. 그 논문은 post-training reasoning data를 “무슨 데이터 객체를 쓰는가, 왜 유용한가, 어떻게 만들고 확장하는가”로 정리한다. 이번 실험에서 SFT가 강하게 먹힌 이유는 새 지식을 무한히 주입했다기보다, LFM2.5가 이미 가지고 있던 shell/code/tool-use 능력을 Terminal ToolBench 형식으로 꺼내 쓰는 경로를 안정화했기 때문이라고 보는 것이 더 맞다.

다만 SFT가 전혀 지식 주입이 아니라고 단정하면 안 된다. 터미널 명령, JSON tool-call, task domain 분포가 충분히 반복되면 모델은 새로운 surface pattern과 일부 절차 지식을 adapter에 저장한다. 그래서 현재 해석은 다음처럼 둔다.

- SFT의 1차 효과: pre-training에 있던 능력을 특정 인터페이스와 답안 형식으로 정렬
- SFT의 2차 효과: 반복되는 terminal/task 절차와 포맷을 adapter에 부분 저장
- RLVR의 기대 효과: 정답 command를 그대로 모방하는 것이 아니라, 실패 출력/에러/검증 결과를 보고 다음 행동을 고르는 정책을 보정

## 7. 예상 관찰 포인트

이번 실험에서 봐야 할 것은 최종 점수만이 아니다.

- Valid JSON 비율이 유지되는가
- First command exact가 무너지지 않는가
- next_action_score가 올라가는가
- verifier reward가 0에서 벗어나는 빈도가 증가하는가
- terminal observation CE가 낮아지는가
- 특정 checkpoint 이후 점수가 급락하지 않는가

RLVR은 오래 돌린다고 항상 좋아지지 않는다. 특히 이미 SFT가 강한 모델에서는 작은 learning rate, 짧은 sync interval, 잦은 평가가 중요하다.

## 8. 현재 수정한 코드

수정 파일:

- `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`
- `Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh`
- `Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh`

핵심 변경:

- vLLM generation request가 LoRA adapter model name을 사용할 수 있게 함
- trainer가 LoRA checkpoint를 저장하고 vLLM에 hot-load하도록 함
- vLLM replica launcher가 LoRA runtime update 모드로 시작하도록 함

검증한 내용:

- vLLM version: `0.19.1`
- `/v1/load_lora_adapter` endpoint 존재 확인
- `/v1/unload_lora_adapter` endpoint 존재 확인
- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`로 runtime LoRA update route 활성화
- `step_000000` LoRA adapter를 vLLM 4개 replica 모두에 load 성공
- `/v1/completions` 요청에서 `model=lfm25-sft1-online`으로 생성 성공

## 9. 문서 표기 원칙

앞으로 결과를 다음처럼 구분한다.

`SFT`: supervised fine-tuning only

`Static-vLLM ECHO-style`: rollout policy가 고정된 상태에서 terminal traces로 학습한 실험

`Online ECHO RLVR`: 최신 LoRA가 vLLM rollout에 주기적으로 반영되는 실험

README 순위표에는 이 구분을 반드시 남긴다.

## 10. 다음 액션

1. vLLM 4개 replica가 모두 ready인지 확인
2. trainer 시작 직후 `step_000000` LoRA sync 성공 확인
3. 5 step 또는 10 step마다 LoRA sync
4. GPU 6으로 checkpoint 평가
5. static-vLLM 결과와 online 결과를 분리해서 문서화
6. Hugging Face에는 adapter, rollout traces, eval results를 분리 업로드

HF 업로드 정책:

- adapter checkpoint는 새 model repo `LLM-OS-Models/LFM2.5-8B-A1B-SFT1-Online-ECHO-RLVR-GRPO-Adapters`에 올린다.
- rollout/eval 결과는 dataset repo `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts` 안에서 `online-sft1/<run_id>` 경로로 분리한다.
- token은 `.env`에서만 읽고 로그/README에는 출력하지 않는다.
- rollout sync는 `logs/train_online.log`와 `run_env.sh`를 읽어 train metrics와 redacted config를 함께 업로드한다.

## 11. 실제 시작 상태

기준 시각: 2026-06-13 10:59 KST

GPU 배치:

- GPU 0: vLLM replica, port 8123, 약 130GB VRAM 점유
- GPU 1: vLLM replica, port 8124, 약 130GB VRAM 점유
- GPU 2: vLLM replica, port 8125, 약 130GB VRAM 점유
- GPU 3: vLLM replica, port 8126, 약 130GB VRAM 점유
- GPU 4: trainer rank 0, 약 21GB VRAM 점유
- GPU 5: trainer rank 1, 약 21GB VRAM 점유
- GPU 6: checkpoint 평가 watcher 대기
- GPU 7: 미사용

데이터:

- 파일: `/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl`
- 총 1,500 rows
- `endless_terminals`: 772 rows, 51.47%
- `openthoughts_agent_v1_rl`: 728 rows, 48.53%

주요 학습 설정:

- `prompts_per_rank=1`
- `num_generations=4`
- `global_rollouts_per_step=8`
- `rollout_workers=8`
- `max_turns=4`
- `max_new_tokens=256`
- `command_timeout=8`
- `verifier_timeout=40`
- `world_model_coeff=0.05`
- `learning_rate=5e-7`
- `warmup_steps=20`
- `max_grad_norm=0.2`
- `LoRA rank=32`, `alpha=64`

초기 로그:

- `step_000000` LoRA sync: vLLM 4개 replica 모두 `load_status=200`
- step 0: reward_mean `-0.1313`, verifier_reward_mean `0.0`, world_loss_mean `2.3551`
- step 1: reward_mean `-0.2000`, verifier_reward_mean `0.0`, world_loss_mean `2.6742`
- step 2: reward_mean `0.4238`, verifier_reward_mean `0.5`, world_loss_mean `2.3092`
- step 3: reward_mean `-0.0375`, verifier_reward_mean `0.0`, world_loss_mean `2.2382`
- 2026-06-13 10:59 KST 기준 latest step: 4

해석:

아직 checkpoint가 없어 점수 판단은 불가능하다. 다만 step 2에서 verifier reward가 0을 벗어났고, vLLM LoRA sync가 이미 2회 기록되어 online path 자체는 작동한다. 첫 의미 있는 TB2-lite 비교는 checkpoint-25 평가가 끝난 뒤 가능하다.

2026-06-13 11:07 KST 추가 확인:

- latest step: 10
- LoRA sync 횟수: 3회 (`step_000000`, `step_000005`, `step_000010`)
- `step_000005`, `step_000010` 모두 vLLM 4개 replica에서 unload/load `200`
- rollout HF sync: `trace_rows_total=88`, `train_steps_logged=11`
- adapter HF sync: checkpoint가 아직 없어서 uploaded checkpoint 0개가 정상
- eval HF sync: 평가 결과가 아직 없어서 result 0개가 정상

즉, 현재는 online 학습/동기화/HF 기록 파이프라인이 모두 살아 있다. 성능 판단은 checkpoint-25 평가 이후 가능하다.

## 12. 예상 완료 시간

초기 5 step은 vLLM warmup, LoRA sync, terminal execution variance가 섞여 있어 step time 추정이 흔들린다. 보수적으로 보면 2,000 steps 전체는 24~36시간 범위로 잡는다. 따라서 현재 run의 hard stop은 2026-06-14 22:55 KST 근처다.

실제 운영에서는 다음 기준을 쓴다.

- checkpoint-25: 첫 sanity 평가
- checkpoint-100: online sync가 점수에 영향을 주는지 1차 판단
- checkpoint-250: SFT1 baseline 대비 상승/하락 방향 확인
- checkpoint-500: ECHO 논문 기본 실험 step 수와 비교 가능한 중간 지점
- checkpoint-1000 이후: 장기 RLVR에서 collapse 또는 late improvement 확인

RL은 길게 돌린다고 항상 좋아지지 않는다. 특히 SFT가 이미 강한 모델에서는 checkpoint별 평가를 보고 최고점을 고르는 방식이 안전하다.

2026-06-13 11:07 KST 기준 속도는 warmup 포함 약 60~80초/step 범위다. 이 속도가 유지되면 checkpoint-25는 대략 2026-06-13 11:25~11:35 KST 사이에 생성될 가능성이 높다. GPU6 watcher는 checkpoint가 생기면 자동 평가한다.

## 13. 참고 논문

- ECHO: Terminal Agents Learn World Models for Free: https://arxiv.org/abs/2605.24517
- A Primer in Post-Training Reasoning Data: What We Know About How It Works: https://arxiv.org/abs/2606.02113

ECHO 논문의 핵심은 terminal stdout/stderr/log/file feedback을 단순 context로만 쓰지 않고, 같은 rollout forward pass에서 environment-token CE loss로도 학습한다는 점이다. 논문 초록 기준 Qwen3-8B는 TerminalBench-2.0 pass@1이 2.70%에서 5.17%로, Qwen3-14B는 5.17%에서 10.79%로 올랐다.

Post-Training Reasoning Data primer는 150개 이상의 공개 연구와 시스템 리포트를 묶어 post-training reasoning data를 `무슨 데이터 객체인가`, `왜 유용한가`, `어떻게 만들었는가`, `어떻게 scaling하는가`로 해석하자는 프레임을 제공한다. 이번 실험 문서에서는 SFT/RLVR 효과를 단순 알고리즘 승패가 아니라 데이터 객체, verifier, trajectory, terminal feedback 품질의 문제로 해석한다.
