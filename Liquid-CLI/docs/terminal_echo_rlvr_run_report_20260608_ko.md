# Terminal ECHO RLVR 실행 보고서 - 2026-06-08

이 문서는 지금 우리가 실제로 무엇을 돌리고 있는지, 어떤 데이터를
쓰는지, 이전 100-step 결과가 왜 약했는지, checkpoint-50 근처에서 왜
멈췄는지, 그리고 현재 4-vLLM-replica 구성이 무엇을 바꿨는지 정리한다.

## 목표

`LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch` 모델에
ECHO 스타일의 live terminal RLVR을 적용한다.

목표 동작은 다음과 같다.

1. 터미널 과제를 샘플링한다.
2. vLLM으로 shell/tool action을 생성한다.
3. 생성된 명령을 rollout별 로컬 작업 공간에서 실제로 실행한다.
4. stdout, stderr, exit code, verifier 결과를 trajectory에 다시 넣는다.
5. verifier reward와 ECHO 스타일 terminal observation loss를 함께 사용해
   LoRA adapter를 학습한다.

즉, 단순한 offline next-command imitation이 아니다. 실제 터미널 명령을
실행하고, 그 피드백을 학습 trajectory 안에 넣는다.

## 현재 코드 경로

메인 학습 코드:

- `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`

학습 런처:

- `Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh`

vLLM replica 런처:

- `Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh`

안전 필터 helper:

- `Liquid-CLI/train_lfm_terminal_live_rlvr_grpo.py`

이번에 반영한 핵심 변경은 다음과 같다.

- vLLM HTTP rollout이 comma-separated base URL을 받을 수 있게 했다.
  그래서 DDP rank별로 서로 다른 vLLM 서버에 요청을 보낸다.
- `SFT_ADAPTER_PATH`로 기존 LoRA adapter에서 재개할 수 있게 했다.
- `GRADIENT_CHECKPOINTING=1`일 때 실제로 gradient checkpointing이 켜지도록
  런처 버그를 고쳤다.
- vLLM URL이 여러 개일 때 health check는 첫 번째 URL로 수행하게 했다.
- no-Docker 안전 필터에서 `/dev/null`을 허용했다. 정상적인 터미널 명령이
  `2>/dev/null`을 자주 쓰는데, 이걸 막으면 `find`, `grep`, `ls` 같은 정상
  flow가 `unsafe_pattern`으로 잘못 막힌다.

## 학습 데이터

현재 live dataset builder는 코드에 연결된 모든 터미널 RL 소스를 읽는다.

- `open-thoughts/OpenThoughts-Agent-v1-RL`: 728 rows
- `endless-terminals`: 512 rows
- `open-thoughts/OpenThoughts-TB-dev`: 70 rows
- `open-thoughts/OpenThoughts-TBLite`: 98 rows

현재 run에서 실제로 로드된 usable row는 총 1,408개다.

2개 row는 invalid 또는 too long으로 제외됐다. 현재 run 기준 prompt 길이는
interaction 전 221-3,957 tokens 범위다.

이 규모는 ECHO 논문 세팅보다 작다. 논문은 더 큰 terminal interaction mix와
SkyRL/Harbor/Docker 기반 원본 stack을 쓴다. 지금 경로는 로컬에서 돌리기
위한 no-Docker 재현/실험 경로이며, 원 논문 시스템을 byte-for-byte로
복제한 것은 아니다.

## 학습 목적 함수

loss는 두 부분으로 구성된다.

- verifier reward로 가중한 assistant action token의 RLVR policy loss
- `WORLD_MODEL_COEFF`로 조절되는 terminal observation token의 ECHO-style CE

현재 coefficient:

- `WORLD_MODEL_COEFF=0.03`

의미는 간단하다. 터미널 출력은 다음 행동을 위한 context일 뿐 아니라,
모델이 터미널의 작동 방식을 배우는 dense supervision으로도 사용된다.

현재 resumed run의 첫 step은 observation token이 실제 loss에 들어가고
있음을 확인해준다.

- step: 0
- reward_mean: 0.05625
- verifier_reward_mean: 0.125
- world_loss_mean: 1.00597
- action_tokens_mean: 795.5
- obs_tokens_mean: 296.0

## 현재 활성 run

Run:

- `run_20260607T222154Z_resume_ckpt50_vllm4rep_train4`

Run directory:

- `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260607T222154Z_resume_ckpt50_vllm4rep_train4`

Output directory:

- `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260607T222154Z_resume_ckpt50_vllm4rep_train4`

Resume adapter:

- `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260607T142109Z_long2d_wm003_g4_tp4_streambackward/checkpoint-50`

GPU 분배:

- GPU 0-3: 독립 vLLM replica 4개, 각 TP1
- GPU 4-7: DDP training rank 4개

vLLM URLs:

- `http://127.0.0.1:8123/v1`
- `http://127.0.0.1:8124/v1`
- `http://127.0.0.1:8125/v1`
- `http://127.0.0.1:8126/v1`

학습 config:

- DDP world size: 4
- prompts per rank: 1
- generations per prompt: 4
- global rollouts per step: 16
- max turns: 12
- max new tokens per turn: 768
- max sequence length: 16,384
- save interval: 50 steps마다
- learning rate: 5e-7
- warmup steps: 50
- max wall time: 47.5 hours
- no Docker

왜 4:4가 현재 맞는가:

- 이 8B 모델은 H200 한 장에 vLLM replica 하나를 올릴 수 있다.
- rollout 요청은 서로 독립적인 HTTP request다. 그래서 하나의 TP4 서버보다
  네 개의 TP1 replica가 request throughput을 더 잘 뽑을 가능성이 높다.
- 학습 쪽은 LoRA trainer가 DDP로 돌고 trajectory/world-model loss가
  메모리를 많이 쓰므로 GPU 4장이 필요하다.
- tensor parallel size는 단일 모델 shard 내부에서는 2, 4, 8 같은 단위가
  중요하지만, 이 workload에서는 rollout replica 병렬성이 더 중요하다.

## no-Docker RLVR의 한계

Docker가 없다고 RLVR이 불가능한 것은 아니다. 다만 Docker 기반
TerminalBench/SkyRL 환경과 품질이 같지는 않다.

되는 것:

- 명령은 실제로 실행된다.
- stdout, stderr, exit code, verifier output이 trajectory에 저장된다.
- observation text가 ECHO world-model loss에 들어간다.
- rollout별 작업 공간이 분리되어 대부분의 task 충돌은 막는다.

주요 한계:

- filesystem isolation이 Docker보다 약하다. 컨테이너가 흡수할 일을
  host-root command 차단 규칙으로 막아야 한다.
- 안전 필터가 정상 명령까지 잘못 막을 수 있다. 실제로 `2>/dev/null` 때문에
  정상적인 `find` flow가 `unsafe_pattern`으로 막힌 사례를 확인했다.
- 일부 TerminalBench 스타일 task는 package install, system-level state,
  clean container image를 전제로 한다. 단순 로컬 workspace는 이를 완전히
  재현하지 못한다.
- 환경 격리가 약하면 reward signal도 Docker 기반 환경보다 덜 깨끗해진다.

Docker와 별개로 더 큰 구현상 한계도 있다.

- 현재 vLLM HTTP 서버는 served base/SFT model로 rollout을 생성한다.
- DDP process는 LoRA adapter를 학습한다.
- 아직 학습 중인 LoRA weight를 매 step vLLM rollout server에 동기화하는
  SkyRL 스타일 weight sync가 없다.

따라서 지금 세팅은 실제 터미널 피드백과 observation-token CE가 들어가는
ECHO-style live terminal RLVR이다. 하지만 원 논문의 완전한 on-policy
ECHO/SkyRL 구현은 아니다.

## 이전 100-step static-vLLM 결과

이전 run:

- `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_7gpu_fileinit_100step`

TB2-lite replay evaluation:

| model | score | avg_command_f1 | first_cmd_exact_pct | steps |
| --- | ---: | ---: | ---: | ---: |
| base_sft | 51.22 | 0.5140 | 50.8 | 303 |
| checkpoint-25 | 50.43 | 0.5083 | 49.5 | 303 |
| checkpoint-50 | 51.50 | 0.5206 | 50.2 | 303 |
| checkpoint-75 | 50.55 | 0.5014 | 51.5 | 303 |
| checkpoint-100 | 51.03 | 0.5181 | 49.2 | 303 |
| final_lora | 51.52 | 0.5251 | 49.2 | 303 |

같은 base SFT 모델의 README leaderboard reference:

- 52.30

해석:

- 같은 run 조건의 base eval보다는 올랐다: 51.22 -> 51.52.
- 상승폭은 작다: +0.30 score.
- 기존 README reference 52.30은 넘지 못했다.
- command F1은 더 명확하게 올랐다: 0.5140 -> 0.5251.
- first command exact match는 떨어졌다: 50.8 -> 49.2.

100-step 결과가 강하지 않았던 이유:

- 100 step은 sparse terminal reward 기준으로 매우 짧은 RL run이다.
- rollout 쪽이 static-vLLM 스타일이라 매 step 최신 LoRA가 vLLM에 동기화되지
  않았다.
- no-Docker 안전 필터가 너무 강해서 정상 명령도 많이 막았다.
- 학습 데이터가 1,408개로 작고 여러 source가 섞여 있다.
- TB2-lite는 next-action replay proxy라 live multi-turn terminal solving과
  분포가 완전히 같지 않다.
- base model이 이미 강해서 작은 LoRA update는 command F1과 exact formatting
  사이에서 trade-off를 만들 수 있다.

그래도 의미 있는 이유:

- 같은 평가 stack에서 final LoRA가 base보다 소폭 상승했다.
- best short-run checkpoint/final 모델은 average command F1을 올렸다.
- vLLM rollout -> 실제 terminal execution -> verifier reward ->
  observation CE -> LoRA checkpoint -> TB2-lite eval이라는 end-to-end loop가
  작동함을 확인했다.

## 이전 long run과 checkpoint-50 근처에서 멈춘 이유

이전 long run:

- `run_20260607T142109Z_long2d_wm003_g4_tp4_streambackward`

Output:

- `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260607T142109Z_long2d_wm003_g4_tp4_streambackward`

이건 "50 epoch"가 아니다. 50 step checkpoint이고, 로그는 step 54까지
이어졌다.

확정 가능한 사실:

- `checkpoint-50`은 존재하고 유효한 LoRA adapter를 포함한다.
- 마지막으로 확인된 train log entry는 step 54다.
- `train.log` 끝에 Python traceback이 없다.
- `train.log` 끝에 명확한 CUDA OOM 라인이 없다.
- stop 이후 training rank들은 사라졌다.
- vLLM TP4 server process들은 GPU 0-3을 계속 점유하고 있었다.

가능성이 높은 원인:

- 확정된 model-training exception은 아니다.
- training process group 또는 launcher/session이 interrupt/cleanup 되었거나,
  DDP rank 중 하나가 traceback 없이 종료됐을 가능성이 높다.
- vLLM은 살아 있고 training rank만 사라졌기 때문에, vLLM startup 문제보다는
  process/session management 문제에 더 가깝게 보였다.

이후 세팅을 바꾼 이유:

- TP4는 GPU 0-3에 하나의 vLLM engine을 걸친다. 독립 rollout 요청이 많은
  상황에서는 병렬성을 충분히 못 쓸 수 있다.
- 네 개의 TP1 replica는 각 rank가 별도 rollout server를 쓰게 만들어 request
  contention을 줄인다.
- 새 run은 앞의 50 step을 버리지 않고 `checkpoint-50`에서 재개한다.

## trace 품질 관찰

이전 long run trace count:

- rank0: 220 rollouts, 138 blocked, 35 verifier successes
- rank1: 223 rollouts, 139 blocked, 20 verifier successes
- rank2: 224 rollouts, 139 blocked, 26 verifier successes
- rank3: 220 rollouts, 131 blocked, 19 verifier successes

합계:

- 887 rollout traces
- 547 blocked traces
- 100 verifier successes

blocked rate가 너무 높다. 이것이 학습 신호가 약했던 이유 중 하나다.
구체적으로 정상적인 `2>/dev/null` redirection이 unsafe로 막힌 사례가
있었다. 이 경로는 이제 safety helper에서 허용했지만, 이미 떠 있는 resumed
process에는 재시작 전까지 반영되지 않는다.

현재 resumed run 초기 trace count:

- 31 rollout traces
- 17 blocked traces
- 2 verifier successes

아직 모델 품질을 판단하기에는 너무 이르다. 이 수치는 live loop가 돌고 있고
terminal feedback이 기록되고 있음을 확인해주는 정도다.

## 다음 단계

1. 현재 4-replica resumed run을 최소 checkpoint-50까지 계속 돌린다.
2. 현재 run의 checkpoint-50을 TB2-lite로 평가한다.
3. blocked rate가 계속 높으면 `/dev/null` safety patch가 활성화된 상태로
   재시작하고, 상대경로 safe command 허용 범위를 더 조정한다.
4. 의미 있게 내부 baseline을 넘기기 전에는 Hugging Face에 업로드하지 않는다.
5. TB2-lite는 빠른 gate로 쓰고, TerminalBench-2.0은 최종 gate로 쓴다.
6. 논문에 더 가까운 재현을 원하면 LoRA weight sync를 vLLM에 구현하거나
   원본 SkyRL/Harbor 경로로 이동해야 한다.

