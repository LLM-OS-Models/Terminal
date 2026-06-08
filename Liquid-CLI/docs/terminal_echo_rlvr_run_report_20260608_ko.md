# Terminal ECHO RLVR 실행 보고서 - 2026-06-08

## 2026-06-08 02:20 UTC 업데이트

최신 목표 배치는 사용 가능한 GPU 0-5만 쓰는 구성이다.

- GPU 0,1,2,3: vLLM TP1 replica 4개
- GPU 4,5: LoRA DDP 학습 rank 2개
- GPU 6,7: 사용하지 않음. 다른 작업용이므로 건드리지 않는다.

현재 활성 실행:

- run id: `run_20260608T023253Z_resume_ckpt50_vllm4rep_train2_total6_rw8_vram092_gpu_filter_false_smi_setsid`
- run dir: `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260608T023253Z_resume_ckpt50_vllm4rep_train2_total6_rw8_vram092_gpu_filter_false_smi_setsid`
- output dir: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260608T023253Z_resume_ckpt50_vllm4rep_train2_total6_rw8_vram092_gpu_filter_false_smi_setsid`
- resume adapter: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260607T223408Z_resume_ckpt50_vllm4rep_train4_devnullfix/checkpoint-50`

이 run은 이전 checkpoint-50 adapter에서 이어서 간다. `MAX_WALL_TIME_HOURS=47.5`,
`SAVE_STEPS=50`이라 정상 진행되면 50 step마다 adapter checkpoint가 저장된다.

중요한 해석:

- 4,5번 GPU가 약 19GB만 쓰는 것은 정상이다. 현재 학습은 full fine-tuning이
  아니라 LoRA adapter 학습이고, `CUDA_VISIBLE_DEVICES=4,5` 때문에 trainer
  내부 로그에는 `cuda:0`, `cuda:1`처럼 보인다. 물리적으로는 GPU 4,5다.
- 0-3번 GPU는 vLLM이 올라가면 각 replica가 KV cache까지 잡기 때문에 약
  133GB 수준까지 VRAM을 쓴다. 따라서 0-3과 4-5의 VRAM 사용량이 크게 다른
  것은 역할 차이다.
- vLLM은 필수다. rollout 생성은 HTTP `/v1/completions`로 vLLM replica에
  보내고, trainer는 실제 터미널 실행 결과를 받아 RLVR/ECHO loss를 계산한다.

이번에 확인한 장애와 원인:

1. `run_20260608T020459Z_resume_ckpt50_vllm4rep_train2_total6_rw8_vram092_gpu_filter_killpg`
   에서는 모델과 adapter가 GPU 4,5에 정상 로드됐고, vLLM 0-3도 요청을 받았다.
   그러나 rank1의 일부 rollout에서 모델 생성 명령이 `nvidia-smi -q -d COMPUTE`
   형태의 GPU 탐색 루프를 만들었다. no-Docker 샌드박스의 `nvidia-smi` wrapper가
   이를 막기는 했지만, wrapper bash process가 수천 개 쌓이면서 rollout이
   사실상 멈췄다.
2. 원인은 학습 GPU가 안 잡힌 것이 아니라, Docker가 없는 환경에서 태스크/모델
   생성 명령이 host GPU probing을 반복할 수 있다는 점이다. 컨테이너라면
   격리로 흡수될 문제가 로컬 workspace에서는 직접 차단 로직이 필요하다.
3. 처음에는 이를 막기 위해 `nvidia-smi` wrapper를 `SIGTERM`에서 `SIGKILL`
   기반 process-group 종료로 강화했다. 그러나 모델이 `nvidia-smi`를 반복
   호출하면 wrapper shell 자체가 대량 생성되는 문제가 남았다. 그래서 최종
   방어는 bash wrapper가 아니라 `nvidia-smi -> /bin/false` symlink로 바꿨다.
   반복 호출이 생겨도 즉시 실패하고 wrapper bash process가 누적되지 않게 하기
   위함이다. 또한 GPU 의존 패턴 필터를 dataset/verifier/command 쪽에 유지한다.
4. `run_20260608T021136Z...sigkill`, `run_20260608T021537Z...stagger`,
   `run_20260608T021947Z...stagger2`에서는
   vLLM launcher가 실제 API process를 안정적으로 유지하지 못했다. 수동
   `python -m vllm.entrypoints.openai.api_server` probe는 정상으로 떠서 모델
   호환성이나 vLLM 자체 문제는 아니었다. 그래서 vLLM replica launcher를
   "4개 동시 기동"에서 "GPU별 순차 기동, `/models` readiness 확인 후 다음
   replica 시작"으로 바꿨다. 또한 orchestrator 자체를 `setsid`로 별도 session에
   올렸을 때 실제 run이 유지됐다. 따라서 현재 운영 방식은 `setsid bash launch.sh`
   형태가 더 안전하다.

현재 잘 되는 것:

- LFM2.5-8B-A1B 모델은 vLLM 0.19.1에서 `--trust-remote-code`,
  `--dtype bfloat16`, `--max-model-len 32768`, `--enforce-eager`로 정상 기동한다.
- 수동 vLLM probe 기준 GPU0에서 약 35초 안에 `/v1/models`가 200 OK를 반환했다.
- 현재 활성 run은 vLLM 4개가 모두 ready가 됐고, GPU 0-3이 각각 약 133GB
  VRAM을 사용한다.
- 현재 활성 run은 train launcher가 붙었고, `optimizer_ready`까지 도달했다.
  GPU 4-5는 LoRA DDP 학습 상태에서 약 19.7GB / 19.1GB를 사용한다.
- false symlink 패치 이후 활성 run의 `nvidia-smi` wrapper process count는 0이다.
- 활성 run의 vLLM replica 4개가 모두 `/v1/completions` POST를 받고 있다.
- 활성 run의 첫 train step이 완료됐다.
  - step: 0
  - loss: -0.00785
  - reward_mean: -0.046875
  - verifier_reward_mean: 0.0
  - world_loss_mean: 1.47626
  - action_tokens_mean: 1060.5
  - obs_tokens_mean: 611.25
  - rollout traces: rank0 8개, rank1 8개
  - 해석: 보상은 아직 낮지만, 실제 터미널 observation token이 ECHO-style
    world-model loss에 들어가고 있다.
- 모델 로딩 후 vLLM 한 replica는 약 15.8GiB weight와 약 110GiB KV cache를
  잡아 총 약 133GB 수준의 VRAM을 사용한다.
- trainer는 checkpoint-50 LoRA adapter를 로드하고 DDP 2-rank로 감쌀 수 있다.
- ECHO-style observation CE는 코드 경로에 들어가 있다. terminal stdout,
  stderr, exit code, verifier result가 trajectory observation으로 들어가고
  `WORLD_MODEL_COEFF=0.03`으로 loss에 반영된다.

아직 고쳐야 하는 것:

- no-Docker 샌드박스에서 GPU probing, background loop, process leak을 더
  강하게 차단해야 한다. 현재는 `nvidia-smi` wrapper와 command text filter로
  막지만, 모델이 shell script를 생성해 우회할 수 있다.
- vLLM rollout server는 현재 학습 중인 LoRA weight를 step마다 동기화하지 않는다.
  즉, rollout policy는 served base/SFT 모델이고, trainer는 LoRA를 업데이트한다.
  이는 엄밀한 SkyRL식 fully on-policy ECHO 구현과 다르다.
- TB2 최종 평가는 proxy eval이 아니라 별도 최종 평가로 남겨야 한다. TB2는 최종용,
  학습 중에는 OpenThoughts RL, Endless Terminals, TB-dev, TBLite train mix를 쓴다.
- 1,408개 local usable row는 논문 규모보다 작다. TerminalBench 1/3 train split을
  안정적으로 더 붙이려면 데이터 변환, verifier 정규화, no-Docker 안전 필터를
  추가해야 한다.

근본적으로 어려운 점:

- Docker 없이 "터미널 RL"을 하면 모델이 만든 명령이 host에서 돈다. 우리가
  workspace root를 분리해도 완전한 syscall/filesystem/network/GPU isolation은
  아니다. 그래서 정상 명령을 너무 많이 막으면 reward가 깨지고, 너무 적게 막으면
  host 자원을 건드린다.
- ECHO 논문의 핵심은 터미널 출력까지 loss로 먹이는 것이지만, 안정적인 장기 RL은
  환경 격리, verifier 품질, rollout server weight sync, checkpoint/eval loop가
  같이 맞아야 한다. 지금은 그중 "실제 터미널 실행 + observation CE + vLLM rollout"
  경로를 로컬 no-Docker 방식으로 구축하는 단계다.
- 그래서 2일 이상 길게 돌리는 것은 가능하지만, 무작정 오래 돌리는 것만으로
  성능이 오른다고 볼 수 없다. reward가 깨끗하고, task mix가 GPU/host 의존성을
  피하고, vLLM throughput이 안정적이어야 "아하 모먼트"가 나온다.

이번 코드 변경:

- `Liquid-CLI/train_lfm_terminal_live_rlvr_grpo.py`
  - system prompt에 GPU/CUDA/NVIDIA probing 금지 문구 추가
  - no-Docker `nvidia-smi` 방어를 bash wrapper에서 `/bin/false` symlink로 변경
  - `nvidia-smi`, `nvcc`, `nvidia-debugdump`, `nvidia-cuda-mps` 등 GPU 의존
    command/task/verifier 필터 유지
- `Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh`
  - vLLM replica를 한 번에 4개 띄우지 않고, 각 replica가 `/models` readiness를
    통과한 뒤 다음 GPU replica를 띄우도록 변경
  - 실패 시 해당 replica log tail을 stderr에 남기고 즉시 실패하도록 변경

운영 원칙:

- 0-3은 vLLM, 4-5는 training, 6-7은 사용 금지.
- `.env` 토큰은 문서나 로그에 쓰지 않는다. 공용 장비이므로 모델 카드/Hub 업로드
  때도 토큰 값을 출력하지 않는다.
- 현재 run이 안정화되어 checkpoint가 생기면 그 checkpoint와 평가 결과를 별도
  섹션으로 추가한다.

이 문서는 지금 우리가 실제로 무엇을 돌리고 있는지, 어떤 데이터를
쓰는지, 이전 100-step 결과가 왜 약했는지, checkpoint-50 근처에서 왜
멈췄는지, 그리고 현재 vLLM replica 구성이 무엇을 바꿨는지 정리한다.

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

`/dev/null` 패치 전 resumed run의 첫 step은 observation token이 실제 loss에
들어가고 있음을 확인해줬다.

- step: 0
- reward_mean: 0.05625
- verifier_reward_mean: 0.125
- world_loss_mean: 1.00597
- action_tokens_mean: 795.5
- obs_tokens_mean: 296.0

## 현재 활성 run

Run:

- `run_20260607T223408Z_resume_ckpt50_vllm4rep_train4_devnullfix`

Run directory:

- `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260607T223408Z_resume_ckpt50_vllm4rep_train4_devnullfix`

Output directory:

- `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260607T223408Z_resume_ckpt50_vllm4rep_train4_devnullfix`

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
- `/dev/null` safety patch active

현재 상태:

- 2026-06-08 01:20 UTC 기준 `step 48`까지 완료했다.
- save interval이 50이므로 `step 49`가 끝나면 `checkpoint-50`이 저장된다.
- checkpoint 저장 직후 자동 전환 watcher가 기존 4-GPU 학습과 4-replica
  vLLM을 정리하고 새 run을 띄운다.

현재 재시작된 run:

- `run_20260608T012844Z_resume_devnullfix_ckpt50_vllm4rep_train2_total6`

GPU 배치:

- vLLM: GPU 0,1,2,3에 TP1 replica 4개
- training: GPU 4,5에 DDP rank 2개
- idle/reserved: GPU 6,7
- prompts per rank: 2
- generations per prompt: 4
- global rollouts per step: 16
- resume adapter: 현재 run의 `checkpoint-50`

첫 step 확인:

- `step 0` 완료
- `reward_mean`: 0.04375
- `verifier_reward_mean`: 0.0625
- `world_loss_mean`: 1.18109
- `action_tokens_mean`: 1,586.625
- `obs_tokens_mean`: 1,049.625

이 값은 현재 run이 단순 GRPO reward만 쓰는 것이 아니라, 실제 터미널 실행
결과(stdout/stderr/exit code에서 만든 observation token)를 ECHO-style
world-model CE loss로 같이 학습하고 있음을 보여준다.

왜 4-vLLM + 2-train, 총 6대인가:

- 이 8B 모델은 H200 한 장에 vLLM replica 하나를 올릴 수 있다.
- rollout 요청은 서로 독립적인 HTTP request다. 그래서 하나의 TP4 서버보다
  여러 개의 TP1 replica가 request throughput을 더 잘 뽑을 가능성이 높다.
- 현재 4-GPU 학습 rank는 VRAM을 약 39-43GB만 쓴다. 병목은 training VRAM이
  아니라 rollout 생성, 터미널 실행, verifier, HTTP 왕복 시간이다.
- 따라서 training rank는 2개면 충분하고, vLLM은 4개 replica를 유지한다.
- 사용 GPU는 총 6대로 제한한다. GPU 6,7은 다음 평가/비교 실험 또는 장애 대응을
  위해 비워 둔다.
- `PROMPTS_PER_RANK=2`로 올려 global rollout 수는 기존 16개를 유지한다.
- tensor parallel size는 단일 모델 shard 내부에서는 2, 4, 8 같은 단위가
  중요하지만, 이 workload에서는 rollout replica 병렬성이 더 중요하다.
- 방금 trainer의 vLLM URL routing도 `rank % num_urls`에서
  `(seed + rank) % num_urls`로 바꿨다. 그래서 2-rank 학습에서도 4개 vLLM
  replica가 seed별 rollout에 분산된다.

예상 시간:

- checkpoint-50 저장: `step 48` 기준 몇 분 내
- 4개 vLLM replica 재기동: 약 1분 내외로 완료됨
- 새 2-GPU 학습 첫 train step: 완료됨
- 이후 checkpoint는 새 run 기준 50 step마다 저장된다.
- 이전 속도는 load 시간을 포함해 약 3.4분/step이었다. 4-vLLM + 2-train은
  첫 10 step 실측 후 다시 계산해야 하지만, 목표는 같은 16 rollout을 더 높은
  vLLM throughput과 낮은 학습 GPU 수로 처리하는 것이다.

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
- no-Docker task/verifier가 host GPU를 직접 건드릴 수 있다. 실제로 한
  TerminalBench-lite task의 test script가 `CUDA_VISIBLE_DEVICES=7 ... --device
  cuda:0` 형태로 idle GPU를 잡으려는 것을 확인했다. 이를 막기 위해 sandbox
  subprocess 환경에서 CUDA/NVIDIA visible-device 변수를 지우고, `python`,
  `python3`, `pytest`, `pip`, `nvidia-smi`에 no-GPU wrapper를 prepend한다.
  또한 모델 command에 CUDA/NVIDIA/GPU 사용 패턴이 나오면 unsafe로 차단한다.
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

이 수치는 `/dev/null` 패치 전 4-replica run
`run_20260607T222154Z_resume_ckpt50_vllm4rep_train4`에서 나온 값이다. 해당
run은 safety patch를 적용하기 위해 step 1에서 중단했다. 현재 활성
`devnullfix` run은 최소 checkpoint-50까지 간 뒤 판단해야 한다.

## 다음 단계

1. 현재 4-replica `devnullfix` resumed run을 최소 checkpoint-50까지 계속
   돌린다.
2. 현재 run의 checkpoint-50을 TB2-lite로 평가한다.
3. blocked rate가 계속 높으면 `/dev/null` safety patch가 활성화된 상태로
   재시작하고, 상대경로 safe command 허용 범위를 더 조정한다.
4. 의미 있게 내부 baseline을 넘기기 전에는 Hugging Face에 업로드하지 않는다.
5. TB2-lite는 빠른 gate로 쓰고, TerminalBench-2.0은 최종 gate로 쓴다.
6. 논문에 더 가까운 재현을 원하면 LoRA weight sync를 vLLM에 구현하거나
   원본 SkyRL/Harbor 경로로 이동해야 한다.
