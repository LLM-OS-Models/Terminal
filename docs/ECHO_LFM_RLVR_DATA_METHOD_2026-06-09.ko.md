# ECHO 계열 LFM Terminal RLVR 데이터/방법 정리

작성일: 2026-06-09

## 현재 결론

우리가 하려는 작업은 ECHO 논문의 Docker/Harbor 전체 환경을 그대로 복제하는 것이 아니라, 그 논문의 핵심 학습 방법론을 현재 LFM no-Docker 터미널 RLVR 환경에 이식하는 것이다.

핵심은 두 가지다.

1. 모델이 터미널 명령을 내린다.
2. 실제 터미널 출력(stdout/stderr/exit code/verifier result)을 다음 turn의 context로만 쓰지 않고, observation token 위치에 cross-entropy loss를 걸어 world-model 신호로 학습한다.

현재 `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`는 이 구조를 이미 구현한다.

- action token: verifier reward 기반 policy loss
- observation token: terminal output prediction CE loss
- 최종 loss: `policy_coeff * policy_loss + world_model_coeff * world_loss`
- 현재 장기 run의 `WORLD_MODEL_COEFF=0.03`

즉 "터미널 피드백을 loss에 넣는가?"에 대한 답은 "그렇다"이다.

## ECHO 논문 데이터와 우리 데이터의 차이

논문 원문 기준 학습 corpus:

- 총 8870 terminal tasks
- 1977개: Endless Terminals
- 723개: OpenThoughts-Agent-v1-RL
- 6170개: 수정된 Endless Terminals pipeline으로 생성한 추가 Harbor-format tasks
- 학습/검증 split: train 8770, val 100
- 실행 환경: Docker + Harbor
- episode: 최대 16 turns
- context: 16k
- turn당 최대 2048 generated tokens
- 학습: 500 GRPO steps, 8x B200

우리 환경에서 바로 확보 가능한 공개/로컬 데이터:

- `open-thoughts/OpenThoughts-Agent-v1-RL`
- `obiwan96/endless-terminals`
- `open-thoughts/OpenThoughts-TB-dev`
- `open-thoughts/OpenThoughts-TBLite`
- TB2는 최종 평가용으로 두고, 학습에는 조심해서 사용한다.

중요한 한계:

- 논문이 추가 생성한 6170개 Harbor export는 현재 공개 repo에서 바로 확인되지 않았다.
- 따라서 우리는 공개 데이터와 로컬에 있는 task archive를 ECHO schema로 맞춰 준비한다.
- 데이터 포맷과 학습 objective는 ECHO식으로 맞추되, Docker/Harbor 대신 로컬 sandbox 실행으로 대체한다.

## 현재 실행 중인 장기 run 상태

Run ID:

`run_20260609T000050Z_patched_sandbox_resume820_vllm4_train2_setsid`

Run directory:

`/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260609T000050Z_patched_sandbox_resume820_vllm4_train2_setsid`

Model:

`LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`

GPU 배치:

- GPU 0-3: vLLM replicas
- GPU 4-5: LoRA RLVR training
- GPU 6-7: 사용하지 않음

현재 run이 로드한 데이터:

- 총 1408 rows
- OpenThoughts RL: 728
- Endless Terminals: 512
- OpenThoughts TB-dev: 70
- OpenThoughts TBLite: 98
- skip: 2개

현재 run은 ECHO식 world loss를 쓰지만, 데이터 입력이 HF loader 중심이라 "준비된 ECHO manifest"와 완전히 고정 연결되지는 않았다. 그래서 아래 데이터 준비 스크립트와 `--prepared-jsonl` 옵션을 추가했다.

## 새로 추가한 데이터 준비 스크립트

다운로드 스크립트:

`Liquid-CLI/scripts/download_echo_public_terminal_data.py`

역할:

- OpenThoughts parquet가 없으면 다운로드한다.
- Endless Terminals task 파일들을 이어받는다.
- Hugging Face 429 rate limit이 나오면 죽지 않고 대기 후 재시도한다.
- `.env`의 HF token을 읽지만 출력하지 않는다.

background 실행:

```bash
mkdir -p /home/work/.data/echo_terminal_data/logs
nohup python Liquid-CLI/scripts/download_echo_public_terminal_data.py \
  --env-file .env \
  --retry-seconds 180 \
  --max-attempts 1000 \
  >/home/work/.data/echo_terminal_data/logs/download_echo_public_terminal_data.log 2>&1 &
```

로그:

`/home/work/.data/echo_terminal_data/logs/download_echo_public_terminal_data.log`

준비 스크립트:

파일:

`Liquid-CLI/scripts/prepare_echo_terminal_data.py`

역할:

1. OpenThoughts parquet의 `task_binary`를 그대로 보존한다.
2. Endless task directory를 tar.gz `task_binary`로 재포장한다.
3. ECHO/SkyRL 호환 parquet를 만든다.
4. 우리 LFM no-Docker trainer가 바로 먹는 JSONL도 만든다.
5. source별 row 수와 비율을 manifest에 저장한다.

생성 명령:

```bash
set -a
source .env >/dev/null 2>&1 || true
set +a

python Liquid-CLI/scripts/prepare_echo_terminal_data.py \
  --tokenizer LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
  --max-prompt-tokens 4096
```

현재 생성 결과:

- 총 1130 rows
- OpenThoughts-Agent-v1-RL: 728 rows, 64.4248%
- Endless Terminals: 402 rows, 35.5752%
- skipped: 0

Endless downloader가 Hugging Face 429 rate limit으로 중간 대기 중이라, 다운로드가 더 진행되면 같은 명령을 다시 돌려 row 수를 늘리면 된다.

## 생성된 산출물

Output directory:

`/home/work/.data/echo_terminal_data/prepared`

Files:

- `echo_terminal_tasks_mixed.parquet`
- `echo_terminal_tasks_openthoughts.parquet`
- `echo_terminal_tasks_endless.parquet`
- `lfm_live_tasks_mixed.jsonl`
- `lfm_live_tasks_mixed.parquet`
- `solution_references_mixed.jsonl`
- `manifest.json`

Schema:

ECHO/SkyRL parquet:

- `prompt`
- `path`
- `task_binary`
- `instruction`
- `source`
- `_data_source`
- `prompt_tokens`

LFM live JSONL:

- `prompt`
- `task_id`
- `source`
- `task_dir`
- `task_binary_b64`
- `instruction`
- `echo_path`
- `prompt_tokens`

`solution_references_mixed.jsonl`은 task archive 안의 `solution/solve.sh`를 따로 뽑은 것이다. 이것은 on-policy RL trajectory가 아니므로 RL reward 데이터처럼 쓰면 안 된다. 나중에 SFT, error analysis, verifier sanity check에 쓰는 것이 맞다.

## Trainer 패치

수정 파일:

- `Liquid-CLI/train_lfm_terminal_live_rlvr_grpo.py`
- `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`
- `Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh`

추가 옵션:

`--prepared-jsonl /path/to/lfm_live_tasks_mixed.jsonl`

wrapper 환경 변수:

- `PREPARED_JSONL=/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl`
- `PREPARED_ONLY=1`이면 HF 자동 loader를 끄고 prepared JSONL만 사용
- `PREPARED_ONLY=0`이면 prepared JSONL과 기존 HF loader를 함께 사용

검증한 dry-run:

```bash
PREPARED_ONLY=1 \
PREPARED_JSONL=/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl \
TRAIN_GPUS= \
MAX_STEPS=1 \
Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh \
  --max-rows 16 \
  --dry-run
```

결과:

- prepared JSONL 로드 성공
- 16 rows 로드
- prompt token range: 485-539

## Docker 없는 RL의 한계

Docker/Harbor가 없는 것이 데이터 준비의 장애물은 아니다. 하지만 실행 안정성에는 영향을 준다.

한계:

1. task 간 격리가 Docker보다 약하다.
2. Dockerfile에 설치 의존성이 숨어 있는 task는 로컬 실행에서 실패할 수 있다.
3. `/workspace`, `/output`, `/logs` path rewrite가 완벽하지 않은 task가 있을 수 있다.
4. 모델이 위험 명령을 내렸을 때 Docker보다 방어선이 약하므로 unsafe command filter가 중요하다.
5. terminal task가 system package, network, service manager, background process에 의존하면 verifier가 흔들릴 수 있다.

현재 대응:

- task archive를 per-rollout local sandbox에 푼다.
- `/workspace`, `/output`, `/logs`를 sandbox 내부로 rewrite한다.
- `sudo`, `tmux`, `screen`, `nohup`, `setsid`, `pkill`, `killall`, root filesystem 탐색 등은 차단한다.
- GPU probe 명령도 task sandbox 안에서는 금지한다.

근본 해결:

- ZeroBox/OpenSandbox 같은 process sandbox를 붙여서 로컬 격리를 강화한다.
- 가능하면 Docker/Harbor가 되는 별도 노드에서는 paper-style harness를 그대로 돌린다.
- 현재 노드에서는 no-Docker 로컬 sandbox + ECHO loss로 빠른 실험을 반복한다.

## 다음 장기 run 권장 설정

현재 run을 계속 살리면서, 다음 새 run에서는 prepared data를 명시적으로 넣는 것이 낫다.

권장:

```bash
PREPARED_JSONL=/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl \
PREPARED_ONLY=0 \
VLLM_BASE_URL=http://127.0.0.1:8123/v1,http://127.0.0.1:8124/v1,http://127.0.0.1:8125/v1,http://127.0.0.1:8126/v1 \
TRAIN_GPUS=4,5 \
NPROC_PER_NODE=2 \
MAX_STEPS=100000 \
MAX_WALL_TIME_HOURS=47.5 \
WORLD_MODEL_COEFF=0.03 \
SAVE_STEPS=10 \
Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh
```

`PREPARED_ONLY=0`을 추천하는 이유:

- prepared JSONL은 source manifest가 명확하다.
- 기존 HF loader는 TB-dev/TBLite 등 추가 데이터를 계속 섞어준다.
- 둘을 같이 쓰면 데이터 다양성이 늘지만 중복 가능성이 있다.

`PREPARED_ONLY=1`을 추천하는 경우:

- 논문 공개 원천 데이터만으로 ablation을 하고 싶을 때
- 데이터 source별 효과를 분리해서 보고 싶을 때

## 왜 이전 결과가 애매했나

이전 100 epoch/짧은 run 결과가 크게 튀지 않은 이유는 다음 가능성이 크다.

1. 터미널 task reward가 매우 sparse하다.
2. 성공 rollouts 비율이 낮으면 GRPO contrast가 약하다.
3. Docker 없는 verifier 실패가 reward noise를 만든다.
4. 짧은 run은 ECHO world loss가 terminal dynamics를 충분히 안정화하기 전에 끝난다.
5. prepared manifest 없이 HF loader를 섞으면 정확히 어떤 source 조합이 먹혔는지 추적이 약하다.

그래도 일부 step에서 verifier reward가 올라간 이유:

- observation CE가 stdout/stderr/exit code 패턴을 계속 학습한다.
- 실패한 trajectory도 world loss에는 신호를 준다.
- OpenThoughts/Endless의 task structure가 현재 모델의 SFT 분포와 잘 맞는다.

## 판단 기준

좋은 checkpoint는 단순 train reward가 아니라 아래를 같이 봐야 한다.

- TB2 final evaluation
- TBLite/TB-dev heldout evaluation
- verifier reward moving average
- world_loss_mean 하락 또는 안정화
- parse error 감소
- timeout 감소
- 평균 command count와 token usage 감소

ECHO 논문도 핵심은 "최종 pass rate"와 "terminal output prediction CE가 실제로 내려가는가"를 함께 본다.
