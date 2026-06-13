# LFM2.5 Online ECHO RLVR 실행 Runbook

작성 시각: 2026-06-13 KST

이 문서는 `LFM2.5-8B-A1B Terminal ToolBench Full SFT 1Epoch` 모델에 online ECHO-style GRPO/RLVR을 돌릴 때 필요한 실행 명령, 파일 위치, step time 조절 변수, 평가/HF 업로드 절차를 정리한다.

## 1. 현재 기준 run

- Run ID: `run_20260613T014646Z_sft1_online_vllm_lora_sync_g4_t4_sync5_wm005`
- Base model: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- Dataset: `/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl`
- Output dir: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__online_echo_grpo_vllm_lora_sync_run_20260613T014646Z_sft1_online_vllm_lora_sync_g4_t4_sync5_wm005`
- Run dir: `/home/work/.data/liquid_cli_sft/live_terminal_echo_online/run_20260613T014646Z_sft1_online_vllm_lora_sync_g4_t4_sync5_wm005`
- GPU layout: `0,1,2,3` vLLM rollout, `4,5` trainer, `6` evaluation watcher, `7` 미사용

주의:

- GPU 7은 이 run에서 쓰지 않는다.
- `.env`에 HF token이 있으므로 토큰을 출력하지 않는다.
- 현재 환경은 Docker 없이 local subprocess sandbox를 쓴다. official TerminalBench Docker/Harbor 평가와 완전히 같지 않다.

## 2. Online RLVR의 의미

여기서 online은 “학습된 최신 LoRA adapter가 다음 rollout 생성에 실제로 반영된다”는 뜻이다.

이전 static-vLLM 실험은 다음 구조였다.

1. vLLM 서버가 base model만 serving한다.
2. trainer는 LoRA를 업데이트한다.
3. 하지만 다음 rollout은 여전히 base policy에서 나온다.
4. 따라서 엄밀한 의미의 on-policy GRPO/RLVR이 아니다.

현재 online 구조는 다음이다.

1. trainer가 terminal task rollout을 만든다.
2. 모델이 낸 shell command를 local sandbox에서 실행한다.
3. stdout/stderr와 verifier 결과를 observation으로 받는다.
4. action token에는 GRPO advantage policy loss를 준다.
5. terminal observation token에는 ECHO-style world-model CE loss를 준다.
6. trainer가 LoRA를 업데이트한다.
7. `VLLM_LORA_SYNC_STEPS`마다 trainer가 transient LoRA snapshot을 저장한다.
8. vLLM replica 4개에 `/unload_lora_adapter`, `/load_lora_adapter`를 호출한다.
9. 다음 rollout은 `lfm25-sft1-online` LoRA가 hot-load된 policy로 생성된다.

현재 구현의 핵심 파일:

- trainer: `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`
- trainer launcher: `Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh`
- vLLM replica launcher: `Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh`
- GPU6 evaluator: `Liquid-CLI/scripts/watch_echo_rlvr_gpu6_eval_queue.sh`
- adapter HF sync: `Liquid-CLI/scripts/sync_echo_adapter_checkpoints_to_hf_model.py`
- rollout HF sync: `Liquid-CLI/scripts/sync_echo_rollouts_to_hf_dataset.py`
- eval HF sync: `Liquid-CLI/scripts/sync_echo_eval_results_to_hf_dataset.py`

## 3. 현재 성능 상태

2026-06-13 20:52 KST 기준:

- latest train step: `503`
- saved checkpoints: `checkpoint-25` through `checkpoint-500`
- GPU6 evaluated: `checkpoint-25~475`, 19 checkpoints
- current best online checkpoint: `checkpoint-425`
- Score: `53.58`
- SFT 1Epoch baseline: `52.30`
- improvement over SFT1: `+1.28`
- historical best in README: `checkpoint-610`, Score `54.05`

해석:

- `checkpoint-425`에서 online RLVR의 의미 있는 상승 spike가 처음 관측됐다.
- 하지만 `checkpoint-450`, `475`는 다시 `52.1` 근처다.
- 따라서 아직 안정적 단조 상승은 아니고, checkpoint selection이 중요하다.
- 다음 핵심 구간은 `checkpoint-500~600`이다.

## 4. 데이터 준비

현재 학습 파일:

```bash
/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl
```

현재 데이터 비율:

- Endless Terminals: `772` rows, `51.47%`
- OpenThoughts-Agent-v1-RL: `728` rows, `48.53%`
- Total: `1,500` rows

관련 파일:

- `Liquid-CLI/scripts/download_echo_public_terminal_data.py`
- `Liquid-CLI/scripts/prepare_echo_terminal_data.py`

데이터를 다시 받을 때:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal

.liquid-sft-env/bin/python Liquid-CLI/scripts/download_echo_public_terminal_data.py
```

데이터를 다시 포맷할 때:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal

.liquid-sft-env/bin/python Liquid-CLI/scripts/prepare_echo_terminal_data.py \
  --output /home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl
```

실제 옵션은 코드 변경에 따라 달라질 수 있으므로 실행 전에는 help를 확인한다.

```bash
.liquid-sft-env/bin/python Liquid-CLI/scripts/prepare_echo_terminal_data.py --help
```

## 5. vLLM rollout replica 실행

GPU 0~3에 vLLM replica 4개를 띄운다. online RLVR에는 runtime LoRA update가 필요하므로 `ENABLE_LORA=1`이 필수다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal

export RUN_ID=run_YYYYMMDDTHHMMSSZ_sft1_online_vllm_lora_sync
export RUN_DIR=/home/work/.data/liquid_cli_sft/live_terminal_echo_online/$RUN_ID
mkdir -p "$RUN_DIR/vllm"

MODEL_PATH=LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
SERVED_MODEL_NAME=lfm25-sft1-base \
VLLM_GPUS=0,1,2,3 \
BASE_PORT=8123 \
ENABLE_LORA=1 \
MAX_LORA_RANK=32 \
MAX_LORAS=1 \
MAX_CPU_LORAS=4 \
MAX_MODEL_LEN=32768 \
GPU_MEMORY_UTILIZATION=0.88 \
LOG_DIR="$RUN_DIR/vllm" \
bash Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh
```

준비 확인:

```bash
curl -fsS http://127.0.0.1:8123/v1/models
curl -fsS http://127.0.0.1:8124/v1/models
curl -fsS http://127.0.0.1:8125/v1/models
curl -fsS http://127.0.0.1:8126/v1/models
```

vLLM 주요 knob:

| 변수 | 파일 | 의미 | 속도/성능 영향 |
| --- | --- | --- | --- |
| `VLLM_GPUS` | `run_lfm25_vllm_replicas_clean.sh` | replica를 띄울 GPU 목록 | 늘리면 rollout throughput 증가. 현재는 `0,1,2,3` |
| `MAX_MODEL_LEN` | 같은 파일 | vLLM context length | 줄이면 memory/compile 부담 감소, 너무 줄이면 긴 prompt 실패 |
| `GPU_MEMORY_UTILIZATION` | 같은 파일 | vLLM KV cache VRAM 비율 | 높이면 batch 여유 증가, 너무 높이면 OOM 위험 |
| `ENABLE_LORA` | 같은 파일 | runtime LoRA load 허용 | online RLVR에는 `1` 필수 |
| `MAX_LORA_RANK` | 같은 파일 | LoRA rank 상한 | adapter rank와 맞춰야 함. 현재 `32` |
| `START_STAGGER_SEC` | 같은 파일 | replica 시작 간격 | 동시 cold-start 부담 완화 |

## 6. Online GRPO/RLVR trainer 실행

GPU 4~5에서 trainer rank 2개를 띄운다. 현재 실험은 update당 prompt 2개, prompt당 generation 4개, global rollout 8개를 만든다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal

export RUN_ID=run_YYYYMMDDTHHMMSSZ_sft1_online_vllm_lora_sync
export RUN_DIR=/home/work/.data/liquid_cli_sft/live_terminal_echo_online/$RUN_ID
export OUT_DIR=/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__online_echo_grpo_vllm_lora_sync_$RUN_ID

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/traces" "$RUN_DIR/sandboxes" "$RUN_DIR/vllm_lora_sync" "$OUT_DIR"

MODEL_PATH=LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
OUTPUT_DIR="$OUT_DIR" \
TRACE_DIR="$RUN_DIR/traces" \
SANDBOX_ROOT="$RUN_DIR/sandboxes" \
PREPARED_JSONL=/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl \
VLLM_BASE_URL=http://127.0.0.1:8123/v1,http://127.0.0.1:8124/v1,http://127.0.0.1:8125/v1,http://127.0.0.1:8126/v1 \
VLLM_SERVED_MODEL=lfm25-sft1-base \
VLLM_LORA_NAME=lfm25-sft1-online \
VLLM_LORA_SYNC_STEPS=5 \
VLLM_LORA_SYNC_DIR="$RUN_DIR/vllm_lora_sync" \
VLLM_LORA_LOAD_INPLACE=0 \
TRAIN_GPUS=4,5 \
NPROC_PER_NODE=2 \
MAX_STEPS=2000 \
MAX_WALL_TIME_HOURS=0 \
PROMPTS_PER_RANK=1 \
NUM_GENERATIONS=4 \
ROLLOUT_WORKERS=8 \
MAX_TURNS=4 \
MAX_NEW_TOKENS=256 \
COMMAND_TIMEOUT=8 \
VERIFIER_TIMEOUT=40 \
MAX_TERMINAL_OUTPUT_CHARS=10000 \
WORLD_MODEL_COEFF=0.05 \
LEARNING_RATE=5e-7 \
WARMUP_STEPS=20 \
MAX_GRAD_NORM=0.2 \
SAVE_STEPS=25 \
bash Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh \
  2>&1 | tee "$RUN_DIR/logs/train_online.log"
```

현재 active run은 `--max-wall-time-hours 36`으로 떠 있지만, future run은 `MAX_WALL_TIME_HOURS=0`을 기본으로 둔다. 종료는 wall-time이 아니라 `MAX_STEPS=2000`으로 제어한다.

trainer 주요 knob:

| 변수 | 파일 | 현재값 | 의미 | step time 영향 |
| --- | --- | ---: | --- | --- |
| `MAX_STEPS` | `run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh` | `2000` | 전체 update 수 | 전체 시간 선형 증가 |
| `MAX_WALL_TIME_HOURS` | 같은 파일 | future `0` | wall-clock hard stop | `0`이면 비활성 |
| `TRAIN_GPUS` | 같은 파일 | `4,5` | trainer GPU | 늘리면 optimizer/load 분산 가능, rollout 병목이면 이득 작음 |
| `NPROC_PER_NODE` | 같은 파일 | `2` | trainer DDP rank 수 | `TRAIN_GPUS` 개수와 맞춤 |
| `PROMPTS_PER_RANK` | 같은 파일 | `1` | rank당 prompt 수 | 늘리면 update당 batch와 step time 증가 |
| `NUM_GENERATIONS` | 같은 파일 | `4` | prompt당 GRPO samples | 줄이면 빠르지만 advantage 품질 감소 |
| `ROLLOUT_WORKERS` | 같은 파일 | `8` | rank별 rollout thread workers | 너무 낮으면 vLLM/GPU 대기, 너무 높으면 terminal contention |
| `MAX_TURNS` | 같은 파일 | `4` | task당 terminal turn 수 | 가장 큰 step time knob 중 하나 |
| `MAX_NEW_TOKENS` | 같은 파일 | `256` | turn당 생성 길이 | 줄이면 빠름, 너무 줄이면 command 누락 |
| `COMMAND_TIMEOUT` | 같은 파일 | `8` | command 실행 timeout | 줄이면 stuck 방지, 너무 줄이면 정상 command 실패 |
| `VERIFIER_TIMEOUT` | 같은 파일 | `40` | verifier timeout | 줄이면 빠름, 너무 줄이면 reward noise |
| `MAX_TERMINAL_OUTPUT_CHARS` | 같은 파일 | `10000` | observation 길이 | 줄이면 world-model CE와 context 부담 감소 |
| `VLLM_LORA_SYNC_STEPS` | 같은 파일 | `5` | vLLM hot-load 주기 | 작으면 더 on-policy, sync overhead 증가 |
| `SAVE_STEPS` | 같은 파일 | `25` | checkpoint 저장 주기 | 작으면 평가 촘촘, IO 증가 |
| `LEARNING_RATE` | 같은 파일 | `5e-7` | LoRA update lr | 품질 knob. 속도에는 영향 작음 |
| `WORLD_MODEL_COEFF` | 같은 파일 | `0.05` | terminal observation CE loss weight | ECHO 신호 강도 |

속도를 빠르게 하고 싶을 때 우선순위:

1. `MAX_TURNS`를 줄인다.
2. `MAX_NEW_TOKENS`를 줄인다.
3. `COMMAND_TIMEOUT`, `VERIFIER_TIMEOUT`을 줄인다.
4. `ROLLOUT_WORKERS`를 vLLM/CPU 상태에 맞춰 조절한다.
5. `NUM_GENERATIONS`를 줄인다. 단, GRPO 품질이 크게 흔들릴 수 있어 마지막 수단이다.
6. vLLM replica 수를 늘린다. 단, GPU 7은 쓰지 않는 원칙을 지킨다.

품질을 안정화하고 싶을 때 우선순위:

1. `NUM_GENERATIONS=4` 이상 유지.
2. `VLLM_LORA_SYNC_STEPS=5` 또는 더 작게 유지.
3. `LEARNING_RATE`를 낮게 유지. 현재 `5e-7`.
4. `MAX_GRAD_NORM=0.2` 유지.
5. `SAVE_STEPS=25`로 checkpoint selection을 촘촘하게 한다.

## 7. GPU6 평가 watcher 실행

GPU 6은 학습 중간 checkpoint를 TB2-lite full replay로 계속 평가한다. README 기준 Score는 `100 * avg_command_f1`이다.

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal

BASE_MODEL=LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
CONT_OUTPUT_DIR="$OUT_DIR" \
RESULTS_DIR=tb2_lite/results/lfm25_echo_online_rlvr_gpu6_eval_20260613 \
SHORT_PREFIX=lfm25-echo-online-sft1-checkpoint- \
GPU=6 \
VLLM_ENV=/home/work/.projects/LLM-OS-Models/Terminal/.vllm-lfm-cu12 \
MAX_MODEL_LEN=32768 \
MAX_TOKENS=1024 \
GPU_MEMORY_UTILIZATION=0.90 \
MAX_NUM_BATCHED_TOKENS=16384 \
POLL_SECONDS=180 \
EVAL_STRIDE=25 \
EVAL_RECENT=8 \
EVAL_EARLY_UNTIL=600 \
EVAL_EARLY_STRIDE=25 \
EVAL_FOCUS_START=400 \
EVAL_FOCUS_END=700 \
EVAL_FOCUS_STRIDE=25 \
EVAL_ORDER=latest_first \
bash Liquid-CLI/scripts/watch_echo_rlvr_gpu6_eval_queue.sh
```

평가 watcher 주요 knob:

| 변수 | 의미 |
| --- | --- |
| `GPU` | 평가에 쓸 GPU. 현재 `6` |
| `RESULTS_DIR` | 평가 JSON과 README가 저장될 위치 |
| `SHORT_PREFIX` | 결과 파일명 prefix |
| `POLL_SECONDS` | 새 checkpoint 확인 주기 |
| `EVAL_STRIDE` | 몇 step 간격 checkpoint를 평가할지 |
| `EVAL_RECENT` | 최신 checkpoint 몇 개를 우선 평가할지 |
| `EVAL_EARLY_UNTIL`, `EVAL_EARLY_STRIDE` | 초반 구간 촘촘 평가 |
| `EVAL_FOCUS_START`, `EVAL_FOCUS_END`, `EVAL_FOCUS_STRIDE` | 관심 구간 촘촘 평가 |
| `MAX_TOKENS` | 평가 생성 길이 |
| `MAX_MODEL_LEN` | 평가 vLLM context |

평가가 끝나면 watcher가 다음을 자동 갱신한다.

- `tb2_lite/results/.../*.json`
- `tb2_lite/results/.../README.md`
- root `README.md`의 GPU6 평가 섹션

## 8. Hugging Face 동기화

HF token은 `.env`에서 읽는다. 토큰을 출력하지 않는다.

Adapter checkpoint model repo sync:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal

.liquid-sft-env/bin/python Liquid-CLI/scripts/sync_echo_adapter_checkpoints_to_hf_model.py \
  --repo-id LLM-OS-Models/LFM2.5-8B-A1B-SFT1-Online-ECHO-RLVR-GRPO-Adapters \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUT_DIR" \
  --env-file /home/work/.projects/LLM-OS-Models/Terminal/.env \
  --interval-sec 900 \
  --loop
```

Rollout trace dataset sync:

```bash
.liquid-sft-env/bin/python Liquid-CLI/scripts/sync_echo_rollouts_to_hf_dataset.py \
  --repo-id LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts \
  --run-dir "$RUN_DIR" \
  --output-dir "$OUT_DIR" \
  --path-in-repo "online-sft1/$RUN_ID" \
  --env-file /home/work/.projects/LLM-OS-Models/Terminal/.env \
  --interval-sec 600 \
  --loop
```

Evaluation result dataset sync:

```bash
.liquid-sft-env/bin/python Liquid-CLI/scripts/sync_echo_eval_results_to_hf_dataset.py \
  --repo-id LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts \
  --results-dir tb2_lite/results/lfm25_echo_online_rlvr_gpu6_eval_20260613 \
  --path-in-repo "eval/tb2_lite_gpu6/online-sft1/$RUN_ID" \
  --env-file /home/work/.projects/LLM-OS-Models/Terminal/.env \
  --interval-sec 900 \
  --loop
```

## 9. 모니터링 명령

학습 로그:

```bash
tail -f "$RUN_DIR/logs/train_online.log"
```

평가 watcher 로그:

```bash
tail -f "$RUN_DIR/logs/eval_gpu6_online_watch.log"
```

최신 checkpoint:

```bash
find "$OUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tail
```

GPU 0~6 상태:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 0,1,2,3,4,5,6
```

현재 best 평가:

```bash
python - <<'PY'
import json, glob, os, re
base='tb2_lite/results/lfm25_echo_online_rlvr_gpu6_eval_20260613'
rows=[]
for p in glob.glob(base+'/*checkpoint-*.json'):
    m=re.search(r'checkpoint-(\d+)', os.path.basename(p))
    if not m:
        continue
    with open(p) as f:
        d=json.load(f)
    agg=d.get('aggregate', {})
    f1=agg.get('avg_command_f1')
    if f1 is None:
        continue
    rows.append((int(m.group(1)), 100*f1, agg.get('avg_command_precision'), agg.get('avg_command_recall')))
rows=sorted(rows)
print('count', len(rows))
print('latest', rows[-1] if rows else None)
print('best', max(rows, key=lambda x:x[1]) if rows else None)
PY
```

## 10. 현재 방식의 한계

이 run은 ECHO 논문의 핵심 아이디어를 현재 장비와 코드에 맞춰 구현한 것이다. 완전한 논문 1:1 재현은 아니다.

동일하게 가져온 핵심:

- terminal command execution feedback을 학습에 사용
- action token policy loss와 observation/world-model CE loss 분리
- GRPO group sampling
- terminal error/stdout/stderr를 버리지 않고 학습 신호로 사용
- online policy rollout을 위해 최신 LoRA를 vLLM에 주기적으로 반영

다른 점:

- Docker sandbox가 아니라 local subprocess sandbox다.
- official TerminalBench environment isolation이 아니다.
- verifier는 현재 local proxy 환경의 검증이다.
- vLLM LoRA sync는 `VLLM_LORA_SYNC_STEPS` 간격이라 완전 매-step on-policy는 아니다.
- checkpoint resume은 adapter weight 중심이다. optimizer state까지 완전 resume하는 구조는 별도 구현이 필요하다.
- TB2-lite Score는 Docker 기반 TerminalBench pass@1이 아니라 빠른 proxy replay score다.

따라서 결과 해석은 다음이 맞다.

- README Score는 checkpoint selection과 방향성 비교용이다.
- 최종 공개 성능은 Docker/Harbor/Terminus 계열의 실제 실행 평가로 다시 검증해야 한다.
- 현재 가장 중요한 관찰은 “online LoRA hot-load가 실제로 작동하고, `checkpoint-425`에서 SFT baseline 대비 `+1.28` spike가 나왔다”는 점이다.

## 11. 문제가 생겼을 때

vLLM server가 죽었는지 확인:

```bash
curl -fsS http://127.0.0.1:8123/v1/models || tail -200 "$RUN_DIR/vllm/vllm_gpu0_port8123.log"
```

LoRA hot-load 실패 확인:

```bash
rg -n "vllm_lora_synced|load_status|unload_status|error" "$RUN_DIR/logs/train_online.log"
```

평가가 멈춘 것 같을 때:

```bash
tail -100 "$RUN_DIR/logs/eval_gpu6_online_watch.log"
pgrep -af 'watch_echo_rlvr_gpu6_eval_queue|replay_eval.py'
```

HF sync 상태:

```bash
tail -100 "$RUN_DIR/logs/sync_online_adapter_loop.log"
tail -100 "$RUN_DIR/logs/sync_online_rollouts_loop.log"
tail -100 "$RUN_DIR/logs/sync_online_eval_loop.log"
```

현재 run을 억지로 재시작해야 할 때:

- trainer를 죽이면 optimizer state 연속성이 끊길 수 있다.
- 마지막 adapter checkpoint에서 continuation은 가능하지만, 엄밀한 optimizer-state resume과는 다르다.
- 따라서 성능이 무너지거나 프로세스가 죽은 경우가 아니면 checkpoint-600 전에는 유지하는 편이 낫다.
