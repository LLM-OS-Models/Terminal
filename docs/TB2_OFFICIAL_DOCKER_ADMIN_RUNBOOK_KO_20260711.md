# LFM2.5 공식 Terminal-Bench 2.0 Docker 관리자 런북

작성일: 2026-07-11 KST
대상 저장소: `/home/ubuntu/Terminal`
재현 디렉터리: `/home/ubuntu/Terminal/tb2_official`

## 1. 목적과 현재 결론

이 문서는 이 저장소의 TB2-lite 순위가 실제 Docker 기반 Terminal-Bench 2.0에서도 유지되는지 검증하기 위한 관리자·실행자 공용 런북이다.

현재 유효한 공식 Terminal-Bench 2.0 점수는 없다. Harbor가 출력한 `Mean: 0.000`은 모델의 0점이 아니라 task container 생성 전에 끝난 인프라 오류다. 해당 실행은 `n_trials=0`, `n_errors=1`이므로 성능 비교에 사용하면 안 된다.

현재까지 완료된 항목:

- H100 80GB에서 SFT 모델과 두 LoRA의 vLLM 로딩
- 세 served model ID의 OpenAI-compatible chat completion
- Harbor 0.18.0의 실제 Terminus-2 JSON prompt/parser 호환성
- 공식 `terminal-bench/terminal-bench-2@1` 데이터셋 해석
- 모델 및 adapter revision과 SHA-256 고정
- Docker/Compose/Harbor/vLLM 전용 환경 설치

남은 유일한 실행 기반 blocker는 현재 workload container가 nested Docker에 필요한 커널 권한을 받지 못했다는 점이다. 모델 서버는 검증 후 종료했으며, 권한이 해결되기 전에는 GPU를 계속 점유할 필요가 없다.

현재 GPU 상태는 H100 memory `0 MiB`, utilization `0%`이고 vLLM port 8000도 닫혀 있다. Docker preflight와 oracle 검증에는 GPU가 필요 없으므로 oracle이 통과한 뒤에만 vLLM을 시작한다.

## 2. 기존 README 점수의 의미

루트 README의 대표값은 다음과 같다.

| 구성 | TB2-lite Score |
| --- | ---: |
| SFT 1Epoch | 52.30 |
| SFT + static/offline ECHO RLVR parent checkpoint-610 | 54.05 |
| SFT + online ECHO RLVR checkpoint-425 | 53.58 |

이 값은 공식 Terminal-Bench 2.0 Accuracy가 아니다.

- 데이터는 50개 task에서 추출한 303개 누적 next-action replay step이다.
- 실제 명령이나 Docker task를 실행하지 않는다.
- `Score = 100 * avg_command_f1`이다.
- 파일시스템 변화, unit test 성공, 장기 오류 복구, 정답과 다른 기능적으로 동등한 풀이를 직접 채점하지 못한다.

따라서 기존 순위는 terminal JSON/action imitation의 proxy로는 유용하지만 실제 TB2 성능 주장으로 사용할 수 없다. 공식 검증은 89개 task를 실제 container에서 수행하고 verifier reward로 채점해야 한다.

## 3. 평가 대상과 고정 manifest

| served model ID | 구성 | revision | weight SHA-256 |
| --- | --- | --- | --- |
| `lfm25-sft1` | SFT 1Epoch full BF16 model | `4ca70e4065d10b171de0a434185a9c436cbe9893` | `c742791bb761580d2b339e946ea8b5f1b8e9b6f742e5ab29fca07b3255f66ead` |
| `lfm25-static-610` | SFT1 + static parent LoRA | `dbd53977c71f1aa842ae6ed558ad5d2221d441ff` | `cca35fd3d624615d67589edb5dee2e70d50479be5a470acdf4b1677d958fdb9e` |
| `lfm25-online-425` | SFT1 + online LoRA | `dae4ec9466fc2b1c233d66d86271236a9acc9986` | `515a9cc24d0be1f30f824ea82c1099372efe37489fac90794ddc66478854a55f` |

두 RLVR 모델은 standalone model이 아니라 SFT1 위에 올리는 PEFT LoRA다. 둘 다 rank 32, alpha 64이며 vLLM runtime LoRA로 평가한다.

static adapter 저장소의 현재 `main/checkpoints/checkpoint-610`을 사용하면 안 된다. 후속 continuation run이 같은 경로를 덮었으며 그 weight SHA-256은 `2d546a2728c3e8190141b5ae89329cbc17ad2e4d677cb77e5b0be31873ae6724`다. README의 54.05 대상은 위 표의 parent revision과 hash다.

## 4. 현재 Docker 실패의 정확한 원인

현재 환경에서 확인된 상태:

```text
Docker client/server: 29.6.1
Docker Compose: 5.3.1
Docker daemon query: 성공
image download: 성공
image layer registration: unshare: operation not permitted
sudo unshare -m true: operation not permitted
root capability bounding set: CAP_SYS_ADMIN 없음
seccomp: mode 2, filter 활성
```

현재 내부 daemon은 `--iptables=false --ip-masq=false --bridge=none --storage-driver=vfs`로 실행 중이다. capability만 해결해도 이 daemon을 그대로 사용하면 일부 task의 DNS, outbound network 또는 Compose network가 실패할 수 있다. privileged runner를 새로 만들 때는 정상 bridge/NAT를 제공하는 daemon을 사용해야 한다.

현재 머신처럼 보이는 workload 자체가 이미 상위 container다. 상위 runtime이 capability bounding set에서 `CAP_SYS_ADMIN`을 제거하고 `unshare()`를 seccomp로 차단했다. 내부의 passwordless sudo는 정상 작동하지만 root도 bounding set 밖의 capability를 새로 얻을 수 없다.

다음 조치는 해결책이 아니다.

- Docker 재설치
- `/var/run/docker.sock`의 단순 `chmod 666`
- 내부 사용자에게 sudo 추가
- 내부 daemon 옵션만 변경
- `docker run --privileged`만 내부에서 호출

마지막 항목의 `--privileged`는 새 child container에 적용할 설정이다. 현재 문제는 그 child container를 생성하는 상위 workload 자체에 권한이 없다는 것이다.

## 5. 관리자 해결 방법

### 5.1 권장: 별도 Docker VM/베어메탈에서 Harbor 실행

가장 안전하고 재현성이 높은 방법이다. H100 workload에는 vLLM만 띄우고 Docker가 정상 작동하는 별도 Ubuntu runner에서 Harbor와 task container를 실행한다. TB2 task 자체는 GPU를 요구하지 않는다.

권장 runner 조건:

- Docker Engine과 Docker Compose가 정상인 Ubuntu VM 또는 베어메탈
- 동시 실행 수 4 기준 16 CPU 이상, RAM 32GB 이상
- 여러 task image를 위한 충분한 로컬 디스크
- container registry 접근을 위한 outbound network
- H100 workload의 vLLM port에 접근 가능한 private network

H100 workload에서는 다음처럼 private interface에 vLLM을 연다.

```bash
cd /home/ubuntu/Terminal
HOST=0.0.0.0 ./tb2_official/run_vllm.sh
```

별도 runner에서는 다음을 사용한다.

```bash
API_BASE=http://<GPU_PRIVATE_IP>:8000/v1 \
  ./tb2_official/run_tb2.sh lfm25-sft1
```

port 8000을 공인 인터넷에 공개하면 안 된다. runner의 private IP만 방화벽에서 허용한다.

### 5.2 현재 workload를 privileged/unconfined로 재생성

현재 container에 사후 추가하는 것이 아니라 workload를 새로 만들어야 한다. Docker 계열 launcher의 핵심 설정은 다음과 같다.

```bash
docker run \
  --gpus all \
  --privileged \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --cgroupns=host \
  --shm-size=16g \
  -v /sys/fs/cgroup:/sys/fs/cgroup:rw \
  -v <persistent-workspace>:/home/ubuntu \
  -v <docker-storage>:/var/lib/docker \
  <workload-image>
```

관리형 Kubernetes라면 최소한 다음 security context가 필요하다.

```yaml
securityContext:
  privileged: true
  allowPrivilegeEscalation: true
  runAsUser: 0
  seccompProfile:
    type: Unconfined
```

추가 요구사항:

- 기존 `/home/ubuntu` persistent volume을 보존해야 모델과 작업물이 유지된다.
- `/var/lib/docker`는 별도 writable volume을 권장한다.
- cluster admission policy가 privileged pod를 차단한다면 해당 namespace에 관리자 예외가 필요하다.
- GPU allocation을 유지한다.
- full Docker에는 mount뿐 아니라 network/cgroup 작업도 필요하므로 capability를 하나씩 추측해 추가하는 것보다 privileged runner가 확실하다.

### 5.3 호스트 Docker socket 전달

호스트가 실제 Docker Engine을 사용한다면 workload에 `/var/run/docker.sock`을 전달할 수 있다. 이 경우 현재 container의 `unshare()`가 막혀 있어도 daemon이 호스트에서 container를 만들므로 동작할 수 있다.

주의사항:

- Kubernetes node가 containerd 전용이면 Docker socket이 없을 수 있다.
- Harbor가 참조하는 workspace path가 daemon 쪽에서도 동일한 절대경로로 보여야 한다.
- socket 접근 사용자는 host root와 사실상 동등한 권한을 얻으므로 보안 승인이 필요하다.
- 내부에서 별도 `dockerd`를 동시에 띄우지 않는다.

`docker_preflight.sh`는 이 방식을 허용하기 위해 `unshare` 실패를 참고 정보로만 기록하고, 실제 `docker run` 성공 여부를 최종 gate로 사용한다.

### 5.4 Harbor cloud sandbox

Harbor가 지원하는 외부 environment credential을 제공하는 방법도 있다. 설치된 Harbor 0.18 기준 Daytona와 Modal은 Compose capability를 제공하지만, E2B는 같은 Compose capability가 없어 전체 TB2의 동등한 대안이라고 단정할 수 없다. 어떤 provider든 비용, image 호환성, network access와 전체 oracle을 먼저 검증해야 한다. 현재 `run_tb2.sh`는 외부 `--env` 전달을 구현하지 않았으므로 이 경로에서는 raw Harbor command를 사용하거나 wrapper를 확장한다.

## 6. 관리자 완료 기준

Harbor를 실행할 동일 사용자와 동일 network namespace에서 확인한다.

```bash
docker info
docker compose version
docker run --rm hello-world
docker run --rm alpine sh -c 'wget -qO- https://github.com >/dev/null'
cd /home/ubuntu/Terminal
./tb2_official/docker_preflight.sh
```

마지막 출력은 다음과 같아야 한다.

```text
Docker preflight passed: daemon, compose, layer extraction, and container run work.
```

`ubuntu`를 docker group에 추가한 직후 기존 shell에는 membership이 반영되지 않을 수 있다. 새 login shell을 시작하거나 다음을 사용한다.

```bash
sg docker -c '/home/ubuntu/Terminal/tb2_official/docker_preflight.sh'
```

preflight가 sudo Docker만 통과시키면 안 된다. Harbor 프로세스 자체가 `docker compose`, `exec`, `cp`, `down`을 호출하므로 Harbor 실행 사용자에게 직접 socket 접근 권한이 있어야 한다. 제공된 실행 스크립트는 group membership이 `/etc/group`에는 있으나 현재 shell에 반영되지 않은 경우 `sg docker`로 한 번 자동 재실행한다.

host socket 방식은 동일 절대경로 bind mount도 확인한다.

```bash
docker run --rm \
  -v /home/ubuntu/Terminal/tb2_official:/probe:ro \
  alpine test -f /probe/README.md
```

`hello-world`가 cache돼 있으면 새 layer extraction 검사가 생략될 수 있으므로 최종 완료 gate는 아래 Harbor oracle 결과다.

## 7. 권한 복구 후 공식 실행 순서

### 7.1 GPU 없이 oracle 1-task와 5-task

먼저 1개 task로 image/network/Compose 경로를 확인한다.

```bash
/home/ubuntu/.tools/bin/harbor run \
  -d terminal-bench/terminal-bench-2@1 \
  -a oracle \
  -l 1 \
  -n 1 \
  --yes \
  --job-name oracle-smoke-1 \
  --jobs-dir /home/ubuntu/Terminal/tb2_official/jobs
```

그다음 공식 smoke 범위인 5개 task를 실행한다.

```bash
/home/ubuntu/.tools/bin/harbor run \
  -d terminal-bench/terminal-bench-2@1 \
  -a oracle \
  -l 5 \
  -n 1 \
  --yes \
  --job-name oracle-smoke-5 \
  --jobs-dir /home/ubuntu/Terminal/tb2_official/jobs
```

성공 조건은 command exit만이 아니다. `result.json`에서 5개 trial이 실제 집계되고 `n_errors=0`이며 oracle reward가 성공이어야 한다. 이 단계에는 GPU와 vLLM이 필요 없다.

### 7.2 vLLM 시작과 endpoint 확인

```bash
cd /home/ubuntu/Terminal
./tb2_official/run_vllm.sh
```

다른 shell에서 확인한다.

```bash
curl -sS http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

다음 ID 세 개가 모두 보여야 한다.

```text
lfm25-sft1
lfm25-static-610
lfm25-online-425
```

#### 7.2.1 구현 파일

vLLM 관련 코드는 다음 세 파일로 분리했다.

- `tb2_official/run_vllm.sh`: base model과 두 LoRA를 한 GPU에 올리는 foreground server
- `tb2_official/smoke_vllm.py`: model 목록, chat completion, balanced JSON과 command schema 검증
- `tb2_official/run_tb2.sh`: Docker preflight 후 Terminus-2를 선택한 vLLM model ID에 연결

runtime artifact는 Git에 넣지 않는다. 기대하는 로컬 구조는 다음과 같다.

```text
tb2_official/models/
├── sft1/
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors
├── static-parent-610/checkpoints/checkpoint-610/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── online-425/checkpoints/checkpoint-425/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

현재 검증된 software 조합:

```text
Python       3.12.13
vLLM         0.19.1
PyTorch      2.10.0+cu128
Transformers 5.13.1
PEFT         0.19.1
HF Hub       1.23.0
GPU          NVIDIA H100 80GB HBM3
```

#### 7.2.2 pinned model 다운로드

현재 파일은 이미 내려받았지만 새 runner에서 복원할 때는 revision을 반드시 고정한다.

```bash
HF=/home/ubuntu/.venvs/tb2-vllm/bin/hf
ROOT=/home/ubuntu/Terminal/tb2_official/models

"$HF" download \
  LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
  --revision 4ca70e4065d10b171de0a434185a9c436cbe9893 \
  --local-dir "$ROOT/sft1"

"$HF" download \
  LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-GRPO-Adapters \
  --revision dbd53977c71f1aa842ae6ed558ad5d2221d441ff \
  --include 'checkpoints/checkpoint-610/*' \
  --local-dir "$ROOT/static-parent-610"

"$HF" download \
  LLM-OS-Models/LFM2.5-8B-A1B-SFT1-Online-ECHO-RLVR-GRPO-Adapters \
  --revision dae4ec9466fc2b1c233d66d86271236a9acc9986 \
  --include 'checkpoints/checkpoint-425/*' \
  --local-dir "$ROOT/online-425"
```

다운로드 후 adapter hash를 확인한다.

```bash
sha256sum \
  "$ROOT/static-parent-610/checkpoints/checkpoint-610/adapter_model.safetensors" \
  "$ROOT/online-425/checkpoints/checkpoint-425/adapter_model.safetensors"
```

각각 `cca35fd3...`와 `515a9cc2...`여야 한다. SFT weight는 약 17GB라 전체 SHA-256 계산에 시간이 걸릴 수 있다.

#### 7.2.3 실제 vLLM serve 코드

`run_vllm.sh`의 핵심 command는 다음과 같다.

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/ubuntu/.venvs/tb2-vllm/bin/vllm serve \
  /home/ubuntu/Terminal/tb2_official/models/sft1 \
  --served-model-name lfm25-sft1 \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 32 \
  --max-loras 2 \
  --max-cpu-loras 2 \
  --lora-modules \
    lfm25-static-610=/home/ubuntu/Terminal/tb2_official/models/static-parent-610/checkpoints/checkpoint-610 \
    lfm25-online-425=/home/ubuntu/Terminal/tb2_official/models/online-425/checkpoints/checkpoint-425
```

주요 옵션의 의미:

| 옵션 | 이유 |
| --- | --- |
| `--served-model-name lfm25-sft1` | base request에 사용할 안정적인 짧은 ID |
| `--dtype bfloat16` | H100과 원 checkpoint dtype에 맞춘 비양자화 실행 |
| `--max-model-len 32768` | adapter 평가와 Terminus `model_info`를 같은 32K로 고정 |
| `--gpu-memory-utilization 0.88` | 80GB H100에서 weight와 KV cache 공간 확보 |
| `--enforce-eager` | hybrid LFM/MoE + 여러 LoRA에서 graph/prefix-cache 실험 변수를 제거해 정확성 우선 |
| `--enable-lora` | 두 51MB adapter를 SFT base 위에서 runtime 적용 |
| `--max-lora-rank 32` | adapter config의 실제 rank와 일치 |
| `--max-loras 2` | 한 batch에서 두 등록 LoRA request를 수용 |
| `--lora-modules name=path` | API의 model ID와 pinned local adapter path를 연결 |

`--enforce-eager`는 throughput보다 재현 안정성을 우선한 선택이다. 나중에 compile/CUDA graph를 켜 성능을 조정할 수 있지만 세 모델 비교 중 server option을 섞으면 안 된다. LFM hybrid cache의 experimental prefix caching도 기준 실행에서는 사용하지 않는다.

실제 script는 경로와 weight 존재 여부를 확인하고 `PYTHONPATH`와 user site package 오염을 차단한 뒤 `exec`한다. 환경변수로 다음 값을 바꿀 수 있다.

```text
ROOT_DIR, VLLM_BIN, CUDA_DEVICE, HOST, PORT,
MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, API_KEY
```

#### 7.2.4 local과 별도 runner 모드

Harbor와 vLLM이 같은 network namespace라면 기본값을 사용한다.

```bash
./tb2_official/run_vllm.sh
```

별도 Docker runner가 접속해야 한다면 private interface에 bind하고 API key를 설정한다.

```bash
export OPENAI_API_KEY='<secret-manager에서 주입한 임의의 긴 값>'
HOST=0.0.0.0 API_KEY="$OPENAI_API_KEY" \
  ./tb2_official/run_vllm.sh
```

runner에서는 동일한 key와 private IP를 사용한다.

```bash
export OPENAI_API_KEY='<동일한 값>'
API_BASE=http://<GPU_PRIVATE_IP>:8000/v1 \
  ./tb2_official/run_tb2.sh lfm25-sft1
```

key를 command, Git remote, 문서나 commit에 직접 기록하지 않는다. vLLM port는 runner private IP에만 허용한다. 현재 script에서 API key를 생략하면 인증 없는 local endpoint가 되므로 `HOST=0.0.0.0`과 무인증을 함께 사용하면 안 된다.

#### 7.2.5 readiness와 세 모델 smoke

server가 준비될 때까지 model route를 확인한 뒤 제공된 smoke code를 실행한다.

```bash
until curl -fsS \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-dummy}" \
  http://127.0.0.1:8000/v1/models >/dev/null; do
  sleep 2
done

API_BASE=http://127.0.0.1:8000/v1 \
OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  python3 ./tb2_official/smoke_vllm.py
```

`smoke_vllm.py`가 하는 일:

1. `/v1/models`에서 base와 두 LoRA ID가 모두 존재하는지 확인한다.
2. 각 ID에 `temperature=0`, `max_tokens=4096` chat completion을 보낸다.
3. LFM의 `<think>...</think>`가 앞에 있어도 첫 balanced JSON object를 추출한다.
4. `analysis`, `plan`, `commands`를 검증한다.
5. 각 `keystrokes`가 실제 실행 가능한 newline으로 끝나는지 확인한다.
6. `finish_reason=stop`, command 수와 completion token 수를 출력한다.

개별 OpenAI-compatible request는 다음처럼 확인할 수 있다.

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-dummy}" \
  -d '{
    "model": "lfm25-static-610",
    "messages": [{
      "role": "user",
      "content": "Return Terminus JSON that creates /tmp/hello.txt with hello followed by a newline."
    }],
    "temperature": 0,
    "max_tokens": 4096
  }' | python3 -m json.tool
```

#### 7.2.6 Terminus-2와 vLLM의 연결

`run_tb2.sh`는 선택한 served ID를 LiteLLM의 OpenAI-compatible provider로 전달한다.

```text
Harbor model_name = openai/<served-model-id>
api_base          = http://127.0.0.1:8000/v1
parser_name       = json
temperature       = 0
max_input_tokens  = 32768
max_output_tokens = 4096
```

실제 Harbor 인자는 다음과 같다.

```bash
harbor run \
  -d terminal-bench/terminal-bench-2@1 \
  -a terminus-2 \
  -m openai/lfm25-static-610 \
  --ak api_base=http://127.0.0.1:8000/v1 \
  --ak parser_name=json \
  --ak temperature=0 \
  --ak 'model_info={"max_input_tokens":32768,"max_output_tokens":4096,"input_cost_per_token":0,"output_cost_per_token":0}' \
  --ak 'llm_kwargs={"max_tokens":4096}'
```

LFM native function-call parser는 사용하지 않는다. 이 모델은 Terminus-2가 요구하는 plain JSON의 `commands[].keystrokes`를 생성하고, Harbor가 그 keystroke를 tmux session으로 보낸다. `<think>`는 JSON 앞 extra text warning이 될 수 있지만 검증한 Terminus-2 parser는 첫 balanced JSON을 정상 추출한다.

#### 7.2.7 장시간 server 운영과 종료

기본 script는 foreground 실행이라 로그와 오류를 즉시 볼 수 있다. 장시간 job에서 background로 관리하려면 ignored `logs/`와 PID 파일을 사용한다.

```bash
mkdir -p tb2_official/logs
nohup ./tb2_official/run_vllm.sh \
  >tb2_official/logs/vllm.log 2>&1 &
echo $! >tb2_official/vllm.pid

tail -f tb2_official/logs/vllm.log
```

종료와 자원 반환 확인:

```bash
kill "$(cat tb2_official/vllm.pid)"
rm -f tb2_official/vllm.pid
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
curl -fsS --max-time 2 http://127.0.0.1:8000/v1/models || true
```

평가가 중단되거나 Docker blocker가 다시 발생하면 vLLM부터 종료해 GPU를 점유하지 않는다.

### 7.3 모델별 1-task smoke

```bash
./tb2_official/run_tb2.sh lfm25-sft1
./tb2_official/run_tb2.sh lfm25-static-610
./tb2_official/run_tb2.sh lfm25-online-425
```

기본값은 `TB2_LIMIT=1`, `TB2_ATTEMPTS=1`, `TB2_CONCURRENCY=1`, `temperature=0`이다.

### 7.4 모델별 89-task 빠른 full pass

명시적 빈 `TB2_LIMIT=`는 limit 옵션을 제거해 전체 89개 task를 선택한다.

```bash
for model in lfm25-sft1 lfm25-static-610 lfm25-online-425; do
  TB2_LIMIT= TB2_ATTEMPTS=1 TB2_CONCURRENCY=4 \
    ./tb2_official/run_tb2.sh "$model"
done
```

이 단계는 모델당 89 trials이며 빠른 방향성 확인용이다.

### 7.5 leaderboard 비교용 5-attempt full run

```bash
for model in lfm25-sft1 lfm25-static-610 lfm25-online-425; do
  TB2_LIMIT= TB2_ATTEMPTS=5 TB2_CONCURRENCY=4 \
    ./tb2_official/run_tb2.sh "$model"
done
```

- 모델당 `89 * 5 = 445` trials
- 세 모델 합계 1,335 trials
- timeout, task CPU/RAM/storage는 비교 중 임의 변경하지 않는다.
- 일부 task timeout은 매우 길 수 있으므로 전체 평가는 장시간 실행될 수 있다.

모델 카드의 proxy 평가와 맞추기 위해 세 구성 모두 `temperature=0`을 사용한다. 공식 TB2 문서는 custom model의 sampling 값을 강제하지 않으므로 결과 보고서에는 이 값을 반드시 명시한다.

## 8. 결과 해석

기본 결과 위치:

```text
tb2_official/jobs/<job-name>/result.json
tb2_official/jobs/<job-name>/<trial-name>/
```

요약 예시:

```bash
jq '
  .stats.evals
  | to_entries[]
  | {
      eval: .key,
      trials: .value.n_trials,
      errors: .value.n_errors,
      metrics: .value.metrics,
      pass_at_k: .value.pass_at_k
    }
' tb2_official/jobs/<job-name>/result.json
```

주의할 점:

- 공식 Accuracy는 89개 task 각각의 5회 binary reward 평균을 다시 평균한 값이다.
- 모든 task의 attempt 수가 5이면 전체 445개 reward 평균과 같다.
- `pass@5`, 즉 5회 중 한 번이라도 성공한 task 비율과 동일하지 않다.
- `n_trials=0`이거나 exception이 있는 run의 `Mean: 0`을 모델 점수로 기록하면 안 된다.
- 89-task 1회 결과는 빠른 추정치일 뿐 공식 5-attempt 결과와 구분한다.

## 9. 빠른 장애 판별표

| 증상 | 원인 | 조치 |
| --- | --- | --- |
| `docker info: permission denied` | 현재 shell에 docker group 미반영 | 재로그인 또는 `sg docker` |
| `unshare: operation not permitted` during layer registration | 제한된 nested Docker | privileged 재생성, host socket, 별도 runner 중 하나 선택 |
| preflight는 sudo만 성공 | Harbor 사용자에게 socket 권한 없음 | Harbor 사용자 직접 접근 권한 부여 |
| `/v1/models` 연결 거절 | vLLM 미실행 또는 잘못된 API address | `run_vllm.sh` 시작, network namespace에 맞게 `API_BASE` 설정 |
| adapter ID 없음 | LoRA path/rank/서버 옵션 오류 | pinned path와 `--max-lora-rank 32` 확인 |
| Terminus JSON warning | `<think>`가 JSON 앞에 존재 | 현 parser는 첫 balanced JSON을 추출하므로 error 여부를 확인 |
| GitHub `401 Bad credentials` | PAT 오타·만료·폐기 | 새 credential 발급 후 API와 repo push 권한 검증 |

## 10. credential 및 저장소 안전 수칙

- `.env`와 `.env.*`는 Git에서 제외한다. 공유 가능한 예시는 `.env.example`만 허용한다.
- credential 파일 권한은 `0600`으로 유지한다.
- PAT 값을 문서, remote URL, shell trace, commit message에 넣지 않는다.
- GitHub credential에는 대상 저장소의 Contents read/write 권한이 필요하다.
- `git add -A` 대신 문서와 작은 실행 파일을 명시적으로 stage한다.
- `tb2_official/.gitignore`는 `models/`, `jobs/`, `logs/`, `cache/`, `*.pid`를 제외한다. 16.9GB model weight와 실행 로그는 커밋하지 않는다.

## 11. 평가 후 정리

- vLLM foreground process를 `Ctrl+C`로 종료하고 `nvidia-smi`에서 memory 사용량을 확인한다.
- privileged DinD/별도 runner가 평가 전용이면 job 결과를 보존한 뒤 종료한다.
- 임시로 연 private firewall rule과 port 8000을 닫는다.
- `/var/lib/docker` volume을 다른 daemon과 동시에 공유하지 않는다.
- 최종 보고서에는 dataset revision, Harbor/vLLM version, model revision/hash, agent, temperature, attempts, concurrency, errors를 함께 기록한다.

## 12. 관련 자료

- Terminal-Bench 2.0 실행 문서: https://www.tbench.ai/docs/run-terminal-bench-2-0
- Harbor Terminus-2: https://www.harborframework.com/docs/agents/terminus-2
- TB2 dataset: https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2
- SFT model: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch
- static adapters: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-GRPO-Adapters
- online adapters: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-SFT1-Online-ECHO-RLVR-GRPO-Adapters
- 상세 재현 상태: `tb2_official/README.md`
