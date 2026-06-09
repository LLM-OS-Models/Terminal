# No-Docker Terminal RLVR Sandbox 안정화 기록

작성일: 2026-06-09

## 결론

현재 서버에서는 Docker 기반 sandbox도 어렵고, `zerobox`의 Linux strict sandbox도 바로 사용할 수 없다.

따라서 이번 안정화는 외부 sandbox runtime으로 갈아타는 대신, 기존 no-Docker local sandbox 실행기를 더 강하게 만드는 방향으로 진행했다.

핵심 변경:

- `/app`를 sandbox 내부 `workspace`로 rewrite
- `/tests`를 sandbox 내부 `tests`로 rewrite
- workspace 안의 생성 파일도 command 실행 후 rewrite
- `tmux`, `screen`, `pkill`, `killall`, `systemctl`, `sudo`, `apt-get` 등 host에 영향을 줄 수 있는 명령 차단
- subprocess 환경변수를 host 환경 상속 방식에서 최소 allowlist 방식으로 변경
- `HOME`, `TMPDIR`, `XDG_RUNTIME_DIR`, cache path를 sandbox 내부로 고정
- GPU 6,7을 실수로 쓰지 않도록 echo no-Docker run script 기본값을 `TRAIN_GPUS=4,5`, `NPROC_PER_NODE=2`로 변경

## zerobox 실험 결과

설치:

```bash
uv tool install zerobox
```

설치된 버전:

```text
zerobox 0.3.3
```

strict sandbox 테스트:

```bash
zerobox --strict-sandbox ...
```

결과:

```text
error: strict sandbox requires bubblewrap but user namespaces are unavailable
```

즉, 이 서버는 user namespace 또는 bubblewrap 기반 strict isolation을 사용할 수 없는 상태다.

non-strict fallback 테스트:

```bash
zerobox --allow-read=... --allow-write=... ...
```

결과:

```text
permission profiles requiring direct runtime enforcement are incompatible with --use-legacy-landlock
```

따라서 현재 커널/권한 설정에서는 `zerobox`를 터미널 RLVR command wrapper로 바로 쓰기 어렵다.

## OpenSandbox 판단

OpenSandbox는 장기적으로는 더 정석적인 방향이다.

하지만 로컬 실행 요구사항에 Docker가 들어간다. 현재 환경에서는 Docker 설치/실행 자체가 막혀 있으므로, 지금 당장 적용할 수 있는 해결책은 아니다.

정리하면 다음과 같다.

- Docker 가능: OpenSandbox 또는 Docker sandbox가 가장 좋다.
- Docker 불가, user namespace 가능: zerobox/bubblewrap 계열이 좋다.
- Docker 불가, user namespace도 불가: 현재처럼 no-Docker 실행기를 강화하는 수밖에 없다.

현재 서버는 세 번째 경우에 해당한다.

## 기존 no-Docker 방식의 문제

기존 실행기는 작업마다 sandbox 폴더를 만들고, `HOME`과 `CUDA_VISIBLE_DEVICES` 정도를 조정했다.

하지만 Docker처럼 다음을 완전히 격리하지는 못했다.

- host process namespace
- `/tmp`와 runtime socket
- tmux/screen session
- host-level service command
- network command
- package manager
- parent process의 여러 환경변수

이 때문에 모델이 생성한 명령이 `tmux`, `pkill`, `systemctl`, `apt-get`, `/tmp` socket 등에 닿을 수 있었다.

## 성능 손실 원인도 있었다

안정성 문제와 별개로, 성능을 깎는 큰 문제가 있었다.

TerminalBench 계열 task는 Docker 내부 기준으로 `/app`, `/tests`, `/output` 같은 경로를 자주 쓴다.

기존 no-Docker 실행기는 `/workspace`, `/output`, `/logs`만 rewrite했다.

그래서 모델이 자연스럽게 다음 같은 명령을 내면:

```bash
cat /app/worker_queue.py
pytest /tests/test_state.py
```

기존 코드에서는 absolute path unsafe rule에 걸려 차단되거나, verifier 내부에서 `/tests/...`를 못 찾아 실패할 수 있었다.

실제 trace에서도 `/app` 명령이 unsafe로 막힌 사례가 있었다.

이번 패치에서 `/app -> sandbox/workspace`, `/tests -> sandbox/tests` 매핑을 추가한 이유가 이것이다.

## 이번 패치의 효과

### 1. host 영향 감소

다음 명령들은 이제 unsafe pattern 또는 PATH wrapper로 막는다.

- `tmux`
- `screen`
- `byobu`
- `pkill`
- `killall`
- numeric PID 대상 `kill`
- `nohup`
- `setsid`
- `systemctl`
- `service`
- `sudo`
- `su`
- `ssh`
- `scp`
- `rsync`
- `nc`
- `ncat`
- `telnet`
- `apt`
- `apt-get`
- `dpkg`
- `nvidia-smi`
- `nvcc`

### 2. host 환경변수 노출 감소

기존에는 `dict(os.environ)` 기반으로 parent 환경을 대부분 들고 들어갔다.

이제 sandbox subprocess는 최소 환경만 받는다.

주요 값:

- `HOME=sandbox/workspace`
- `TMPDIR=sandbox/tmp`
- `XDG_RUNTIME_DIR=sandbox/runtime`
- `XDG_CACHE_HOME=sandbox/.cache`
- `PIP_CACHE_DIR=sandbox/.cache/pip`
- `UV_CACHE_DIR=sandbox/.cache/uv`
- `PYTHONPYCACHEPREFIX=sandbox/.cache/python`
- `CUDA_VISIBLE_DEVICES=`
- `NVIDIA_VISIBLE_DEVICES=none`
- `TMUX=`
- `STY=`
- `SSH_AUTH_SOCK=`

즉, HF token, tmux socket, SSH agent, CUDA device 같은 host context가 task command로 흘러가는 위험을 줄였다.

### 3. `/app`, `/tests` 호환성 개선

명령 실행 전:

- `/app` -> `sandbox/workspace`
- `/tests` -> `sandbox/tests`
- `/workspace` -> `sandbox/workspace`
- `/output` -> `sandbox/output`
- `/logs` -> `sandbox/logs`

테스트 파일과 seed 파일에도 같은 rewrite를 적용한다.

또한 command 실행 후 workspace 내부 텍스트 파일도 다시 rewrite한다. 모델이 Python script를 생성하면서 내부에 `/app/...`를 박아 넣는 경우를 줄이기 위해서다.

## Smoke test 결과

검증 내용:

- `tmux ls`가 unsafe로 잡히는가
- `cat /app/foo.txt`는 unsafe가 아닌가
- `/app/foo.txt`가 실제 workspace path로 rewrite되는가
- Python script 내부 `/app/bar.txt`도 rewrite되어 workspace에 파일을 쓰는가

결과:

```text
unsafe_tmux True
unsafe_app_cat False
stdout hello
stderr_first tmux is disabled in no-Docker RLVR sandboxes.
bar_exists True
```

즉, `/app` 호환성과 host-sensitive command 차단이 모두 작동한다.

## 현재 run에 대한 영향

이미 떠 있는 학습 프로세스는 Python code를 메모리에 로드한 상태라, 이번 코드 변경이 자동으로 반영되지는 않는다.

반영하려면 최신 checkpoint에서 새 run으로 재시작해야 한다.

현재 안정적인 전환 방법:

1. 기존 run이 다음 checkpoint를 저장할 때까지 기다린다.
2. trainer process만 종료한다.
3. vLLM replicas 0-3은 그대로 둔다.
4. 최신 checkpoint를 `SFT_ADAPTER_PATH`로 지정해서 새 run을 시작한다.
5. 새 run은 patched no-Docker sandbox 실행기를 사용한다.

이 방식이면 vLLM 서버 재시작 비용을 줄이면서 sandbox 안정화만 적용할 수 있다.

## Hugging Face rollout dataset sync

진행 중 생성되는 rollout trace, train step log, checkpoint manifest는 다음 dataset repo에 계속 동기화한다.

```text
LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts
```

주의할 점:

- `.env`는 공용 서버의 token 파일이므로 shell에서 `source .env` 하지 않는다.
- sync script가 `--env-file`로 token만 읽게 한다.
- 이전 sync는 `checkpoint-620` 근처에서 멈췄기 때문에, `checkpoint-820` 저장 후 다시 sync를 시작해야 한다.
- patched sandbox로 재시작한 새 run도 별도 `runs/<run_id>` prefix에 sync한다.

이렇게 해야 학습 중 생성된 실패/성공 trajectory와 터미널 관측값을 나중에 다시 SFT/RL 데이터로 재사용할 수 있다.

## 장기 실행 launch 안정화

이번 서버에서는 일반적인 `nohup ... &` 실행이 항상 충분히 안정적이지 않았다. 일부 background process는 shell 실행이 끝난 뒤 바로 사라졌고, 로그도 0바이트로 남았다.

반면 간단한 생존 테스트는 `setsid -f`로 통과했다.

```bash
setsid -f bash -lc 'sleep 5; date -u > /tmp/codex_setsid_survive_test.txt'
```

따라서 장기 trainer와 HF sync는 `setsid -f`로 별도 session에 분리해서 띄운다.

주의할 점도 있었다.

처음에는 로그 redirection을 다음처럼 잡았다.

```bash
> "$TRACE_DIR/../train.log"
```

하지만 redirection은 script 본문이 `TRACE_DIR`를 만들기 전에 shell이 먼저 처리한다. 이때 `traces/..` 경로의 중간 디렉토리인 `traces`가 아직 없으면 파일 열기가 실패해서 script가 시작 전 종료된다.

따라서 run 시작 전 다음을 보장한다.

- `RUN_DIR`, `TRACE_DIR`, `SANDBOX_ROOT`를 미리 만든다.
- 로그는 `"$RUN_DIR/train.log"`처럼 이미 존재하는 parent path로 직접 연다.
- 장기 trainer와 sync는 `setsid -f env ... bash -lc ...` 형태로 띄운다.

## 한계

이 패치는 Docker나 kernel-level sandbox를 완전히 대체하지 않는다.

여전히 한계는 있다.

- process namespace가 완전 격리되지 않는다.
- root filesystem mount가 Docker처럼 재구성되지 않는다.
- network syscall 자체를 커널 수준에서 차단하지 못한다.
- `/app`를 진짜 mount하는 것이 아니라 문자열 rewrite로 흉내 낸다.
- binary file 내부 경로는 rewrite하지 못한다.

그래도 현재 서버 조건에서는 가장 현실적인 개선이다.

특히 `/app`/`/tests` 호환성은 성능에도 직접적인 이득이 있을 가능성이 높다.

## 다음 권장 작업

1. 최신 checkpoint에서 patched sandbox run으로 재시작한다.
2. checkpoint-820 또는 그 이후를 기준으로 새 run을 만든다.
3. 100~200 step 정도 돌린 뒤 TB2-lite 중간 평가를 한다.
4. 기존 no-Docker run과 비교한다.
5. blocker 비율, verifier reward mean, timeout 비율, `/app` unsafe block 감소 여부를 본다.

성능을 높이려면 단순히 더 오래 돌리는 것보다, reward가 깨끗하게 들어가는 환경을 먼저 만드는 것이 중요하다.

이번 패치는 그 방향의 안정화 작업이다.
