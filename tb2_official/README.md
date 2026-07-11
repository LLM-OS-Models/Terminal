# LFM2.5 Terminal-Bench 2.0 공식 재현 기록

> 상태: 진행 중
> 시작: 2026-07-11 KST
> 작업 디렉터리: `/home/ubuntu/Terminal/tb2_official`

이 문서는 `Terminal` 저장소의 TB2-lite 주장과 실제 Docker 기반 Terminal-Bench 2.0 성능을 구분하여 재현하기 위한 실행 기록이다. 설치 버전, 모델 revision, 실행 명령, 실패 원인, 결과를 모두 이 파일에 계속 누적한다.

관리자용 해결·재개 절차는 `docs/TB2_OFFICIAL_DOCKER_ADMIN_RUNBOOK_KO_20260711.md`에 정리했다.

## 현재 결론

- 아직 유효한 Terminal-Bench 2.0 점수는 없다.
- 공식 하네스 Harbor 설치와 TB2.0 데이터셋 해석은 성공했다.
- SFT 본체와 두 pinned LoRA를 H100의 단일 vLLM 서버에 함께 올렸고, 세 model ID 모두 OpenAI-compatible chat completion과 Terminus-2 JSON 파싱을 통과했다.
- 공식 문서대로 oracle 1-task 스모크를 실행했지만, 모델이나 Harbor가 아니라 현재 작업 컨테이너의 중첩 Docker 권한 때문에 task container 생성 전에 실패했다.
- Harbor 출력의 `Mean: 0.000`은 모델 점수 0점이 아니다. `Trials=0`, `Exceptions=1`인 인프라 오류이므로 성능표에 사용하면 안 된다.

## 조사 대상

1. SFT 모델
   - `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
2. static/offline ECHO RLVR adapter 모음
   - `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-GRPO-Adapters`
3. online ECHO RLVR adapter 모음
   - `LLM-OS-Models/LFM2.5-8B-A1B-SFT1-Online-ECHO-RLVR-GRPO-Adapters`

공식 평가에서는 최소 다음 세 구성을 비교할 예정이다.

- SFT 1Epoch 단독
- SFT 1Epoch + static/offline ECHO RLVR의 README 최고 checkpoint
- SFT 1Epoch + online ECHO RLVR의 README 최고 checkpoint

## 로컬 저장소 해석

`/home/ubuntu/Terminal`은 정식 Terminal-Bench 2.0 실행 저장소가 아니다. 학습 코드와 `tb2_lite`라는 자체 proxy evaluator가 중심이다.

- `tb2_lite`는 Docker 환경에서 태스크를 실제 수행해 pass/fail을 채점하지 않는다.
- 저장된 terminal trajectory의 다음 명령을 예측하고 command token F1을 계산한다.
- README의 `52.30`, `54.05`, `53.58`은 303-step TB2-lite replay 점수이며 공식 TB2.0 pass@1 백분율이 아니다.
- 따라서 README의 “타 모델 압도” 문구는 같은 로컬 TB2-lite proxy 안에서만 성립하며, 공식 TB2.0 성능으로 읽으면 안 된다.

관련 로컬 코드:

- `tb2_lite/scripts/replay_eval.py`: vLLM/PEFT 기반 replay 생성
- `tb2_lite/scripts/prompt_builder.py`: chat template 및 fallback prompt
- `tb2_lite/scripts/replay_metrics.py`: command F1/JSON 채점
- `Liquid-CLI/scripts/run_lfm25_vllm_server_clean.sh`: LFM2.5 vLLM 단일 서버
- `Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh`: LoRA를 활성화할 수 있는 vLLM replica 실행

## 하드웨어와 기본 환경

- GPU: NVIDIA H100 80GB HBM3 1장
- GPU driver: `580.105.08`
- 표시 CUDA 호환 버전: `13.0`
- CPU: 124 logical CPUs
- RAM: 약 1.73 TiB
- `/home/ubuntu` 여유 디스크: 약 1.6 TiB
- OS: Ubuntu 22.04.5 LTS
- Docker client/server: `29.6.1`
- Docker Compose: `v5.3.1`
- Docker storage driver: `vfs`

`ubuntu` 사용자를 `docker` 그룹에 추가했다. 현재 셸에서는 `sg docker -c '<command>'`로 새 그룹을 적용한다.

## 설치 기록

시스템 Python에 venv 지원이 없어 다음 패키지를 설치했다.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
```

도구는 기존 프로젝트 Python과 분리했다.

- uv: `0.11.28`
- uv 실행 환경: `/home/ubuntu/.venvs/uv`
- uv-managed Python: CPython `3.12.13`
- Harbor: `0.18.0`
- Harbor 실행 파일: `/home/ubuntu/.tools/bin/harbor`
- Harbor tool dir: `/home/ubuntu/.tools/uv/harbor`
- vLLM environment: `/home/ubuntu/.venvs/tb2-vllm`
- vLLM: `0.19.1`
- PyTorch: `2.10.0+cu128`
- Transformers: `5.13.1`
- PEFT: `0.19.1`
- Hugging Face Hub: `1.23.0`

vLLM 환경에서 H100/CUDA 인식도 확인했다.

```text
torch.cuda.is_available() = True
GPU = NVIDIA H100 80GB HBM3
```

설치 명령:

```bash
python3 -m venv /home/ubuntu/.venvs/uv
/home/ubuntu/.venvs/uv/bin/pip install --upgrade uv

UV_TOOL_BIN_DIR=/home/ubuntu/.tools/bin \
UV_TOOL_DIR=/home/ubuntu/.tools/uv \
UV_PYTHON_INSTALL_DIR=/home/ubuntu/.tools/python \
/home/ubuntu/.venvs/uv/bin/uv tool install --python 3.12 harbor

/home/ubuntu/.venvs/uv/bin/uv venv \
  --python /home/ubuntu/.tools/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 \
  /home/ubuntu/.venvs/tb2-vllm

/home/ubuntu/.venvs/uv/bin/uv pip install \
  --python /home/ubuntu/.venvs/tb2-vllm/bin/python \
  'vllm==0.19.1' \
  'peft==0.19.1' \
  'huggingface_hub>=1.0'
```

## adapter revision 주의사항

HF adapter 저장소는 run별 namespace 없이 여러 학습 run이 모두 `checkpoints/checkpoint-N` 경로를 사용했다. 같은 번호가 후속 run에서 덮였으므로 `main/checkpoints/checkpoint-610`만 받으면 README 1위 adapter가 아니다.

확인된 이력:

- README의 static/offline 최고 결과: parent run `checkpoint-610`, TB2-lite `54.05`
- 해당 adapter를 처음 업로드한 HF commit: `dbd53977c71f1aa842ae6ed558ad5d2221d441ff`
- 후속 continuation run이 같은 `checkpoint-610` 경로를 덮은 HF commit: `4e3681739bcb4d0c31cfe789023458f98541d8e7`
- 현재 `main/checkpoints/checkpoint-610`의 adapter config는 SFT base를 가리키지만, 문서상 continuation checkpoint의 TB2-lite 점수는 `50.12`다.

따라서 static/offline README 최고점을 재현할 때는 반드시 아래처럼 revision을 고정해야 한다.

```text
repo: LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-GRPO-Adapters
revision: dbd53977c71f1aa842ae6ed558ad5d2221d441ff
subfolder: checkpoints/checkpoint-610
base: LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch
```

온라인 최고 결과는 문서상 `checkpoint-425`, TB2-lite `53.58`이며 base는 SFT 1Epoch다. 이 checkpoint도 최종 실행 전에 HF commit과 파일 hash를 고정한다.

고정 manifest:

| 구성 | HF revision | weight LFS SHA-256 | 크기 |
| --- | --- | --- | ---: |
| SFT 1Epoch | `4ca70e4065d10b171de0a434185a9c436cbe9893` | `c742791bb761580d2b339e946ea8b5f1b8e9b6f742e5ab29fca07b3255f66ead` | 16,936,006,912 B |
| static parent `checkpoint-610` | `dbd53977c71f1aa842ae6ed558ad5d2221d441ff` | `cca35fd3d624615d67589edb5dee2e70d50479be5a470acdf4b1677d958fdb9e` | 51,493,768 B |
| static `main/checkpoint-610` (사용 금지) | `main` | `2d546a2728c3e8190141b5ae89329cbc17ad2e4d677cb77e5b0be31873ae6724` | 51,493,768 B |
| online `checkpoint-425` | `dae4ec9466fc2b1c233d66d86271236a9acc9986` | `515a9cc24d0be1f30f824ea82c1099372efe37489fac90794ddc66478854a55f` | 51,493,768 B |

online `checkpoint-425`는 확인 시점의 `main`과 LFS hash가 같아 덮어쓰기 오염이 없었다. 그래도 평가 manifest에는 최초 업로드 commit을 고정한다.

## vLLM 및 Terminus-2 연결 검증

`run_vllm.sh`는 한 H100에 다음 세 ID를 동시에 제공한다.

- `lfm25-sft1`: SFT 1Epoch base
- `lfm25-static-610`: pinned static/offline parent LoRA
- `lfm25-online-425`: pinned online LoRA

```bash
./tb2_official/run_vllm.sh
curl -sS http://127.0.0.1:8000/v1/models | python3 -m json.tool
python3 ./tb2_official/smoke_vllm.py
```

실측 검증 결과:

- vLLM이 `Lfm2MoeForCausalLM`을 native architecture로 인식했다.
- base weight 로드는 약 `15.79 GiB`, LoRA 활성화 서버는 약 `16.85 GiB` GPU memory를 사용했다.
- 세 ID 모두 chat completion `finish_reason=stop`으로 정상 종료했다.
- Harbor 0.18.0의 실제 `terminus-json-plain.txt` 전체 prompt로 세 ID를 호출했다.
- 세 응답 모두 balanced JSON을 추출해 `analysis`, `plan`, `commands`와 newline이 포함된 `keystrokes`를 파싱했다.
- LFM의 `<think>...</think>`는 JSON 앞의 extra text 경고만 만들며, 현행 Terminus-2 parser는 뒤의 첫 balanced JSON을 정상 추출한다.

평가 설정은 현재 데이터셋 revision `1`, Terminus-2 JSON parser, 모델 카드의 TB2-lite 권장값과 같은 `temperature=0`, 32K input context, turn당 최대 4096 output tokens로 고정한다. `run_tb2.sh` 기본값은 권한 복구 후 안전한 1-task/1-attempt smoke다.

```bash
./tb2_official/run_tb2.sh lfm25-sft1

# 89-task 1회 full pass
TB2_LIMIT= TB2_ATTEMPTS=1 TB2_CONCURRENCY=4 \
  ./tb2_official/run_tb2.sh lfm25-sft1

# 공식 leaderboard와 같은 task당 5회: 89 x 5 = 445 trials
TB2_LIMIT= TB2_ATTEMPTS=5 TB2_CONCURRENCY=4 \
  ./tb2_official/run_tb2.sh lfm25-sft1
```

같은 명령의 model ID만 `lfm25-static-610`, `lfm25-online-425`로 바꾸면 동일 agent/sampling/timeout 조건으로 비교된다.

## 공식 Harbor oracle 스모크

공식 문서의 5-task oracle smoke에서 먼저 1개 task만 실행했다.

```bash
mkdir -p /home/ubuntu/Terminal/tb2_official/jobs

sg docker -c '/home/ubuntu/.tools/bin/harbor run \
  -d terminal-bench/terminal-bench-2@1 \
  -a oracle \
  -l 1 \
  -n 1 \
  --yes \
  --job-name oracle-smoke \
  --jobs-dir /home/ubuntu/Terminal/tb2_official/jobs'
```

결과:

- Harbor가 `terminal-bench/terminal-bench-2`와 첫 태스크 `make-mips-interpreter`를 정상 해석했다.
- Docker image 다운로드까지 진행했다.
- 레이어 압축 해제 단계에서 `failed to register layer: unshare: operation not permitted`로 실패했다.
- 결과 JSON: `jobs/oracle-smoke/result.json`
- 상세 예외: `jobs/oracle-smoke/make-mips-interpreter__3eVD72Q/exception.txt`

원인은 현재 머신이 이미 제한된 상위 Docker container 안에서 실행되고 있기 때문이다.

- PID 1 환경은 container로 감지된다.
- effective/bounding capability에 `CAP_SYS_ADMIN`이 없다.
- seccomp filter가 활성화되어 있다.
- `unshare -Ur true`와 `unshare -m true`가 모두 `Operation not permitted`다.
- root/sudo도 bounding capability 밖의 권한을 얻을 수 없으므로, 이 컨테이너 안에서 Docker daemon 옵션만 바꿔 해결할 수 있는 문제가 아니다.

필요한 조치 중 하나:

1. 이 작업 컨테이너를 nested Docker가 가능한 privileged 모드로 다시 시작
2. 호스트 Docker socket을 정상적으로 전달하고 task container 생성 권한 부여
3. Harbor가 지원하는 Compose-capable 외부 sandbox 제공 후 전체 oracle 검증

권한이 해결되면 먼저 아래 oracle 5-task가 통과해야 한다.

```bash
sg docker -c '/home/ubuntu/.tools/bin/harbor run \
  -d terminal-bench/terminal-bench-2@1 \
  -a oracle \
  -l 5 \
  -n 1 \
  --yes \
  --job-name oracle-smoke-5 \
  --jobs-dir /home/ubuntu/Terminal/tb2_official/jobs'
```

## 다음 작업

- [x] SFT와 두 best adapter의 HF revision/hash manifest 고정
- [x] H100에 맞는 vLLM 전용 환경 설치
- [x] SFT 단독 OpenAI-compatible endpoint 기동 및 chat completion 확인
- [x] pinned static/offline checkpoint-610 endpoint 확인
- [x] pinned online checkpoint-425 endpoint 확인
- [x] Harbor Terminus-2가 local vLLM endpoint에 접근하도록 config 작성 및 parser 확인
- [ ] Docker 권한 해결 후 oracle 5-task 통과
- [ ] 각 모델 동일 task subset smoke 평가
- [ ] 89-task, 동일 agent/prompt/sampling/timeout으로 전체 평가
- [ ] pass@1, 오류율, task별 차이와 TB2-lite 순위의 상관관계 분석

## 참고 링크

- Terminal-Bench 2.0 실행 문서: https://www.tbench.ai/docs/run-terminal-bench-2-0
- Harbor Terminal-Bench 문서: https://www.harborframework.com/docs/tutorials/running-terminal-bench
- Harbor Terminus-2 문서: https://www.harborframework.com/docs/agents/terminus-2
- Terminal-Bench 2.0 dataset: https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2
- SFT 모델: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch
- static/offline adapters: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-GRPO-Adapters
- online adapters: https://huggingface.co/LLM-OS-Models/LFM2.5-8B-A1B-SFT1-Online-ECHO-RLVR-GRPO-Adapters
