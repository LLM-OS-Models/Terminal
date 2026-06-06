# Harness-1 -> LFM2.5 Training Plan

작성 시각: `2026-06-06`

## 결론

`LiquidAI/LFM2.5-8B-A1B`로 Harness-1식 검색 에이전트 학습을 시도할 수 있다. 다만 논문 핵심은 full fine-tuning이 아니라 `LoRA SFT warm-start -> on-policy RL`이다. 그래서 로컬 준비도 full SFT가 아니라 `all-linear LoRA r32`로 맞춘다.

현재 구현된 것은 바로 학습 가능한 SFT warm-start 파이프라인이다.

- Harness-1 trajectory JSON/JSONL -> LFM conversations 변환기
- LFM2.5용 `all-linear` LoRA SFT trainer
- H200 8GPU 설정 파일
- NVIDIA 장기 평가가 끝난 뒤 자동으로 LFM Harness SFT를 시작하는 chain script
- BrowseComp+ public source clone 준비 스크립트

RL은 별도 단계다. Harness-1의 RL은 live retrieval environment, reward computation, rollout sampling, Chroma backend가 필요하므로 단순 데이터셋 변환만으로 끝나지 않는다. SFT checkpoint를 만든 뒤 SEC-family query rollout 환경을 붙이는 것이 다음 작업이다.

## 논문에서 확인한 학습 레시피

논문/공개 코드 기준 Harness-1은 다음 구조다.

| Stage | 설정 |
| --- | --- |
| Base model | `openai/gpt-oss-20b` |
| SFT | LoRA rank `32`, LR `5e-6`, batch `128`, max sequence `32768`, `3` epochs |
| SFT data | GPT-5.4 teacher가 Harness 환경 안에서 생성한 trajectory |
| SFT 규모 | raw 약 `1K`, final recall `0.10` gate 이후 `899` trajectories |
| SFT expansion | trajectory별 turn-conditional datum으로 확장, 약 `26K` examples |
| SFT selected checkpoint | `step 550` sampler weights |
| RL | LoRA rank `32`, on-policy CISPO, LR `1e-5` |
| RL data/domain | SEC train split `3,453` queries |
| RL rollout | group `8`, `1024` rollouts/step, `80` steps, 총 약 `82K` rollouts |
| RL turn cap | `40` turns |
| RL reward | terminal reward, set recall/trajectory recall/final-answer evidence/tool diversity/turn penalty 계열 |

SFT와 RL 데이터는 같지 않다. SFT는 BC+/Web/Patents/SEC를 섞어 Harness interface 사용법을 가르치고, RL은 SEC에서 긴 문서 검색/큐레이션 전략을 실제 rollout으로 최적화한다. 논문이 강조하는 전이는 SEC RL만으로도 다른 도메인에서 개선되는 점이다.

GPU 모델, 실제 wall time, 실제 비용은 논문에 공개되어 있지 않다. checklist에서는 Tinker managed training service를 사용했기 때문에 물리 GPU worker 종류와 메모리 할당 같은 low-level compute가 플랫폼에 의해 추상화됐다고 설명한다. 따라서 논문 수치로 "몇 시간/몇 달러"를 확정할 수는 없다.

## LFM2.5에 맞춘 설계

LFM2.5 문서의 ChatML 계열 template와 tool-use 흐름에 맞춰 다음처럼 변환한다.

- system: Harness-1 검색 에이전트 역할과 도구 설명
- user: 원 query, 사용 가능한 도구, 이전 Harness state/turn 요약
- assistant: 짧은 reasoning + `tool_calls` JSON
- 기존 terminal SFT pipeline처럼 `tokenizer.apply_chat_template()`를 사용
- `train_on_responses_only()`로 assistant target만 학습

학습 포맷은 full trajectory transcript를 그대로 늘어놓는 방식이 아니라, 논문처럼 각 turn을 하나의 supervised sample로 확장한다. 이전 turn은 user context 안에 최근 `5`턴 상세 + 오래된 turn summary로 넣고, target은 현재 turn의 next Harness action만 둔다. 이렇게 해야 모델이 "현재 state에서 다음 tool action을 고르는 정책"으로 학습된다.

Tool call은 LFM chat template가 받아들일 수 있게 OpenAI-style function call JSON으로 저장한다.

```json
[
  {
    "type": "function",
    "function": {
      "name": "fan_out_search",
      "arguments": {
        "queries": ["..."]
      }
    }
  }
]
```

## 추가된 파일

| 파일 | 역할 |
| --- | --- |
| `Liquid-CLI/scripts/build_lfm_harness1_dataset.py` | Harness trajectory를 LFM conversations dataset으로 변환 |
| `Liquid-CLI/train_unsloth_processed_lora.py` | processed conversations를 `all-linear` LoRA로 SFT |
| `Liquid-CLI/configs/sft_h200_8gpu_lfm25_8b_a1b_harness1_lora.env` | H200 8GPU Harness SFT 설정 |
| `Liquid-CLI/scripts/run_lfm_harness1_lora_sft.sh` | trajectory 생성 옵션, dataset build, LoRA SFT 실행 |
| `Liquid-CLI/scripts/chain_nemotron_then_lfm_harness1_sft.sh` | NVIDIA 평가 완료 감지 후 SFT 실행 |
| `Liquid-CLI/scripts/prepare_harness1_sources.sh` | BrowseComp+ public source 준비 |

## 데이터 상태

준비 완료:

- Harness-1 source tree: `/home/work/.projects/LLM-OS-Models/harness-1`
- BrowseComp+ public repo clone: `/home/work/.data/harness1/external/BrowseComp-Plus`
- LFM Harness SFT trajectory target dir: `/home/work/.data/harness1/sft_data`

아직 필요한 것:

- `/home/work/.projects/LLM-OS-Models/harness-1/.env.local`
- BrowseComp+ decrypted answer file path
- Chroma corpus/index
- `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE` 등 Harness generator가 요구하는 credentials

중요: `pat-jj/harness-1` Hugging Face model repo에는 모델 weights만 있고 SFT trajectory dataset은 들어있지 않다. 따라서 공개 trajectory를 단순 다운로드해서 바로 학습하는 경로는 현재 확인되지 않았다. 학습 데이터는 upstream generator로 생성하거나, 별도 확보한 trajectory JSON/JSONL을 `HARNESS_INPUT`으로 넣어야 한다.

## 실행 방법

소스 준비:

```bash
bash Liquid-CLI/scripts/prepare_harness1_sources.sh
```

trajectory가 이미 있는 경우:

```bash
HARNESS_INPUT=/path/to/harness_sft_json_or_dir \
bash Liquid-CLI/scripts/run_lfm_harness1_lora_sft.sh \
  --config Liquid-CLI/configs/sft_h200_8gpu_lfm25_8b_a1b_harness1_lora.env
```

Harness-1 generator로 trajectory 생성까지 같이 하는 경우:

```bash
bash Liquid-CLI/scripts/run_lfm_harness1_lora_sft.sh \
  --config Liquid-CLI/configs/sft_h200_8gpu_lfm25_8b_a1b_harness1_lora.env \
  --generate-trajectories
```

NVIDIA GGUF 평가 완료 후 자동으로 학습 시작:

```bash
nohup bash Liquid-CLI/scripts/chain_nemotron_then_lfm_harness1_sft.sh \
  --train-config Liquid-CLI/configs/sft_h200_8gpu_lfm25_8b_a1b_harness1_lora.env \
  > /home/work/.data/liquid_cli_sft/logs/chain_nemotron_then_harness1_sft.manual.log 2>&1 &
```

현재는 `.env.local`/trajectory가 없으면 chain이 학습 단계에서 멈춘다. 무의미한 synthetic 데이터로 H200 8대를 태우지 않기 위해 이 fail-fast가 맞다.

## 예상 시간

논문은 실제 GPU/time/cost를 공개하지 않았으므로 아래는 우리 H200 8GPU 기준 추정이다.

SFT:

- 899 trajectories -> 약 26K turn examples가 논문 규모다.
- 설정상 effective batch는 `8 * 1 * 16 = 128`, 논문 batch와 맞다.
- 3epoch면 optimizer step은 대략 `26K / 128 * 3 ~= 610`이다.
- 평균 context가 8K 근처면 수 시간 이하, 16K~32K에 가까우면 수 시간에서 반나절까지 볼 수 있다.
- 우리 LFM2.5-8B-A1B는 20B gpt-oss보다 작고 activated params도 작지만, 32K context와 LoRA all-linear라 메모리/attention 비용은 무시하면 안 된다.

RL:

- 논문 규모 그대로면 약 `82K` rollout, 각 rollout 최대 `40` turn이다.
- live retrieval, Chroma, rerank/verify, generation이 병목이라 단순 SFT보다 훨씬 오래 걸린다.
- H200 8GPU에서 모델 inference 자체는 줄어도 retrieval/API가 끼면 며칠 단위로 보는 게 안전하다.
- RL은 GPU를 놀리지 않기보다 rollout producer/learner/reward evaluator를 분리해서 queue를 계속 채우는 구조가 필요하다.

## 왜 이 작업이 LFM2.5에 맞는가

LFM2.5-8B-A1B는 terminal TB2-lite에서 base Score `36.53`, ToolBench Full SFT 1epoch Score `52.30`으로 이미 tool/action 형식에 잘 반응했다. Harness-1도 본질은 "긴 transcript를 외우는 모델"이 아니라 "정리된 state에서 다음 tool action을 고르는 모델"을 만드는 작업이다. 따라서 LFM2.5에 맞춘 LoRA SFT warm-start는 충분히 해볼 가치가 있다.

다만 SFT만으로 논문 성능을 기대하면 안 된다. 논문에서도 SFT는 interface discipline이고, 성능의 핵심은 SEC에서의 on-policy RL이다. SFT 결과가 나오면 다음 순서는 다음과 같다.

1. LFM2.5 LoRA SFT checkpoint를 Harness-1 inference 경로에 붙인다.
2. BrowseComp+/SEC smoke eval로 tool-call parsing과 state update가 맞는지 확인한다.
3. SEC train query에서 rollout generator를 돌린다.
4. reward는 curated recall, trajectory recall, answer evidence miss penalty, tool diversity, turn penalty를 재현한다.
5. GRPO/GDPO로 근사하지 말고 가능하면 Harness episode reward를 직접 쓰는 on-policy loop를 만든다.

## 리스크

- trajectory dataset이 공개되어 있지 않아 generator 환경 구축이 먼저다.
- Chroma index 없이는 search/grep/read가 의미 있게 동작하지 않는다.
- LFM template는 Harness-1의 GPT-OSS Harmony template와 다르므로 tool-call 문자열이 그대로 호환되지 않는다. 이번 변환기는 LFM 문서의 JSON tool-call 스타일에 맞춰 `tool_calls` 필드로 정규화한다.
- SFT는 좋아도 RL reward/rollout이 없으면 Harness-1 논문 성능의 핵심을 재현했다고 볼 수 없다.
- 32K context LoRA는 빠른 8B 모델이라도 데이터 길이가 길면 생각보다 오래 걸릴 수 있다.
