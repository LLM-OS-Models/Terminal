# Terminal RL/DPO Plan After LFM2.5, Qwen 2B, and KoHRM Runs

작성일: `2026-06-06`

이 문서는 TB2-lite corrected 303-step replay 결과를 기준으로, 다음 학습을 단순 SFT 추가 epoch가 아니라 RL/DPO 계열로 넘길 때의 실행 기준을 정리한다. 결론부터 말하면 **LFM2.5와 Qwen 2B를 먼저 RL/DPO 대상으로 잡고, HRM/KoHRM은 serving/decoding 안정화와 JSON contract 연구를 먼저 한다.**

## 1. 현재 점수 기준

| 모델 | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | 해석 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch` | 52.30 | 0.5230 | 0.5854 | 0.5431 | 49.5% | 76.9% | 현재 전체 1위, RL baseline 1순위 |
| `LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-2Epoch` | 50.48 | 0.5048 | 0.5695 | 0.5296 | 49.2% | 74.9% | 1epoch보다 하락, 추가 epoch의 위험 신호 |
| `KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch2` | 45.90 | 0.4590 | 0.5031 | 0.5098 | 44.9% | 68.3% | KoHRM current best |
| `Qwen3.5-2B-Terminal-ToolCall-FullConv-FastContinue-1Epoch` | 44.79 | 0.4479 | 0.5266 | 0.4701 | 34.3% | 83.2% | JSON 강함, First Cmd 약함 |
| `KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch3` | 43.57 | 0.4357 | 0.4703 | 0.5003 | 45.5% | 61.7% | 추가 epoch에서 contract regression |
| `Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 39.52 | 0.3952 | 0.5082 | 0.4101 | 33.0% | 82.2% | fast-continue 이전 Qwen 2B 기준점 |

핵심 패턴:

- LFM2.5는 가장 강하지만 2epoch에서 1epoch보다 내려갔다. 단순 추가 SFT보다 reward/selection이 필요하다.
- Qwen 2B는 Valid JSON이 `83.2%`로 좋지만 First Cmd가 `34.3%`라 첫 행동 보상이 먹힐 여지가 크다.
- KoHRM은 Epoch2까지 크게 올랐지만 Epoch3에서 Valid JSON과 Precision이 내려갔다. 같은 SFT 데이터 반복은 이제 위험하다.

## 2. 왜 RL/DPO인가

TB2-lite Score는 JSON valid rate가 아니라 `100 * avg_command_f1`이다. 따라서 학습 목표는 다음 네 가지를 동시에 맞춰야 한다.

1. `commands[].keystrokes`가 reference command set과 많이 겹쳐야 한다.
2. 첫 command가 맞아야 한다.
3. JSON schema를 깨면 안 된다.
4. `task_complete=true`를 너무 빨리 내면 안 된다.

SFT는 reference trajectory를 그대로 밀어 넣는 데 강하지만, 모델이 이미 어느 정도 배운 뒤에는 두 문제가 생긴다.

- command를 더 안전하고 짧게 내는 쪽으로 수렴해서 Recall이 줄 수 있다.
- JSON 형식은 좋아져도 실제 command coverage가 늘지 않을 수 있다.
- 반대로 command를 많이 내게 만들면 JSON validity와 Precision이 흔들릴 수 있다.

그래서 다음 단계는 단순 next-token loss보다 **candidate output 간 선택**을 학습하는 쪽이 맞다. DPO/KTO/ORPO는 빠르게 시작하기 좋고, GRPO/PPO류 online RL은 rollout infrastructure가 안정된 뒤 쓰는 편이 낫다.

## 3. 우선순위

### 3.1 1순위: LFM2.5-8B-A1B 1epoch

이유:

- 현재 Score `52.30`으로 가장 높다.
- First Cmd `49.5%`, Recall `0.5431`, Valid JSON `76.9%`가 모두 강하다.
- 2epoch에서 점수가 내려간 전례가 있어, 무작정 더 학습하지 말고 reward로 약점만 보정해야 한다.
- vLLM/chat-template 경로가 안정적이라 rollout 생성과 평가 반복이 빠르다.

목표:

- Score `53~55` 구간 탐색.
- invalid JSON `70/303`과 empty command `19/303` 줄이기.
- `code` group, late bucket, premature `task_complete=true` 줄이기.

위험:

- JSON penalty를 너무 세게 걸면 모델이 짧고 안전한 command만 내서 Recall이 떨어진다.
- First Cmd reward를 과하게 주면 이후 command set coverage가 줄 수 있다.

### 3.2 2순위: Qwen3.5-2B fast-continue fullconv

이유:

- Score `44.79`로 Qwen 2B 계열 최고점이고 KoHRM Epoch2 `45.90`에 근접했다.
- Valid JSON `83.2%`라 형식 안정성은 이미 강하다.
- First Cmd `34.3%`가 낮아 reward로 개선할 공간이 크다.
- 모델이 작아 rollout/training 비용이 낮고 반복이 빠르다.

목표:

- First Cmd를 `34.3% -> 40%+`.
- Recall을 `0.4701 -> 0.50+`.
- Score를 KoHRM Epoch2 `45.90` 이상으로 올리는지 확인.

위험:

- JSON이 이미 강한 모델이라 JSON reward만 주면 점수 상승이 거의 없을 수 있다.
- first-command only 최적화가 전체 command F1을 깎을 수 있다.

### 3.3 HRM/KoHRM은 RL 직행보다 decoding/serving 안정화 먼저

이유:

- current best는 Epoch2 `45.90`이지만 Epoch3에서 Score `43.57`, Valid JSON `61.7%`로 내려갔다.
- PrefixLM 구조라 vLLM 일반 causal/chat fast path를 바로 타지 못한다.
- 로컬 evaluator는 SDPA/nocompile 조건에서 동작하지만 반복 실험 속도가 느리다.
- RL rollout을 대량 생성하려면 serving이 먼저 빨라져야 한다.

HRM에서 먼저 해야 할 일:

- Epoch2 checkpoint를 기준으로 고정.
- JSON schema constrained decoding 또는 output repair를 붙여 invalid JSON을 줄인다.
- `task_complete=true` 조기 출력 penalty/guard를 적용한다.
- stop token/EOA 조기 종료를 안정화해서 `max_tokens=1024` 전량 생성 낭비를 줄인다.
- PrefixLM 전용 batched generation을 만든 뒤 RL rollout으로 넘어간다.

## 4. Reward 설계

기본 reward는 TB2-lite scoring과 같은 방향이어야 한다.

```text
reward =
  1.00 * command_f1
  + 0.15 * first_cmd_exact
  + 0.10 * valid_json
  - 0.15 * invalid_json
  - 0.10 * premature_task_complete
  - 0.05 * empty_commands
  - 0.03 * excessive_explanation_or_think_leak
```

세부 설계:

- `command_f1`: 핵심 reward. Precision과 Recall 중 하나만 높아서는 안 된다.
- `first_cmd_exact`: 첫 행동 선택은 downstream step 전체에 영향을 준다.
- `valid_json`: 필요하지만 과대 가중하면 짧은 안전 출력으로 수렴한다.
- `premature_task_complete`: reference가 command를 요구하는데 task_complete만 내는 경우 강하게 벌점.
- `empty_commands`: JSON은 valid지만 `commands=[]`인 경우 벌점.
- `think_leak`: JSON 앞뒤 설명문, `<think>`, markdown fence가 붙으면 벌점.

권장 가중치:

- LFM2.5: command_f1 중심, valid_json은 보조. 이미 점수가 높으므로 작은 보정으로 시작한다.
- Qwen 2B: first_cmd_exact와 command_recall을 조금 더 준다. JSON은 이미 높으므로 과하게 주지 않는다.
- HRM: valid_json/contract reward를 더 강하게 둘 수 있지만, RL 전에는 decoding guard부터 한다.

## 5. 데이터 구성

TB2-lite 303-step corrected replay는 개발/평가 기준으로 유지한다. RL 학습 데이터는 이 평가셋을 그대로 암기시키면 안 된다.

권장 데이터:

- LFM25/ToolBench terminal/toolcall train trajectories.
- 기존 full-conversation terminal/toolcall data.
- late-step/error-recovery 중심 추가 샘플.
- dependency/security/model_training 약점 영역 oversampling.

Preference pair 생성:

1. 현재 SFT 모델로 rollout을 여러 개 생성한다.
2. 각 rollout을 같은 parser/scorer로 채점한다.
3. reference 또는 높은-score output을 chosen으로 둔다.
4. invalid JSON, low F1, premature complete output을 rejected로 둔다.
5. 같은 prompt 안에서 chosen/rejected 길이 차이가 너무 큰 pair는 제외한다.

Rollout 조건:

- temperature `0.0`, `0.2`, `0.5`를 섞되 평가 기준은 t=0.0 유지.
- max_tokens는 현재 `1024` 기준을 유지하되, stop/EOA가 안정되면 줄인다.
- candidate 수는 prompt당 `4~8`개부터 시작한다.

## 6. 학습 순서

### Phase A: Offline DPO

목표는 빠른 검증이다.

- 대상: LFM2.5 1epoch, Qwen 2B fast-continue.
- GPU: 각 모델별 4GPU 또는 8GPU, 데이터 크기에 따라 선택.
- LR: SFT보다 작게 시작. 너무 큰 LR은 JSON contract를 깬다.
- Epoch: 0.25~1.0 epoch sweep.
- checkpoint마다 TB2-lite full replay로 검증.

성공 기준:

- LFM2.5: Score `+1.0` 이상 또는 Valid JSON 개선과 Score 유지.
- Qwen 2B: Score `+1.5` 이상 또는 First Cmd `+3%p` 이상.

### Phase B: Rejection Sampling SFT

DPO 전에 더 간단히 볼 수 있는 방법이다.

- rollout 중 score가 높은 output만 골라 SFT.
- invalid JSON과 empty command는 제거.
- reference보다 command F1이 높은 generated output은 유지.

장점:

- 구현이 쉽다.
- reward model 없이 바로 가능하다.

단점:

- 다양성이 줄어들 수 있다.
- over-selection으로 command recall이 줄 수 있다.

### Phase C: Online GRPO/PPO류

vLLM rollout이 안정된 모델에만 적용한다.

- LFM2.5와 Qwen 2B는 가능성이 높다.
- HRM은 serving이 느려서 후순위다.
- reward hacking 방지를 위해 JSON validity만 보상하지 않는다.

## 7. 평가와 게이트

항상 full 303-step replay 기준으로 판단한다. partial 점수는 진행 추정용으로만 쓴다.

필수 기록:

- Score
- Cmd F1
- Precision
- Recall
- First Cmd
- Valid JSON
- invalid JSON count
- empty command count
- average predicted commands
- source group F1
- early/mid/late bucket F1
- sec/step

배포 기준:

- LFM2.5는 1epoch Score `52.30`을 넘지 못하면 대표를 바꾸지 않는다.
- Qwen 2B는 fast-continue Score `44.79`와 KoHRM Epoch2 `45.90` 둘 다 비교한다.
- HRM은 Epoch2 `45.90`이 기준이고, Epoch3 `43.57`보다 높아도 Epoch2를 못 넘으면 대표를 바꾸지 않는다.

## 8. 지금 결정

1. NVIDIA Nemotron GGUF 평가는 계속 돌려 외부 strong model 기준을 확보한다.
2. 그 다음 학습 실험은 LFM2.5 1epoch와 Qwen 2B fast-continue를 RL/DPO 우선순위로 둔다.
3. HRM은 Epoch2를 대표로 유지하고, 추가 SFT epoch보다 JSON/PrefixLM serving 연구로 분리한다.
4. 모든 결과는 README와 모델카드에 score-first로 반영하고, 결과 JSON과 문서 변경은 커밋한다.
