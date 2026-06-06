# KoHRM / HRM Usage Research Notes

작성일: `2026-06-06`

이 문서는 KoHRM-Text/HRM 계열을 terminal next-action 모델로 어떻게 써야 하는지, 왜 Epoch3에서 성능이 내려갔는지, 앞으로 무엇을 연구해야 하는지 정리한다. 결론은 **KoHRM은 terminal/tool action을 배울 수 있지만, 현재는 Epoch2를 대표 checkpoint로 유지하고, 추가 epoch보다 PrefixLM 전용 decoding/serving과 JSON contract 안정화를 먼저 해야 한다**는 것이다.

## 1. 모델과 평가 조건

KoHRM-Text는 일반 chat causal LM이 아니라 HRM-Text PrefixLM stack 기반 모델이다. 따라서 일반 vLLM chat-template 모델처럼 바로 서빙/평가하기 어렵다.

현재 평가 조건:

- 평가셋: corrected TB2-lite 303-step full replay.
- 점수: `Score = 100 * avg_command_f1`.
- 보조 지표: Precision, Recall, First Cmd, Valid JSON.
- KoHRM full SFT evaluator: local PrefixLM export runtime.
- Epoch3 최종 평가 조건: `KOHRM_FORCE_SDPA_KVCACHE=1`, `KOHRM_DISABLE_INFERENCE_COMPILE=1`.
- 이유: local flash-attention build가 append-KV cache를 지원하지 않았고, compile guard error가 있어 SDPA + no-compile이 안정 경로였다.

## 2. 현재 성능 요약

| 모델 | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | 비고 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `KoHRM-Text-1.4B-stage4d direct` | 11.48 | 0.1148 | 0.1995 | 0.0961 | 5.9% | 38.9% | base direct |
| `terminal-tool-core-r64 LoRA` | 29.11 | 0.2911 | 0.3988 | 0.2768 | 22.1% | 63.4% | 최고 LoRA |
| `terminal-comp-jsonfix-r64 LoRA` | 28.80 | 0.2880 | 0.3834 | 0.2878 | 24.4% | 66.7% | JSON+terminal |
| `behavior-jsonfix-r32 LoRA` | 26.23 | 0.2623 | 0.3807 | 0.2507 | 21.1% | 71.0% | JSON은 최고지만 Score는 낮음 |
| `Top2 Terminal Tool Merge Epoch1` | 31.59 | 0.3159 | 0.3859 | 0.3415 | 24.8% | 73.3% | full SFT, top2 merge |
| `LFM25 Terminal ToolBench Epoch1` | 38.56 | 0.3856 | 0.4262 | 0.4341 | 37.0% | 55.1% | LFM25 data 1 pass |
| `LFM25 Terminal ToolBench Epoch2` | 45.90 | 0.4590 | 0.5031 | 0.5098 | 44.9% | 68.3% | current best |
| `LFM25 Terminal ToolBench Epoch3` | 43.57 | 0.4357 | 0.4703 | 0.5003 | 45.5% | 61.7% | 추가 epoch에서 하락 |

대표 checkpoint:

```text
LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch2
```

Epoch3는 배포 대표로 쓰지 않는다.

## 3. JSON 안정성은 올랐나

Epoch3에서는 오르지 않았다.

```text
Epoch1 Valid JSON: 55.1%
Epoch2 Valid JSON: 68.3%
Epoch3 Valid JSON: 61.7%
```

Epoch2에서는 command coverage와 JSON validity가 같이 올랐다. 이때는 추가 SFT가 실제 terminal action 분포와 출력 contract를 동시에 개선했다.

Epoch3에서는 다른 양상이 나왔다.

- First Cmd는 `44.9% -> 45.5%`로 소폭 상승했다.
- Recall은 `0.5098 -> 0.5003`으로 약간 하락했다.
- Precision은 `0.5031 -> 0.4703`으로 크게 하락했다.
- Valid JSON은 `68.3% -> 61.7%`로 내려갔다.
- Score는 `45.90 -> 43.57`로 내려갔다.

즉 첫 행동 감각은 유지됐지만, JSON schema를 끝까지 닫고 정확한 command set을 복원하는 능력이 약해졌다.

## 4. 왜 Epoch3가 내려갔나

가장 가능성이 높은 해석은 **contract overfit**이다.

근거:

- train loss는 `0.5271`, train accuracy는 `0.86567`까지 좋아졌다.
- 하지만 held-out replay Score와 Valid JSON은 내려갔다.
- 같은 LFM25/ToolBench target을 한 번 더 반복하면서 모델이 더 정확해진 것이 아니라, 일부 command 후보와 완료 판단을 더 좁게 잡은 것으로 보인다.
- PrefixLM 출력 contract가 chat model보다 얇게 강제되어 있어, 반복 학습이 JSON closing, array 유지, `task_complete` 판단을 안정화하지 못했다.

전형적인 암기형 과적합과 다른 점:

- Recall은 크게 무너지지 않았다. `0.5098 -> 0.5003`으로 소폭 하락이다.
- First Cmd는 오히려 소폭 올랐다.
- 크게 내려간 쪽은 Precision과 Valid JSON이다.

따라서 단순히 "데이터를 외웠다"기보다, **더 자신 있게 잘못된 command를 넣거나 JSON contract를 흔드는 방향으로 출력 분포가 이동했다**고 보는 게 맞다.

## 5. KoHRM이 잘하는 것

1. LoRA 반응이 크다.
   - base direct Score `11.48`에서 최고 LoRA `29.11`까지 오른다.
   - terminal/tool 데이터가 들어가면 출력 계약을 상당히 배운다.

2. full SFT 반응이 더 크다.
   - top2 full SFT `31.59`.
   - LFM25 Epoch1 `38.56`.
   - LFM25 Epoch2 `45.90`.

3. 1.4B급 모델로 Qwen 2B fast-continue에 근접하거나 넘는다.
   - KoHRM Epoch2 `45.90`.
   - Qwen 2B fast-continue `44.79`.

4. command recall은 작은 모델치고 강하다.
   - Epoch2 Recall `0.5098`.
   - Qwen 2B fast-continue Recall `0.4701`.

## 6. KoHRM이 못하는 것

1. vLLM fast path를 바로 타지 못한다.
   - HRM PrefixLM stack이 일반 HF causal/chat model 구조와 다르다.
   - 반복 평가/rollout 비용이 크다.

2. JSON contract가 불안정하다.
   - Epoch2는 `68.3%`까지 올랐지만 Epoch3에서 `61.7%`로 하락했다.
   - JSON-fix LoRA는 Valid JSON을 올리지만 command F1을 충분히 올리지는 못했다.

3. late bucket과 repo 수정형 task가 약하다.
   - 후반 수정/검증/마무리 command를 생략하거나 짧게 낸다.
   - `swe`, `dependency_management`, `security`, `data_processing` 계열이 약하다.

4. 속도가 느리다.
   - LoRA local evaluator는 `15~17 sec/step`.
   - full SFT export도 vLLM 모델들보다 느리다.
   - RL rollout에는 불리하다.

## 7. 현재 권장 사용법

### 7.1 대표 모델

대표는 Epoch2다.

```text
LLM-OS-Models/KoHRM-Text-1.4B-FullSFT-LFM25-Terminal-ToolBench-Epoch2
```

Epoch3는 다음 이유로 대표에서 제외한다.

- Score가 `45.90 -> 43.57`로 하락.
- Valid JSON이 `68.3% -> 61.7%`로 하락.
- Precision이 `0.5031 -> 0.4703`로 하락.
- First Cmd 개선폭은 `+0.6%p`라 전체 하락을 상쇄하지 못한다.

### 7.2 decoding 원칙

- temperature는 기본 `0.0`.
- JSON-only output을 강하게 요구한다.
- `<think>`, markdown fence, 설명문을 금지한다.
- `commands` array가 비면 retry한다.
- reference가 command를 요구하는 단계에서는 `task_complete=true` 조기 출력을 벌점/guard로 막는다.
- stop token/EOA를 정확히 잡아 긴 생성 낭비를 줄인다.

### 7.3 후처리

실사용에서는 output parser를 반드시 둔다.

권장 후처리:

- JSON parse 실패 시 repair retry.
- top-level field 외 텍스트 제거.
- `commands[].keystrokes` 없는 command 제거.
- command가 하나도 없고 task_complete가 false면 재생성.
- JSON은 valid지만 shell command가 너무 짧으면 optional second-pass generation.

후처리는 점수를 부풀리기 위한 꼼수가 아니라, HRM PrefixLM output contract가 아직 chat model만큼 안정적이지 않기 때문에 필요한 runtime guard다.

## 8. vLLM을 쓰는 방향

목표는 HRM을 vLLM 일반 chat model처럼 억지로 다루는 것이 아니라, PrefixLM 구조를 유지하면서 빠르게 batch generation하는 것이다.

가능한 경로:

1. HF export 구조 점검
   - `config.json`, model class, tokenizer, generation config가 vLLM에서 인식 가능한지 확인.
   - 안 되면 vLLM custom model plugin 경로가 필요하다.

2. PrefixLM 전용 batched inference
   - 지금 local evaluator를 batch-first 구조로 바꾼다.
   - KV cache reuse, SDPA cache, stop token handling을 안정화한다.
   - 8-shard replay가 아니라 true batched generation으로 바꾼다.

3. constrained decoding
   - llama.cpp grammar 같은 강제 JSON decoding을 HRM 경로에도 붙일 수 있는지 검토.
   - 최소한 top-level JSON start/end와 `commands` array contract는 강제한다.

4. invalid retry server
   - 모델 생성 1회로 끝내지 않고 parser failure 시 같은 GPU process에서 빠르게 retry한다.
   - retry prompt는 짧게 유지해 latency를 줄인다.

## 9. 다음 실험 제안

### A. Epoch2 decoding sweep

Epoch2 checkpoint 고정.

변수:

- temperature `0.0`, `0.1`, `0.2`.
- max_tokens `512`, `768`, `1024`.
- JSON repair retry on/off.
- `task_complete=true` guard on/off.
- stop token/EOA 조기 종료 on/off.

목표:

- Score 유지 또는 상승.
- Valid JSON `68.3% -> 75%+`.
- sec/step 감소.

### B. JSON-only small SFT

단순 extra epoch가 아니라 contract-only 보강.

데이터:

- invalid JSON이 많았던 source group.
- `<think>`/설명문 제거한 assistant target.
- `commands` array 유지 예제.
- late-step 검증 command 예제.

주의:

- JSON validity만 올리면 `behavior-jsonfix-r32`처럼 Score가 낮을 수 있다.
- 반드시 command F1을 같이 검증한다.

### C. Preference/DPO for HRM

조건:

- serving/rollout 속도 개선 후 시작.

chosen/rejected 구성:

- chosen: Epoch2 output 중 command F1 높은 것 또는 reference.
- rejected: invalid JSON, empty command, premature complete, low-recall output.

목표:

- Epoch2 Score `45.90` 초과.
- Valid JSON 회복.
- Precision 하락 없이 First Cmd 유지.

### D. Late-step targeted data

Epoch3의 약점은 후반 안정성이다.

보강할 것:

- 테스트 실행 후 수정 command.
- dependency 설치 후 검증 command.
- security/data_processing 후처리 command.
- 파일 생성/수정 후 `cat`, `pytest`, `python`, `grep` 확인 흐름.

## 10. 운영 규칙

- 모델카드에는 항상 base model, relation, score, command metrics, evaluation path를 쓴다.
- LoRA는 `base_model_relation: adapter`, full SFT는 `base_model_relation: finetune`로 구분한다.
- current best가 아닌 checkpoint는 명확히 "not current best"라고 쓴다.
- README에는 full replay 완료 점수만 전체 순위에 넣는다.
- partial 점수는 운영 스냅샷에만 적고 최종 순위에는 넣지 않는다.
- 결과 JSON과 문서 변경은 선별 stage 후 커밋한다.

## 11. 결론

KoHRM은 실패가 아니다. base `11.48`에서 LoRA `29.11`, full SFT Epoch2 `45.90`까지 올라간 것은 terminal/tool action을 강하게 학습할 수 있다는 증거다. 다만 Epoch3 `43.57`은 같은 데이터를 더 돌리는 방식의 한계를 보여준다.

따라서 지금 방향은 다음과 같다.

1. KoHRM 대표는 Epoch2로 고정한다.
2. Epoch3는 분석용으로 보존하고 대표로 올리지 않는다.
3. HRM은 vLLM/PrefixLM serving, JSON constrained decoding, output guard를 먼저 연구한다.
4. RL/DPO는 LFM2.5와 Qwen 2B에서 먼저 빠르게 검증한다.
5. HRM RL은 serving 속도와 JSON contract가 안정된 뒤 들어간다.
