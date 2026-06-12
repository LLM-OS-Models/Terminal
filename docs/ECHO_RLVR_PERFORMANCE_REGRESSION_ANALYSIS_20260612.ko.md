# LFM2.5 ECHO RLVR 성능 하락 분석

업데이트: 2026-06-12 06:12 UTC / 2026-06-12 15:12 KST

관련 문서:

- 실행/평가 종합 기록: [`docs/LFM25_ECHO_RLVR_RUNBOOK_KO_20260612.md`](LFM25_ECHO_RLVR_RUNBOOK_KO_20260612.md)
- GPU6 평가 기록: [`docs/ECHO_RLVR_GPU6_EVAL_20260612.md`](ECHO_RLVR_GPU6_EVAL_20260612.md)
- ECHO paper: <https://arxiv.org/abs/2605.24517>
- ECHO code: <https://github.com/microsoft/echo-rl>
- 현재 상태 노트: [`docs/LFM25_ECHO_RLVR_CURRENT_STATUS_KO_20260612.md`](LFM25_ECHO_RLVR_CURRENT_STATUS_KO_20260612.md)
- Post-training reasoning data primer: <https://arxiv.org/abs/2606.02113>

## 한 줄 결론

RLVR이 무의미한 것이 아니다. 현재 평가된 checkpoint 중 일부는 SFT 1Epoch baseline을 넘었다.

다만 이미 강하게 SFT된 terminal model에 no-docker local sandbox, sparse verifier reward, TB2-lite replay metric, 그리고 vLLM rollout-policy 동기화 문제가 겹치면서 대부분 checkpoint는 SFT가 가진 command imitation 능력을 흔들었다. 그래서 "계속 오래 돌리면 자동으로 오른다"가 아니라 "best checkpoint selection과 on-policy/metric 정렬이 없으면 떨어진다"가 현재 결론이다.

## README 기준 점수 정정

README의 주 지표는 다음이다.

```text
Score = 100 * avg_command_f1
```

평가 JSON에는 `next_action_score`도 들어 있다. 이 값은 현재 코드상 다음 blend다.

```text
next_action_score = 100 * (0.7 * avg_command_f1 + 0.3 * first_cmd_exact_pct / 100)
```

따라서 README 순위와 분석은 `100 * avg_command_f1` 기준으로 봐야 한다. `next_action_score`는 first command를 섞은 보조 지표다.

## 현재 관측 숫자

비교 기준:

- SFT 1Epoch baseline: Score `52.30`
- SFT 2Epoch: Score `50.48`
- LiquidAI raw base: Score `36.53`

RLVR 평가 현황:

- 평가 완료 checkpoint: `141`개
- 현재 best checkpoint: parentrun `checkpoint-610`
- 현재 best Score: `54.05`
- SFT 1Epoch `52.30` 대비 best gain: `+1.75`
- 새 paper-aligned HF on-policy run: first checkpoint pending

현재 README 기준 RLVR 상위권:

| checkpoint | Score | next_action_score | First Cmd | Valid JSON | early F1 | mid F1 | late F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parentrun `checkpoint-610` | `54.05` | `54.18` | `54.5%` | `77.9%` | `60.76` | `52.24` | `50.17` |
| parentrun `checkpoint-490` | `53.76` | `53.17` | `51.8%` | `77.2%` | - | - | - |
| parentrun `checkpoint-650` | `53.65` | `52.91` | `51.2%` | `76.2%` | `61.54` | `50.04` | `50.60` |
| parentrun `checkpoint-230` | `53.43` | `52.76` | `51.2%` | `77.9%` | `63.97` | `51.21` | `46.72` |
| parentrun `checkpoint-440` | `53.32` | `53.16` | `52.8%` | `75.2%` | - | - | - |
| continuation `checkpoint-220` | `53.26` | `52.34` | `50.2%` | `77.2%` | `63.68` | `51.72` | `45.96` |
| parentrun `checkpoint-760` | `53.23` | `52.02` | `49.2%` | `76.6%` | - | - | - |

즉 최고점만 보면 RLVR은 올랐다. SFT 1Epoch 대비 최고 `+1.75`점이다. 하지만 평균적으로는 SFT baseline보다 낮다. 이 차이가 핵심이다.

## 왜 오른 checkpoint도 있나

RLVR이 일부 영역에서는 실제로 도움을 준다. 89개 checkpoint 전체를 baseline SFT 1Epoch와 비교하면 평균적으로 다음 영역이 좋아졌다.

| source group | 평균 delta | win rate |
| --- | ---: | ---: |
| `code` | `+17.48` | `98.8%` |
| `data_science` | `+4.24` | `84.9%` |
| `data_querying` | `+3.41` | `81.4%` |
| `model_training` | `+2.65` | `73.3%` |
| `data_processing` | `+1.25` | `68.6%` |

bucket 기준으로는 `late`가 평균 `+0.59` 올라갔다. 이건 ECHO-style observation loss가 아예 무의미하지 않다는 신호다. 터미널 출력(stdout/stderr, ls/cat 결과, 에러 메시지)을 auxiliary CE로 맞추게 하니, 긴 상호작용의 후반부나 코드/데이터 탐색처럼 observation을 보고 다음 명령을 고르는 영역에서는 이득이 생긴다.

가장 좋은 예시는 continuation `checkpoint-250`이다.

- `code`: baseline `16.13` -> `36.76`, `+20.63`
- `file_operations`: baseline `52.39` -> `61.17`, `+8.78`
- `model_training`: baseline `51.76` -> `59.18`, `+7.42`
- `mid`: baseline `51.72` -> `53.30`, `+1.58`
- `late`: baseline `44.55` -> `46.49`, `+1.94`

즉 RLVR은 "후반 상태 추적", "파일/코드 관찰", "실행 결과 기반 수정" 쪽으로는 신호가 있다.

## 왜 대부분 떨어지나

반대로 평균적으로 크게 떨어지는 영역도 뚜렷하다.

| source group | 평균 delta | win rate |
| --- | ---: | ---: |
| `scientific_computing` | `-5.05` | `1.2%` |
| `swe` | `-4.43` | `10.5%` |
| `security` | `-4.06` | `20.9%` |
| `system_administration` | `-3.89` | `4.7%` |
| `debugging` | `-3.24` | `8.1%` |
| `software_engineering` | `-3.22` | `17.4%` |

bucket 기준:

- `early`: 평균 `-1.52`, win rate `11.6%`
- `mid`: 평균 `-1.09`, win rate `22.1%`
- `late`: 평균 `+0.59`, win rate `57.0%`

이 패턴은 "terminal world model은 일부 후반부 reasoning을 돕지만, 이미 SFT로 잘 맞춰진 초반 명령 습관과 benchmark imitation 분포를 깨고 있다"는 뜻이다.

TB2-lite replay는 정답 command와 모델 command의 token F1을 본다. live terminal RLVR의 verifier reward는 "실제로 task가 풀렸는가"를 본다. 두 목표는 비슷하지만 동일하지 않다. 예를 들어 verifier 관점에서는 `find`, `ls`, `grep`, `python - <<...` 조합이 모두 기능적으로 가능할 수 있지만, replay metric은 gold command와 토큰이 다르면 F1이 떨어진다.

그래서 RLVR이 live task 성공 방향으로 command 스타일을 바꾸면, TB2-lite replay imitation 점수는 오히려 떨어질 수 있다.

## 이미 SFT가 많이 된 모델이라는 점

현재 base는 그냥 raw LFM2.5가 아니다.

- raw `LiquidAI/LFM2.5-8B-A1B`: Score `36.53`
- Terminal ToolBench Full SFT 1Epoch: Score `52.30`
- Terminal ToolBench Full SFT 2Epoch: Score `50.48`

1Epoch SFT가 이미 `+15.77`점을 만든 상태다. 게다가 2Epoch는 1Epoch보다 낮다. 이건 이 모델이 이미 TB2-lite command distribution에 강하게 맞춰져 있고, 추가 학습이 항상 이득이 아니라는 직접 증거다.

따라서 "SFT가 많이 된 모델은 RL하기 부적합하다"가 정확한 결론은 아니다. 더 정확히는 다음이다.

```text
이미 잘 맞춰진 SFT 모델은 RLVR의 이득 폭이 작고,
reward/rollout/eval이 조금만 어긋나도 SFT 분포가 깨져 점수가 떨어지기 쉽다.
```

즉 RLVR은 가능하지만, 이 경우에는 낮은 LR, KL/SFT anchor, on-policy rollout, metric-aligned validation, best-checkpoint selection이 필수다.

## 학습 reward가 약하고 noisy하다

현재 active continuation run의 학습 로그:

- steps: `735`
- 평균 reward: `-0.033`
- 평균 verifier reward: `0.107`
- verifier reward가 양수인 step 비율: `29.4%`

parent run의 전체 로그:

- steps: `1889`
- 평균 reward: `-0.027`
- 평균 verifier reward: `0.111`
- verifier reward가 양수인 step 비율: `30.6%`

최근 continuation step 예시에서는 verifier reward가 계속 `0.0`으로 나오는 구간도 있다.

```text
step 730 reward=-0.0875 verifier=0.0
step 731 reward=-0.1625 verifier=0.0
step 732 reward=-0.2625 verifier=0.0
step 733 reward=0.0    verifier=0.0
step 734 reward=-0.2875 verifier=0.0
```

GRPO는 group 안의 상대 reward 차이로 업데이트한다. 지금은 `num_generations=2`라 한 prompt당 비교 샘플이 2개뿐이다. reward가 sparse하고 noisy한데 그룹도 작으니 advantage가 흔들리기 쉽다. 이런 조건에서는 checkpoint가 `52 -> 49 -> 51`처럼 출렁이는 것이 자연스럽다.

## 가장 큰 기술 리스크: vLLM rollout policy 동기화

현재 실행 상태를 보면 vLLM 서버 4개는 다음 모델을 고정 서빙한다.

```text
LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch
```

반면 train rank는 다음 adapter를 로드해서 업데이트한다.

```text
--sft-adapter-path .../parentrun.../checkpoint-1880
```

그리고 rollout HTTP payload는 다음 필드만 보낸다.

```python
payload = {
    "model": args.vllm_served_model or args.model_path,
    "prompt": prompt,
    ...
}
```

LoRA adapter id/name을 vLLM request에 싣는 코드가 없다. 즉 vLLM rollout은 현재 학습 중인 RLVR LoRA가 아니라 SFT 1Epoch base에서 생성될 가능성이 크다.

이건 ECHO/SkyRL 원 논문 구현 방향과 다르다. ECHO repo README도 SkyRL/vLLM을 쓰는 이유 중 하나로 batching, sampling, weight synchronization, trajectory construction 제어를 언급한다. 원래 RL은 rollout policy와 update policy가 맞아야 한다. 지금 구조는 고정 SFT rollout으로 만든 trajectory를 다른 LoRA policy에 업데이트하는 off-policy에 가깝다.

이게 현재 성능 하락의 가장 큰 기술 원인 후보다.

2026-06-12 05:00 UTC 이후 새로 시작한 run은 이 리스크를 피하기 위해 SFT 1Epoch base에서 adapter 없이 다시 시작했고, `--rollout-backend hf`로 학습 policy가 직접 rollout을 생성한다. 이 방식은 vLLM보다 느리지만 rollout-policy/update-policy mismatch를 줄인다.

해결책:

1. vLLM에 `--enable-lora`를 켜고 현재 checkpoint adapter를 request마다 명시한다.
2. 학습 중 LoRA weight를 vLLM server에 주기적으로 reload/sync한다.
3. 이것이 어렵다면 rollout backend를 임시로 HF generate로 돌려서 on-policy 여부를 먼저 검증한다.
4. 또는 step 단위 on-policy는 포기하더라도 checkpoint 단위 policy iteration으로 명확히 분리한다. 예: fixed policy rollout -> adapter update -> vLLM adapter reload -> next rollout.

이 문제를 해결하지 않고 2일 더 돌리면, GPU 시간은 쓰지만 RL 신호가 policy에 정확히 물리지 않을 가능성이 크다.

## no-docker local sandbox의 한계

Docker가 없다고 RLVR이 불가능한 것은 아니다. 실제로 command 실행, stdout/stderr 수집, verifier 실행, observation CE loss는 동작한다.

다만 ECHO paper의 Harbor/Docker backend와 비교하면 다음 노이즈가 생긴다.

- `/app`, `/tests` path rewrite가 필요하다.
- host-sensitive command를 차단해야 한다.
- command가 tmux/server/local process 상태를 건드릴 수 있다.
- package install, filesystem permission, timeout, subprocess cleanup이 task마다 달라진다.
- verifier가 원 논문 환경보다 더 자주 false negative/timeout을 만들 수 있다.

따라서 no-docker가 "유일한 원인"은 아니지만, reward noise를 키우는 원인은 맞다. 특히 현재처럼 verifier positive가 30% 안팎이면 sandbox 안정성이 곧 학습 품질이다.

zerobox/OpenSandbox 같은 대체 sandbox를 검토하는 이유도 여기 있다. 목표는 보안 그 자체보다 "매 rollout마다 같은 task가 같은 조건에서 재현되는가"다.

## ECHO 방식 자체는 들어가 있나

들어가 있다.

현재 trainer는 terminal observation token에 CE loss를 건다.

```python
policy_loss = -(token_logp * action_next).sum() / action_tokens * adv
world_loss = -(token_logp * obs_next).sum() / obs_tokens
loss = policy_coeff * policy_loss + world_model_coeff * world_loss
```

현재 active turbo run의 `world_model_coeff=0.05`다.

즉 "터미널 피드백을 다음 context로만 읽고 버리는 vanilla GRPO"가 아니라, terminal output token 자체도 loss에 들어간다. 다만 논문 원본과 다른 점은 SkyRL/FSDP/Harbor/weight sync 조합이 아니라, local no-docker trainer와 고정 vLLM server 조합이라는 것이다.

새 turbo run에서는 `world_model_coeff=0.05`, `num_generations=4`, `max_turns=6`, `max_new_tokens=512`, `max_terminal_output_chars=12000`, `verifier_timeout=45`로 빠른 실험 사이클에 맞췄다. 직전 paper-aligned slow run의 `num_generations=16`, `max_turns=16`, `max_new_tokens=2048`은 첫 step만 15분 이상 걸려 중단했다. official ECHO 내부 parquet 데이터와 Harbor/Docker 환경은 아직 로컬에 없으므로 완전 동일 재현은 아니다.

## A Primer in Post-Training Reasoning Data 관점

`A Primer in Post-Training Reasoning Data`는 reasoning post-training 데이터를 단순 `prompt -> answer`가 아니라 다음 구조로 봐야 한다고 정리한다.

```text
task/context -> trace/actions -> answer/artifact -> verifier/reward/environment -> attribution metadata
```

이 관점에서 현재 ECHO RLVR 결과를 보면, "RL을 길게 돌리면 언젠가 오른다"보다 다음 조건이 더 중요하다.

- verifier가 실제 terminal task 성공을 안정적으로 반영해야 한다.
- 환경이 rollout마다 재현되어야 한다.
- 너무 쉬운 문제나 너무 어려운 문제보다 모델이 가끔 맞히는 경계 난이도 데이터가 gradient를 만든다.
- 이미 강하게 SFT된 모델은 reward/eval metric이 조금만 어긋나도 기존 command imitation 능력이 흔들린다.
- trace/actions, terminal observations, verifier result, metadata를 모두 저장해야 나중에 RLVR 재학습이나 SFT 데이터로 재사용할 수 있다.

따라서 현재 실험의 해석은 "RLVR 무효"가 아니라 "데이터 객체와 검증 계약이 아직 완전히 논문 수준으로 정렬되지 않은 상태에서, 일부 checkpoint만 SFT baseline을 넘긴다"가 더 정확하다.

## 지금 결론

현재 결과를 한 문장으로 쓰면 다음이다.

```text
ECHO-style RLVR은 일부 checkpoint와 일부 영역에서는 SFT baseline을 넘겼지만,
현재 구현/환경에서는 평균적으로 SFT 1Epoch의 안정적인 command imitation 능력을 깨는 쪽이 더 많이 관측된다.
```

이건 "RLVR이 안 된다"가 아니라 "지금 조건에서는 긴 학습보다 정렬과 안정화가 먼저"라는 뜻이다.

## 다음 실험 우선순위

1. vLLM LoRA sync를 고친다.
   - 현재 최고 우선순위다.
   - rollout policy와 train policy를 맞춰야 한다.

2. SFT 1Epoch에서 clean-start RLVR을 다시 한다.
   - parent `checkpoint-1880`에서 이어가는 run은 이미 drift가 들어간 policy에서 시작한다.
   - SFT 1Epoch -> RLVR 300/600/1000 step clean curve가 필요하다.

3. KL/SFT anchor를 넣는다.
   - 이미 잘 된 SFT 모델이므로 RL만 걸면 command style이 무너질 수 있다.
   - 작은 SFT CE loss 또는 reference KL을 같이 둬야 한다.

4. world model coefficient sweep을 한다.
   - 후보: `0.0`, `0.01`, `0.03`, `0.05`
   - 현재 `0.03`은 일부 late/code 영역에 도움을 주지만 전체 imitation에는 과할 수 있다.

5. GRPO group size와 rollout 길이를 분리해 튜닝한다.
   - paper-like: `num_generations=16`, `max_turns=16`, `max_new_tokens=2048`은 재현성은 높지만 너무 느리다.
   - fast: `num_generations=8`, `max_turns=8`, `max_new_tokens=768`은 첫 step 약 4분대로 줄었다.
   - turbo: `num_generations=4`, `max_turns=6`, `max_new_tokens=512`는 checkpoint/eval 사이클을 빠르게 보기 위한 현재 설정이다.
   - 성능 곡선이 보이면 다시 `num_generations=8` 이상으로 올려 안정성을 확인한다.
   - 가능하면 `4` 또는 `8`로 올려 reward variance를 줄인다.

6. LR을 더 낮춘다.
   - 현재 `5e-7`
   - 후보: `1e-7`, `2e-7`
   - SFT가 이미 강한 모델은 작은 update가 맞다.

7. TB2-lite replay metric을 validation으로 계속 본다.
   - long-run final이 아니라 best checkpoint를 선택한다.
   - 현재처럼 10-step dense eval은 유지한다.

8. train/eval mismatch를 분리한다.
   - live terminal success metric과 TB2 replay metric을 둘 다 기록한다.
   - RL이 live success를 올리는데 replay F1만 내리는지, 둘 다 내리는지 구분해야 한다.

9. sandbox 안정화 후 장기 run을 다시 한다.
   - zerobox/OpenSandbox 후보를 실험한다.
   - Docker가 안 되면 최소한 process isolation, timeout cleanup, cwd/path rewrite를 더 강하게 고정해야 한다.

## 운영 판단

현재 checkpoint 중 README 기준 최고는 parentrun `checkpoint-610` Score `54.05`다. 이 checkpoint는 전체 리더보드에 올릴 후보가 될 수 있다. 다만 TB2 최종/전체 평가 전에는 "확정 1위"로 적지 않는다.

현재 active continuation은 계속 돌릴 수 있지만, 그대로 2일 더 돌리는 것보다 vLLM LoRA sync 문제를 해결한 run을 새로 시작하는 것이 더 효율적이다. 지금 구조에서는 오래 돌릴수록 좋아진다는 보장이 약하다.
