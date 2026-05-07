# 모델 평가 재계산 및 분석 리포트 (2026-05-08)

이 문서는 모델 평가 결과를 같은 기준으로 다시 해석하기 위한 리포트다. 현재는 LFM 계열 재학습 모델들의
TB2-lite 결과를 정리했고, 이후 Qwen/Gemma/Nemotron/Ouro 등 다른 모델 재평가 결과도 같은 파일에 이어서
추가할 수 있도록 범용 이름으로 관리한다.

이번 LFM 계열 내용은 이미 끝난 vLLM 평가 결과 JSON을 다시 읽어서 점수만 새 기준으로 재계산한 기록이다.
평가를 다시 돌린 것은 아니다. GPU를 다시 쓰지 않았고, 기존 생성 결과를 그대로 사용했다.

결과 원본 경로:

```text
/home/work/.data/tb2_lite_eval/20260507T_lfm_family_corrected_after_24b_eval
```

이번 LFM 결과에 사용된 평가 조건:

- 평가 방식: vLLM 전용 LFM 평가 코드
- 프롬프트: 각 모델 tokenizer의 `chat_template` 사용
- raw prompt fallback: 사용하지 않음
- 평가셋: 수정된 누적 대화형 TB2-lite replay
- 평가 row 수: `303 steps / 50 tasks`
- 평가 대상: LFM 계열 4개 모델의 epoch별 checkpoint 8개

중요한 전제:

- 이 결과는 README 기존 표의 `386 step` legacy 결과와 직접 비교하면 안 된다.
- 이번 결과는 JSON reference 오류와 빈 command step을 제거한 `303 step` 기준이다.
- prompt 구성도 기존 raw prompt 방식이 아니라 누적 conversation + chat template 방식이다.

## LFM 계열: 왜 점수를 다시 계산했나

기존 요약 파일은 아래 식으로 순위를 매겼다.

```text
next_action_score = 100 * (0.7 * avg_command_f1 + 0.3 * first_cmd_exact)
```

이 방식은 `first_cmd_exact`에 30% 가중치를 준다. 이 가중치는 현재 평가 목적에는 과하다.

`first_cmd_exact`는 첫 번째 command가 정답과 완전히 같은지만 보는 binary 지표다. 즉, 첫 command가
완전히 같으면 1점이고 아니면 0점이다. command 전체가 얼마나 비슷한지, 뒤 command들이 얼마나 맞는지는
이 지표만으로는 충분히 보지 못한다.

반대로 `avg_command_f1`은 예측 command들과 정답 command들의 token overlap을 평균적으로 본다. 그래서
전체 command 품질을 볼 때는 `avg_command_f1`이 더 직접적인 주 지표다.

따라서 이 문서에서는 아래 기준으로 다시 계산했다.

```text
새 점수 = 100 * avg_command_f1
```

이 문서에서 `first_cmd_exact_pct`는 순위 계산에서 제외하고 보조 진단 지표로만 둔다.

## LFM 계열: 최종 체크포인트 기준 순위

학습 결과로 실제 사용할 모델을 고르는 기준은 최종 체크포인트가 우선이다. 따라서 아래 표가 가장 중요한
최종 결과표다.

| 순위 | 모델 | 최종 체크포인트 | Epoch | 재계산 점수 | Cmd F1 | Precision | Recall | First Cmd Exact | Valid JSON |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LFM2-24B-A2B | 1460 | 2 | 33.35 | 0.3335 | 0.4242 | 0.3590 | 26.1% | 66.3% |
| 2 | LFM2-2.6B | 1090 | 2 | 32.85 | 0.3285 | 0.4194 | 0.3336 | 31.0% | 55.4% |
| 3 | LFM2-8B-A1B | 1660 | 2 | 31.02 | 0.3102 | 0.4063 | 0.3201 | 29.0% | 54.8% |
| 4 | LFM2.5-1.2B | 1090 | 2 | 28.64 | 0.2864 | 0.3775 | 0.2904 | 29.0% | 50.5% |

결론:

- 최종 체크포인트만 보면 `LFM2-24B-A2B checkpoint-1460`이 1위다.
- `LFM2-2.6B checkpoint-1090`은 24B 최종 모델과 차이가 작다. 점수 차이는 `0.50점`이다.
- `LFM2-8B-A1B checkpoint-1660`은 2epoch에서 성능이 떨어졌다.
- `LFM2.5-1.2B checkpoint-1090`은 2epoch에서 조금 좋아졌지만 전체적으로는 큰 모델들보다 낮다.

다만 최종 체크포인트만 보는 표와 실사용 후보 선정은 약간 다르게 해석해야 한다. `LFM2-8B-A1B`는 final인
`checkpoint-1660`이 아니라 epoch 1인 `checkpoint-830`이 더 좋다. 이 모델은 A1B 구조라서 active parameter
관점의 효율이 좋기 때문에, 성능이 조금 낮아도 운영 후보로는 꽤 매력적이다.

## LFM 계열: 모든 체크포인트 기준 순위

아래 표는 epoch 1과 epoch 2 체크포인트를 모두 포함한 순위다. 학습이 어느 시점에서 제일 좋았는지 보기
위한 표다.

| 순위 | 모델 | 체크포인트 | Epoch | 재계산 점수 | Cmd F1 | Precision | Recall | First Cmd Exact | Valid JSON | 기존 Weighted 점수 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LFM2-24B-A2B | 730 | 1 | 33.46 | 0.3346 | 0.4363 | 0.3549 | 31.7% | 65.0% | 32.93 |
| 2 | LFM2-24B-A2B | 1460 | 2 | 33.35 | 0.3335 | 0.4242 | 0.3590 | 26.1% | 66.3% | 31.17 |
| 3 | LFM2-2.6B | 1090 | 2 | 32.85 | 0.3285 | 0.4194 | 0.3336 | 31.0% | 55.4% | 32.29 |
| 4 | LFM2-8B-A1B | 830 | 1 | 32.41 | 0.3241 | 0.4327 | 0.3220 | 29.4% | 56.8% | 31.51 |
| 5 | LFM2-2.6B | 545 | 1 | 31.86 | 0.3186 | 0.4102 | 0.3321 | 30.0% | 55.8% | 31.30 |
| 6 | LFM2-8B-A1B | 1660 | 2 | 31.02 | 0.3102 | 0.4063 | 0.3201 | 29.0% | 54.8% | 30.41 |
| 7 | LFM2.5-1.2B | 1090 | 2 | 28.64 | 0.2864 | 0.3775 | 0.2904 | 29.0% | 50.5% | 28.75 |
| 8 | LFM2.5-1.2B | 545 | 1 | 28.10 | 0.2810 | 0.3615 | 0.2941 | 29.4% | 50.5% | 28.49 |

전체 체크포인트 기준으로는 `LFM2-24B-A2B checkpoint-730`이 가장 높다. 다만 이건 epoch 1 체크포인트다.
최종 결과를 중요하게 본다면 `checkpoint-1460`을 최종 모델로 보는 것이 맞다.

## LFM 계열: Epoch별 변화

| 모델 | Epoch 1 점수 | Epoch 2 점수 | 변화량 | 가장 좋은 체크포인트 |
| --- | ---: | ---: | ---: | --- |
| LFM2-24B-A2B | 33.46 | 33.35 | -0.11 | epoch 1 / checkpoint-730 |
| LFM2-2.6B | 31.86 | 32.85 | +0.99 | epoch 2 / checkpoint-1090 |
| LFM2-8B-A1B | 32.41 | 31.02 | -1.39 | epoch 1 / checkpoint-830 |
| LFM2.5-1.2B | 28.10 | 28.64 | +0.54 | epoch 2 / checkpoint-1090 |

해석:

- `LFM2-24B-A2B`는 epoch 1과 epoch 2가 거의 비슷하다. 평균 F1 차이는 `0.11점`뿐이다.
- 24B의 epoch 2는 JSON 형식 준수와 recall이 더 좋지만, precision과 first command exact가 낮아졌다.
- `LFM2-2.6B`는 2epoch가 확실히 더 좋다. 이번 학습에서는 2epoch까지 가는 것이 도움이 됐다.
- `LFM2-8B-A1B`는 2epoch에서 성능이 내려갔다. 이 모델은 epoch 1 checkpoint를 따로 보관할 가치가 있다.
- `LFM2.5-1.2B`는 2epoch에서 조금 좋아졌지만, 절대 성능은 낮다.

여기서 중요한 점은 “최종 checkpoint가 항상 제일 좋지는 않다”는 것이다. 특히 `LFM2-8B-A1B`와
`LFM2-24B-A2B`는 epoch 1이 더 높다. 반면 `LFM2-2.6B`와 `LFM2.5-1.2B`는 epoch 2가 더 높다.

따라서 앞으로 모델 업로드/배포를 할 때는 두 가지를 분리해야 한다.

- 최종 산출물: 학습이 끝난 final checkpoint
- 실사용 추천 checkpoint: 평가상 가장 좋은 checkpoint

LFM 계열에서는 이 둘이 완전히 일치하지 않는다.

## LFM 계열: 기존 Weighted 점수가 왜 헷갈렸나

24B 결과를 보면 왜 `first_cmd_exact` 30% 가중치가 문제인지 명확하다.

| 체크포인트 | Cmd F1 기반 점수 | First Cmd Exact | 기존 Weighted 점수 |
| ---: | ---: | ---: | ---: |
| 730 | 33.46 | 31.7% | 32.93 |
| 1460 | 33.35 | 26.1% | 31.17 |

평균 command F1 차이는 `0.11점`뿐이다. 그런데 기존 weighted 점수에서는 `1.76점` 차이로 커졌다.
이 차이 대부분은 `first_cmd_exact`에서 나온 것이다.

즉, 기존 점수식은 첫 번째 command exact match를 너무 크게 반영했다. 전체 command 품질을 보려면
`avg_command_f1`을 주 지표로 보는 것이 더 맞다.

앞으로 권장하는 순위 기준:

```text
1순위 지표: avg_command_f1
동점 또는 근소 차이 판단: avg_command_recall, avg_command_precision, valid_json_pct
보조 진단 지표: first_cmd_exact_pct
```

## LFM 계열: 모델별 장단점 비교

### LFM2-24B-A2B

최종 체크포인트 기준 1위 모델이다. `checkpoint-1460`의 재계산 점수는 `33.35`이고, 전체 checkpoint까지
포함하면 epoch 1인 `checkpoint-730`이 `33.46`으로 가장 높다.

장점:

- 최종 checkpoint 기준 가장 높은 평균 command F1을 보인다.
- `valid_json_pct`가 `66.3%`로 4개 모델 중 가장 높다.
- recall이 `0.3590`으로 가장 높아서 정답 command 요소를 넓게 커버하는 편이다.
- security, dependency management, scientific computing 쪽에서 상대적으로 강하다.

단점:

- 2epoch로 가면서 first command exact가 `31.7% -> 26.1%`로 떨어졌다.
- 평균 F1 기준 epoch 1과 epoch 2 차이가 `0.11점`뿐이라, 2epoch 추가 학습 이득이 거의 없다.
- 24B급 모델인데 2.6B와 점수 차이가 `0.50점`이라 비용 대비 압도적이라고 보긴 어렵다.
- code/swe 영역과 late step에서는 여전히 약하다.

판단:

- 성능 최우선 최종 모델로는 `checkpoint-1460`을 사용한다.
- checkpoint sweep 최고점만 보면 `checkpoint-730`도 반드시 보관해야 한다.

### LFM2-2.6B

이번 결과에서 가장 효율이 좋아 보이는 dense 후보. 최종 checkpoint인 `checkpoint-1090`이 `32.85`로
24B final과 `0.50점` 차이다.

장점:

- 24B와 매우 가까운 점수를 낸다.
- epoch 1에서 epoch 2로 `31.86 -> 32.85`, `+0.99점` 개선됐다.
- scientific computing, security, data science에서 강하다.
- final checkpoint를 그대로 써도 되는 구조라 관리가 단순하다.
- 크기 대비 성능이 가장 좋다.

단점:

- `valid_json_pct`가 `55.4%`로 24B보다 낮다.
- code/swe/data_querying 영역이 약하다.
- late step 평균 F1이 `0.2328`이라 긴 trajectory 후반부에서는 불안정할 수 있다.

판단:

- 실사용 가성비 기준으로 가장 중요한 후보 중 하나다.
- 24B가 부담되면 `LFM2-2.6B checkpoint-1090`이 가장 합리적인 대체 모델이다.

### LFM2-8B-A1B

이 모델은 final만 보면 손해다. `checkpoint-1660`은 `31.02`지만, epoch 1인 `checkpoint-830`은 `32.41`이다.
그리고 A1B 구조라서 active parameter 기준 효율이 좋다. 그래서 단순 final 순위보다 실사용 매력도가 높다.

장점:

- epoch 1 checkpoint는 `32.41`로 2.6B final보다 `0.44점` 낮을 뿐이다.
- 24B final과도 `0.94점` 차이다.
- A1B라서 추론 비용/속도 측면에서 매력적이다.
- early step 성능이 매우 강하다. final 기준 early F1이 `0.5054`로 네 모델 중 가장 높다.
- `checkpoint-830`을 쓰면 성능과 효율 균형이 좋다.

단점:

- epoch 2에서 `32.41 -> 31.02`로 `-1.39점` 하락했다.
- final checkpoint를 그대로 쓰면 손해다.
- mid/late 단계로 갈수록 약해진다.
- code/swe/data_querying이 약하다.

판단:

- 이 모델은 반드시 `checkpoint-830`을 별도 실사용 후보로 봐야 한다.
- “최종 checkpoint 리더보드”에서는 3위지만, “효율 포함 실사용 후보”로는 2.6B와 같이 강하게 고려할 만하다.

### LFM2.5-1.2B

4개 중 확실히 한 단계 낮다. final checkpoint인 `checkpoint-1090` 점수는 `28.64`다.

장점:

- 가장 작고 가벼운 후보로 볼 수 있다.
- epoch 1에서 epoch 2로 `28.10 -> 28.64`, `+0.54점` 개선됐다.
- security, scientific computing, file operations에서는 비교적 괜찮다.

단점:

- 24B final보다 `4.71점`, 2.6B final보다 `4.21점` 낮다.
- `valid_json_pct`가 `50.5%`로 가장 낮다.
- code/swe/math 쪽이 특히 약하다.
- 전체적으로 주력 모델로 쓰기엔 부족하다.

판단:

- 경량 실험용 또는 작은 모델 baseline으로는 의미가 있다.
- 성능 주력 모델로는 2.6B/8B-A1B/24B보다 명확히 밀린다.

## LFM 계열: 실사용 후보 관점 정리

최종 checkpoint만 보면 순위는 아래와 같다.

```text
1위: LFM2-24B-A2B checkpoint-1460
2위: LFM2-2.6B checkpoint-1090
3위: LFM2-8B-A1B checkpoint-1660
4위: LFM2.5-1.2B checkpoint-1090
```

하지만 실사용 후보 관점에서는 다음처럼 보는 것이 더 정확하다.

| 목적 | 추천 모델 | 이유 |
| --- | --- | --- |
| 최고 성능 final 모델 | LFM2-24B-A2B checkpoint-1460 | 최종 checkpoint 기준 1위, JSON 안정성 최고 |
| 최고 checkpoint 점수 | LFM2-24B-A2B checkpoint-730 | 전체 checkpoint sweep 1위 |
| 크기 대비 성능 | LFM2-2.6B checkpoint-1090 | 24B와 0.50점 차이, final 사용 가능 |
| 추론 효율/성능 균형 | LFM2-8B-A1B checkpoint-830 | A1B 구조, 2.6B final과 0.44점 차이 |
| 경량 baseline | LFM2.5-1.2B checkpoint-1090 | 가장 작지만 성능은 확실히 낮음 |

핵심 판단:

- `1.2B`는 확실히 약하다.
- `24B`, `2.6B`, `8B-A1B epoch1`은 생각보다 비슷하다.
- `8B-A1B checkpoint-830`은 A1B 구조 때문에 점수 이상의 운영 매력이 있다.
- `2.6B checkpoint-1090`은 가장 균형 잡힌 dense 후보로 보인다.
- `24B`는 최고 성능이지만, 비용 대비 압도적인 차이는 아니다.

## LFM 계열: 최종 체크포인트의 위치별 성능

아래는 replay 진행 위치별 평균 command F1이다.

| 모델 | Early | Mid | Late |
| --- | ---: | ---: | ---: |
| LFM2-24B-A2B checkpoint-1460 | 0.4754 | 0.2848 | 0.2620 |
| LFM2-2.6B checkpoint-1090 | 0.4389 | 0.3302 | 0.2328 |
| LFM2-8B-A1B checkpoint-1660 | 0.5054 | 0.2599 | 0.1953 |
| LFM2.5-1.2B checkpoint-1090 | 0.4368 | 0.2588 | 0.1865 |

공통적으로 late 단계 성능이 낮다. 즉, 초반 next action은 비교적 잘 맞추지만, 누적 context가 길어지고
작업 후반으로 갈수록 정확도가 떨어진다.

이건 모델만의 문제라기보다 현재 TB2-lite replay 평가 구조의 한계도 있다. 지금 평가는 전체 task 성공 여부를
완전히 재현하는 평가라기보다는, 각 step에서 다음 command를 얼마나 비슷하게 내는지 보는 imitation 성격이 강하다.

## LFM 계열: 최종 체크포인트별 강점과 약점

아래 표는 source group별 평균 command F1 기준으로 강한 영역과 약한 영역을 정리한 것이다.

| 모델 | 강한 영역 | 약한 영역 |
| --- | --- | --- |
| LFM2-24B-A2B checkpoint-1460 | security 0.4015; dependency_management 0.3884; scientific_computing 0.3796 | debugging 0.3023; swe 0.2631; code 0.1024 |
| LFM2-2.6B checkpoint-1090 | scientific_computing 0.4602; security 0.4532; data_science 0.4134 | data_querying 0.2291; swe 0.1298; code 0.0492 |
| LFM2-8B-A1B checkpoint-1660 | scientific_computing 0.4498; system_administration 0.3919; data_science 0.3887 | data_querying 0.1832; swe 0.1615; code 0.1341 |
| LFM2.5-1.2B checkpoint-1090 | scientific_computing 0.4027; security 0.3998; file_operations 0.3965 | math 0.2014; swe 0.1177; code 0.0000 |

공통 약점:

- `code`
- `swe`
- 긴 context의 late 단계

가능한 이유:

- code/swe 영역은 단순 command 한 줄보다 패치 계획, 파일 확인, 수정 순서가 중요하다.
- 모델이 긴 설명이나 broader plan을 내면 replay evaluator에서는 다음 command exact/F1이 낮게 잡힐 수 있다.
- 현재 평가는 실제 task 성공률이 아니라 reference command와의 유사도를 보는 방식이다.

## LFM 계열: 이번 결과의 최종 판단

최종 모델 기준:

```text
1위: LFM2-24B-A2B checkpoint-1460
2위: LFM2-2.6B checkpoint-1090
3위: LFM2-8B-A1B checkpoint-1660
4위: LFM2.5-1.2B checkpoint-1090
```

전체 checkpoint sweep 기준:

```text
1위: LFM2-24B-A2B checkpoint-730
2위: LFM2-24B-A2B checkpoint-1460
3위: LFM2-2.6B checkpoint-1090
```

운영 관점 추천:

- 최종 산출물로는 `LFM2-24B-A2B checkpoint-1460`을 1순위로 둔다.
- 단, TB2-lite next-action 점수만 놓고 최고 checkpoint를 고르면 `LFM2-24B-A2B checkpoint-730`도 따로 보관한다.
- `LFM2-2.6B checkpoint-1090`은 크기 대비 성능이 좋으므로 반드시 보관한다.
- 기존 README 386-step 결과와는 분리해서 관리한다.

## 앞으로 보고서 작성 기준

앞으로는 결과를 두 표로 나눠서 기록하는 것이 맞다.

1. 최종 모델 리더보드
   - final checkpoint만 포함
   - `avg_command_f1 * 100` 기준 정렬

2. checkpoint sweep 리더보드
   - epoch별 checkpoint 모두 포함
   - 어느 epoch가 가장 좋은지 확인

그리고 `first_cmd_exact_pct`는 순위 점수에 직접 섞지 않고, 보조 지표로만 기록한다.

## 이후 추가할 평가

이 파일은 LFM 전용 리포트가 아니라 전체 모델 평가 재계산 리포트로 계속 확장한다. 이후 GPU가 비었을 때
README 기존 모델들도 같은 방식으로 재평가하고, 아래 섹션을 추가한다.

- Qwen 계열 재평가
- Gemma 계열 재평가
- Nemotron-Terminal 재평가
- Ouro 계열 재평가
- legacy 386-step 결과와 corrected 303-step 결과의 분리 비교

다른 모델을 평가할 때도 원칙은 동일하다.

- vLLM 평가를 우선 사용한다.
- 모델별 공식 chat template 또는 tokenizer chat template을 적용한다.
- raw prompt fallback 결과는 랭킹에서 제외한다.
- primary score는 `avg_command_f1 * 100`으로 둔다.
- final checkpoint 리더보드와 checkpoint sweep 리더보드를 분리한다.
