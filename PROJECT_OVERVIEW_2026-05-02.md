# Terminal Agent

Nemotron-Terminal-Corpus 기반 터미널 에이전트 학습과 평가를 위한 작업 폴더입니다.

현재 이 저장소는 크게 세 축으로 운영됩니다.

1. `eval/` 기반 단발 프록시 평가
2. `tb2_lite/` 기반 멀티턴 replay 평가
3. `Liquid-CLI + Unsloth` / `Qwen3.5 + Unsloth` / `Qwen3.5~3.6 + HF+FSDP` 기반 SFT 및 학습

기준 날짜: `2026-05-02`

## 오늘 기준 핵심 상태

### 1. TB2-lite 평가

`tb2_lite`는 full Terminal-Bench 2.0보다 훨씬 빠르게 모델을 걸러내기 위한 replay 평가 레이어입니다.

주요 문서:

- 루트 성능 요약:
  [PERFORMANCE_SUMMARY_2026-04-26.md](./PERFORMANCE_SUMMARY_2026-04-26.md)
- 통합 결과:
  [TB2_LITE_RESULTS_2026-04-26.md](./tb2_lite/docs/TB2_LITE_RESULTS_2026-04-26.md)
- 소형/체크포인트 스윕:
  [TB2_LITE_CHECKPOINT_SWEEP_2026-04-26.md](./tb2_lite/docs/TB2_LITE_CHECKPOINT_SWEEP_2026-04-26.md)
- 미실험/스킵:
  [TB2_LITE_UNTESTED_2026-04-25.md](./tb2_lite/docs/TB2_LITE_UNTESTED_2026-04-25.md)
- `vLLM` 이슈:
  [TB2_LITE_VLLM_ISSUES_2026-04-26.md](./tb2_lite/docs/TB2_LITE_VLLM_ISSUES_2026-04-26.md)

vLLM 호환성 (2026-05-02 기준):

| 모델 계열 | vLLM 버전 | torch | CUDA | 비고 |
| --- | --- | --- | --- | --- |
| Qwen 계열 전체 | 0.7.3 | 2.7+ | 12.x | 안정 |
| LFM2 계열 | 0.14.1 | 2.9.1+cu128 | 12.8 | `Lfm2MoeForCausalLM` 지원 |
| Gemma4 계열 | 0.19.1 | 2.10.0+cu128 | 12.8 | `Gemma4ForCausalLM` 지원, vLLM 0.14.1은 미지원 |
| Gemma4 계열 | 0.20.0 | 2.11.0+cu130 | 13.0 | CUDA 13 필요, 현재 드라이버 미지원 |

- Gemma4와 LFM2는 서로 다른 vLLM 버전이 필요해서 동시 평가 시 별도 환경으로 실행해야 합니다.
- `PYTHONNOUSERSITE=1` + 직접 python 바이너리 경로 사용 필수. `source activate` + `PYTHONNOUSERSITE=1`만으로는 user-site torch가 누출됩니다.

#### TB2-lite 평가 방식

현재 빠른 모델 비교는 `tb2_lite` replay 평가를 기준으로 합니다. 이 평가는 full Terminal-Bench 2.0 전체 실행이 아니라, 멀티턴 터미널 trajectory에서 **현재 step의 다음 행동을 얼마나 정확하게 고르는지**를 빠르게 보는 프록시입니다.

순서는 아래와 같습니다.

1. `tb2_lite/data/replay_full.jsonl`을 읽습니다.
   - 각 row는 하나의 replay step입니다.
   - 현재 기준 `386 step / 50 task`입니다.
   - row에는 그 시점까지의 누적 prompt와 정답 행동 `ref_raw`가 같이 들어 있습니다.
2. `tb2_lite/scripts/replay_eval.py`가 모델에게 각 prompt당 응답 1개를 생성하게 합니다.
   - 기본 엔진은 `vLLM`입니다.
   - multimodal config가 섞인 모델은 `language-model-only`로 text-only 추론을 강제합니다.
3. `tb2_lite/scripts/replay_metrics.py`가 응답을 파싱합니다.
   - 먼저 JSON을 파싱합니다.
   - JSON 내부 `commands[].keystrokes`를 우선적으로 명령어로 해석합니다.
   - JSON 파싱이 실패하면 fallback regex로 `"keystrokes"` 패턴만 제한적으로 회수합니다.
4. 각 step마다 정답 명령과 예측 명령을 shell token 단위로 비교합니다.
   - `first_cmd_exact`
   - `command_precision`
   - `command_recall`
   - `command_f1`
   를 계산합니다.
5. 전체 평균을 집계합니다.
   - `valid_json_pct`
   - `first_cmd_exact_pct`
   - `avg_command_f1`
   - `by_bucket` (`early/mid/late`)
   - `by_source_group`
6. 최종 점수는 `next_action_score` 하나로 요약합니다.
   - 공식:
   - `next_action_score = 100 * (0.7 * avg_command_f1 + 0.3 * first_cmd_exact_pct / 100)`

이 평가는 잘 보는 것과 못 보는 것이 분명합니다.

- 잘 보는 것:
  - 짧고 정확한 첫 명령 선택
  - 현재 step에서 바로 필요한 탐색/수정 행동
  - JSON 포맷 순응도
- 잘 못 보는 것:
  - 실제 command execution 성공 여부
  - 파일 결과 correctness
  - 장기 계획과 에러 복구를 포함한 진짜 closed-loop agent 능력

즉 `tb2_lite`는 **빠른 비교용 프록시**입니다. 특히 큰 모델이 장황한 분석, transcript continuation, 결과 서술로 흐르면 실제 잠재력보다 훨씬 낮게 나올 수 있습니다.

#### 용어 설명

- `trajectory`
  - 하나의 태스크를 해결하는 전체 멀티턴 터미널 기록입니다.
  - 예를 들어 `ls -> cat -> edit -> run -> verify`처럼 여러 step이 이어진 전체 흐름을 뜻합니다.
- `replay step`
  - trajectory 전체 중 한 시점을 잘라서, 그 문맥 다음에 어떤 행동을 해야 하는지 묻는 평가 단위입니다.
- `prompt`
  - 현재 step까지 모델에게 보여주는 누적 문맥입니다.
  - 이전 사용자 요청, assistant 응답, 터미널 출력이 함께 들어갑니다.
- `ref_raw`
  - 그 step에서 데이터셋이 정답으로 가지고 있는 assistant 원문입니다.
  - evaluator는 여기서 reference command를 추출합니다.
- `commands[].keystrokes`
  - evaluator가 가장 우선적으로 읽는 명령 필드입니다.
  - 실제 shell command나 엔터 단위 입력이 여기에 들어갑니다.
- `valid_json_pct`
  - 모델 응답 중 evaluator가 JSON으로 정상 파싱한 비율입니다.
  - 이 수치가 낮으면 command를 거의 못 읽었을 가능성이 큽니다.
- `first_cmd_exact_pct`
  - 모델이 낸 첫 명령이 정답 첫 명령과 완전히 같았던 비율입니다.
  - 현재 점수는 이 수치에 꽤 민감합니다.
- `avg_command_f1`
  - 예측 명령과 정답 명령을 token 단위로 비교했을 때의 평균 F1입니다.
  - 명령 일부만 맞아도 부분 점수가 반영됩니다.
- `next_action_score`
  - 현재 저장소에서 쓰는 대표 요약 점수입니다.
  - `avg_command_f1`과 `first_cmd_exact_pct`를 합쳐 한 숫자로 만든 값입니다.


현재 `tb2_lite` 점수 확정 모델은 `57개`입니다.


현재 `tb2_lite` 전체 비교 순위:

| 순위 | 모델 | Score | Cmd F1 | First Cmd Exact | Sec/Step | Load(s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `gyung/LFM2-8B-Terminal-SFT-Unsloth` | 30.14 | 0.2965 | 31.3% | 0.029 | 46.6 |
| 2 | `nvidia/Nemotron-Terminal-8B` | 30.02 | 0.2969 | 30.8% | 0.078 | 49.8 |
| 3 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount` | 29.77 | 0.2912 | 31.3% | 0.030 | 52.6 |
| 4 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-1Epoch-FullFT-SameCount` | 29.66 | 0.2917 | 30.8% | 0.029 | 52.2 |
| 5 | `nvidia/Nemotron-Terminal-32B` | 29.13 | 0.2872 | 30.1% | 0.281 | 99.1 |
| 6 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-1Epoch-FullFT-2BData` | 28.89 | 0.3185 | 22.0% | 0.063 | 85.2 |
| 7 | `LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 28.84 | 0.3065 | 24.6% | 0.170 | 99.7 |
| 8 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData` | 28.80 | 0.3094 | 23.8% | 0.084 | 73.9 |
| 9 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth` | 28.78 | 0.2821 | 30.1% | 0.025 | 30.3 |
| 10 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth` | 28.51 | 0.2796 | 29.8% | 0.025 | 32.5 |
| 11 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth` | 28.32 | 0.2781 | 29.5% | 0.029 | 29.1 |
| 12 | `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.32 | 0.2756 | 30.1% | 0.030 | 34.0 |
| 13 | `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.30 | 0.2778 | 29.5% | 0.025 | 30.3 |
| 14 | `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth-checkpoint-27` | 28.15 | 0.2779 | 29.0% | 0.027 | 40.3 |
| 15 | `nvidia/Nemotron-Terminal-14B` | 27.72 | 0.2751 | 28.2% | 0.108 | 68.9 |
| 16 | `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-1Epoch-FullFT-2BData` | 27.45 | 0.3021 | 21.0% | 0.083 | 86.5 |
| 17 | `Qwen/Qwen3.5-9B` | 27.04 | 0.2808 | 24.6% | 0.072 | 90.6 |
| 18 | `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData` | 26.80 | 0.2907 | 21.5% | 0.065 | 60.3 |
| 19 | `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 26.75 | 0.2956 | 20.2% | 0.226 | 108.5 |
| 20 | `Qwen/Qwen3.5-2B` | 26.52 | 0.2743 | 24.4% | 0.024 | 84.7 |
| 21 | `Qwen/Qwen3.5-4B` | 26.36 | 0.2745 | 23.8% | 0.055 | 77.2 |
| 22 | `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | 26.35 | 0.2744 | 23.8% | 0.282 | 113.4 |
| 23 | `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 26.20 | 0.2865 | 20.5% | 0.060 | 439.0 |
| 24 | `google/gemma-4-26B-A4B-it` | 25.95 | 0.2631 | 25.1% | 0.094 | 123.9 |
| 25 | `LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 25.74 | 0.2721 | 22.3% | 0.166 | 99.9 |
| 26 | `LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 25.74 | 0.2700 | 22.8% | 0.065 | 768.9 |
| 27 | `Qwen/Qwen3.6-27B` | 25.60 | 0.2702 | 22.3% | 0.282 | 118.5 |
| 28 | `LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 25.58 | 0.2698 | 22.3% | 0.066 | 547.6 |
| 29 | `Qwen/Qwen3.6-35B-A3B-FP8` | 25.21 | 0.2658 | 22.0% | 0.082 | 120.5 |
| 30 | `google/gemma-4-31B-it` | 24.70 | 0.2594 | 21.8% | 0.404 | 101.4 |
| 31 | `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 24.41 | 0.2720 | 17.9% | 0.222 | 103.6 |
| 32 | `LiquidAI/LFM2-2.6B` | 24.12 | 0.2379 | 24.9% | 0.032 | 36.4 |
| 33 | `google/gemma-4-E4B-it` | 23.43 | 0.2391 | 22.3% | 0.051 | 129.7 |
| 34 | `LiquidAI/LFM2.5-1.2B-Instruct` | 23.36 | 0.2381 | 22.3% | 0.021 | 31.7 |
| 35 | `LiquidAI/LFM2-8B-A1B` | 23.19 | 0.2336 | 22.8% | 0.025 | 58.2 |
| 36 | `google/gemma-4-E2B-it` | 23.05 | 0.2359 | 21.8% | 0.032 | 116.0 |
| 37 | `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-Unsloth` | 22.84 | 0.2586 | 15.8% | 0.028 | 89.9 |
| 38 | `LiquidAI/LFM2-24B-A2B` | 22.80 | 0.2323 | 21.8% | 0.050 | 81.6 |
| 39 | `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 22.40 | 0.2433 | 17.9% | 0.060 | 302.8 |
| 40 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 18.12 | 0.2135 | 10.6% | 0.180 | 383.6 |
| 41 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 16.76 | 0.1961 | 10.1% | 0.182 | 384.6 |
| 42 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 14.08 | 0.1621 | 9.1% | 0.079 | 303.9 |
| 43 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-2BData (rp=1.05, min_p=0.15)` | 15.41 | 0.1734 | 10.9% | 0.094 | 303.9 |
| 44 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-2BData (rp=1.05, min_p=0.15)` | 13.24 | 0.1592 | 7.0% | 0.101 | 262.0 |
| 45 | `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 13.09 | 0.1570 | 7.0% | 0.080 | 262.0 |
| 44 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData (rp=1.05)` | 17.95 | 0.2132 | 10.1% | 0.187 | 383.6 |
| 45 | `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData (rp=1.05)` | 16.01 | 0.1975 | 7.3% | 0.190 | 384.6 |
| 46 | `LLM-OS-Models/Ouro-2.6B-Terminal-SFT` (transformers) | 13.39 | 0.1368 | 12.7% | 195.079 | 1.9 |
| 47 | `LLM-OS-Models/Ouro-1.4B-Terminal-SFT` (transformers) | 11.45 | 0.1134 | 11.7% | 98.671 | 1.4 |
| 48 | `ByteDance/Ouro-1.4B` | 9.08 | 0.0954 | 8.0% | 0.212 | 105.5 |
| 49 | `ByteDance/Ouro-2.6B-Thinking` | 7.22 | 0.0731 | 7.0% | 0.910 | 143.5 |
| 50 | `ByteDance/Ouro-2.6B` | 6.68 | 0.0676 | 6.5% | 0.659 | 116.4 |
| 51 | `ByteDance/Ouro-1.4B-Thinking` | 6.56 | 0.0658 | 6.5% | 0.411 | 120.6 |
| 52 | `LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT` (transformers) | 6.49 | 0.0649 | 6.5% | 23.482 | 2.3 |
| 53 | `LLM-OS-Models/Ouro-2.6B-Thinking-Terminal-SFT` (transformers) | 6.64 | 0.0662 | 6.7% | 66.200 | 2.2 |
| 50 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-DDP-4GPU` | 6.79 | 0.0691 | 6.5% | 0.039 | 165.2 |
| 50 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 6.54 | 0.0656 | 6.5% | 0.119 | 326.0 |
| 51 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 6.51 | 0.0652 | 6.5% | 0.123 | 326.0 |
| 52 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-1Epoch-DDP-4GPU` | 6.65 | 0.0672 | 6.5% | 0.039 | 165.0 |
| 53 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` | 6.49 | 0.0648 | 6.5% | 0.821 | 82.3 |
| 54 | `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-1Epoch-HF-FSDP-2BData` | 6.49 | 0.0648 | 6.5% | 0.821 | 99.4 |
| 55 | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-2Epoch-DDP-4GPU` | 6.49 | 0.0648 | 6.5% | 0.075 | 163.6 |
| 56 | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-1Epoch-DDP-4GPU` | 6.49 | 0.0648 | 6.5% | 0.075 | 163.3 |

#### LFM corrected 재학습 vLLM 재평가 (2026-05-07)

요청에 따라 기존 README 순위표와 최대한 같은 조건으로 다시 평가했습니다.

평가 조건:

- 데이터: git HEAD 기준 기존 `tb2_lite/data/replay_full.jsonl` (`386 step / 50 task`)
- 평가 코드: git HEAD 기준 기존 `replay_eval.py` + `replay_metrics.py`
- 백엔드: `vLLM`
- 생성 설정: `temperature=0.0`, `max_tokens=1024`, `max_model_len=8192`
- 컨텍스트 확인: LFM tokenizer 기준 최대 prompt 길이 `5245 tokens`, 따라서 `8192` 안에 들어가며 overflow 없음
- 주의: README의 LFM2 권장 vLLM 0.14.1 환경은 현재 torch ABI 불일치로 `vllm._C` import에서 실패. 데이터/코드/metric/sampling은 기존 조건 그대로 두고, 실제 동작 가능한 vLLM 0.19.1로 평가함

기존 386-step 조건 결과:

| 모델 | checkpoint | Score | Cmd F1 | First Cmd Exact | Valid JSON | 기존표 기준 위치 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `LFM2-8B-A1B corrected` | `checkpoint-830` | **15.21** | 0.1475 | 16.3% | 19.9% | `15.41` 아래, `14.08` 위 |
| `LFM2.5-1.2B corrected` | `checkpoint-545` | **14.99** | 0.1430 | 16.6% | 11.9% | `14.08` 위 |
| `LFM2-8B-A1B corrected` | `checkpoint-1660` | **14.98** | 0.1463 | 15.8% | 18.7% | `14.08` 위 |
| `LFM2.5-1.2B corrected` | `checkpoint-1090` | **14.63** | 0.1413 | 15.8% | 13.0% | `14.08` 위 |

해석:

- 기존표 기준으로는 네 모델 모두 `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-2BData`의 `14.08`보다 높고, `LFM2-24B-A2B e2 (rp=1.05, min_p=0.15)`의 `15.41`보다 낮습니다.
- 동일 386-step 조건에서 과거 Liquid-CLI 재학습 `sft_h200_7gpu_processed_checkpoint_831`도 `15.13`이었습니다. 따라서 이번 corrected 8B의 `15.21`은 같은 legacy evaluator에서는 소폭 개선이지만, 기존 README 1위 `gyung/LFM2-8B-Terminal-SFT-Unsloth`의 `30.14`를 재현하지는 못했습니다.
- 가장 큰 하락 원인은 legacy evaluator에서 JSON 순응도가 낮게 잡힌 점입니다. 8B best checkpoint도 `valid_json 19.9%`, 1.2B는 `11.9-13.0%`라서 evaluator가 command를 안정적으로 회수하지 못합니다.
- bucket 기준으로는 `early`는 비교적 높지만(`8B checkpoint-830 early F1 0.3194`), `mid`가 `0.029`로 거의 무너집니다. legacy 386 프롬프트 구조에서 중간 step의 문맥/다음 행동 정렬이 특히 약합니다.
- 카테고리별로는 `swe`, `math`, `model_training`, `security`가 상대적으로 낫고, `data_querying`, `system_administration`, `scientific_computing`, `data_science`가 약합니다.
- 8B는 epoch 1(`checkpoint-830`)이 epoch 2(`checkpoint-1660`)보다 좋습니다. 1.2B도 epoch 1(`checkpoint-545`)이 epoch 2(`checkpoint-1090`)보다 좋습니다. 이 조건에서는 2 epoch가 포맷/첫 명령 정확도를 더 개선하지 못하고 약간 과적합 또는 출력 스타일 drift가 생긴 것으로 봅니다.

별도로, corrected 303-step cumulative replay에서는 같은 모델들이 전혀 다른 점수를 보였습니다.

| 모델 | checkpoint | Score | Cmd F1 | First Cmd Exact | Valid JSON | 비교 가능성 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `LFM2-8B-A1B corrected` | `checkpoint-830` | **31.51** | 0.3241 | 29.4% | 56.8% | 기존 386-step 순위표와 직접 비교 금지 |
| `LFM2-8B-A1B corrected` | `checkpoint-1660` | **30.41** | 0.3102 | 29.0% | 54.8% | 기존 386-step 순위표와 직접 비교 금지 |
| `LFM2.5-1.2B corrected` | `checkpoint-1090` | **28.75** | 0.2864 | 29.0% | 50.5% | 기존 386-step 순위표와 직접 비교 금지 |
| `LFM2.5-1.2B corrected` | `checkpoint-545` | **28.49** | 0.2810 | 29.4% | 50.5% | 기존 386-step 순위표와 직접 비교 금지 |

이 303-step 결과는 invalid/empty reference 제거, cumulative conversation prompt, 통합 prompt builder 적용 후의 결과입니다. 모델 자체가 완전히 망가진 것이 아니라, legacy 386 evaluator의 prompt/reference 구조와 corrected template 학습 결과가 강하게 충돌한다는 신호로 봐야 합니다.

현재 corrected run의 학습 상태:

- 학습 완료: `LFM2-8B-A1B`, `LFM2.5-1.2B`
- 아직 미완료: `LFM2-2.6B`, `LFM2-24B-A2B`
- `24B`는 corrected pipeline에서 아직 학습 결과가 없습니다. README의 기존 `24B SFT 14.08`은 이전 HF+FSDP 실험 결과이며, Qwen-format text 재사용, assistant-only masking 누락, 짧은 `MAX_SEQ_LENGTH`, 평가 prompt mismatch가 섞인 낮은 점수입니다. corrected 24B의 성능 판단에는 쓰면 안 됩니다.

핵심 해석:

- `Qwen3.5-2B`는 LoRA보다 full FT가 압도적으로 좋았습니다.
- `Qwen3.5-4B`는 `1 epoch`가 좋았고, `2 epoch final`은 오히려 내려갔습니다.
- `Qwen3.5-9B`는 반대로 `2 epoch final`이 더 좋았습니다.
- `Qwen3.5-27B`는 학습은 잘 됐지만, 점수는 `2 epoch 26.75`, `1 epoch 24.41`로 기대보다 낮았습니다.
- `Qwen3.5-35B-A3B`도 학습/평가는 성공했지만, `2 epoch 26.20`, `1 epoch 22.40`으로 기대보다 낮았습니다.
- `Qwen3.6-27B`는 `2 epoch 28.84`로 크게 회복됐고, 현재 전체 `7위`입니다. 다만 `1 epoch 25.74`는 아직 낮아서, 이 모델도 `2 epoch`까지 가야 성능이 붙었습니다.
- `Qwen3.6-35B-A3B`는 `1 epoch 25.58`, `2 epoch 25.74`로 아주 조금만 좋아졌고, 현재 전체 `26위` 수준입니다. 즉 `3.6`으로 올려도 `35B-A3B` 계열은 여전히 큰 개선이 없었습니다.
- `Gemma 4 E2B/E4B` DDP 실험은 **출력 포맷 적합성에서 무너진 케이스**입니다. JSON `commands` 대신 transcript continuation, 로그 복사, 결과 서술로 흘렀습니다. 수치로 보면 `E2B 2 epoch valid_json 1.0%`, `E4B 2 epoch valid_json 2.8%`라서 evaluator가 실제 명령을 거의 못 잡았습니다.

#### Gemma4-26B-A4B / LFM2-24B-A2B SFT 결과 (2026-05-02)

이번 HF+FSDP 학습으로 Gemma4-26B-A4B, Gemma4-31B, LFM2-24B-A2B 세 모델을 추가 평가했습니다.

결과 요약:

| 모델 | Score | Cmd F1 | First Cmd Exact | Valid JSON | Sec/Step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Gemma4-26B-A4B` e2 | 18.12 | 0.2135 | 10.6% | 22.5% | 0.180 |
| `Gemma4-26B-A4B` e1 | 16.76 | 0.1961 | 10.1% | 12.4% | 0.182 |
| `LFM2-24B-A2B` e2 | 14.08 | 0.1621 | 9.1% | 47.7% | 0.079 |
| `LFM2-24B-A2B` e1 | 13.09 | 0.1570 | 7.0% | 54.7% | 0.080 |
| `Gemma4-31B` e1/e2 | 6.49 | 0.0648 | 6.5% | 0.0% | 0.821 |
| `Gemma4-E4B` e2 (HF+FSDP) | 14.75 | 0.1708 | 9.3% | 13.5% | 67.9 |
| `Gemma4-E4B` e1 (HF+FSDP) | 10.50 | 0.1234 | 6.2% | 7.3% | 69.6 |
| `Qwen3.5-27B` e1 (transformers) | 16.42 | 0.1935 | 9.6% | 53.9% | 71.7 |
| `Gemma4-31B` e1 (transformers) | 8.06 | 0.0941 | 4.9% | 9.1% | 85.5 |
| `Gemma4-31B` e2 (transformers) | 7.46 | 0.0864 | 4.7% | 16.3% | 93.0 |

왜 기대보다 낮게 나왔는가:

- **Gemma4-26B-A4B**는 base 모델(`google/gemma-4-26B-A4B-it`)이 `Score 25.95`였는데, SFT 후 `18.12`로 오히려 하락했습니다. 원인은 `valid_json 22.5%`로 base의 JSON 포맷 순응도를 크게 잃었고, `first_cmd_exact 10.6%`도 base의 `25.1%`보다 절반 이하입니다. SFT 데이터의 포맷이 모델 본래 출력 스타일과 충돌해 포맷 능력이 훼손된 것으로 보입니다. 학습은 성공했지만 **SFT 데이터 포맷 → TB2-lite evaluator 기대 포맷 간 불일치**가 핵심 원인입니다.
- **LFM2-24B-A2B**도 마찬가지 패턴입니다. base `LiquidAI/LFM2-24B-A2B`가 `Score 22.80`인데 SFT 후 `14.08`로 하락. `valid_json 47.7%`는 base보다 낮고, `first_cmd_exact 9.1%`도 base의 `21.8%`보다 크게 낮습니다. 다만 base 대비 JSON 포맷 유지율은 Gemma4보다 높아(LFM2 47.7% vs Gemma4 22.5%), 모델 자체의 포맷 학습 능력 차이가 있습니다.
- **Gemma4-31B**는 **완전히 고장**났습니다. `valid_json 0.0%`, `first_cmd_exact 6.5%`, e1과 e2가 완전히 동일한 점수입니다. 원인은 31B의 아키텍처가 26B와 다르기 때문입니다. 26B는 MoE 구조(`experts` 레이어)인데, 31B는 dense 구조(`mlp` + `layer_scalar`)입니다. 학습 코드의 `load_gemma4_text_only_model()` 가중치 추출이 31B 아키텍처를 제대로 처리하지 못해 **모델 가중치가 손상**된 것으로 판단됩니다. 출력이 반복적인 무의미 텍스트("la la la de la la")로 나옵니다.

공통 결론:

- **Gemma4/LFM2 대형 모델 SFT는 현재 데이터셋과 포맷 설정에서 역효과**입니다. base 모델이 이미 어느 정도 JSON 출력을 할 수 있는데, SFT가 오히려 이 능력을 훼손합니다.
- Qwen 계열은 SFT 후 점수가 올랐는데(예: Qwen3.5-2B base 26.52 → SFT 29.77), Gemma4/LFM2은 내려갔습니다. 이는 **모델 계열별 SFT 데이터 호환성 차이**입니다.
- Gemma4-E4B HF+FSDP 재학습 평가 완료. DDP 대비 개선 (6.49→14.75 e2, 6.49→10.50 e1), 하지만 base(23.43) 대비 여전히 하락. `valid_json 13.5%`로 DDP(2.8%)보다 개선됐으나 여전히 낮음. E2B HF+FSDP 학습도 완료됨.
- Qwen3.5-27B e1 transformers 평가 = 16.42 (기존 vLLM 평가는 24.41). transformers 백엔드에서 점수 하락.
- 근본적으로 **Gemma 4 계열은 Nemotron-Terminal-Corpus 포맷과 맞지 않는 출력 스타일**을 가지고 있으며, 데이터셋 포맷을 Gemma 4 chat template에 맞게 변환하거나 평가 파이프라인을 조정해야 합니다.

#### ByteDance Ouro 모델 평가 결과 (최종: 2026-05-06)

Ouro는 ByteDance의 루프드(Looped) 언어 모델로, 동일 24층 스택을 4회 반복(total_ut_steps=4)하는 recurrent depth 아키텍처입니다. Thinking 변형은 수학/과학 추론 데이터로 추가 SFT된 버전입니다.

**Base 모델 결과:**

| 모델 | Score | Cmd F1 | First Cmd Exact | Valid JSON | Sec/Step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Ouro-1.4B` | 9.08 | 0.0954 | 8.0% | 9.6% | 0.212 |
| `Ouro-2.6B` | 6.68 | 0.0676 | 6.5% | 6.5% | 0.659 |
| `Ouro-2.6B-Thinking` | 7.22 | 0.0731 | 7.0% | 9.1% | 0.910 |
| `Ouro-1.4B-Thinking` | 6.56 | 0.0658 | 6.5% | 10.4% | 0.411 |

**SFT 결과:**

| 모델 | Score | Cmd F1 | First Cmd Exact | Valid JSON | 변화 |
| --- | ---: | ---: | ---: | ---: | --- |
| `Ouro-1.4B` base | 9.08 | 0.0954 | 8.0% | 9.6% | - |
| `Ouro-1.4B` SFT | **11.45** | **0.1134** | **11.7%** | 9.8% | **+26%** |
| `Ouro-2.6B` base | 6.68 | 0.0676 | 6.5% | 6.5% | - |
| `Ouro-2.6B` SFT | **13.39** | **0.1368** | **12.7%** | **17.9%** | **+100%** |
| `Ouro-1.4B-Thinking` base | 6.56 | 0.0658 | 6.5% | 10.4% | - |
| `Ouro-1.4B-Thinking` SFT | 6.49 | 0.0649 | 6.5% | 6.5% | **-1%** |
| `Ouro-2.6B-Thinking` SFT | 6.64 | 0.0662 | 6.7% | 8.0% | **-8%** |

**HF 업로드**: `LLM-OS-Models/Ouro-1.4B-Terminal-SFT`, `LLM-OS-Models/Ouro-2.6B-Terminal-SFT`, `LLM-OS-Models/Ouro-1.4B-Thinking-Terminal-SFT`, `LLM-OS-Models/Ouro-2.6B-Thinking-Terminal-SFT`

---

**Ouro가 다른 모델들보다 현저히 성능이 낮은 원인 분석**

참조: 상위 모델 Qwen3.5-2B SFT = 29.77점, 최하위 Ouro-2.6B = 6.68점. **약 4.5배 격차.**

**1. 구조적 문제: 루프드 아키텍처의 터미널 태스크 부적합**

Ouro는 24층 transformer를 4회 반복(recurrent depth)하는 아키텍처입니다. 이 구조는 수학/논리 추론처럼 "깊은 사고"가 필요한 태스크에서는 이점이 있지만, 터미널 명령 생성에는 오히려 방해가 됩니다:
- `ls -la` 같은 단순 명령을 생성하는 데 4번의 recurrent pass가 과도한 연산
- 모델이 간단한 명령을 "과대사고"하여 불필요하게 복잡한 응답을 생성
- Sec/Step이 타 모델 대비 매우 높음 (Ouro-1.4B: 0.212s vs Qwen3.5-2B: 0.030s)

**2. 근본 원인: JSON 구조화 출력 불가 (valid_json < 11%)**

이것이 가장 치명적인 문제입니다. 다른 모델들은 JSON 포맷으로 명령을 출력하지만, Ouro는 **90%가 넘는 스텝에서 JSON 파싱에 실패**합니다:
- Qwen3.5-2B base: valid_json 2.4% → SFT 후 **100%**
- Ouro-1.4B base: valid_json 9.6% → SFT 후 **9.8%** (거의 변화 없음)
- SFT 데이터가 JSON 포맷을 가르치지만, Ouro의 recurrent 구조가 포맷 준수를 어렵게 만듦
- JSON 파싱 실패 시 `score_commands()`가 제대로 작동하지 않아 F1도 0에 가깝게 나옴

**3. 학습 데이터 편향: 수학/과학에 특화, 터미널 경험 전무**

Ouro의 사전학습 데이터 구성:
- 수학: 3.5M 샘플 | 코드: 3.2M | 과학: 808K
- 터미널/시스템 관리 도메인: 사실상 0
- 결과적으로 `system_administration`, `file_operations`, `data_querying` 등 터미널 핵심 카테고리에서 F1 ≈ 0

**4. 모델 크기 vs 성능 역상**

| 모델 | 파라미터 | Base | SFT | 변화 |
|------|---------|------|-----|------|
| Ouro-1.4B | 1.4B | **9.08** | 11.45 | +26% |
| Ouro-2.6B | 2.6B | 6.68 | **13.39** | **+100%** |
| Ouro-1.4B-Thinking | 1.4B | 6.56 | 6.49 | -1% |
| Ouro-2.6B-Thinking | 2.6B | 7.22 | 6.64 | **-8%** |

더 큰 모델이 더 낮은 점수를 기록합니다. 이는 일반적인 모델 스케일링 법칙과 반대되는 현상으로:
- 루프드 아키텍처에서 모델이 커질수록 과도한 연산량이 더 심화됨
- Thinking 변형은 수학 추론 데이터로 추가 학습되어 터미널 태스크에서 더 큰 편향을 보임

**5. SFT 효과 분석**

**2.6B base (성공, +100%)**: base 대비 정확히 2배 향상. 모든 지표 개선: F1 0.0676→0.1368 (+102%), first_cmd_exact 6.5%→12.7% (+95%), valid_json 6.5%→17.9% (+175%). 하지만 절대값은 여전히 낮음 — 상위 Qwen3.5-2B SFT(29.77)와 비교하면 2.2배 격차. base가 워낙 낮아서(6.68) 상대적 향상폭은 크지만 실질적 차이는 제한적.

**1.4B base (성공, +26%)**: 모델이 작아 SFT 데이터로 빠르게 적응. 하지만 절대적 향상폭은 작음 (+2.37점). Qwen3.5-2B의 +3.25점과 비교하면 근본적 한계가 큼.

**1.4B-Thinking (실패, -1%)**: Thinking 모델은 수학/과학 추론으로 강하게 편향되어 있어, 2 epoch SFT로는 터미널 도메인 적응이 거의 불가능:
- `system_administration` F1: base 0.0 → SFT 0.0001 (변화 없음)
- `file_operations` F1: base 0.0057 → SFT 0.0 (오히려 하락)
- `data_science`, `data_querying` 등: 여전히 F1 = 0.0
- 반면 `swe`, `math`, `code`는 SFT 전후 동일 → 이 카테고리들은 원래 학습 데이터와 겹쳐서 이미 고정됨

**2.6B-Thinking SFT (실패, -8%)**: base 7.22 → SFT 6.64. 1.4B-Thinking(-1%)와 동일한 패턴. Thinking 모델은 수학/과학 추론에 강하게 편향되어 있어 터미널 SFT로 도메인 전환 불가. 오히려 SFT가 기존 능력을 일부 훼손.

**6. 카테고리별 패턴: 추론 도메인 vs 실행 도메인**

Ouro가 상대적으로 나은 카테고리: `swe` (0.24), `math` (0.22), `model_training` (0.20), `code` (0.13) → 이들은 수학/코드 사전학습과 겹치는 "추론 중심" 태스크

완전히 실패하는 카테고리: `system_administration` (0.00), `data_querying` (0.00), `data_science` (0.00), `file_operations` (0.01) → 터미널 "실행 중심" 태스크. 실제 셸 명령을 정확히 생성해야 하는데, 이 경험이 전무함.

**결론**: Ouro의 루프드 아키텍처는 터미널 에이전트 태스크에 근본적으로 부적합합니다. JSON 구조화 출력 학습이 안 되고, 사전학습 데이터가 터미널 도메인을 전혀 커버하지 않으며, SFT로도 이 두 가지 근본적 한계를 극복하기 어렵습니다. Qwen3.5-2B 대비 1/3 수준의 성능은 아키텍처적 한계로 판단됩니다.

---

**Ouro 논문 기반 심층 분석** (arXiv:2510.25741)

Ouro는 ByteDance에서 개발한 Looped Language Model (LoopLM)로, 24층 transformer를 4회 반복(recurrent depth=4)하는 아키텍처입니다. 7.7T 토큰으로 사전학습되었으며, 1.4B/2.6B 파라미터로 4B~12B급 표준 transformer와 동등한 성능을 달성하는 것이 목표입니다.

**논문이 증명한 핵심: LoopLM의 강점은 "지식 조작"이지 "지식 저장"이 아님**

논문 Section 6에서 통제 실험으로 증명한 내용:
- 루핑을 해도 **파라미터당 ~2비트**의 지식 용량은 변하지 않음 (looped/non-looped 동일)
- 강점은 이미 아는 지식을 "조합/추론"하는 능력 (multi-hop QA, modular arithmetic)
- 터미널 명령은 **사전학습에 없던 새로운 지식**이 필요 → 루핑으로 해결 안 됨
- 즉, Ouro의 아키텍처적 우위는 "모르는 것을 잘하는" 게 아니라 "아는 것을 잘 조합하는" 것에 국한

**LoopLM이 강한 태스크 (논문 Table 7-9)**

| 벤치마크 | Ouro-2.6B | 비교 모델 | 비고 |
|----------|-----------|---------|------|
| MATH500 | **90.85** | Qwen3-8B: 62.30 | 수학 추론 1.46배 |
| BBH | **80.46** | Qwen3-8B: 77.65 | 논리 추론 |
| GSM8K | 78.92 | Qwen3-8B: 85.19 | 수학 응용 |
| AIME25 pass@1 | 50.3 | Qwen3-4B: 51.3 | 수학 경시 |
| OlympiadBench | **76.44** | Qwen3-8B: 75.25 | 올림피아드 |

수학/논리/과학 추론에서 8B급 모델과 동등하거나 우세. 이는 "깊은 사고"가 필요한 태스크에서 루핑이 유효함을 보여줌.

**LoopLM이 약한 태스크 — 터미널 에이전트가 여기에 해당**

터미널 명령 생성은 논문이 강점으로 주장하는 능력과 정반대되는 특성을 가집니다:

| LoopLM 강점 | 터미널 태스크 특성 | 불일치 |
|-------------|-----------------|--------|
| 복잡한 다단계 추론 | 단순한 다음 명령 1개 | 과도한 연산 |
| 지식 조합/재구성 | 정확한 명령어 회상 | 지식 저장 필요 |
| "깊은 사고"에 유리 | 빠르고 정확한 반응 | 속도 저하 |
| 반복적 정제가 도움 | 한 번에 정확해야 함 | 오버피팅 |
| Latent reasoning | 구조화된 JSON 출력 | 포맷 준수 불가 |

**RL 불가 — 논문 자체에서도 실패 인정**

논문 Section 4.5에서 DAPO/GRPO 시도 결과:
> "These attempts did not yield significant performance gains over the final SFT checkpoint."

원인: vLLM/SGLang이 LoopLM의 동적 early-exit을 지원하지 않아 RL rollout 불가. Fixed 4-round RL은 학습은 되지만 SFT 체크포인트를 넘지 못함. 인프라 호환성 + SFT 이후 한계치 도달이 복합 원인.

**Mid-training (사전강화학습) 가능성**

논문의 Stage 4 mid-training은 이미 존재 (300B 토큰, 고 품질 SFT 포맷 데이터). 하지만 터미널 데이터는 전무. 추가 mid-training으로 셸 명령/JSON 데이터를 수십억 토큰 주입하면:
- 예상 효과: **현재 13.39 → 20~25점** 수준 (50~80% 향상)
- 한계: 지식 용량이 ~2 bits/param으로 고정되어 Qwen3.5-2B 수준(29.77) 도달은 어려움
- RL은 논문에서도 포기한 영역이라 현실적으로 불가

**Ouro의 적합/부적합 도메인 요약**

적합: 수학 경시, 논리 퍼즐, 과학 추론, 코드 생성(복잡한 알고리즘), 안전성 검증(루핑이 안전 향상)
부적합: 터미널 에이전트, 구조화 출력(JSON), 실시간 반응, 간단한 명령 생성, 도메인 특화 지식 필요 태스크

---

**파이프라인 근본 문제점 (전 모델에 영향)**

Ouro 분석 외에도 전체 파이프라인에 구조적 문제가 있어, 상위 모델들의 실제 잠재력보다 점수가 낮게 나올 가능성이 큽니다.

**1. 학습/평가 카테고리 불일치 68.4%**

`prepare_dataset.py`의 필터가 평가 데이터 386스텝의 **264스텝(68.4%)**을 제외합니다:

| 분류 | 카테고리 | 스텝 | 비율 |
|------|---------|------|------|
| 학습 포함 | file_operations, data_processing, data_querying, dependency_management, security | 122 | 31.6% |
| 명시적 제외 | debugging(51), software_engineering(43), data_science(24), scientific_computing(23) | 141 | 36.5% |
| 암시적 드롭 | swe(30), system_administration(26), code(23), model_training(22), math(22) | 123 | 31.9% |

평가의 2/3가 OOD(out-of-distribution)이므로 SFT 효과 측정이 본질적으로 부정확합니다.

**2. HF FSDP 대형 모델 프롬프트 마스킹 누락**

`train_qwen_hf_fsdp.py`의 "text" 모드에서 `labels = [ids.copy() for ids in input_ids]`로 전체 시퀀스에 loss 계산. user 프롬프트까지 학습하여 모델이 user 발화를 재생산하도록 훈련됨. 영향: Gemma4-26B, Gemma4-31B, LFM2-24B, Qwen 27B/35B 전부.

**3. MAX_SEQ_LENGTH=1024 (HF FSDP)**

터미널 trajectory 첫 user 메시지가 ~1,500-2,000 토큰. MAX_SEQ_LENGTH=1024는 거의 모든 실제 trajectory를 truncate. 대형 모델 SFT 실패의 큰 원인.

**4. 공유 SFT 데이터의 Qwen 전용 토큰**

Gemma4, LFM2 대형 모델이 모두 `qwen35_27b_processed_2bdata`로 학습. 이 데이터에 `<|im_start|>`/`<|im_end|>` Qwen 전용 토큰이 포함되어 있어 Gemma4/LFM2는 존재하지 않는 토큰으로 학습. README의 "Gemma4 valid_json 22.5%로 JSON 순응도 상실"과 일치.

**5. 평가 백엔드 불일치**

| 백엔드 | 챗 템플릿 | 영향 |
|--------|----------|------|
| replay_eval.py (vLLM) | `apply_chat_template` 적용 | 정상 |
| replay_eval_transformers.py | **raw prompt 그대로** | 챗 템플릿 미적용 |

Ouro, Gemma4-31B 등 transformers로 평가한 모델은 챗 템플릿 없이 평가되어 점수 과소평가 가능. 예: Qwen3.5-27B vLLM 24.41 → transformers 16.42 (-33%).

**예상 최고 점수 (파이프라인 수정 시)**

위 문제들을 모두 수정하면(필터 확장, 프롬프트 마스킹, 시퀀스 길이 확장, 데이터 포맷 정렬, 평가 백엔드 통일):

| 모델 | 현재 Score | 예상 Score | 근거 |
|------|-----------|-----------|------|
| Gemma4-26B-A4B SFT | 18.12 | **25-28** | base=25.95, SFT 데이터 포맷만 맞추면 base 수준 회복 + α |
| LFM2-24B SFT | 14.08 | **22-25** | base=22.80, 마스킹+길이 수정 시 base 이상 가능 |
| Qwen3.5-27B SFT | 26.75 | **28-30** | vLLM=24.41에서 이미 회복, 마스킹+길이 수정 시 추가 상승 |
| Qwen3.6-27B SFT | 28.84 | **29-31** | 이미 28.84, 여유 있음 |
| Ouro-2.6B SFT | 13.39 | **20-25** | mid-training으로 터미널 지식 주입 시 (하지만 RL 불가, 한계 존재) |

---

**다음 작업 계획 (우선순위순)**

파이프라인 수정으로 가장 큰 성능 향상을 기대할 수 있는 작업부터 정리합니다.

**Phase 1: 데이터/학습 수정 (예상 소요: 1-2일, 효과: 최대)**

| # | 작업 | 대상 | 효과 | 난이도 |
|---|------|------|------|--------|
| 1 | **데이터 필터 재검토** — `EXCLUDED_PREFIXES`에서 `software_engineering`, `debugging` 제거. `NON_CODING_PREFIXES`에 `system_administration`, `model_training` 추가. `dataset_adapters` split도 로드에 포함 | `prepare_dataset.py` 3종 | 평가 커버리지 31.6% → 80%+ | 낮 |
| 2 | **HF FSDP "text" 모드 프롬프트 마스킹** — `tokenize_text_dataset()`에서 `labels[:prompt_len] = [-100]` 처리. ChatML 포맷에서 `<\|im_start\|>assistant\n` 이전까지 마스킹 | `train_qwen_hf_fsdp.py` | 대형 모델 SFT 품질 대폭 개선 | 중 |
| 3 | **MAX_SEQ_LENGTH=1024 → 4096** — HF FSDP 대형 config 전체. gradient checkpointing 켜면 메모리 증가 최소화 | `qwen_sft/configs/*hf_fsdp*.env` | trajectory truncation 해소 | 낮 |
| 4 | **Gemma4 마스킹 토큰 수정** — `<\|turn\|>user\n` → `<start_of_turn>user\n`, `<\|turn\|>model\n` → `<start_of_turn>model\n` | `gemma4_sft/scripts/train_sft_unsloth_ddp.py` | Gemma4 E2B/E4B 마스킹 실패 해소 | 낮 |
| 5 | **SFT 데이터 계열별 분리** — Gemma4/LFM2 전용 데이터를 각 모델의 chat template으로 재생성 (`<\|im_start\|>` → `<start_of_turn>` 등) | `qwen35_27b_processed_2bdata` 분기 | Gemma4/LFM2 SFT 근본 개선 | 중 |

**Phase 2: 평가 파이프라인 수정 (예상 소요: 반나절)**

| # | 작업 | 효과 |
|---|------|------|
| 6 | **transformers 평가에 챗 템플릿 적용** — `replay_eval_transformers.py`에서 `tokenizer.apply_chat_template([{"role":"user","content":prompt}])` 사용 | vLLM/transformers 점수 일치. 현재 transformers 평가가 과소평가 |
| 7 | **step ≥ 1 평가에 시스템 instruction 재주입** — step 0의 시스템 instruction을 모든 step의 prompt 앞에 prepend | base 모델 valid_json 정상화 |
| 8 | **replay_metrics `lower()` 제거** — `tokenize_command()`에서 `.lower()` 제거 또는 옵션화 | Linux case-sensitive 경로 정확도 |
| 9 | **순위표에 백엔드 컬럼 추가** — vLLM/transformers/sampling 구분 표시 | 비교 가능성 |

**Phase 3: 재학습 (Phase 1 완료 후, 예상 소요: 2-3일)**

| 모델 | 수정사항 | 예상 결과 |
|------|---------|----------|
| **Gemma4-26B-A4B** | 필터 확장 + 마스킹 토큰 수정 + MAX_SEQ=4096 + Gemma4 포맷 데이터 | 18.12 → **25-28** |
| **LFM2-24B-A2B** | 필터 확장 + 프롬프트 마스킹 + MAX_SEQ=4096 | 14.08 → **22-25** |
| **Gemma4-E2B/E4B** | 마스킹 토큰 수정만으로도 큰 개선 | 6.49 → **15-20** (valid_json 회복) |
| **Gemma4-31B** | 가중치 로딩 버그 수정 (`dense` 구조 처리) 후 재학습 | 6.49 → **20-24** (체크포인트 복구) |
| **Qwen3.5-27B** | 프롬프트 마스킹 + MAX_SEQ=4096 | 26.75 → **28-30** |

**Phase 4: 코드 정리 (예상 소요: 반나절)**

| # | 작업 |
|---|------|
| 10 | `prepare_dataset.py` 3종 → 1종 통합 + `--family` 인자 |
| 11 | `train_sft_unsloth_ddp.py` 3종 → 1종 + `--instruction-part`/`--response-part` 인자 |
| 12 | 빈 placeholder 폴더 제거 (`scripts/`, `train/`, `tests/`) |
| 13 | `ouro_sft/`에 README + `prepare_dataset.py` 추가 (다른 SFT 폴더와 대칭) |

큰 모델이 오히려 떨어진 이유:

**1. 평가 구조: `next_action_score = 70% × Cmd F1 + 30% × first_cmd_exact`**

이 평가는 `다음 한두 개 명령을 얼마나 정확하게 고르느냐`에 크게 반응합니다. 큰 모델이 길게 설명하거나 여러 명령을 제안해도 첫 행동이 빗나가면 손해가 큽니다. 이건 편향이기도 하지만, 실제 약점이기도 합니다.

**2. 첫 명령 정확도 하락 (`first_cmd_exact`)**

| 모델 | first_cmd_exact | Score |
| --- | ---: | ---: |
| Qwen3.5-2B SFT | 31.3% | 29.77 |
| Qwen3.6-27B SFT | 24.6% | 28.84 |
| Qwen3.5-27B SFT | 20.2% | 26.75 |
| Qwen3.6-35B-A3B SFT | 22.8% | 25.74 |
| Gemma4-26B base | 25.1% | 25.95 |
| Gemma4-26B SFT | 10.6% | 18.12 |
| LFM2-24B base | 21.8% | 22.80 |
| LFM2-24B SFT | 9.1% | 14.08 |

모델이 클수록 첫 명령 정확도가 떨어지는 경향이 명확합니다. 특히 Gemma4/LFM2는 SFT 후 base 대비 절반 이하로 추락.

**3. 과한 계획/과한 행동열**

`Qwen3.5-27B`는 평균 예측 명령 수가 `5.11`로 너무 많았습니다. `2B`는 `0.43`, `35B`는 `1.49`였습니다. 큰 모델은 실제로는 꽤 맞는 명령을 여러 개 내지만, 이 벤치는 **과한 계획보다 짧고 맞는 첫 행동**을 더 좋게 칩니다.

실제 출력 패턴: 작은 상위 모델은 바로 JSON 안에 짧은 명령을 넣는데, 큰 모델은 문제 재서술, 장황한 계획, 잘못된 하위 과제 설정이 먼저 나오는 비율이 높습니다. `swe` 샘플에서 큰 모델은 `rm` 동작 수정 이슈를 보자마자 긴 설명과 계획을 먼저 쓰고, 어떤 경우엔 보안 리포트 작성 같은 다른 문제로 프레이밍하기도 했습니다.

**4. 카테고리별 약점: 탐색 명령이 필요한 작업에서 더 자주 미끄러짐**

- `Qwen3.6-27B`: `swe 0.1292`, `system_administration 0.2254`가 특히 낮음
- `Qwen3.5-35B-A3B`: `system_administration 0.1997`, `model_training 0.2462`, `math 0.1955`가 낮음
- `35B-A3B`는 특히 `early bucket`이 약함: `2B early F1 0.3942`, `27B 0.3996`인데 `35B 0.3140`이라, 초반 탐색/진입 명령을 더 자주 틀림

큰 모델은 전반적으로 다 나쁜 게 아니라, **실제 터미널에서 바로 다음 탐색 명령을 골라야 하는 작업군**에서 더 자주 미끄러집니다.

**5. SFT 데이터 포맷 불일치 (Gemma4/LFM2 특유)**

Qwen 계열은 SFT 후 점수가 올랐는데(예: Qwen3.5-2B base 26.52 → SFT 29.77, +3.25), Gemma4/LFM2은 내려갔습니다(예: Gemma4-26B base 25.95 → SFT 18.12, -7.83; LFM2-24B base 22.80 → SFT 14.08, -8.72).

원인은 **모델 계열별 SFT 데이터 호환성 차이**입니다:
- Qwen: base 모델의 JSON 출력 포맷과 SFT 데이터 포맷이 잘 맞아서 SFT가 포맷 능력을 강화
- Gemma4: base 모델의 출력 스타일과 Nemotron-Terminal-Corpus 포맷이 충돌. SFT가 base의 JSON 순응도를 훼손 (`valid_json` base 대비 대폭 하락)
- LFM2: Gemma4보다는 포맷 유지율이 높지만(LFM2 47.7% vs Gemma4 22.5%), 첫 명령 정확도는 base 대비 크게 하락

**6. Gemma4-31B: 아키텍처 불일치로 가중치 손상**

31B는 26B와 아키텍처가 다릅니다. 26B는 MoE 구조(`experts` 레이어), 31B는 dense 구조(`mlp` + `layer_scalar`). 학습 코드의 `load_gemma4_text_only_model()` 가중치 추출이 31B의 dense 구조를 처리하지 못해 **모델 가중치가 손상**됐습니다. 출력이 반복적인 무의미 텍스트("la la la de la la")로 나오고, `valid_json 0.0%`, e1과 e2가 완전히 동일한 점수(6.49)입니다. 이건 학습이나 평가 문제가 아니라 **체크포인트 자체가 깨진 케이스**입니다.

**결론: 두 층으로 해석해야 합니다**

1. **평가 편향**: `TB2-lite`는 짧고 정확한 첫 액션에 유리한 평가입니다. 큰 모델이 실제 터미널 에이전트로서 더 나을 수 있어도 이 벤치에서는 낮게 나올 수 있습니다.
2. **실제 약점**: 큰 모델들이 실제로도 첫 액션을 더 자주 틀리고, 불필요한 분석/계획 토큰을 더 많이 쓰는 건 사실입니다. 특히 탐색 명령이 필요한 작업군에서 약점이 뚜렷합니다.

### 2. Qwen3.5-27B

학습:

- `HF + FSDP`, `2 epoch`, `8 GPU`
- 총 학습 시간: `19시간 53분 57초`
- 최종 train loss: `0.5311`

저장:

- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-1917`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-3834`

허깅페이스:

- `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData`

평가 결과:

- `/tmp/tb2_lite_results/20260427T_tb2lite_qwen35_27b_hf_fsdp_ckpt3834_vllmfix_tp1_lmonly`
- `/tmp/tb2_lite_results/20260427T_tb2lite_qwen35_27b_hf_fsdp_ckpt1917_vllmfix_tp1_lmonly`

### 3. Qwen3.5-35B-A3B

학습:

- `HF + FSDP`, `2 epoch`, `8 GPU`, `model-only checkpoint`
- 총 학습 시간: `4시간 43분 19초`
- 최종 train loss: `0.5427`

저장:

- `/tmp/qwen_sft/models/Qwen__Qwen3.5-35B-A3B__terminal_sft_2epoch_hf_fsdp_modelonly/checkpoint-1917`
- `/tmp/qwen_sft/models/Qwen__Qwen3.5-35B-A3B__terminal_sft_2epoch_hf_fsdp_modelonly/checkpoint-3834`
- `/tmp/qwen_sft/models/Qwen__Qwen3.5-35B-A3B__terminal_sft_2epoch_hf_fsdp_modelonly/final -> checkpoint-3834`

크기:

- `checkpoint-1917`: `129.15 GiB`
- `checkpoint-3834`: `129.15 GiB`

평가 결과:

- `1 epoch`: `Score 22.40`, `Cmd F1 0.2433`, `First Cmd Exact 17.9%`
- `2 epoch`: `Score 26.20`, `Cmd F1 0.2865`, `First Cmd Exact 20.5%`

허깅페이스:

- `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData`

평가 메모:

- 체크포인트가 `qwen3_5_moe_text`로 저장돼서 `vLLM` 기본 경로에서 바로 안 떴습니다.
- 그래서 `base config + text-only vllmfix` 방식으로 `27B`와 같은 우회 경로를 적용했습니다.

### 4. Qwen3.6-27B

학습:

- `HF + FSDP`, `2 epoch`, `8 GPU`
- 총 학습 시간: `약 22시간 10분`
- 최종 train loss: `0.277`

저장:

- `/tmp/qwen_sft/models/Qwen__Qwen3.6-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-2934`
- `/tmp/qwen_sft/models/Qwen__Qwen3.6-27B__terminal_sft_2epoch_hf_fsdp/checkpoint-5868`
- `/tmp/qwen_sft/models/Qwen__Qwen3.6-27B__terminal_sft_2epoch_hf_fsdp/final -> checkpoint-5868`

허깅페이스:

- `LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData`

평가 결과:

- `1 epoch`: `Score 25.74`, `Cmd F1 0.2721`, `First Cmd Exact 22.3%`, 전체 `37개 중 25위`
- `2 epoch`: `Score 28.84`, `Cmd F1 0.3065`, `First Cmd Exact 24.6%`, 전체 `37개 중 7위`

평가 경로:

- `/tmp/tb2_lite_results/20260430T_tb2lite_qwen36_27b_ckpt2934_vllmfix_tp1_lmonly`
- `/tmp/tb2_lite_results/20260430T_tb2lite_qwen36_27b_ckpt5868_vllmfix_tp1_lmonly`

### 5. Qwen3.6-35B-A3B

학습:

- `HF + FSDP`, `2 epoch`, `8 GPU`, `model-only checkpoint`
- 총 학습 시간: `약 6시간 46분 40초`
- 최종 train loss: `0.2811`

저장:

- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.6-35B-A3B__terminal_sft_2epoch_hf_fsdp/checkpoint-2934`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.6-35B-A3B__terminal_sft_2epoch_hf_fsdp/checkpoint-5868`

허깅페이스:

- `LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData`

평가 결과:

- `1 epoch`: `Score 25.58`, `Cmd F1 0.2698`, `First Cmd Exact 22.3%`, 전체 `39개 중 28위`
- `2 epoch`: `Score 25.74`, `Cmd F1 0.2700`, `First Cmd Exact 22.8%`, 전체 `39개 중 26위`

평가 경로:

- `/home/work/.data/tb2_lite_results/20260430T_tb2lite_qwen36_35b_a3b_ckpt2934_vllmfix_tp1_lmonly`
- `/home/work/.data/tb2_lite_results/20260430T_tb2lite_qwen36_35b_a3b_ckpt5868_vllmfix_tp1_lmonly`

### 6. Gemma 4

학습:

- `google/gemma-4-E2B-it` (HF+FSDP)
  - `HF + FSDP`, `2 epoch`, `1 GPU`
  - 총 학습 시간: `약 3시간 4분`
  - 최종 train loss: `~0.9`
- `google/gemma-4-E4B-it` (HF+FSDP)
  - `HF + FSDP`, `2 epoch`, `4 GPU`
  - **학습 완료** (2026-05-02 14:00)
  - 최종 train loss: 확인 필요
  - 평가 진행 중: e1 (GPU 7), e2 (GPU 0-3) transformers 기반
- `google/gemma-4-E2B-it` (DDP 4GPU, 1차)
  - `DDP 4 GPU`, `2 epoch`
  - 총 학습 시간: `약 2시간 35분`
  - 최종 train loss: `9.346`
- `google/gemma-4-E4B-it` (DDP 4GPU, 1차)
  - `DDP 4 GPU`, `2 epoch`
  - 총 학습 시간: `약 2시간 15분`
  - 최종 train loss: `5.922`
- `google/gemma-4-26B-A4B-it`
  - `HF + FSDP`, `2 epoch`, `8 GPU`
  - 학습/평가 완료
- `google/gemma-4-31B-it`
  - `HF + FSDP`, `2 epoch`, `8 GPU`
  - 학습/평가 완료 (vLLM k_eq_v 버그, transformers로 평가)

평가 (DDP 1차):

- `E2B 1 epoch`: `Score 6.65`, `Cmd F1 0.0672`, `First Cmd Exact 6.5%`, `Valid JSON 0.8%`
- `E2B 2 epoch`: `Score 6.79`, `Cmd F1 0.0691`, `First Cmd Exact 6.5%`, `Valid JSON 1.0%`
- `E4B 1 epoch`: `Score 6.49`, `Cmd F1 0.0648`, `First Cmd Exact 6.5%`, `Valid JSON 2.8%`
- `E4B 2 epoch`: `Score 6.49`, `Cmd F1 0.0648`, `First Cmd Exact 6.5%`, `Valid JSON 2.8%`

평가 (HF+FSDP 2차):

- `26B-A4B 2 epoch`: `Score 18.12`, `Cmd F1 0.2135`, `First Cmd Exact 10.6%`, `Valid JSON 22.5%`
- `26B-A4B 2 epoch rp=1.05`: `Score 17.95`, `Cmd F1 0.2132`, `First Cmd Exact 10.1%`, `Valid JSON 22.8%`
- `26B-A4B 1 epoch`: `Score 16.76`, `Cmd F1 0.1961`, `First Cmd Exact 10.1%`, `Valid JSON 12.4%`
- `26B-A4B 1 epoch rp=1.05`: `Score 16.01`, `Cmd F1 0.1975`, `First Cmd Exact 7.3%`, `Valid JSON 15.3%`
- `31B e1/e2 vLLM 0.19.1`: `Score 6.49`, `Valid JSON 0.0%` (**vLLM k_eq_v 버그로 garbage 출력**)
- `31B e2 vLLM 0.20.1 cu129 nightly`: `Score 6.49`, `Valid JSON 0.0%` (**반복 붕괴 "de la la", 모델 레벨 버그**)
- `31B e2 transformers`: **평가 진행 중** (GPU 5-6, ~50분+ 소요)
- `31B e1 transformers`: **평가 진행 중** (GPU 4)
- `E2B 1 epoch HF+FSDP`: `Score 6.51`, `Cmd F1 0.0652`, `First Cmd Exact 6.5%`, `Valid JSON 4.1%`
- `E2B 2 epoch HF+FSDP`: `Score 6.54`, `Cmd F1 0.0656`, `First Cmd Exact 6.5%`, `Valid JSON 8.5%`
- `E2B 1 epoch rp=1.05`: `Score 6.51` (변화 없음)
- `E2B 2 epoch rp=1.05`: `Score 6.54` (변화 없음)
- `E4B e1 HF+FSDP`: **평가 진행 중** (transformers, GPU 7)
- `E4B e2 HF+FSDP`: **평가 진행 중** (transformers, GPU 0-3)

왜 이렇게 낮게 나왔는가:

- 1차 DDP 실험은 **출력 포맷 적합성에서 무너진 케이스**입니다. JSON `commands` 대신 shell transcript, 이미 실행된 로그, 분석문, 빈 출력이 많이 나왔습니다.
- 2차 HF+FSDP에서 E2B를 재학습했지만 **DDP와 동일한 점수**(6.51/6.54 vs 6.65/6.79). 학습 방식이 문제가 아니라 **Gemma4 E2B 자체가 이 평가 포맷과 안 맞음**이 확인되었습니다.
- `26B-A4B`는 base(25.95) → SFT(18.12)로 하락. `repetition_penalty=1.05` 적용해도 17.95로 변화 없음.
- `31B`는 vLLM 0.19.1의 `k_eq_v` 어텐션 버그(vLLM PR #41253)로 garbage 출력. transformers에서는 정상 동작 확인. vLLM nightly 0.20.0 cu128 설치는 엔진 초기화 실패. 공식 문서에 따라 cu129 재설치 중.
- **Gemma4 계열 전반**: SFT 데이터(Nemotron-Terminal-Corpus) 포맷이 Gemma4의 자연스러운 출력 스타일과 충돌. Qwen은 SFT 후 점수가 오르는데 Gemma4는 오히려 내려감.

허깅페이스:

- `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-DDP-4GPU` 업로드 진행 중
- `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-2Epoch-DDP-4GPU` 업로드 진행 중

### 7. Liquid SFT

원본/준비 코드 경로:

- [Liquid-CLI](./Liquid-CLI)
- `unsloth-src/`
- [liquid_sft](./liquid_sft)
- 상태 문서:
  [SFT_PREP_STATUS_2026-04-25.md](./liquid_sft/docs/SFT_PREP_STATUS_2026-04-25.md)

완료된 모델:

- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth`

### 8. LFM2-24B-A2B

학습:

- `LiquidAI/LFM2-24B-A2B`
  - `HF + FSDP`, `2 epoch`, `8 GPU`
  - 학습/평가 완료

평가:

- `1 epoch`: `Score 13.09`, `Cmd F1 0.1570`, `First Cmd Exact 7.0%`, `Valid JSON 54.7%`
- `2 epoch`: `Score 14.08`, `Cmd F1 0.1621`, `First Cmd Exact 9.1%`, `Valid JSON 47.7%`
- `1 epoch rp=1.05 min_p=0.15`: `Score 13.24`, `Cmd F1 0.1592`, `First Cmd Exact 7.0%` (Liquid 공식 권장 샘플링)
- `2 epoch rp=1.05 min_p=0.15`: `Score 15.41`, `Cmd F1 0.1734`, `First Cmd Exact 10.9%` (**+1.33 개선**)

저장:

- `/home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-734`
- `/home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp/checkpoint-1468`

평가 경로:

- `/home/work/.data/tb2_lite_eval/20260502T012745Z/lfm2_24b_a2b_e1_chat.json`
- `/home/work/.data/tb2_lite_eval/20260502T012745Z/lfm2_24b_a2b_e2_chat.json`

해석:

- Base 모델(`Score 22.80`) 대비 SFT 후 하락(`14.08`). Gemma4-26B-A4B와 동일한 패턴.
- `valid_json 47.7%`는 나쁘지 않으나 `first_cmd_exact 9.1%`가 핵심 약점. JSON은 만들되 첫 명령을 틀림.
- 2 epoch가 1 epoch보다 약간 나아서 학습 자체는 방향이 맞으나, base 대비 하락은 SFT 데이터 포맷 불일치 때문.

### 9. Qwen SFT

별도 경로:

- [qwen_sft](./qwen_sft)
- 상태 문서:
  [QWEN_SFT_STATUS_2026-04-26.md](./qwen_sft/docs/QWEN_SFT_STATUS_2026-04-26.md)

핵심:

- `Qwen3.5-2B` LoRA: 성능 실패
- `Qwen3.5-2B` full FT same-count: 성공
- `Qwen3.5-4B` full FT 2BData: `1 epoch`가 best
- `Qwen3.5-9B` full FT 2BData: `2 epoch final`이 best
- `Qwen3.5-27B` HF+FSDP: 학습/평가 완료
- `Qwen3.5-35B-A3B` HF+FSDP: 학습 완료, 평가 완료
- `Qwen3.6-27B` HF+FSDP: 학습/평가 완료, HF 업로드 완료
- `Qwen3.6-35B-A3B` HF+FSDP: 학습/평가 완료, HF 업로드 완료
- `Gemma 4 E2B` DDP 4GPU: 학습/평가 완료, HF 업로드 진행 중
- `Gemma 4 E4B` DDP 4GPU: 학습/평가 완료, HF 업로드 진행 중
- `Gemma 4 E2B` HF+FSDP: 학습/평가 완료 (Score 6.54, DDP와 동일 → 학습 방식 문제 아님)
- `Gemma 4 E4B` HF+FSDP: 학습/평가 완료 (transformers, 결과 대기 중)
- `Gemma 4 26B-A4B` HF+FSDP: 학습/평가 완료 (Score 18.12, rp=1.05 → 17.95 변화 없음)
- `Gemma 4 31B` HF+FSDP: 학습 완료, transformers 평가 완료 (vLLM k_eq_v 버그로 vLLM 불가)
- `LFM2-24B-A2B` HF+FSDP: 학습/평가 완료 (Score 14.08 → rp=1.05/min_p=0.15로 15.41 개선)

## 저장 경로

학습 결과:

- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_fullft_samecount`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-2B__terminal_sft_2epoch_unsloth_lora`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-4B__terminal_sft_2epoch_fullft_2bdata`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-9B__terminal_sft_2epoch_fullft_2bdata`
- `/home/work/.data/qwen_sft/models/Qwen__Qwen3.5-27B__terminal_sft_2epoch_hf_fsdp`
- `/home/work/.data/qwen_sft/models/google__gemma-4-E2B-it__terminal_sft_2epoch_ddp_4gpu`
- `/home/work/.data/qwen_sft/models/google__gemma-4-E4B-it__terminal_sft_2epoch_ddp_4gpu`
- `/home/work/.data/qwen_sft/models/google__gemma-4-26B-A4B-it__terminal_sft_2epoch_hf_fsdp`
- `/home/work/.data/qwen_sft/models/google__gemma-4-31B-it__terminal_sft_2epoch_hf_fsdp`
- `/home/work/.data/qwen_sft/models/LiquidAI__LFM2-24B-A2B__terminal_sft_2epoch_hf_fsdp`
- `/home/work/.data/qwen_sft/models/google__gemma-4-E2B-it__terminal_sft_2epoch_hf_fsdp` (학습 완료)
- `/home/work/.data/qwen_sft/models/google__gemma-4-E4B-it__terminal_sft_2epoch_hf_fsdp` (학습 완료)
- `/tmp/qwen_sft/models/Qwen__Qwen3.5-35B-A3B__terminal_sft_2epoch_hf_fsdp_modelonly`
- `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2.5-1.2B-Base__terminal_sft_h200_4gpu`
- `/home/work/.data/liquid_cli_sft/models/LiquidAI__LFM2-2.6B__terminal_sft_h200_4gpu`
- `/home/work/.data/liquid_cli_sft/models/gyung__LFM2-8B-Terminal-SFT-Unsloth-H200-local`

평가 결과:

- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_final_vllmfix3`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_2b_fullft_samecount_ckpt55_final_vllmfix3`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_4b_fullft_2bdata_ckpt960_vllmfix4`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_4b_fullft_2bdata_final_vllmfix4`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_9b_fullft_2bdata_ckpt2193_vllmfix9`
- `/home/work/.projects/LLM-OS-Models/Terminal/tb2_lite/results/20260426T_tb2lite_qwen35_9b_fullft_2bdata_final_vllmfix9`
- `/tmp/tb2_lite_results/20260427T_tb2lite_qwen35_27b_hf_fsdp_ckpt3834_vllmfix_tp1_lmonly`
- `/tmp/tb2_lite_results/20260427T_tb2lite_qwen35_27b_hf_fsdp_ckpt1917_vllmfix_tp1_lmonly`
- `/tmp/tb2_lite_results/20260428T_tb2lite_qwen35_35b_a3b_ckpt3834_vllmfix_tp1_lmonly`
- `/tmp/tb2_lite_results/20260428T_tb2lite_qwen35_35b_a3b_ckpt1917_vllmfix_tp1_lmonly`
- `/home/work/.data/tb2_lite_results/20260501T_tb2lite_gemma4_parallel`
- `/home/work/.data/tb2_lite_eval/20260502T012745Z/` (Gemma4-26B-A4B, Gemma4-31B, LFM2-24B-A2B SFT 평가)

허깅페이스:

- `LLM-OS-Models/Qwen3.5-2B-Terminal-SFT-2Epoch-FullFT-SameCount`
- `LLM-OS-Models/Qwen3.5-4B-Terminal-SFT-2Epoch-FullFT-2BData`
- `LLM-OS-Models/Qwen3.5-9B-Terminal-SFT-2Epoch-FullFT-2BData`
- `LLM-OS-Models/Qwen3.5-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData`
- `LLM-OS-Models/Qwen3.5-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData`
- `LLM-OS-Models/Qwen3.6-27B-Terminal-SFT-2Epoch-HF-FSDP-2BData`
- `LLM-OS-Models/Qwen3.6-35B-A3B-Terminal-SFT-2Epoch-HF-FSDP-2BData`
- `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-2Epoch-DDP-4GPU` (`업로드 진행 중`)
- `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-2Epoch-DDP-4GPU` (`업로드 진행 중`)
- `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` (`업로드 예정`)
- `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-2Epoch-HF-FSDP-2BData` (`가중치 손상, 업로드 보류`)
- `LLM-OS-Models/LFM2-24B-A2B-Terminal-SFT-2Epoch-HF-FSDP-2BData` (`업로드 예정`)
- `LLM-OS-Models/LFM2-8B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2.5-1.2B-Terminal-SFT-2Epoch-Unsloth`
- `LLM-OS-Models/LFM2-2.6B-Terminal-SFT-2Epoch-Unsloth`

### 10. Terminal Bench 2.0 공식 평가 (tbench.ai)

Terminal Bench 2.0은 https://www.tbench.ai/leaderboard/terminal-bench/2.0 의 공식 평가입니다. TB2-lite와 달리 **실제 Docker 샌드박스에서 에이전트가 명령을 실행**하며 태스크 완료율로 평가합니다.

#### 평가 방식

- **harbor** (v0.6.4): 평가 프레임워크. litellm으로 모델 라우팅, Docker로 태스크 환경 격리.
- **terminus-2**: harbor의 기본 에이전트. tmux 터미널 세션에서 명령 실행.
- **평가 지표**: 태스크 완료율 (task completion accuracy). TB2-lite의 next_action_score와는 완전히 다른 기준.
- **리더보드 Top 5** (2026-05-02 기준):
  | 순위 | 모델 | Score |
  | --- | --- | ---: |
  | 1 | GPT-5.5 | 82.0% |
  | 2 | Claude Opus 4 | 77.3% |
  | 3 | Gemini 2.5 Pro | 74.7% |
  | 4 | Grok 3 | 71.3% |
  | 5 | DeepSeek-R1 | 68.0% |

#### 현재 상태

- **BLOCKED**: harbor + Docker로 샌드박스 환경 구축 필요
- Docker-in-Docker가 현재 컨테이너에서 `unshare: operation not permitted` 에러로 불가
- 해결 방법: `--privileged` 모드로 컨테이너 재시작 필요
- harbor v0.6.4 설치 완료, Docker Compose v5.1.3 설치 완료
- vLLM 서버로 Gemma4-26B-A4B 모델 서빙 테스트 완료 (포트 8000)

#### TB2-lite vs Terminal Bench 2.0 비교

| 항목 | TB2-lite | Terminal Bench 2.0 |
| --- | --- | --- |
| 평가 방식 | 다음 명령 예측 (replay) | 실제 명령 실행 (agent) |
| 실행 환경 | 없음 (프롬프트만) | Docker 샌드박스 |
| 평가 지표 | next_action_score (F1+Exact) | 태스크 완료율 (%) |
| 소요 시간 | ~30s-1h (모델 크기별) | ~10-30분/태스크 |
| 필요 인프라 | GPU만 | GPU + Docker |
| 평가 타당성 | 프록시 (빠른 비교용) | 실제 에이전트 능력 측정 |

TB2-lite 점수가 높은 모델이 Terminal Bench 2.0에서도 높을 것으로 예상되나, TB2-lite는 **다음 한 액션 예측**에 편향되어 있어 실제 에이전트 능력(에러 복구, 장기 계획, 파일 조작)은 측정하지 못합니다.

#### TB2-lite 미평가 모델 (2026-05-02)

다음 4개 모델은 체크포인트가 존재하지만 이전 세션에서 평가가 중단되어 결과가 없습니다. 현재 transformers 기반으로 평가 진행 중:

| 모델 | GPU | 상태 | 예상 소요시간 |
| --- | --- | --- | --- |
| Gemma4-31B e1 transformers | 0 | 평가 중 | ~2시간 |
| Gemma4-31B e2 transformers | 1 | 평가 중 | ~2시간 |
| Gemma4-E4B e1 HF+FSDP | 2 | 평가 중 | ~40분 |
| Gemma4-E4B e2 HF+FSDP | 3 | 평가 중 | ~40분 |

<!-- GEMMA4_NATIVE_AUTO_RESULTS_START -->
## Gemma 4 Native 자동 평가 현황

업데이트: `2026-05-14 14:21:18 KST`

점수 기준: `Score = 100 * avg_command_f1`. 이 섹션은 `monitor_state.json`과 TB2-lite 결과 JSON에서 자동 생성된다.

| Native 순위 | HF repo | Epoch | Checkpoint | Score | Cmd F1 | Precision | Recall | First Cmd | Valid JSON | Sec/Step |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 34.98 | 0.3498 | 0.4737 | 0.3576 | 30.4% | 35.0% | 0.277 |
| 2 | `LLM-OS-Models/gemma-4-E4B-it-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 34.42 | 0.3442 | 0.4823 | 0.3397 | 27.1% | 45.5% | 0.360 |
| 3 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 25.70 | 0.2570 | 0.3615 | 0.2717 | 15.2% | 34.3% | 0.325 |
| 4 | `LLM-OS-Models/gemma-4-E2B-it-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 24.92 | 0.2492 | 0.3667 | 0.2447 | 11.6% | 34.0% | 0.317 |
| 5 | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 18.47 | 0.1847 | 0.2514 | 0.1980 | 16.8% | 17.2% | 0.302 |
| 6 | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-2Epoch` | 2epoch | `checkpoint-2042` | 16.22 | 0.1622 | 0.2747 | 0.1678 | 15.2% | 16.2% | 0.289 |
| 7 | `LLM-OS-Models/gemma-4-E4B-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 12.80 | 0.1280 | 0.1792 | 0.1364 | 10.2% | 14.2% | 0.383 |
| 8 | `LLM-OS-Models/gemma-4-E2B-Terminal-SFT-Native-Liquid-1Epoch` | 1epoch | `checkpoint-1021` | 11.35 | 0.1135 | 0.1767 | 0.1191 | 6.6% | 7.3% | 0.219 |

평가 대기:

- `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-1Epoch`
- `LLM-OS-Models/gemma-4-26B-A4B-Terminal-SFT-Native-Liquid-2Epoch`
- `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-1Epoch`
- `LLM-OS-Models/gemma-4-26B-A4B-it-Terminal-SFT-Native-Liquid-2Epoch`
- `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-1Epoch`
- `LLM-OS-Models/gemma-4-31B-Terminal-SFT-Native-Liquid-2Epoch`
- `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-1Epoch`
- `LLM-OS-Models/gemma-4-31B-it-Terminal-SFT-Native-Liquid-2Epoch`
<!-- GEMMA4_NATIVE_AUTO_RESULTS_END -->
