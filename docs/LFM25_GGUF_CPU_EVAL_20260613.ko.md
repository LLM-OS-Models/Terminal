# LFM2.5 Terminal SFT 1Epoch GGUF CPU 평가 기록

작성 시각: 2026-06-13 09:03 KST

이 문서는 `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF`의 4-bit GGUF를 GPU 없이 CPU 환경에서 검증한 기록이다.

목표는 README의 `LFM2.5-8B-A1B Terminal ToolBench Full SFT 1Epoch` 평가와 최대한 같은 TB2-lite 조건으로 비교하는 것이다.

## 비교 대상

- 원본 HF 모델: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- GGUF 저장소: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF`
- GGUF 파일: `LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.Q4_K_M.gguf`
- 로컬 원본 파일:
  `/home/work/.data/liquid_cli_sft/gguf/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.Q4_K_M.gguf`
- 평가용 metadata patch copy:
  `/home/work/.data/liquid_cli_sft/gguf/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF/eval/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.Q4_K_M.llamacpp_eval.gguf`

원본 GGUF 파일은 덮어쓰지 않았다. 평가용 copy만 만들어서 `llama-cpp-python` 호환성 문제를 우회했다.

## 왜 평가용 copy가 필요한가

`llama-cpp-python==0.3.20`은 GGUF metadata에 포함된 `tokenizer.chat_template`를 모델 로드 시 Jinja2로 컴파일한다.

그런데 Liquid LFM2.5 계열 chat template에는 `{% generation %}` 태그가 포함되어 있고, 현재 로컬 `llama-cpp-python` Jinja 환경은 이 태그를 알지 못한다.

처음 원본 Q4_K_M 파일을 그대로 로드했을 때 다음 오류가 발생했다.

```text
jinja2.exceptions.TemplateSyntaxError: Encountered unknown tag 'generation'.
```

TB2-lite 평가 스크립트는 이미 Hugging Face tokenizer로 prompt 문자열을 만든 뒤 `llama_cpp.Llama()`에 completion prompt를 넣는다. 즉, GGUF 내부 chat template은 평가 경로에서 필요하지 않다.

따라서 `gguf_new_metadata`로 `tokenizer.chat_template`만 단순 템플릿으로 치환한 평가용 copy를 만들었다.

```bash
python -m gguf.scripts.gguf_new_metadata \
  /home/work/.data/liquid_cli_sft/gguf/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.Q4_K_M.gguf \
  /home/work/.data/liquid_cli_sft/gguf/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF/eval/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.Q4_K_M.llamacpp_eval.gguf \
  --chat-template "{{ bos_token }}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}" \
  --force
```

## BOS 중복 문제

처음 문자열 prompt를 그대로 `llama_cpp`에 넘기면 다음 경고가 나왔다.

```text
Detected duplicate leading "<|startoftext|>" in prompt
```

원인은 HF chat template이 이미 `<|startoftext|>`를 prompt 앞에 넣는데, `llama-cpp-python` 문자열 completion 경로도 BOS를 자동으로 추가하기 때문이다.

이를 피하기 위해 `tb2_lite/scripts/replay_eval_llamacpp.py`에 `--manual-prompt-tokenize` 옵션을 추가했다. 이 옵션은 prompt 문자열을 `llm.tokenize(..., add_bos=False, special=True)`로 직접 토큰화한 뒤 token id list를 `Llama()`에 넣는다. token id list 입력은 `llama-cpp-python`의 자동 BOS 추가 경로를 타지 않는다.

## CPU-only 설정

GGUF 평가는 GPU를 전혀 쓰지 않도록 다음 조건을 강제했다.

- `CUDA_VISIBLE_DEVICES=""`
- `--n-gpu-layers 0`
- `--no-offload-kqv`
- `--n-threads 64`
- `--n-threads-batch 64`

실제 실행 중 측정한 RSS는 약 10.8~12.0GB였다. 시스템 RAM은 충분했지만, 16GB CPU 환경에서는 OS와 기타 프로세스 메모리를 고려해 `Q4_K_M` + context 4096~8192를 기본 추천으로 유지하는 것이 맞다.

## 평가 조건

README와 같은 TB2-lite replay를 사용한다.

- 평가 파일: `tb2_lite/data/replay_full.jsonl`
- 전체 step 수: 303
- temperature: 0.0
- top_p: 1.0
- max_tokens: 1024
- full 평가 context: 32768

README의 vLLM 평가 기록은 `max_model_len=49152`였지만, 실제 prompt 통계는 다음과 같다.

- prompt token max: 22593
- prompt token p99: 19905
- output budget: 1024

따라서 `32768` context도 전체 TB2-lite 입력을 자르지 않는다. CPU 메모리와 속도를 위해 GGUF full 평가는 32768로 실행한다.

## 10-step smoke 결과

10-step smoke는 CPU-only 설정에서 정상 완료됐다.

- generated steps: 10
- complete: true
- Score: 54.00
- Cmd F1: 0.5143
- First Cmd: 60.0%
- Valid JSON: 70.0%
- load time: 5.2s
- gen time: 283.4s
- avg sec/step: 28.34s
- RSS: 약 12GB

10-step은 표본이 작으므로 README 순위에는 넣지 않는다. 이 결과는 “Q4_K_M GGUF가 CPU에서 로드되고 JSON/명령 생성이 가능하다”는 smoke validation으로만 해석한다.

## 진행 중인 full 평가

full 303-step CPU 평가는 2026-06-13 09:02 KST에 시작했다.

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1 \
.liquid-sft-env/bin/python tb2_lite/scripts/replay_eval_llamacpp.py \
  --repo-id LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF \
  --filename LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.Q4_K_M.llamacpp_eval.gguf \
  --model-path /home/work/.data/liquid_cli_sft/gguf/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF/eval/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.Q4_K_M.llamacpp_eval.gguf \
  --tokenizer-path LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
  --model-short lfm25-sft1-gguf-q4km-cpu-full-manualtok-ctx32768 \
  --eval-path tb2_lite/data/replay_full.jsonl \
  --output-dir tb2_lite/results/20260613T_lfm25_sft1_gguf_q4km_cpu_eval \
  --max-model-len 32768 \
  --max-tokens 1024 \
  --temperature 0 \
  --top-p 1 \
  --n-gpu-layers 0 \
  --no-offload-kqv \
  --n-threads 64 \
  --n-threads-batch 64 \
  --n-batch 1024 \
  --n-ubatch 512 \
  --save-every 10 \
  --manual-prompt-tokenize
```

10-step 속도 기준 단순 예상 시간은 약 2시간 24분이다. 긴 프롬프트 구간이 뒤쪽에 더 많으면 2.5~3시간까지 걸릴 수 있다.

예상 완료: 2026-06-13 11:25~12:00 KST.

## 해석 기준

최종 full 결과가 나오면 다음과 같이 비교한다.

- vLLM/Transformers SFT 1Epoch: Score 52.30
- GGUF Q4_K_M CPU full: pending

만약 GGUF 점수가 vLLM보다 낮으면 가능한 원인은 세 가지다.

1. 4-bit quantization 손실
2. llama.cpp/LFM2 MoE runtime과 Transformers/vLLM runtime의 샘플링/토크나이저 세부 차이
3. chat template metadata 우회 및 manual tokenization 경로의 미세한 차이

반대로 점수가 비슷하면 `Q4_K_M`가 16GB급 CPU 로컬 환경에서도 충분히 쓸 수 있는 terminal-agent 포맷 보존력을 가진다는 근거가 된다.

## HF 모델 카드 반영 예정

full 결과 이후 GGUF 모델 카드에 다음을 반영한다.

- 16GB CPU 권장 파일은 BF16이 아니라 `Q4_K_M`
- `llama-cpp-python==0.3.20`에서는 GGUF 내 chat template의 `{% generation %}` 태그로 로드 오류가 날 수 있음
- raw completion 방식 또는 최신 llama.cpp/llama-cpp-python 사용을 권장
- TB2-lite full CPU 결과가 나오면 Score와 sec/step을 명시

