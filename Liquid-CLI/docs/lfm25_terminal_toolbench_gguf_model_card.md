---
language:
- en
- ko
library_name: llama-cpp
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- lfm2_moe
- terminal
- sft
- toolbench
- tb2-lite
- cpu-inference
base_model: LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch
---

# LFM2.5-8B-A1B Terminal ToolBench Full SFT 1Epoch GGUF

GGUF builds for `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`.

This is a local/CPU-friendly packaging of the current #1 Terminal README model. The original model is a terminal automation SFT model trained to read a task and terminal state, then emit JSON containing the next terminal action.

## Source Model

- Source model: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- Base model: `LiquidAI/LFM2.5-8B-A1B`
- Architecture: `Lfm2MoeForCausalLM`, `model_type=lfm2_moe`
- Training setup of source model: full SFT, 1 epoch, Terminal + ToolBench conversation data
- Corrected TB2-lite score of source model: `52.30`
- Original source snapshot converted here: `4ca70e4065d10b171de0a434185a9c436cbe9893`

## Files

Recommended files:

- `Q4_K_M` (`4.801 GiB`): best first choice for a 16GB RAM CPU machine. Small enough for local use while preserving reasonable quality.
- `Q5_K_M` (`5.616 GiB`): better quality than Q4, still practical on many CPU machines with enough free RAM.
- `Q6_K` (`6.482 GiB`): higher-quality local option when RAM is comfortable.
- `Q8_0` (`8.391 GiB`): high-fidelity quantization. Use when you have more RAM or want a quality-preserving GGUF.
- `Q4_0` (`4.512 GiB`): simple 4-bit compatibility option. Prefer `Q4_K_M` when your runtime supports K-quants.
- `BF16` (`15.783 GiB`): conversion master used to produce the quantized files. This is not the default recommendation for 16GB CPU usage.

Actual file sizes and SHA256 checksums are in `SHA256SUMS` and `checksums.json`.

## Quickstart: llama.cpp

Build llama.cpp with LFM2 MoE support:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF
cmake --build build --target llama-cli -j
```

Download the 4-bit file:

```bash
pip install -U huggingface_hub
huggingface-cli download \
  LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF \
  --include "*Q4_K_M.gguf" \
  --local-dir ./lfm-terminal-gguf
```

Run a terminal-agent style prompt:

```bash
./llama.cpp/build/bin/llama-cli \
  -m ./lfm-terminal-gguf/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.Q4_K_M.gguf \
  -c 8192 \
  -n 512 \
  --temp 0 \
  -p '<|startoftext|><|im_start|>system
You are a terminal automation assistant. Return JSON only.<|im_end|>
<|im_start|>user
Inspect the current directory and list Python files.<|im_end|>
<|im_start|>assistant
'
```

Expected output format:

```json
{
  "analysis": "brief reasoning about the next terminal action",
  "plan": "short execution plan",
  "commands": [
    {"keystrokes": "ls -la\n", "duration": 0.1}
  ],
  "task_complete": false
}
```

## Quickstart: Python llama-cpp-python

Install:

```bash
pip install -U llama-cpp-python huggingface_hub
```

Run:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./lfm-terminal-gguf/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch.Q4_K_M.gguf",
    n_ctx=8192,
    n_threads=8,
    verbose=False,
)

prompt = """<|startoftext|><|im_start|>system
You are a terminal automation assistant. Return JSON only.<|im_end|>
<|im_start|>user
Inspect the current directory and list Python files.<|im_end|>
<|im_start|>assistant
"""

out = llm(prompt, max_tokens=512, temperature=0.0, stop=["<|im_end|>"])
print(out["choices"][0]["text"])
```

## 16GB CPU Guidance

Use `Q4_K_M` first.

Recommended starting settings:

- Context: `-c 4096` or `-c 8192`
- Generation: `-n 256` to `-n 1024`
- Temperature: `--temp 0` for deterministic terminal JSON
- Threads: set near your physical CPU core count, for example `-t 8`

If memory is tight, lower context first. Context/KV cache memory is separate from the GGUF file size, so a model that fits at `-c 4096` may fail at a much longer context.

## Important Safety Note

This model emits terminal commands. Do not run generated commands directly on a host machine without a sandbox, allowlist, or human review. Treat outputs as proposed actions, not trusted shell input.

## Conversion Notes

The conversion used a llama.cpp build with `Lfm2MoeForCausalLM` / `MODEL_ARCH.LFM2MOE` support. The source tokenizer uses the Liquid LFM2 BPE pre-tokenizer pattern, but this SFT model has a newer tokenizer hash:

`9e454714343b69b99b71795c1d27a68c2a1d15dab111f4d353109f966af29da7`

The conversion script maps that hash to `lfm2`, matching the existing Liquid LFM2 tokenizer handling.

Reproduction script in the Terminal repo:

```bash
MODEL_ID=LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
OUT_DIR=/home/work/.data/liquid_cli_sft/gguf/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch-GGUF \
LLAMA_CPP_DIR=/home/work/.cache/llama.cpp-mtp \
QUANTS="Q4_K_M Q5_K_M Q8_0" \
bash Liquid-CLI/scripts/convert_lfm25_terminal_gguf.sh
```

## Validation Notes

The GGUF files were produced successfully from the source safetensors and checksummed.

A local `llama-cli` generation smoke test with this specific cached llama.cpp build produced repeated prompt-marker output (`>`), so the uploaded model card does not claim a passing generation benchmark for GGUF runtime yet. Use a recent llama.cpp build with LFM2 MoE support and validate your target runtime locally. The original Transformers/vLLM source model remains the evaluated model for the reported TB2-lite score.

## 한국어 요약

이 저장소는 `LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch` 모델의 GGUF 변환본입니다.

16GB RAM CPU 환경에서는 `Q4_K_M`을 먼저 쓰는 것을 권장합니다. 품질을 조금 더 원하면 `Q5_K_M`, 원본 품질 보존에 가깝게 쓰고 싶으면 `Q8_0`을 쓰면 됩니다.

이 모델은 터미널 자동화용 모델입니다. 디렉터리 상태나 이전 터미널 출력을 보고 다음에 실행할 명령을 JSON 형태로 내는 용도입니다. 실제 명령 실행은 반드시 샌드박스, allowlist, 사람 검토 같은 안전장치를 거쳐야 합니다.
