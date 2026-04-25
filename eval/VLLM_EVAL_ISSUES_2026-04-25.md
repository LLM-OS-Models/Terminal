# vLLM Eval Issues (2026-04-25)

Active run:
- Run ID: `20260425T094140Z`
- Logs: [eval/logs/20260425T094140Z](/home/work/.projects/LLM-OS-Models/Terminal/eval/logs/20260425T094140Z)
- Results: [eval/results/20260425T094140Z](/home/work/.projects/LLM-OS-Models/Terminal/eval/results/20260425T094140Z)
- Mode: `vllm` Python API, GPUs `0-6` in use, GPU `7` reserved

Skipped / Failed

1. `principled-intelligence/Qwen3.5-2B-text-only`
- Status: skipped after runtime failure
- Reason: `vllm 0.19.1` raises `TypeError` during model init because it expects `Qwen3_5Config` but receives `Qwen3_5TextConfig`
- Evidence: [principled-intelligence_qwen3.5-2b-text-only.log](/home/work/.projects/LLM-OS-Models/Terminal/eval/logs/20260425T094140Z/principled-intelligence_qwen3.5-2b-text-only.log)

2. `principled-intelligence/Qwen3.5-4B-text-only`
- Status: skipped after runtime failure
- Reason: same `Qwen3_5TextConfig` incompatibility as above
- Evidence: [principled-intelligence_qwen3.5-4b-text-only.log](/home/work/.projects/LLM-OS-Models/Terminal/eval/logs/20260425T094140Z/principled-intelligence_qwen3.5-4b-text-only.log)

3. `principled-intelligence/Qwen3.5-9B-text-only`
- Status: skipped after runtime failure
- Reason: same `Qwen3_5TextConfig` incompatibility as above
- Evidence: [principled-intelligence_qwen3.5-9b-text-only.log](/home/work/.projects/LLM-OS-Models/Terminal/eval/logs/20260425T094140Z/principled-intelligence_qwen3.5-9b-text-only.log)

4. `principled-intelligence/gemma-4-E2B-it-text-only`
- Status: skipped
- Reason: vLLM resolves the model as `TransformersEmbeddingModel` with pooling/embed runner instead of a generative text model, so it is not suitable for this command-generation eval path
- Evidence: [principled-intelligence_gemma-4-e2b-it-text-only.log](/home/work/.projects/LLM-OS-Models/Terminal/eval/logs/20260425T094140Z/principled-intelligence_gemma-4-e2b-it-text-only.log)

5. `principled-intelligence/gemma-4-E4B-it-text-only`
- Status: skipped
- Reason: same pooling/embed resolution as above
- Evidence: [principled-intelligence_gemma-4-e4b-it-text-only.log](/home/work/.projects/LLM-OS-Models/Terminal/eval/logs/20260425T094140Z/principled-intelligence_gemma-4-e4b-it-text-only.log)

6. `Jiunsong/supergemma4-26b-abliterated-multimodal`
- Status: failed during weight load
- Reason: vLLM weight mapping fails with `KeyError: 'layers.0.experts.switch_glu.down_proj.weight'`
- Evidence: [jiunsong_supergemma4-26b-abliterated-multimodal.log](/home/work/.projects/LLM-OS-Models/Terminal/eval/logs/20260425T094140Z/jiunsong_supergemma4-26b-abliterated-multimodal.log)

Known constraints

- `Qwen/Qwen3.6-35B-A3B-FP8` should be run with `--gdn-triton` on this stack to avoid the `flashinfer.gdn_prefill` import issue seen in prior logs.
- vLLM GGUF support exists, but the official docs describe it as highly experimental and limited to single-file GGUF loading. Multi-file GGUF repos should be treated as unsupported in this run unless merged first.
