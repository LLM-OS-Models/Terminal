# Gemma 4 native SFT

This folder is for the Gemma 4 rerun after the TB2-lite template audit.

Principles:

- Build data from raw `conversations`, not preformatted Qwen/ChatML text caches.
- Use Gemma 4 turn tokens through the tokenizer chat template.
- Inject the matching `*-it` chat template for base Gemma 4 models because the base tokenizers do not ship a `chat_template`.
- Strip previous thinking blocks from history before rendering prompts.
- Train only on the current assistant JSON command response. System/user/history tokens are masked with `-100`.
- Keep tokenized caches model-specific until tokenizer/template identity is proven safe.

This is intentionally not the Qwen SFT path:

- Do not use `/home/work/.data/qwen_sft/datasets/*processed*` text caches.
- Do not train on `<|im_start|>` or `<|im_end|>`.
- Do not use Qwen ChatML formatting.
- Do not let the trainer regenerate data from a generic text cache.
- Do not train on previous assistant thoughts, shell transcripts, or terminal output continuations.

Native Gemma data format:

- E2B/E4B instruct template: `<bos><|turn>system...<turn|>\n<|turn>user...<turn|>\n<|turn>model\n`
- 26B/31B instruct template: same turn format, plus the native empty thought channel in non-thinking mode.
- Base models: tokenizer vocab is compatible, but `chat_template` is absent, so the matching instruct template is injected before tokenization.
- Supervised sequence: `generation_prompt + assistant JSON + <turn|>`.
- Labels: only the current assistant JSON command object and closing turn token are supervised.

Prepared small-model caches:

| Model | Dataset path | Rows | Template injected |
| --- | --- | ---: | --- |
| `google/gemma-4-E2B-it` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E2B-it__liquid_raw_json_masked_8192` | 16,322 | false |
| `google/gemma-4-E4B-it` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E4B-it__liquid_raw_json_masked_8192` | 16,322 | false |
| `google/gemma-4-E2B` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E2B__liquid_raw_json_masked_8192` | 16,322 | true |
| `google/gemma-4-E4B` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-E4B__liquid_raw_json_masked_8192` | 16,322 | true |
| `google/gemma-4-26B-A4B-it` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-26B-A4B-it__liquid_raw_json_masked_8192` | 16,311 | false |
| `google/gemma-4-26B-A4B` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-26B-A4B__liquid_raw_json_masked_8192` | 16,311 | true |
| `google/gemma-4-31B-it` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-31B-it__liquid_raw_json_masked_8192` | 16,311 | false |
| `google/gemma-4-31B` | `/home/work/.data/gemma4_native_sft/datasets/google__gemma-4-31B__liquid_raw_json_masked_8192` | 16,311 | true |

Execution order:

1. Run an E2B-it smoke train for a few steps to verify Gemma 4 text-only load, collator, labels, FSDP, and checkpoint save.
2. Train the four small models in parallel, two GPUs each:
   - GPUs 0,1: `google/gemma-4-E2B-it`
   - GPUs 2,3: `google/gemma-4-E4B-it`
   - GPUs 4,5: `google/gemma-4-E2B`
   - GPUs 6,7: `google/gemma-4-E4B`
3. Evaluate every epoch checkpoint on corrected 303-step TB2-lite with the patched Gemma 4 evaluator.
4. Promote the best small-model setting to 26B-A4B-it and 26B-A4B.
5. Only after 31B save/load is explicitly validated, train 31B-it and 31B.

Prepared large-model configs:

- `configs/sft_gemma4_26b_a4b_it_native_8gpu.env`
- `configs/sft_gemma4_26b_a4b_base_native_8gpu.env`
- `configs/smoke_gemma4_31b_it_native_8gpu.env`
- `configs/sft_gemma4_31b_it_native_8gpu.env`
- `configs/sft_gemma4_31b_base_native_8gpu.env`

The 31B path starts with the smoke config because the previous 31B SFT checkpoints collapsed to non-JSON repeated text. Do not start the full 31B run until the smoke checkpoint loads in vLLM and emits sane command JSON on a small replay subset.

Evaluation and upload helpers:

- `scripts/eval_native_checkpoint.sh`: runs corrected TB2-lite replay with Gemma 4 native prompt options.
- `scripts/stage_model_repo.py`: copies a selected checkpoint to a clean staging folder and writes a model card.
- `scripts/upload_model_repo.py`: uploads the staged folder to Hugging Face. It can read `.env` without printing token values.

Initial small-model targets:

1. `google/gemma-4-E2B-it`
2. `google/gemma-4-E4B-it`
3. `google/gemma-4-E2B`
4. `google/gemma-4-E4B`

Large targets after the full evaluation finishes:

1. `google/gemma-4-26B-A4B-it`
2. `google/gemma-4-26B-A4B`
3. `google/gemma-4-31B-it`
4. `google/gemma-4-31B`
