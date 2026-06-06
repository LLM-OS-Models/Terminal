# Harness-1 Retrieval RLVR For LFM2.5

This folder implements the local path for training an LFM2.5 retrieval agent
with Harness-1 style rewards.

## Goal

The target behavior is not terminal command execution. The model should become
better at retrieval curation:

- inspect a query and candidate document pool,
- keep every document needed to answer or verify the query,
- avoid dropping answer-supporting documents after they have been found,
- return a structured curated set that can be scored by recall and precision.

This follows the core Harness-1 idea from arXiv:2606.02373: move bookkeeping
into the environment and train the model to make high-level retrieval decisions.

## Data Split Discipline

Harness-1 does not train SFT and RL on the same distribution.

- SFT prior: mixed-domain trajectories from BrowseComp+/Web/Patents/SEC.
- RL: SEC-only search episodes.
- Evaluation: source-domain and held-out transfer benchmarks.

The local LFM2.5 SFT adapter already covers the mixed-domain warm start. RL must
therefore not be reported as paper-aligned if it only uses BrowseComp+.

## Public Proxy Stage

The executable public proxy is candidate-pool curation over BrowseComp+ because
the released files include query, gold, evidence, and negative documents:

`/home/work/.data/harness1/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl`

Each row contains a query, gold documents, evidence documents, and negative
documents. The builder emits GRPO prompts with a shuffled candidate pool. The
model must output:

```json
{"curated_doc_ids":["doc_id_1","doc_id_2"],"reasoning":"brief evidence-based reason"}
```

The reward is programmatic and does not require OpenAI:

- `F2(gold recall, precision)` for recall-heavy evidence selection,
- gold document recall,
- answer/evidence document recall,
- penalties for invalid IDs, empty curation, and over-selection,
- a smaller strict-JSON reward.

This proxy is useful for plumbing and reward-smoke tests only. It is not the
paper SEC RL stage.

## Paper-Aligned SEC Stage

The paper-aligned config is:

`Liquid-CLI/configs/rlvr_h200_8gpu_lfm25_8b_a1b_harness1_sec_grpo.env`

It targets the Harness-1 SEC train RL split:

- dataset class: `sec`
- HF path from upstream code: `kellyhongg/1_18_sec_train`
- split IDs: `harness-1/datagen/splits/sec_splits.json`
- split key: `rl_query_ids`
- Chroma collection expected by upstream code: `sec_1_4`

The cloned Harness-1 release does not bundle the SEC corpus or Chroma index, and
the current environment only has an `HF_TOKEN`; it has no `CHROMA_API_KEY`,
`CHROMA_DATABASE`, or accessible SEC HF dataset. For this reason the SEC config
intentionally fails fast until the SEC data/index is provided. This prevents
accidentally burning GPUs on BrowseComp+ while calling it SEC RL.

For this machine, the practical SEC-only path is to generate SEC tasks with
`context-1-data-gen`, then convert its JSON outputs:

```bash
Liquid-CLI/scripts/prepare_context1_sec_rlvr_data.sh \
  --env-file /path/to/private/sec.env \
  --identity "Name email@example.com" \
  --output-dir /home/work/.data/harness1/sec/context1_output \
  --output-jsonl /home/work/.data/harness1/sec/context1_sec_rlvr.jsonl \
  --overwrite-jsonl
```

If Context-1 SEC JSON files already exist, skip generation and only convert:

```bash
python Liquid-CLI/scripts/convert_context1_sec_to_rlvr_jsonl.py \
  --input-dir /home/work/.data/harness1/sec/context1_output \
  --output-jsonl /home/work/.data/harness1/sec/context1_sec_rlvr.jsonl \
  --overwrite
```

Then train with:

```bash
Liquid-CLI/scripts/run_lfm_harness1_retrieval_rlvr.sh \
  --config Liquid-CLI/configs/rlvr_h200_8gpu_lfm25_8b_a1b_harness1_sec_context1_grpo.env
```

## Run

```bash
Liquid-CLI/scripts/run_lfm_harness1_retrieval_rlvr.sh \
  --config Liquid-CLI/configs/rlvr_h200_8gpu_lfm25_8b_a1b_harness1_retrieval_grpo.env
```

The default config uses 8 H200 GPUs and starts from the reloadable SFT adapter:

`/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B__harness1_lora_sft_r32_h200_8gpu_reloadable_3epoch/final_lora`

The default config trims candidate snippets to keep the first RL run moving on
the current non-vLLM path. It uses `MAX_DOC_CHARS=900`,
`MAX_PROMPT_LENGTH=8192`, and `MAX_COMPLETION_LENGTH=512`; checkpoints are saved
every 25 GRPO steps so partial progress is inspectable.

The output is:

`/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B__harness1_retrieval_rlvr_grpo_r32_browsecomp_v1`

The runner automatically evaluates `final_lora` after training with
`Liquid-CLI/scripts/eval_harness1_retrieval_curation.py` and writes metrics to:

`$OUTPUT_DIR/eval/metrics.json`

## Why Not Use The Upstream RL Script Directly

The upstream Harness-1 RL code is designed around Tinker and a live retrieval
stack. It also has optional LLM verification paths. For this machine, the local
route is more reliable: use available gold labels and train directly on
curation quality. This preserves the important retrieval objective while
removing external API dependence from the RL loop.

## Next Stage

The full paper path needs true multi-turn tool use:

1. `fan_out_search` / `search_corpus` over a local BM25 or embedding index,
2. `read_document` to expose full text,
3. `curate` to update working memory,
4. `end_search` to submit the curated set,
5. terminal reward using the same recall, answer-recall, and missed-found-doc
   penalties.
