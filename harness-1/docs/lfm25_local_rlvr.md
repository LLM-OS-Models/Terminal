# LFM2.5 Local Harness RLVR

This repo keeps two RL paths separate.

- `training/train_rl.py`: original Harness-1 Tinker/Chroma RL. It expects Tinker checkpoints, Chroma Cloud retrieval, and Harness-1 SEC/web/patents datasets.
- `training/train_lfm25_local_rlvr_grpo.py`: local Hugging Face training for `LiquidAI/LFM2.5-8B-A1B` with PEFT LoRA and TRL GRPO.

The local path starts from the pure base model by default. It only uses an SFT adapter if `SFT_ADAPTER_PATH` is explicitly set.

## Paper SEC Requirements

The paper-aligned SEC RL path is not just a local JSON file. It needs:

- SEC query dataset: `kellyhongg/1_18_sec_train`, split `train`.
- SEC RL split IDs: `datagen/splits/sec_splits.json`, key `rl_query_ids`.
- Retrieval corpus/index: Chroma collection `sec_1_4`.
- Real candidate pools from the same corpus/index.

The local builder intentionally fails if only gold SEC facts are available, because training on gold-only candidates is not paper-aligned RL.

Current local status checked on 2026-06-18:

- root `.env` contains `HF_TOKEN`, but no `CHROMA_API_KEY`, `CHROMA_DATABASE`, or `TINKER_API_KEY`.
- `kellyhongg/1_18_sec_train` is not accessible with the current HF token.
- `/home/work/.data/harness1/sec` exists, but no materialized SEC candidate JSONL or local HF cache for `kellyhongg/1_18_sec_train` was found.
- GPUs were idle after the failed SEC data check; no full training process was started.

So the original SEC run is blocked by data/access, not by the LFM2.5 trainer path.

## Local SEC JSONL Contract

If SEC candidates are materialized from `sec_1_4`, use:

```bash
DATASET_KIND=sec_jsonl \
SEC_JSONL=/home/work/.data/harness1/sec/sec_rlvr_candidates.jsonl \
MAX_STEPS=300 \
bash training/launch_lfm25_local_rlvr.sh
```

Each JSONL row should contain:

- `query_id`
- `query`
- `answer`
- `gold_docs`: list of `{docid,title,text}` or Harness-1 `document_ids` fact items
- `candidate_docs`: retrieved candidates from the SEC index, including the gold docs

Optional row fields:

- `evidence_docs`
- `negative_docs`

Rows with only gold candidates are rejected by default. Use `ALLOW_GOLD_ONLY_CANDIDATES=1` only for a wiring smoke test.

## Launch

From `harness-1`:

```bash
DATASET_KIND=sec_hf \
MAX_STEPS=300 \
bash training/launch_lfm25_local_rlvr.sh
```

For a wiring-only smoke check with gold-only SEC candidates:

```bash
DATASET_KIND=sec_hf \
ALLOW_GOLD_ONLY_CANDIDATES=1 \
LIMIT=16 \
MIN_CANDIDATES=1 \
MAX_STEPS=1 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
bash training/launch_lfm25_local_rlvr.sh
```

For the public BrowseComp+ proxy, explicitly opt in:

```bash
DATASET_KIND=browsecomp \
MAX_STEPS=300 \
bash training/launch_lfm25_local_rlvr.sh
```

BrowseComp+ is useful for code and reward wiring, but it is not the paper SEC RL run.

## Outputs

By default:

- dataset: `/home/work/.data/harness1/lfm25_local_rlvr/...`
- model LoRA: `/home/work/.data/harness1/models/.../final_lora`
- logs: `/home/work/.data/harness1/logs/...`
