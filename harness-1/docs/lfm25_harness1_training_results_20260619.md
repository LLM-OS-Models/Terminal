# LFM2.5 Harness-1 Local Training Results - 2026-06-19

This note records the local LFM2.5-8B-A1B Harness-style retrieval experiments run on the BrowseComp fallback dataset.

## Working Directory

- Main project: `/home/work/.projects/LLM-OS-Models/Terminal`
- Harness repo: `harness-1`
- Dataset: `/home/work/.data/harness1/lfm25_local_rlvr/browsecomp_lfm25_harness1_browsecomp_fallback_20260618T103645Z`
- Base model: `LiquidAI/LFM2.5-8B-A1B`

## Best Current Result

The strongest current model is the mixed direct/agentic SFT adapter trained from the previous JSON-RL adapter:

`/home/work/.data/harness1/models/LFM2.5-8B-A1B__mixed_agentic_sft_lora_20260619_lfm25_pool_agentic_mixed_from_best_v1/final_lora`

Direct evidence-curation evaluation on 120 examples:

| Metric | Value |
| --- | ---: |
| Mean total reward | 2.5757 |
| Mean retrieval reward | 1.9257 |
| Strict JSON reward | 0.6500 |
| Valid JSON rate | 1.0000 |
| Fallback rate | 0.0000 |
| Recall | 0.8557 |
| Precision | 0.9601 |
| F2 | 0.8666 |
| All gold found rate | 0.6000 |
| Mean selected docs | 2.7333 |
| Mean invalid docs | 0.1333 |

Summary file:

`/home/work/.data/harness1/evals/20260619_lfm25_pool_agentic_mixed_from_best_v1_direct_eval/summary.json`

Predictions:

`/home/work/.data/harness1/evals/20260619_lfm25_pool_agentic_mixed_from_best_v1_direct_eval/predictions.jsonl`

## Result History

| Run | Adapter | Eval mode | Count | Recall | Precision | F2 | All gold |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| JSON SFT | `...__rlvr_json_sft_lora_20260619_lfm25_rlvr_json_sft_v1/final_lora` | direct JSON | 120 | 0.6933 | 0.8606 | 0.7080 | 0.3500 |
| JSON SFT + RLVR | `...__local_harness_rlvr_http_vllm_lora_browsecomp_jsonsft_v1/final_lora` | direct JSON | 120 | 0.7799 | 0.8491 | 0.7791 | 0.5083 |
| Mixed agentic from JSON-RL | `...__mixed_agentic_sft_lora_20260619_lfm25_mixed_agentic_from_jsonrl_v1/final_lora` | direct JSON | 120 | 0.8161 | 0.9286 | 0.8266 | 0.5167 |
| Mixed agentic from JSON-RL | same | agentic pool prompt | 80 | 0.8042 | 0.8787 | 0.7961 | 0.6250 |
| Pool-agentic mixed from best | `...__mixed_agentic_sft_lora_20260619_lfm25_pool_agentic_mixed_from_best_v1/final_lora` | direct JSON | 120 | 0.8557 | 0.9601 | 0.8666 | 0.6000 |
| Pool-agentic mixed from best | same | direct-preferred agentic, 3 turns | 80 | 0.7513 | 0.9173 | 0.7641 | 0.5250 |

## What Was Trained

The successful path was not pure multi-turn tool use. It was evidence curation over a fixed candidate document pool:

1. Build direct JSON SFT rows from the local Harness-style dataset.
2. Train LoRA SFT so the model outputs strict JSON:
   `{"curated_doc_ids":[...],"reasoning":"..."}`
3. Continue with local HTTP/vLLM RLVR using retrieval rewards.
4. Add mixed SFT rows that include direct JSON curation and agentic trajectories.
5. Keep direct rows heavily repeated so the model does not lose strict JSON retrieval behavior.

The best mixed run used:

- Parent adapter: `/home/work/.data/harness1/models/LFM2.5-8B-A1B__mixed_agentic_sft_lora_20260619_lfm25_mixed_agentic_from_jsonrl_v1/final_lora`
- Output adapter: `/home/work/.data/harness1/models/LFM2.5-8B-A1B__mixed_agentic_sft_lora_20260619_lfm25_pool_agentic_mixed_from_best_v1/final_lora`
- Direct rows: 828
- Agentic rows: 4140
- Mixed rows after repeat: 10764
- Direct repeat: 8
- Agentic repeat: 1
- Epochs: 0.5
- Learning rate: 1e-5
- Training GPUs: 0,1,2,3
- Train loss: 0.1879
- Runtime: 520.9 seconds

## Agentic Evaluation Diagnosis

The model can do evidence retrieval well when asked in the same strict JSON format used for SFT/RLVR. It is weaker when the prompt starts with tool-use instructions.

Observed failure modes:

- The model sometimes emits `fan_out_search` instead of final evidence IDs.
- With a one-turn limit, search actions cannot recover because there is no next turn to curate.
- With three turns, valid action rate reaches 1.0, but mean selected docs stays near 2.0, so recall is lower than direct JSON.
- The old mixed adapter reached F2 0.796 in agentic mode because it often used a direct curation shortcut instead of a real multi-turn loop.

Current conclusion:

- Use direct JSON evidence curation as the official score path.
- Treat agentic tool-loop evaluation as a secondary diagnostic until the action policy is retrained.
- The model is not failing at evidence selection in general; it is failing to preserve recall under tool-loop prompting.

## Code Changes

Added or modified local scripts:

- `harness-1/training/build_lfm25_agentic_sft.py`
- `harness-1/training/launch_lfm25_agentic_sft.sh`
- `harness-1/training/mix_lfm25_sft_jsonl.py`
- `harness-1/training/launch_lfm25_mixed_agentic_sft.sh`
- `harness-1/training/train_lfm25_rlvr_json_sft.py`
- `harness-1/eval_scripts/eval_lfm25_agentic_vllm.py`
- `harness-1/eval_scripts/eval_lfm25_agentic_checkpoint_vllm.sh`

The latest agentic evaluator now:

- Includes the full candidate document pool in the initial prompt.
- Accepts direct `curated_doc_ids` as a final curation action.
- Handles `fan_out_search`, `search`, `lookup`, and `web_search` aliases.
- Treats emitted `doc_ids` from search-like actions as curated evidence when appropriate.
- Strongly prefers direct evidence JSON when the candidate pool is already present.

## Next Training Direction

To improve beyond the current direct F2 0.8666:

1. Keep the current best adapter as the base.
2. Build more high-recall direct JSON SFT rows with all intermediate evidence, not only final answer pages.
3. Retrain with direct rows dominant and agentic rows small.
4. For real multi-turn agentic behavior, generate trajectories where every search action is followed by a high-recall curate action.
5. Reward emitted doc IDs in the executor, not only final `curated_doc_ids`, so useful search actions are not wasted.
6. Run longer RLVR only after the SFT action distribution is stable.

For Qwen3.5-9B follow-up:

- First reproduce the direct JSON SFT/RLVR path.
- Only then add agentic trajectories.
- Compare against the LFM2.5 best direct score using the same 120-example direct evaluation.

## Current Recommendation

Use this adapter for current best LFM2.5 retrieval experiments:

`/home/work/.data/harness1/models/LFM2.5-8B-A1B__mixed_agentic_sft_lora_20260619_lfm25_pool_agentic_mixed_from_best_v1/final_lora`

Use this result as the primary reported score:

`/home/work/.data/harness1/evals/20260619_lfm25_pool_agentic_mixed_from_best_v1_direct_eval/summary.json`

Use this result only as the current agentic diagnostic:

`/home/work/.data/harness1/evals/20260619_lfm25_pool_agentic_mixed_from_best_v1_agentic_directpref_limit80_turn3/summary.json`
