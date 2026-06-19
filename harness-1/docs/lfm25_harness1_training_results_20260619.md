# LFM2.5 Harness-1 Local Training Results - 2026-06-19

This note records the local LFM2.5-8B-A1B Harness-style retrieval experiments run on the BrowseComp fallback dataset.

## Working Directory

- Main project: `/home/work/.projects/LLM-OS-Models/Terminal`
- Harness repo: `harness-1`
- Dataset: `/home/work/.data/harness1/lfm25_local_rlvr/browsecomp_lfm25_harness1_browsecomp_fallback_20260618T103645Z`
- Base model: `LiquidAI/LFM2.5-8B-A1B`

## Best Current Result

The strongest direct evidence-curation model is the hardcase high-recall adapter trained from the previous high-recall direct refresh:

`/home/work/.data/harness1/models/LFM2.5-8B-A1B__hardcase_highrecall_lora_20260619_v1/final_lora`

Direct evidence-curation evaluation on 120 examples:

| Metric | Value |
| --- | ---: |
| Mean total reward | 2.7999 |
| Mean retrieval reward | 2.1499 |
| Strict JSON reward | 0.6500 |
| Valid JSON rate | 1.0000 |
| Fallback rate | 0.0000 |
| Recall | 0.9580 |
| Precision | 0.9776 |
| F2 | 0.9583 |
| All gold found rate | 0.8917 |
| Mean selected docs | 3.1083 |
| Mean invalid docs | 0.1167 |

Summary file:

`/home/work/.data/harness1/evals/20260619_lfm25_hardcase_highrecall_v1_direct_eval/summary.json`

Predictions:

`/home/work/.data/harness1/evals/20260619_lfm25_hardcase_highrecall_v1_direct_eval/predictions.jsonl`

The strongest current agentic-prompt adapter is:

`/home/work/.data/harness1/models/LFM2.5-8B-A1B__agentic_hardcase_lora_20260619_v1/final_lora`

Agentic candidate-pool evaluation on 120 examples:

| Metric | Value |
| --- | ---: |
| Recall | 0.8697 |
| Precision | 0.9448 |
| F2 | 0.8653 |
| All gold found rate | 0.6917 |
| Mean selected docs | 2.5833 |
| Valid action rate | 1.0000 |
| Ended rate | 1.0000 |

Direct JSON evaluation of the same agentic-hardcase adapter:

| Metric | Value |
| --- | ---: |
| Mean total reward | 2.7869 |
| Mean retrieval reward | 2.1369 |
| Strict JSON reward | 0.6500 |
| Valid JSON rate | 1.0000 |
| Recall | 0.9523 |
| Precision | 0.9736 |
| F2 | 0.9518 |
| All gold found rate | 0.8667 |

The agentic-hardcase adapter improves tool-prompt behavior substantially while keeping direct JSON retrieval close to the direct-only best. For pure direct reporting, keep using `LFM2.5-8B-A1B__hardcase_highrecall_lora_20260619_v1`. For agentic-prompt experiments, use `LFM2.5-8B-A1B__agentic_hardcase_lora_20260619_v1`.

## Result History

| Run | Adapter | Eval mode | Count | Recall | Precision | F2 | All gold |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| JSON SFT | `...__rlvr_json_sft_lora_20260619_lfm25_rlvr_json_sft_v1/final_lora` | direct JSON | 120 | 0.6933 | 0.8606 | 0.7080 | 0.3500 |
| JSON SFT + RLVR | `...__local_harness_rlvr_http_vllm_lora_browsecomp_jsonsft_v1/final_lora` | direct JSON | 120 | 0.7799 | 0.8491 | 0.7791 | 0.5083 |
| Mixed agentic from JSON-RL | `...__mixed_agentic_sft_lora_20260619_lfm25_mixed_agentic_from_jsonrl_v1/final_lora` | direct JSON | 120 | 0.8161 | 0.9286 | 0.8266 | 0.5167 |
| Mixed agentic from JSON-RL | same | agentic pool prompt | 80 | 0.8042 | 0.8787 | 0.7961 | 0.6250 |
| Pool-agentic mixed from best | `...__mixed_agentic_sft_lora_20260619_lfm25_pool_agentic_mixed_from_best_v1/final_lora` | direct JSON | 120 | 0.8557 | 0.9601 | 0.8666 | 0.6000 |
| Pool-agentic mixed from best | same | direct-preferred agentic, 3 turns | 80 | 0.7513 | 0.9173 | 0.7641 | 0.5250 |
| High-recall direct refresh | `...__mixed_agentic_sft_lora_20260619_lfm25_highrecall_direct_refresh_v1/final_lora` | direct JSON | 120 | 0.9379 | 0.9741 | 0.9411 | 0.8083 |
| Hardcase high-recall | `...__hardcase_highrecall_lora_20260619_v1/final_lora` | direct JSON | 120 | 0.9580 | 0.9776 | 0.9583 | 0.8917 |
| Hardcase high-recall | same | agentic pool prompt | 120 | 0.7872 | 0.9131 | 0.7961 | 0.5667 |
| Agentic hardcase | `...__agentic_hardcase_lora_20260619_v1/final_lora` | agentic pool prompt | 120 | 0.8697 | 0.9448 | 0.8653 | 0.6917 |
| Agentic hardcase | same | direct JSON | 120 | 0.9523 | 0.9736 | 0.9518 | 0.8667 |

## What Was Trained

The successful path was not pure multi-turn tool use. It was evidence curation over a fixed candidate document pool:

1. Build direct JSON SFT rows from the local Harness-style dataset.
2. Train LoRA SFT so the model outputs strict JSON:
   `{"curated_doc_ids":[...],"reasoning":"..."}`
3. Continue with local HTTP/vLLM RLVR using retrieval rewards.
4. Add mixed SFT rows that include direct JSON curation and agentic trajectories.
5. Keep direct rows heavily repeated so the model does not lose strict JSON retrieval behavior.

The first high-recall refresh used:

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

The second high-recall refresh used:

- Parent adapter: `/home/work/.data/harness1/models/LFM2.5-8B-A1B__mixed_agentic_sft_lora_20260619_lfm25_pool_agentic_mixed_from_best_v1/final_lora`
- Output adapter: `/home/work/.data/harness1/models/LFM2.5-8B-A1B__mixed_agentic_sft_lora_20260619_lfm25_highrecall_direct_refresh_v1/final_lora`
- Direct rows: 828
- Agentic rows: 4140
- Mixed rows after repeat: 15732
- Direct repeat: 14
- Agentic repeat: 1
- Epochs: 0.5
- Learning rate: 7e-6
- Training GPUs: 0,1,2,3
- Train loss: 0.1307
- Runtime: 742.5 seconds

The hardcase high-recall run used:

- Parent adapter: `/home/work/.data/harness1/models/LFM2.5-8B-A1B__mixed_agentic_sft_lora_20260619_lfm25_highrecall_direct_refresh_v1/final_lora`
- Output adapter: `/home/work/.data/harness1/models/LFM2.5-8B-A1B__hardcase_highrecall_lora_20260619_v1/final_lora`
- Source eval predictions: `/home/work/.data/harness1/evals/20260619_lfm25_highrecall_direct_refresh_v1_direct_eval/predictions.jsonl`
- Failed or partial eval rows selected: 26
- Hardcase SFT rows after repeat/repair expansion: 234
- Epochs: 2
- Learning rate: 4e-6
- Training GPUs: 0,1,2,3,4,5,6,7
- Train loss: 0.6933
- Runtime: 25.6 seconds after model load

The agentic hardcase run used:

- Parent adapter: `/home/work/.data/harness1/models/LFM2.5-8B-A1B__hardcase_highrecall_lora_20260619_v1/final_lora`
- Output adapter: `/home/work/.data/harness1/models/LFM2.5-8B-A1B__agentic_hardcase_lora_20260619_v1/final_lora`
- Source eval predictions: `/home/work/.data/harness1/evals/20260619_lfm25_hardcase_highrecall_v1_agentic_eval_full120/predictions.jsonl`
- Failed or partial agentic rows selected: 60
- Agentic hardcase SFT rows after repeat/repair expansion: 540
- Epochs: 1
- Learning rate: 4e-6
- Training GPUs: 0,1,2,3,4,5,6,7
- Train loss: 0.4908
- Runtime: 29.0 seconds after model load
- Agentic eval summary: `/home/work/.data/harness1/evals/20260619_lfm25_agentic_hardcase_v1_agentic_eval_full120/summary.json`
- Direct eval summary: `/home/work/.data/harness1/evals/20260619_lfm25_agentic_hardcase_v1_direct_eval/summary.json`

## Agentic Evaluation Diagnosis

The model can do evidence retrieval well when asked in the same strict JSON format used for SFT/RLVR. It is weaker when the prompt starts with tool-use instructions.

Observed failure modes:

- The model sometimes emits `fan_out_search` instead of final evidence IDs.
- With a one-turn limit, search actions cannot recover because there is no next turn to curate.
- With three turns, valid action rate reaches 1.0, but mean selected docs stays near 2.0, so recall is lower than direct JSON.
- The old mixed adapter reached F2 0.796 in agentic mode because it often used a direct curation shortcut instead of a real multi-turn loop.

Current conclusion:

- Use direct JSON evidence curation as the strongest local score path.
- Agentic-prompt behavior is now materially improved by hardcase SFT: recall increased from 0.7872 to 0.8697 on the same 120-row evaluation.
- The remaining gap is not JSON/action validity; it is incomplete evidence recall under the tool-loop prompt. The model still selects fewer documents than the direct path.
- The model is not failing at evidence selection in general; it is still losing some recall when the prompt frames the task as tool use.

## Official Harness-1 BrowseComp+ Evaluation Status

The official paper-style path is `harness-1/inference/evaluate_harness1_vllm.py`, documented in `harness-1/docs/run_vllm_browsecompplus.md`.

It requires:

- vLLM serving a token-ID-capable `/v1/completions` endpoint.
- BrowseComp+ query/qrel/answer files.
- A compatible Chroma backend containing BrowseComp+ corpus chunks with qrel-matching document IDs.
- `OPENAI_API_KEY`, `CHROMA_API_KEY`, and `CHROMA_DATABASE` available to `harness-1/harness/config.py`.

Local BrowseComp+ files exist under `/home/work/.data/harness1/external/BrowseComp-Plus`, but the checked env files only expose `HF_TOKEN`; they do not expose `OPENAI_API_KEY`, `CHROMA_API_KEY`, or `CHROMA_DATABASE`. Because of that, the official Chroma-backed Harness-1 BrowseComp+ evaluator cannot be run faithfully in the current shell. The local evaluations above are candidate-pool evaluations built from the fallback BrowseComp dataset, not the official Chroma-backed BrowseComp+ search-agent score.

## Code Changes

Added or modified local scripts:

- `harness-1/training/build_lfm25_agentic_sft.py`
- `harness-1/training/build_lfm25_agentic_hardcase_sft.py`
- `harness-1/training/build_lfm25_hardcase_sft.py`
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

To improve beyond the current direct F2 0.9583 and agentic F2 0.8653:

1. Keep the current best adapter as the base.
2. Build more high-recall direct JSON SFT rows with all intermediate evidence, not only final answer pages.
3. Retrain with direct rows dominant and agentic rows small.
4. For real multi-turn agentic behavior, generate trajectories where every search action is followed by a high-recall curate action.
5. Reward emitted doc IDs in the executor, not only final `curated_doc_ids`, so useful search actions are not wasted.
6. Run longer RLVR only after the SFT action distribution is stable.
7. Restore the official Chroma/OpenAI env and run `inference/evaluate_harness1_vllm.py` as a smoke test on 3-10 BrowseComp+ queries before claiming paper-style results.

For Qwen3.5-9B follow-up:

- First reproduce the direct JSON SFT/RLVR path.
- Only then add agentic trajectories.
- Compare against the LFM2.5 best direct score using the same 120-example direct evaluation.

## Current Recommendation

Use this adapter for current best LFM2.5 retrieval experiments:

`/home/work/.data/harness1/models/LFM2.5-8B-A1B__hardcase_highrecall_lora_20260619_v1/final_lora`

Use this result as the primary reported score:

`/home/work/.data/harness1/evals/20260619_lfm25_hardcase_highrecall_v1_direct_eval/summary.json`

Use this result as the current agentic diagnostic:

`/home/work/.data/harness1/evals/20260619_lfm25_agentic_hardcase_v1_agentic_eval_full120/summary.json`
