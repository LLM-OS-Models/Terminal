# LFM2.5 Raw ECHO RLVR Restart Log

Updated: 2026-06-12 11:38 UTC

This note tracks the clean restart from the raw `LiquidAI/LFM2.5-8B-A1B` base model. It is separate from the earlier SFT-based RLVR runs.

## Current Decision

The previous SFT-based continuation run has been stopped. The active experiment now trains a fresh LoRA adapter on top of the raw `LiquidAI/LFM2.5-8B-A1B` model, with no SFT adapter loaded.

GPU allocation:

- GPUs `0,1,2,3`: vLLM rollout servers, ports `8123-8126`
- GPUs `4,5`: LoRA/GRPO training
- GPU `6`: TB2-lite replay evaluation watcher
- GPU `7`: excluded from this work

## Why Restart From Raw

The previous experiments started from `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`, which is already strong on the terminal replay metric. That makes RLVR gains harder to interpret.

The new question is narrower:

```text
Can raw LFM2.5 learn terminal behavior from ECHO-style RLVR and terminal feedback?
```

The raw base rerun score is `39.92` on the same TB2-lite replay setup, while the SFT 1Epoch baseline is `52.30`. This gives the raw run much more headroom.

## vLLM Failure Causes

The idle GPU issue was not caused by the model or the trainer. It was caused by vLLM runtime setup.

1. User-site package contamination

`.vllm-lfm-cu12` was accidentally loading `torch 2.12.0.dev20260407+cu128` from user-site paths. Its vLLM extension matches `torch 2.10.0+cu128`, so the API server failed with a C++ ABI error.

Fix: add `PYTHONNOUSERSITE=1` to vLLM, training, and evaluation launchers.

2. CUDA13 environment mismatch

`.vllm-lfm` can import vLLM, but it is a CUDA13 build. The current host driver is CUDA 12.9 era, so GPU init failed with a driver-too-old error.

Fix: use `.vllm-lfm-cu12`.

3. Process lifetime

Starting vLLM in the background from a short health-check shell caused the servers to be cleaned up when the command ended.

Fix: keep vLLM inside the long-running `run_lfm25_vllm_replicas_clean.sh` launcher, which waits on the server processes.

## Code Changes

The following scripts were patched:

- `Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh`
- `Liquid-CLI/scripts/run_lfm25_vllm_server_clean.sh`
- `Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh`
- `Liquid-CLI/scripts/watch_echo_rlvr_gpu6_eval_queue.sh`
- `Liquid-CLI/scripts/evaluate_echo_rlvr_checkpoints_gpu6.sh`

The training launcher also now supports `VLLM_SERVED_MODEL`, so the raw vLLM server can be addressed as `lfm25-raw`.

## Data

The active run uses:

```text
/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl
```

Composition:

- Total: `1,500` rows
- `openthoughts_agent_v1_rl`: `728` rows, `48.53%`
- `endless_terminals`: `772` rows, `51.47%`
- Prompt token min/max: `362 / 1568`
- Skipped long/invalid: `0`

TB2-lite replay data is not included in training. This is not a byte-for-byte reproduction of the internal ECHO paper data or Harbor/Docker backend. It is a local no-Docker implementation of the core ECHO objective: verifier RL plus CE loss on terminal observation tokens.

## Stopped First Raw Run

Stopped run:

```text
run_20260612T112155Z_echo_raw_lfm25_vllm4_train2_g4_t6_tok512_save50_wm005_2k
```

It used `max_turns=6`, `max_new_tokens=512`, and `save_steps=50`. It was stopped at step `4`, before any checkpoint was saved.

Reason:

- `action_tokens_mean` ranged from `1496` to `3101`.
- Step time was roughly `2.0-2.5 min/step`.
- A 2000-step run would likely exceed the 48-hour budget.

## Active Fast Raw Run

Active run:

```text
run_20260612T113238Z_echo_raw_lfm25_vllm4_train2_g4_t4_tok256_save25_wm005_2k
```

Base model:

```text
LiquidAI/LFM2.5-8B-A1B
```

SFT adapter: none

Key settings:

- LoRA rank: `32`
- Trainable parameters: `12,867,584`
- World-model coefficient: `0.05`
- Learning rate: `1e-6`
- Warmup steps: `10`
- Max steps: `2000`
- Max wall time: `48h`
- Max turns: `4`
- Max new tokens: `256`
- Command timeout: `8s`
- Verifier timeout: `40s`
- Save interval: every `25` steps
- Global rollouts per step: `8`

Initial step `0-5` observations:

| step | reward_mean | verifier_reward_mean | action_tokens_mean | obs_tokens_mean |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `-0.1650` | `0.0` | `528.0` | `391.25` |
| 1 | `-0.2000` | `0.0` | `1043.5` | `92.0` |
| 2 | `-0.1662` | `0.0` | `914.75` | `140.75` |
| 3 | `-0.1700` | `0.0` | `1044.0` | `90.5` |
| 4 | `-0.1350` | `0.0` | `634.75` | `288.0` |
| 5 | `-0.1887` | `0.0` | `979.5` | `121.0` |

Interpretation:

- The raw model has not yet produced verifier-positive behavior in the first few steps.
- Action length is now much lower than the stopped first raw run.
- Early speed is roughly `30-35 sec/step`.
- The first checkpoint should appear around step `25`.
- A 2000-step run is roughly `17-20 hours` at the current measured speed.

## Upload and Evaluation

Hugging Face sync:

- Rollout dataset repo: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts`
- Adapter model repo: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-GRPO-Adapters`
- Path prefix: `raw-lfm25/{run_id}`
- Sync interval: `1800s`

GPU6 evaluation:

- Script: `Liquid-CLI/scripts/watch_echo_rlvr_gpu6_eval_queue.sh`
- Base model: `LiquidAI/LFM2.5-8B-A1B`
- Early stride: every `25` steps
- Result dir: `tb2_lite/results/lfm25_echo_rlvr_gpu6_eval_20260612`

README score remains:

```text
Score = 100 * avg_command_f1
```

## Checkpoint-25 to Checkpoint-225 TB2-lite Results

Updated: 2026-06-12 13:45 UTC.

GPU6 evaluated checkpoint-25 through checkpoint-225 from the current raw clean-start run using the same TB2-lite replay score used by the root README.

| checkpoint | README Score | next_action_score | first_cmd_exact | valid_json |
| ---: | ---: | ---: | ---: | ---: |
| 25 | `37.89` | `39.00` | `41.6%` | `59.4%` |
| 50 | `40.34` | `40.90` | `42.2%` | `58.1%` |
| 75 | `39.19` | `39.22` | `39.3%` | `57.1%` |
| 100 | `40.34` | `40.81` | `41.9%` | `58.4%` |
| 125 | `41.01` | `41.67` | `43.2%` | `57.4%` |
| 150 | `40.43` | `40.39` | `40.3%` | `59.4%` |
| 175 | `40.02` | `40.19` | `40.6%` | `58.1%` |
| 200 | `39.67` | `39.95` | `40.6%` | `59.7%` |
| 225 | `41.06` | `40.92` | `40.6%` | `57.4%` |

Current raw-run best is `checkpoint-225` with Score `41.06`.

Interpretation:

- Compared with the raw base rerun Score `39.92`, checkpoint-225 is `+1.14`.
- Compared with the SFT 1Epoch baseline Score `52.30`, checkpoint-225 is still `-11.24`.
- Checkpoint-225 barely beats checkpoint-125, but the first 225 steps still look like a small recovery/search phase rather than a clear aha moment.
- Valid JSON remains in the `57.1%` to `59.7%` band, which confirms that raw terminal JSON/tool formatting is still weak.
- Verifier reward is still near `0.0`, so this early phase is likely driven more by ECHO observation loss and shaping penalties than by sparse verifier success.

This is not a final negative result. The raw run does recover above raw base at checkpoint-50/100/125/225, so the terminal-feedback signal is not dead. The decisive checkpoints are still 500, 1000, and 2000.

## Risks

- Raw model format following is weak, so verifier reward may remain sparse.
- No-Docker local sandbox differs from the ECHO paper's Harbor/Docker backend.
- The training reward and TB2-lite command-F1 metric are related but not identical.
- Longer RL is not automatically better; checkpoint curves matter.

Decision points:

- `checkpoint-25`: verify that raw behavior does not collapse.
- `checkpoint-50`: check format and reward trends.
- `checkpoint-100`: compare TB2-lite score against raw baseline `39.92`.
- `checkpoint-200+`: decide whether an "aha moment" is emerging.

If verifier reward remains zero through checkpoint `100`, pure raw RLVR is likely too sparse and a small format/trajectory warmup should be used before another RLVR run.
