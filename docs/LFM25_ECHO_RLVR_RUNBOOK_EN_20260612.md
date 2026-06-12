# LFM2.5 ECHO RLVR Runbook

Updated: 2026-06-12 02:34 UTC / 2026-06-12 11:34 KST

## Short Read

The current result is not "RLVR is useless." The best checkpoint so far, continuation `checkpoint-250`, reaches TB2-lite replay Score `52.88`, above the SFT 1Epoch baseline `52.30`.

However, the curve is noisy. The latest checked continuation checkpoint, `checkpoint-630`, scores `49.85`. So the current finding is: RLVR can produce a better checkpoint, but this run does not yet show a stable long-run aha moment. We should pick the best checkpoint, not the final checkpoint.

Step-axis distinction:

- `parentrun checkpoint-N` means local step `N` from the previous RLVR run started on 2026-06-09. Its range is `10` through `1880`.
- `continue checkpoint-M` means local step `M` in the active continuation run that resumed from parent `checkpoint-1880`.
- Therefore continuation `checkpoint-250` is roughly cumulative step `1880 + 250 = 2130`.
- Looking only at the previous final `parentrun checkpoint-1880` can miss the actual best point. GPU6 is therefore evaluating the full parent run from `checkpoint-10, 20, 30, ... 1880`.

## Active Run

- Base model: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- Active run: `run_20260611T094438Z_echo_public1500_continue_from_1880_vllm4_train2`
- Output dir: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260611T094438Z_echo_public1500_continue_from_1880_vllm4_train2`
- Resume adapter: previous ECHO RLVR `checkpoint-1880`
- Save interval: every 10 train steps
- Latest observed train step: `670`
- Latest observed saved checkpoint: `checkpoint-670`
- Expected wall-time stop: around 2026-06-13 09:14 UTC / 2026-06-13 18:14 KST

## GPU Layout

- GPU 0-3: four vLLM rollout servers, ports `8123-8126`
- GPU 4-5: two RLVR training ranks
- GPU 6: TB2-lite replay evaluation
- GPU 7: excluded from this job

Low GPU utilization during parts of the run is expected. This is live terminal RLVR, not dense SFT. Each step waits on generation, JSON parsing, real shell execution, stdout/stderr capture, verifier execution, trace writing, and then LoRA updates.

## Training Data

Active training file:

- `/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl`
- Total rows: `1,500`
- `endless_terminals`: `772`, `51.47%`
- `openthoughts_agent_v1_rl`: `728`, `48.53%`

This is not an exact reproduction of the ECHO paper infrastructure. The paper uses Harbor/Docker. This run uses a local no-Docker sandbox, but keeps the ECHO core idea: terminal observation tokens receive an auxiliary world-model cross-entropy loss.

TB2-lite is currently held out for checkpoint evaluation. The active run does not directly train on TB-dev/TBLite rows.

## Method

The trainer is `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`.

The loop:

1. The model emits JSON shell commands.
2. Commands execute in a local sandbox.
3. stdout, stderr, exit codes, and verifier results are appended as terminal observations.
4. The full trajectory is forwarded again.
5. Assistant action tokens receive reward-weighted policy loss.
6. Terminal observation tokens receive ECHO-style world-model CE loss.

Current key settings:

- LoRA rank: `32`
- World-model coefficient: `0.03`
- Learning rate: `5e-7`
- Max turns: `8`
- Max new tokens: `512`
- Max sequence length: `12288`
- Command timeout: `20s`
- Verifier timeout: `90s`
- vLLM rollout replicas: `4`
- Training ranks: `2`

## Current Evaluation

Baselines:

- SFT 1Epoch: Score `52.30`
- SFT 2Epoch: Score `50.48`
- LiquidAI base: Score `36.53`
- Parent ECHO RLVR standalone `checkpoint-1880`: Score `50.05`
- Parentrun sweep `checkpoint-10`: Score `51.14`
- Parentrun sweep `checkpoint-1830`: Score `51.94`
- Parentrun sweep `checkpoint-1880`: Score `51.86`

Best RLVR checkpoint so far:

- continuation `checkpoint-250`: Score `52.88`, Cmd F1 `0.5305`, First Cmd `52.5%`, Valid JSON `74.9%`

GPU6 sweep status:

- continuation run: `38` evaluated out of `67` saved checkpoints, `29` remaining
- parent run: `11` evaluated out of `188` saved checkpoints, `177` remaining
- parent run is currently sweeping upward from the beginning: `10, 20, 30, 40, 50, ...`

Main interpretation:

- Positive: RLVR has produced a checkpoint above SFT baseline.
- Negative: later checkpoints regress and the score is not monotonic.
- Current action: keep evaluating every 10 steps and preserve the best checkpoint separately.
- Unknown: most parent-run checkpoints from `10` to `1780` are still pending. The previous 1880-step run may have a better intermediate checkpoint than its final checkpoint.

## Why There Is No Clean Aha Moment Yet

- The SFT baseline is already strong, so remaining headroom is small.
- The live verifier reward is not identical to TB2-lite replay command-F1.
- The local no-Docker sandbox adds path, package, timeout, and verifier noise.
- The run continues from a previous RLVR adapter, which may already be drifted.
- The current active data is a 1,500-row public task mix, not the full paper-scale Harbor corpus.
- Verifier rewards are sparse, so GRPO advantages are noisy.

## Next Experiments

- Evaluate all 10-step checkpoints and latest checkpoints on GPU 6.
- Select best checkpoint, not final checkpoint.
- Run a clean start from SFT 1Epoch without the parent RLVR adapter.
- Sweep world-model coefficient: `0.00`, `0.01`, `0.03`, `0.05`.
- Add reward shaping closer to TB2-lite command-F1 / first-command behavior.
- Test a more stable sandbox layer such as `zerobox` or `OpenSandbox`.
- Add TerminalBench 1/3-style train data only with clear train/eval separation.
