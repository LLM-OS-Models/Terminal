# Terminal ECHO RLVR Run Report - 2026-06-08

This note records what we are actually running, what data is used, why the
earlier run was weak, why the checkpoint-50 run stopped, and what the current
vLLM replica setup changes.

## Goal

Train `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch` with
ECHO-style live terminal RLVR.

The target behavior is:

1. sample a terminal task,
2. generate shell/tool actions with vLLM,
3. execute those commands in a per-rollout local workspace,
4. feed stdout, stderr, exit code, and verifier output back into the trajectory,
5. train a LoRA adapter with both verifier reward and ECHO-style terminal
   observation loss.

This is not just offline next-command imitation. The runner executes actual
terminal commands and writes terminal feedback into the training trajectory.

## Current Code Path

Main trainer:

- `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`

Launcher:

- `Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh`

vLLM replica launcher:

- `Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh`

Safety helper imported from:

- `Liquid-CLI/train_lfm_terminal_live_rlvr_grpo.py`

The important local changes are:

- vLLM HTTP rollout now accepts comma-separated base URLs and routes each DDP
  rank to a different vLLM server.
- The launcher can resume from an existing LoRA adapter through
  `SFT_ADAPTER_PATH`.
- Gradient checkpointing is now actually enabled when
  `GRADIENT_CHECKPOINTING=1`.
- vLLM health checks use the first URL when multiple vLLM base URLs are passed.
- `/dev/null` is now allowed by the no-Docker safety filter. This matters
  because normal terminal commands often use `2>/dev/null`; blocking that
  incorrectly punishes otherwise valid `find`, `grep`, and `ls` flows.

## Training Data

The current local live dataset builder loads all terminal RL sources currently
wired into the trainer:

- `open-thoughts/OpenThoughts-Agent-v1-RL`: 728 rows
- `endless-terminals`: 512 rows
- `open-thoughts/OpenThoughts-TB-dev`: 70 rows
- `open-thoughts/OpenThoughts-TBLite`: 98 rows

Total usable rows in the current run: 1,408.

Two rows were skipped as invalid or too long. Prompt lengths in the active run
range from 221 to 3,957 tokens before interaction.

This is smaller than the ECHO paper setup. The paper uses a larger terminal
interaction mix and the original repo stack around SkyRL/Harbor/Docker. Our
current path is a practical local no-Docker reproduction path, not a byte-for-
byte reproduction of the original training system.

## Objective

The loss has two pieces:

- RLVR policy loss on assistant action tokens, weighted by verifier reward.
- ECHO-style cross entropy on terminal observation tokens, controlled by
  `WORLD_MODEL_COEFF`.

Current coefficient:

- `WORLD_MODEL_COEFF=0.03`

This means terminal output is not only context for the next action. It also
becomes dense supervision for the model's internal terminal world model.

The first step of the pre-`/dev/null` resumed run confirmed observation tokens
were entering the loss:

- step: 0
- reward_mean: 0.05625
- verifier_reward_mean: 0.125
- world_loss_mean: 1.00597
- action_tokens_mean: 795.5
- obs_tokens_mean: 296.0

## Current Active Run

Run:

- `run_20260607T223408Z_resume_ckpt50_vllm4rep_train4_devnullfix`

Run directory:

- `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260607T223408Z_resume_ckpt50_vllm4rep_train4_devnullfix`

Output directory:

- `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260607T223408Z_resume_ckpt50_vllm4rep_train4_devnullfix`

Resume adapter:

- `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260607T142109Z_long2d_wm003_g4_tp4_streambackward/checkpoint-50`

GPU split:

- GPU 0-3: four independent vLLM replicas, TP1 each
- GPU 4-7: four DDP training ranks

vLLM URLs:

- `http://127.0.0.1:8123/v1`
- `http://127.0.0.1:8124/v1`
- `http://127.0.0.1:8125/v1`
- `http://127.0.0.1:8126/v1`

Training config:

- DDP world size: 4
- prompts per rank: 1
- generations per prompt: 4
- global rollouts per step: 16
- max turns: 12
- max new tokens per turn: 768
- max sequence length: 16,384
- save interval: every 50 steps
- learning rate: 5e-7
- warmup steps: 50
- max wall time: 47.5 hours
- no Docker
- `/dev/null` safety patch active

Current state:

- As of 2026-06-08 01:20 UTC, the run has completed `step 48`.
- The save interval is 50, so `checkpoint-50` is written after `step 49`.
- After that checkpoint appears, an automatic watcher stops the current
  4-GPU training job and 4-replica vLLM pool, then starts the next long run.

Next-run plan:

- vLLM: six TP1 replicas on GPUs 0,1,2,3,6,7
- training: two DDP ranks on GPUs 4,5
- prompts per rank: 2
- generations per prompt: 4
- global rollouts per step: 16
- resume adapter: `checkpoint-50` from the current run

Why switch to 6-vLLM + 2-train:

- The model is small enough that one H200 can serve one vLLM replica.
- Rollouts are independent HTTP requests, so multiple TP1 replicas usually give
  better request throughput than one TP4 server.
- The current four training ranks only use roughly 39-43GB per H200. The
  bottleneck is not training VRAM. It is rollout generation, terminal
  execution, verifier work, and HTTP round trips.
- Moving two GPUs from training to vLLM is therefore more sensible for the long
  run.
- `PROMPTS_PER_RANK=2` keeps the global rollout count at 16 despite using only
  two training ranks.
- Tensor parallel sizes being powers of two matters inside one sharded model,
  but this workload benefits more from replica parallelism on the rollout side.
- The trainer's vLLM URL routing was also changed from `rank % num_urls` to
  `(seed + rank) % num_urls`. That lets a two-rank training job use all six
  vLLM replicas across rollout seeds.

Expected timing:

- checkpoint-50 save: within minutes from `step 48`
- six-vLLM restart: roughly 5-10 minutes
- first train step of the new two-GPU run: roughly 15-25 minutes after the
  checkpoint is written
- checkpoints after that: every 50 steps in the new run
- the previous observed speed was about 3.4 minutes/step including load time.
  The 6-vLLM + 2-train run needs its first 10 steps before we can recalculate a
  reliable ETA, but the target is to process the same 16 rollouts with higher
  vLLM throughput.

## No-Docker RLVR: What Works And What Is Limited

No-Docker RLVR is possible. It is not the same quality of environment as
Docker-based TerminalBench/SkyRL.

What works:

- Commands really execute.
- stdout, stderr, exit code, and verifier output are written into trajectories.
- The observation text is used in the ECHO world-model loss.
- Per-rollout workspaces prevent most tasks from colliding with each other.

Main limitations:

- Filesystem isolation is weaker than Docker. We must block host-root commands
  instead of letting a container absorb them.
- Safety filters can accidentally block valid commands. We saw this with
  normal shell redirection like `2>/dev/null`, which caused valid `find` flows
  to be marked as `unsafe_pattern`.
- Some TerminalBench-style tasks expect package installs, system-level state, or
  a clean container image. A plain local workspace cannot reproduce all of that.
- The reward signal is less clean when the environment is not fully isolated.

The larger implementation limitation is separate from Docker:

- The current vLLM HTTP servers generate rollouts from the served base/SFT model.
- The DDP process trains the LoRA adapter.
- We do not yet have SkyRL-style per-step weight synchronization from the
  training LoRA into the vLLM rollout servers.

So the current setup is ECHO-style live terminal RLVR with real terminal
feedback and observation-token CE, but it is not the full original on-policy
ECHO/SkyRL implementation.

## Previous 100-Step Static-vLLM Result

Earlier run:

- `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_7gpu_fileinit_100step`

TB2-lite replay evaluation:

| model | score | avg_command_f1 | first_cmd_exact_pct | steps |
| --- | ---: | ---: | ---: | ---: |
| base_sft | 51.22 | 0.5140 | 50.8 | 303 |
| checkpoint-25 | 50.43 | 0.5083 | 49.5 | 303 |
| checkpoint-50 | 51.50 | 0.5206 | 50.2 | 303 |
| checkpoint-75 | 50.55 | 0.5014 | 51.5 | 303 |
| checkpoint-100 | 51.03 | 0.5181 | 49.2 | 303 |
| final_lora | 51.52 | 0.5251 | 49.2 | 303 |

README leaderboard reference for the same base SFT model:

- 52.30

Interpretation:

- It did improve over the same-run base eval: 51.22 -> 51.52.
- The improvement is small: +0.30 score.
- It did not beat the stronger existing README reference: 52.30.
- Command F1 improved more clearly: 0.5140 -> 0.5251.
- First command exact match dropped: 50.8 -> 49.2.

Why the 100-step result was not strong:

- 100 steps is a very short RL run for sparse terminal reward.
- The rollout side was static-vLLM style and did not sync the latest LoRA into
  vLLM every step.
- The no-Docker safety filter was too strict and blocked many otherwise normal
  commands.
- The training data is mixed and relatively small at 1,408 local tasks.
- TB2-lite is a next-action replay proxy, not the same distribution as live
  multi-turn terminal solving.
- The base model is already strong, so small LoRA updates can easily trade
  command F1 against exact formatting.

Why it still counts as a useful signal:

- The same evaluation stack showed a positive final LoRA delta over base.
- The best short-run checkpoint/final models improved average command F1.
- The run proved the end-to-end loop: vLLM rollout -> real terminal execution ->
  verifier reward -> observation CE -> LoRA checkpoint -> TB2-lite eval.

## Previous Long Run And Why It Stopped Near Checkpoint 50

Previous long run:

- `run_20260607T142109Z_long2d_wm003_g4_tp4_streambackward`

Output:

- `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260607T142109Z_long2d_wm003_g4_tp4_streambackward`

This was not "50 epochs". It was a checkpoint at step 50, with logs continuing
through step 54.

Confirmed facts:

- `checkpoint-50` exists and contains a valid LoRA adapter.
- The last observed training log entry was step 54.
- There is no Python traceback at the end of `train.log`.
- There is no clear CUDA OOM line at the end of `train.log`.
- After the stop, the training ranks were gone.
- The vLLM TP4 server processes were still occupying GPU 0-3.

Likely cause:

- Not a confirmed model-training exception.
- Most likely the training process group or launching session was interrupted,
  cleaned up, or one DDP rank exited without a clean Python traceback.
- Because vLLM remained alive and the training ranks disappeared, this looked
  more like process/session management failure than a vLLM startup failure.

Why we changed the setup after that:

- TP4 puts one vLLM engine across GPU 0-3. For many independent rollout
  requests, that can underuse available parallelism.
- Four TP1 replicas give each rank a dedicated rollout server and reduce
  request contention.
- The new run resumes from `checkpoint-50` instead of throwing away the useful
  first 50 steps.

## Trace Quality Observations

Previous long run trace counts:

- rank0: 220 rollouts, 138 blocked, 35 verifier successes
- rank1: 223 rollouts, 139 blocked, 20 verifier successes
- rank2: 224 rollouts, 139 blocked, 26 verifier successes
- rank3: 220 rollouts, 131 blocked, 19 verifier successes

Total:

- 887 rollout traces
- 547 blocked traces
- 100 verifier successes

This block rate is too high. It explains part of the weak learning signal.
One concrete blocker was valid commands with `2>/dev/null`. That path is now
allowed in the safety helper, but the already-running resumed process must be
restarted before that specific patch affects training.

Current resumed run early trace counts at the time this note was written:

- 31 rollout traces
- 17 blocked traces
- 2 verifier successes

These counts came from the pre-`/dev/null` 4-replica run
`run_20260607T222154Z_resume_ckpt50_vllm4rep_train4`, which was stopped at step
1 so the safety patch could take effect. The active `devnullfix` run should be
judged after it reaches at least checkpoint 50.

## Next Steps

1. Let the current 4-replica `devnullfix` resumed run continue to at least
   checkpoint 50.
2. Evaluate current-run checkpoint-50 against TB2-lite.
3. If the block rate remains high, restart with the `/dev/null` safety patch
   active and consider relaxing relative-path safe commands further.
4. Do not upload to Hugging Face unless a checkpoint beats the current internal
   baseline in a meaningful way.
5. Use TB2-lite for fast gating and TerminalBench-2.0 as the final gate.
6. For a closer paper reproduction, implement LoRA weight sync into vLLM or move
   to the original SkyRL/Harbor path.
