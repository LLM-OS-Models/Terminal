# Terminal ECHO RLVR Run Report - 2026-06-08

## 2026-06-08 02:20 UTC Update

The target GPU allocation is capped to GPUs 0-5 only.

- GPUs 0,1,2,3: four TP1 vLLM replicas
- GPUs 4,5: two LoRA DDP training ranks
- GPUs 6,7: unused/reserved for other work; do not touch them

Current active run:

- run id: `run_20260608T023253Z_resume_ckpt50_vllm4rep_train2_total6_rw8_vram092_gpu_filter_false_smi_setsid`
- run dir: `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260608T023253Z_resume_ckpt50_vllm4rep_train2_total6_rw8_vram092_gpu_filter_false_smi_setsid`
- output dir: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260608T023253Z_resume_ckpt50_vllm4rep_train2_total6_rw8_vram092_gpu_filter_false_smi_setsid`
- resume adapter: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260607T223408Z_resume_ckpt50_vllm4rep_train4_devnullfix/checkpoint-50`

This run resumes from the previous checkpoint-50 adapter.
`MAX_WALL_TIME_HOURS=47.5` and `SAVE_STEPS=50`, so a healthy run writes an
adapter checkpoint every 50 steps.

Important interpretation:

- GPUs 4 and 5 using about 19GB each is normal. This is LoRA adapter training,
  not full fine-tuning. Because the trainer process uses
  `CUDA_VISIBLE_DEVICES=4,5`, internal logs say `cuda:0` and `cuda:1`; those map
  to physical GPUs 4 and 5.
- GPUs 0-3 use much more VRAM once vLLM is ready because each replica reserves a
  large KV cache. A healthy replica reaches about 133GB VRAM on H200 with
  `gpu_memory_utilization=0.92`.
- vLLM is required. Rollout generation goes through vLLM HTTP
  `/v1/completions`; the trainer then executes terminal commands and computes
  RLVR/ECHO losses from the real terminal feedback.

Failures observed and root causes:

1. `run_20260608T020459Z_resume_ckpt50_vllm4rep_train2_total6_rw8_vram092_gpu_filter_killpg`
   loaded the model and adapter onto GPUs 4 and 5 correctly, and vLLM replicas
   on GPUs 0-3 were receiving requests. The failure was not GPU placement.
   A rank1 rollout generated a repeated `nvidia-smi -q -d COMPUTE` probing loop.
   The no-Docker `nvidia-smi` wrapper blocked the GPU call, but thousands of
   wrapper bash processes accumulated and stalled rollout progress.
2. The underlying issue is no-Docker terminal execution. Without container
   isolation, generated commands can probe host GPUs or leave background loops.
   We must enforce this in the local sandbox layer.
3. The first mitigation strengthened the `nvidia-smi` wrapper from SIGTERM to
   process-group SIGKILL. That still left one problem: if the generated command
   repeatedly invokes `nvidia-smi`, the wrapper shell itself can be spawned in
   large numbers before the group is killed. The final local mitigation changes
   `nvidia-smi` from a bash wrapper to a `/bin/false` symlink. Repeated probes
   fail immediately without accumulating wrapper bash processes. GPU-dependent
   task/verifier/command filtering remains enabled.
4. `run_20260608T021136Z...sigkill`, `run_20260608T021537Z...stagger`, and
   `run_20260608T021947Z...stagger2` showed
   unstable vLLM launcher behavior. A direct
   `python -m vllm.entrypoints.openai.api_server` probe worked, so the model and
   vLLM compatibility were not the issue. The replica launcher was changed from
   launching four servers at once to starting each GPU replica sequentially and
   waiting for `/models` readiness before starting the next one. The active run
   is also launched with `setsid bash launch.sh`, which keeps the orchestrator in
   its own session and avoided the parent-shell/job-control failure mode.

What is working:

- LFM2.5-8B-A1B starts correctly on vLLM 0.19.1 with `--trust-remote-code`,
  `--dtype bfloat16`, `--max-model-len 32768`, and `--enforce-eager`.
- A manual GPU0 vLLM probe reached `/v1/models` 200 OK in about 35 seconds.
- The active run brought all four vLLM replicas to ready state, with GPUs 0-3
  each using about 133GB VRAM.
- The active run attached the train launcher and reached `optimizer_ready`.
  GPUs 4-5 use about 19.7GB / 19.1GB during LoRA DDP training.
- After the false-symlink patch, the active run's `nvidia-smi` wrapper process
  count is 0.
- All four active vLLM replicas are receiving `/v1/completions` POSTs.
- The active run completed its first train step.
  - step: 0
  - loss: -0.00785
  - reward_mean: -0.046875
  - verifier_reward_mean: 0.0
  - world_loss_mean: 1.47626
  - action_tokens_mean: 1060.5
  - obs_tokens_mean: 611.25
  - rollout traces: 8 for rank0 and 8 for rank1
  - interpretation: reward is still low, but real terminal observation tokens
    are entering the ECHO-style world-model loss.
- One vLLM replica loads about 15.8GiB of model weights and reserves about
  110GiB for KV cache, reaching roughly 133GB VRAM total.
- The trainer can load the checkpoint-50 LoRA adapter and wrap it with 2-rank
  DDP.
- ECHO-style observation CE is active in the code path. Terminal stdout,
  stderr, exit code, and verifier result enter the trajectory as observation
  tokens and are weighted by `WORLD_MODEL_COEFF=0.03`.

What still needs work:

- The no-Docker sandbox must block GPU probing, background loops, and process
  leaks more aggressively. A text filter plus wrappers are useful, but generated
  shell scripts can still route around simple command-text checks.
- The vLLM rollout server does not yet synchronize the freshly updated LoRA
  weights at every training step. Rollouts are served by the base/SFT model
  while the trainer updates the adapter. This is not a fully on-policy SkyRL
  implementation.
- TB2 should remain the final evaluation target. Training currently uses the
  OpenThoughts RL, Endless Terminals, TB-dev, and TBLite train mix.
- The 1,408 usable local rows are smaller than the paper-scale mix. To add
  TerminalBench 1/3 train data safely, we need task conversion, verifier
  normalization, and no-Docker safety filtering.

Fundamental limitation:

- Without Docker, terminal RL commands run on the host. Per-rollout workspaces
  help, but they are not complete syscall/filesystem/network/GPU isolation. If
  filters are too strict, reward quality degrades; if they are too loose, the
  model can touch host resources.
- ECHO's core idea is to train on terminal observations, but long stable RL
  also requires clean environment isolation, verifier quality, rollout weight
  synchronization, and a reliable checkpoint/evaluation loop.
- A two-day run can be useful only if the reward stream is clean and the vLLM
  rollout path remains stable. Longer RL by itself is not enough.

Code changes in this update:

- `Liquid-CLI/train_lfm_terminal_live_rlvr_grpo.py`
  - add an explicit system-prompt instruction not to run GPU/CUDA/NVIDIA probes
  - replace the no-Docker `nvidia-smi` bash wrapper with a `/bin/false` symlink
  - keep GPU-dependent command/task/verifier filtering for `nvidia-smi`, `nvcc`,
    `nvidia-debugdump`, `nvidia-cuda-mps`, and CUDA/NVIDIA patterns
- `Liquid-CLI/scripts/run_lfm25_vllm_replicas_clean.sh`
  - launch vLLM replicas sequentially
  - wait for each `/models` readiness check before starting the next replica
  - print the failing replica log tail on startup failure

Operating rules:

- GPUs 0-3 for vLLM, GPUs 4-5 for training, GPUs 6-7 untouched.
- Do not print `.env` tokens in docs, logs, commits, or model cards.
- Once a stable checkpoint and evaluation result exist, append them as a new
  section.

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

Current restarted run:

- `run_20260608T012844Z_resume_devnullfix_ckpt50_vllm4rep_train2_total6`

GPU split:

- vLLM: four TP1 replicas on GPUs 0,1,2,3
- training: two DDP ranks on GPUs 4,5
- idle/reserved: GPUs 6,7
- prompts per rank: 2
- generations per prompt: 4
- global rollouts per step: 16
- resume adapter: `checkpoint-50` from the current run

First step confirmed:

- `step 0` complete
- `reward_mean`: 0.04375
- `verifier_reward_mean`: 0.0625
- `world_loss_mean`: 1.18109
- `action_tokens_mean`: 1,586.625
- `obs_tokens_mean`: 1,049.625

This confirms the current run is not just using sparse GRPO reward. Real
terminal execution feedback, converted into observation tokens, is entering the
ECHO-style world-model CE loss.

Why switch to 4-vLLM + 2-train, six GPUs total:

- The model is small enough that one H200 can serve one vLLM replica.
- Rollouts are independent HTTP requests, so multiple TP1 replicas usually give
  better request throughput than one TP4 server.
- The current four training ranks only use roughly 39-43GB per H200. The
  bottleneck is not training VRAM. It is rollout generation, terminal
  execution, verifier work, and HTTP round trips.
- Two training ranks are therefore enough, while four vLLM replicas keep the
  rollout side parallel.
- Total GPU usage is capped at six. GPUs 6 and 7 are left idle/reserved for
  evaluation, comparison runs, or recovery.
- `PROMPTS_PER_RANK=2` keeps the global rollout count at 16 despite using only
  two training ranks.
- Tensor parallel sizes being powers of two matters inside one sharded model,
  but this workload benefits more from replica parallelism on the rollout side.
- The trainer's vLLM URL routing was also changed from `rank % num_urls` to
  `(seed + rank) % num_urls`. That lets a two-rank training job use all four
  vLLM replicas across rollout seeds.

Expected timing:

- checkpoint-50 save: within minutes from `step 48`
- four-vLLM restart: completed in roughly one minute
- first train step of the new two-GPU run: complete
- checkpoints after that: every 50 steps in the new run
- the previous observed speed was about 3.4 minutes/step including load time.
  The 4-vLLM + 2-train run needs its first 10 steps before we can recalculate a
  reliable ETA, but the target is to process the same 16 rollouts with higher
  vLLM throughput and fewer training GPUs.

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
- No-Docker tasks/verifiers can try to touch host GPUs. We observed one
  TerminalBench-lite test script running `CUDA_VISIBLE_DEVICES=7 ... --device
  cuda:0`, which could steal an idle GPU. The subprocess environment now clears
  CUDA/NVIDIA visible-device variables and prepends no-GPU wrappers for
  `python`, `python3`, `pytest`, `pip`, and `nvidia-smi`. Model-generated
  commands containing CUDA/NVIDIA/GPU-use patterns are also blocked as unsafe.
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
