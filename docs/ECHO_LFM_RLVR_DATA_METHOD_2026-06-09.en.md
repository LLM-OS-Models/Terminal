# ECHO-Style LFM Terminal RLVR Data and Method Notes

Date: 2026-06-09

## Current conclusion

The goal is not to clone the full Docker/Harbor setup from the ECHO paper. The goal is to adapt the paper's core learning method to our current LFM no-Docker terminal RLVR environment.

The core method is:

1. The model emits terminal actions.
2. Real terminal feedback, including stdout, stderr, exit codes, and verifier results, is appended to the next-turn context.
3. Those observation tokens are also used as cross-entropy targets, so terminal feedback becomes a world-model supervision signal instead of context only.

`Liquid-CLI/train_lfm_terminal_echo_live_grpo.py` already implements this path.

- Action tokens: policy loss driven by verifier reward.
- Observation tokens: terminal-output prediction CE loss.
- Final loss: `policy_coeff * policy_loss + world_model_coeff * world_loss`.
- Current long run uses `WORLD_MODEL_COEFF=0.03`.

So the current answer to "does terminal feedback enter the loss?" is yes.

## ECHO paper corpus vs. local corpus

The paper reports:

- 8870 terminal tasks total.
- 1977 tasks from Endless Terminals.
- 723 tasks from OpenThoughts-Agent-v1-RL.
- 6170 additional Harbor-format tasks generated with a modified Endless Terminals pipeline.
- Split: train 8770, val 100.
- Runtime: Docker + Harbor.
- Episode length: up to 16 turns.
- Context: 16k.
- Max generated tokens per turn: 2048.
- Training: 500 GRPO steps on 8x B200.

Public/local sources we can use now:

- `open-thoughts/OpenThoughts-Agent-v1-RL`
- `obiwan96/endless-terminals`
- `open-thoughts/OpenThoughts-TB-dev`
- `open-thoughts/OpenThoughts-TBLite`
- TB2 should remain primarily a final evaluation benchmark.

Important limitation:

- The paper's additional 6170 generated Harbor exports are not directly available from the public repos we inspected.
- We therefore prepare the public/local task archives into the ECHO schema and run them through our local no-Docker harness.
- The data format and objective are ECHO-style; the runtime isolation is our local sandbox instead of Docker/Harbor.

## Current long run

Run ID:

`run_20260609T000050Z_patched_sandbox_resume820_vllm4_train2_setsid`

Run directory:

`/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260609T000050Z_patched_sandbox_resume820_vllm4_train2_setsid`

Model:

`LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`

GPU placement:

- GPUs 0-3: four vLLM replicas.
- GPUs 4-5: two LoRA RLVR training ranks.
- GPUs 6-7: not used.

Dataset loaded by the current run:

- 1408 rows total.
- OpenThoughts RL: 728.
- Endless Terminals: 512.
- OpenThoughts TB-dev: 70.
- OpenThoughts TBLite: 98.
- Skipped rows: 2.

The current run uses ECHO-style world loss, but its data path is still driven by the HF/local loaders. To make the data source fully explicit for future runs, we added a prepared-data script and `--prepared-jsonl` support.

Latest checked status:

- Latest log step checked: 375.
- Latest saved checkpoint: `checkpoint-370`.
- Save interval: 10 steps.
- Current GPU placement: four vLLM replicas on GPUs 0-3, training ranks on GPUs 4-5.
- GPUs 6-7 are reserved for another job and must not be used.

## New data preparation script

Download script:

`Liquid-CLI/scripts/download_echo_public_terminal_data.py`

Responsibilities:

- Download OpenThoughts parquet if missing.
- Resume Endless Terminals task-file downloads.
- Sleep and retry instead of exiting on Hugging Face 429 rate limits.
- Read the HF token from `.env` without printing it.

Background command:

```bash
mkdir -p /home/work/.data/echo_terminal_data/logs
nohup python Liquid-CLI/scripts/download_echo_public_terminal_data.py \
  --env-file .env \
  --retry-seconds 180 \
  --max-attempts 1000 \
  >/home/work/.data/echo_terminal_data/logs/download_echo_public_terminal_data.log 2>&1 &
```

Log:

`/home/work/.data/echo_terminal_data/logs/download_echo_public_terminal_data.log`

Preparation script:

File:

`Liquid-CLI/scripts/prepare_echo_terminal_data.py`

Responsibilities:

1. Preserve OpenThoughts `task_binary` archives directly from parquet.
2. Repack Endless task directories as gzipped tar `task_binary` archives.
3. Emit ECHO/SkyRL-compatible parquet.
4. Emit local LFM no-Docker trainer JSONL.
5. Write a source-count manifest with percentages.

Command:

```bash
set -a
source .env >/dev/null 2>&1 || true
set +a

python Liquid-CLI/scripts/prepare_echo_terminal_data.py \
  --tokenizer LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
  --max-prompt-tokens 4096
```

Current output:

- 1500 rows total.
- Endless Terminals: 772 rows, 51.4667%.
- OpenThoughts-Agent-v1-RL: 728 rows, 48.5333%.
- Skipped: 0.

Current local raw-data status:

- Endless task directories: 774.
- Endless files: 13914.
- OpenThoughts parquet: present.

Two Endless task directories were incomplete or filtered out, so 772 of the 774 local Endless directories became training rows. The downloader retries after Hugging Face 429 rate limits. Re-run the preparation command as more tasks finish downloading.

## Generated outputs

Output directory:

`/home/work/.data/echo_terminal_data/prepared`

Files:

- `echo_terminal_tasks_mixed.parquet`
- `echo_terminal_tasks_openthoughts.parquet`
- `echo_terminal_tasks_endless.parquet`
- `lfm_live_tasks_mixed.jsonl`
- `lfm_live_tasks_mixed.parquet`
- `solution_references_mixed.jsonl`
- `manifest.json`

ECHO/SkyRL parquet schema:

- `prompt`
- `path`
- `task_binary`
- `instruction`
- `source`
- `_data_source`
- `prompt_tokens`

LFM live JSONL schema:

- `prompt`
- `task_id`
- `source`
- `task_dir`
- `task_binary_b64`
- `instruction`
- `echo_path`
- `prompt_tokens`

`solution_references_mixed.jsonl` extracts `solution/solve.sh` from task archives. It is not on-policy RL data. Use it for optional SFT, analysis, or verifier sanity checks, not as an RL trajectory.

## Trainer changes

Modified files:

- `Liquid-CLI/train_lfm_terminal_live_rlvr_grpo.py`
- `Liquid-CLI/train_lfm_terminal_echo_live_grpo.py`
- `Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh`

New option:

`--prepared-jsonl /path/to/lfm_live_tasks_mixed.jsonl`

Wrapper environment variables:

- `PREPARED_JSONL=/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl`
- `PREPARED_ONLY=1` disables the automatic HF loaders and uses only prepared JSONL.
- `PREPARED_ONLY=0` mixes prepared JSONL with the existing HF loaders.

Verified dry-run:

```bash
PREPARED_ONLY=1 \
PREPARED_JSONL=/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl \
TRAIN_GPUS= \
MAX_STEPS=1 \
Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh \
  --max-rows 16 \
  --dry-run
```

Result:

- Prepared JSONL loaded correctly.
- 16 rows loaded.
- Prompt token range: 485-539.

## No-Docker RL limitations

Docker/Harbor is not required for data preparation, but it matters for execution stability.

Limitations:

1. Task isolation is weaker than Docker.
2. Tasks with hidden Dockerfile package dependencies can fail locally.
3. `/workspace`, `/work`, `/home/user`, `/output`, and `/logs` rewriting may not cover every task.
4. Unsafe command filtering is critical because there is no container boundary.
5. Tasks depending on services, background processes, system packages, networking, or process managers can produce noisy verifier results.

Current mitigations:

- Each task archive is unpacked into a per-rollout local sandbox.
- `/workspace`, `/work`, and `/home/user` are rewritten to the sandbox `workspace`.
- `/output` and `/logs` are rewritten into sandbox output/log paths.
- `/tmp` is rewritten to the sandbox `tmp` directory.
- `sudo`, `tmux`, `screen`, `nohup`, `setsid`, `pkill`, `killall`, root filesystem scans, and similar unsafe operations are blocked.
- GPU probe commands are blocked inside task sandboxes.

2026-06-09 fix:

- The previous rewriter used cascading string replacement, so `/home/user/output` could become the broken path `/home/user/home/work/.../output` after only the `/output` segment was rewritten.
- The unsafe checker also blocked normal `/home/user/...` and `/tmp/...` absolute paths, which incorrectly rejected common commands such as `mkdir`, `ls`, `stat`, and `wc`.
- A later trace showed the same issue for Endless-style `/work/tickets/...` paths and read-only commands such as `find /home/user ...` and `cat /home/user/...`.
- Added `rewrite_known_paths()` to perform one regex substitution pass for `/home/user`, `/work`, `/tmp`, `/workspace`, `/output`, `/logs`, `/tests`, and `/app`.
- Read-only traversal commands such as `find`, `ls`, `cat`, and `wc` now pass when all absolute paths are inside allowed task paths; host-sensitive paths such as `/`, `/etc`, `/proc`, and `/sys` remain blocked.
- Smoke check now allows `mkdir -p /home/user/diagnostics`, `ls -la /home/user/data`, `stat /home/user/data/status.json`, and `wc -l /tmp/out.txt`, while `rm -rf /` remains blocked.

Longer-term fixes:

- Add a local process sandbox such as ZeroBox or OpenSandbox.
- Use a Docker/Harbor-capable node for a closer paper-style run when available.
- Keep this node focused on fast no-Docker sandbox iteration plus ECHO-style loss.

## Recommended next long run

Keep the current run alive. For the next new run, explicitly include prepared data. The `--prepared-jsonl` path has already passed a dry-run check:

```bash
PREPARED_JSONL=/home/work/.data/echo_terminal_data/prepared/lfm_live_tasks_mixed.jsonl \
PREPARED_ONLY=0 \
VLLM_BASE_URL=http://127.0.0.1:8123/v1,http://127.0.0.1:8124/v1,http://127.0.0.1:8125/v1,http://127.0.0.1:8126/v1 \
TRAIN_GPUS=4,5 \
NPROC_PER_NODE=2 \
MAX_STEPS=100000 \
MAX_WALL_TIME_HOURS=47.5 \
WORLD_MODEL_COEFF=0.03 \
SAVE_STEPS=10 \
Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh
```

Use `PREPARED_ONLY=0` when we want diversity from TB-dev/TBLite and existing HF loaders. Use `PREPARED_ONLY=1` for cleaner source ablations.

## Why earlier short results were noisy

Likely causes:

1. Terminal rewards are sparse.
2. Low solve rates weaken GRPO group contrast.
3. No-Docker verifier failures add reward noise.
4. Short runs can stop before ECHO world loss stabilizes terminal dynamics.
5. Without a prepared manifest, source attribution is weaker.

Why the run can still improve:

- Observation CE learns stdout/stderr/exit-code structure even from failed rollouts.
- Failed trajectories still contain world-model supervision.
- OpenThoughts/Endless task format is close to the model's existing terminal SFT distribution.

## Checkpoint selection

Select checkpoints by multiple signals:

- TB2 final evaluation.
- TBLite/TB-dev heldout evaluation.
- Verifier reward moving average.
- World loss trend and stability.
- Parse error rate.
- Timeout rate.
- Command count and token usage.

The ECHO paper also evaluates both task pass rate and terminal-output prediction cross-entropy.
