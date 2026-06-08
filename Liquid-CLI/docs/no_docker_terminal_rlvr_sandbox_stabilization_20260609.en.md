# No-Docker Terminal RLVR Sandbox Stabilization Notes

Date: 2026-06-09

## Summary

Docker-based sandboxes are not available on this server, and `zerobox` cannot be used as a strict Linux sandbox either. The practical path is therefore to harden the existing no-Docker local sandbox runner instead of switching runtimes.

Main changes:

- Rewrite `/app` into the per-task sandbox `workspace`.
- Rewrite `/tests` into the per-task sandbox `tests` directory.
- Rewrite generated workspace text files after each command execution.
- Block host-sensitive commands such as `tmux`, `screen`, `pkill`, `killall`, `systemctl`, `sudo`, and `apt-get`.
- Replace broad host environment inheritance with a minimal allowlist environment.
- Pin `HOME`, `TMPDIR`, `XDG_RUNTIME_DIR`, and cache paths inside the sandbox.
- Change the no-Docker ECHO run script defaults to `TRAIN_GPUS=4,5` and `NPROC_PER_NODE=2` so GPUs 6 and 7 are not used accidentally.

## zerobox Result

Installed with:

```bash
uv tool install zerobox
```

Version:

```text
zerobox 0.3.3
```

Strict sandbox failed with:

```text
error: strict sandbox requires bubblewrap but user namespaces are unavailable
```

The non-strict fallback also failed with:

```text
permission profiles requiring direct runtime enforcement are incompatible with --use-legacy-landlock
```

So the current kernel/runtime settings do not allow zerobox to be used as the command wrapper for Terminal RLVR.

## OpenSandbox Result

OpenSandbox is a better long-term direction for agent/RL isolation, but its local runtime path depends on Docker. Since Docker is unavailable here, it is not a short-term fix.

Decision matrix:

- Docker available: use Docker/OpenSandbox.
- Docker unavailable but user namespaces available: use zerobox/bubblewrap-style isolation.
- Docker unavailable and user namespaces unavailable: harden the no-Docker runner.

This server is in the third category.

## Why The Old No-Docker Runner Hurt Both Stability And Score

TerminalBench-style tasks often assume Docker-style absolute paths such as `/app`, `/tests`, and `/output`.

The old runner only rewrote `/workspace`, `/output`, and `/logs`. As a result, commands like:

```bash
cat /app/worker_queue.py
pytest /tests/test_state.py
```

could either be blocked by the absolute-path safety rule or fail because the verifier could not find the expected files.

This is a score issue, not just a safety issue. If a correct model action is blocked because the path layout differs from TerminalBench, the rollout reward becomes artificially bad.

## Patch Effects

Host-sensitive commands are now blocked through unsafe-pattern checks and PATH wrappers:

- `tmux`
- `screen`
- `byobu`
- `pkill`
- `killall`
- numeric-PID `kill`
- `nohup`
- `setsid`
- `systemctl`
- `service`
- `sudo`
- `su`
- `ssh`
- `scp`
- `rsync`
- `nc`
- `ncat`
- `telnet`
- `apt`
- `apt-get`
- `dpkg`
- `nvidia-smi`
- `nvcc`

The subprocess environment is now minimal. Important values:

- `HOME=sandbox/workspace`
- `TMPDIR=sandbox/tmp`
- `XDG_RUNTIME_DIR=sandbox/runtime`
- `XDG_CACHE_HOME=sandbox/.cache`
- `PIP_CACHE_DIR=sandbox/.cache/pip`
- `UV_CACHE_DIR=sandbox/.cache/uv`
- `PYTHONPYCACHEPREFIX=sandbox/.cache/python`
- `CUDA_VISIBLE_DEVICES=`
- `NVIDIA_VISIBLE_DEVICES=none`
- `TMUX=`
- `STY=`
- `SSH_AUTH_SOCK=`

This reduces leakage of host context such as HF tokens, tmux sockets, SSH agents, and CUDA device visibility into task commands.

## Smoke Test

The smoke test checked:

- `tmux ls` is blocked.
- `cat /app/foo.txt` is allowed.
- `/app/foo.txt` is rewritten into the sandbox workspace.
- Python-created `/app/bar.txt` is also rewritten and lands in the workspace.

Result:

```text
unsafe_tmux True
unsafe_app_cat False
stdout hello
stderr_first tmux is disabled in no-Docker RLVR sandboxes.
bar_exists True
```

## Effect On The Existing Run

An already-running Python trainer has loaded the old code in memory. The patch does not affect that process automatically.

Safe transition:

1. Wait for the next checkpoint, for example `checkpoint-820`.
2. Stop only the trainer processes.
3. Keep vLLM replicas on GPUs 0-3 running.
4. Start a new run from the latest checkpoint with the patched code.
5. Train on GPUs 4-5 only.

This avoids paying the vLLM startup cost again while applying the sandbox fixes.

## Hugging Face Rollout Dataset Sync

Rollout traces, parsed train-step logs, checkpoint manifests, and redacted environment metadata are synced to:

```text
LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ECHO-RLVR-Rollouts
```

Important:

- Do not `source .env` on a shared machine.
- Let the sync script read the token through `--env-file`.
- The old sync had stopped near `checkpoint-620`, so it must be restarted after `checkpoint-820`.
- The patched run should be synced under its own `runs/<run_id>` prefix.

This keeps the generated terminal observations and trajectories reusable for later SFT/RL experiments.

## Limitations

This patch is not a replacement for Docker or kernel-level sandboxing.

Known limits:

- No full process namespace isolation.
- No Docker-style root filesystem mount.
- No syscall-level network blocking.
- `/app` is emulated by string rewriting rather than a real mount.
- Binary files are not rewritten.

Still, given the current server constraints, this is the most realistic stabilization path.

## Next Recommended Work

1. Restart from `checkpoint-820` with the patched sandbox.
2. Run 100-200 steps and compare with the previous no-Docker run.
3. Track blocker rate, verifier reward mean, timeout rate, and `/app` unsafe-block frequency.
4. Use TB2-lite as a fast mid-run check, and reserve full TB2 for final evaluation.
