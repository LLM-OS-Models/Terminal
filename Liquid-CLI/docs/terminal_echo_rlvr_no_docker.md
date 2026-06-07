# Terminal ECHO-Style Live RLVR, No Docker Path

This path trains LoRA adapters for:

- Base/SFT model: `LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch`
- Objective: verifier reward on assistant action tokens plus ECHO-style cross
  entropy on terminal observation tokens
- Runtime: local per-rollout workspaces, no Docker or Harbor
- Rollout server: vLLM OpenAI-compatible `/v1/completions`

## What Is Implemented

`Liquid-CLI/train_lfm_terminal_echo_live_grpo.py` runs real multi-turn terminal
episodes:

1. sample terminal tasks,
2. ask the model for shell/tool actions,
3. execute commands in a local sandbox workspace,
4. append stdout, stderr, exit code, and verifier feedback as observations,
5. optimize LoRA adapters with policy reward plus observation-token CE.

This matches the core ECHO idea from Microsoft Research: terminal output is not
only context for the next action, it is also dense supervision for the policy's
world model.

## Difference From The ECHO Paper

The original `echo-rl` repository uses SkyRL plus Harbor/Docker:

- 8,770 train tasks, 100 held-out validation tasks,
- up to 16 turns,
- 16k to 32k context depending on evaluation,
- vLLM rollout engines with training weight sync,
- Docker isolation for terminal actions.

This local path intentionally avoids Docker. It therefore uses weaker
filesystem isolation and must block host-root commands. The current vLLM HTTP
server produces fast base-model rollouts; LoRA weights are trained by the DDP
process. This is ECHO-style live terminal RLVR, but not the full SkyRL
weight-synchronized on-policy implementation.

## Training Data Used Locally

The current live builder loads every locally wired terminal RL source:

- `open-thoughts/OpenThoughts-Agent-v1-RL`: 728 rows
- `endless-terminals`: 512 rows
- `open-thoughts/OpenThoughts-TB-dev`: 70 rows
- `open-thoughts/OpenThoughts-TBLite`: 98 rows

Total loaded rows in the current run: 1,408.

## No-Docker Safety

Because commands run on the host from a local sandbox directory, the base live
terminal runner now blocks unsafe absolute-path access outside:

- `/workspace`
- `/output`
- `/logs`

It also starts commands in a separate process group and kills the whole process
group on timeout. This prevents stalled commands such as `find / ...` from
surviving after the shell timeout.

## Current Long Run

Started on `2026-06-07`:

- Run dir: `/home/work/.data/liquid_cli_sft/live_terminal_echo_vllm/run_20260607T142109Z_long2d_wm003_g4_tp4_streambackward`
- Output dir: `/home/work/.data/liquid_cli_sft/models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch__echo_live_grpo_vllm_r32_run_20260607T142109Z_long2d_wm003_g4_tp4_streambackward`
- vLLM GPUs: `0,1,2,3`
- vLLM tensor parallel size: `4`
- training GPUs: `4,5,6,7`
- DDP world size: `4`
- rollout workers per rank: `4`
- global rollouts per step: `16`
- max wall time: `47.5h`
- save interval: every `50` steps
- max turns: `12`
- max new tokens per turn: `768`
- max seq length: `16384`
- max terminal output chars: `12000`
- world model coefficient: `0.03`
- learning rate: `5e-7`

First completed step from this run:

- `step`: 0
- `reward_mean`: 0.040625
- `verifier_reward_mean`: 0.125
- `world_loss_mean`: 1.6652
- `action_tokens_mean`: 1334.25
- `obs_tokens_mean`: 765.75

The previous 32k run OOMed before its first checkpoint. The trainer now
backpropagates each trajectory immediately instead of keeping all rollout
graphs alive until the end of the step. The launcher also keeps gradient
checkpointing enabled and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

GPU utilization is phase-based: GPUs `0,1,2,3` are hot during vLLM rollout
generation, while GPUs `4,5,6,7` hold the train ranks and become active during
trajectory loss/backward. A single `nvidia-smi` snapshot can therefore show
the vLLM side at high utilization while the training side is waiting for
rollout results.

## Previous 100-Step Result

Earlier 100-step static-vLLM rollout experiment:

- base under same 32k LoRA eval path: `51.22`
- `checkpoint-25`: `50.43`
- `checkpoint-50`: `51.50`
- `checkpoint-75`: `50.55`
- `checkpoint-100`: `51.03`
- `final_lora`: `51.52`
- README leaderboard reference for the base SFT model: `52.30`

Interpretation: the short run showed a small positive delta over the same-run
base evaluation, but did not beat the README leaderboard score. Longer training
needs the safe live-run setup above and should be evaluated checkpoint by
checkpoint before updating the root leaderboard.

## Launch Pattern

vLLM server:

```bash
ROOT_DIR=/home/work/.projects/LLM-OS-Models/Terminal \
MODEL_PATH=LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
VLLM_GPUS=0,1,2,3 \
TENSOR_PARALLEL_SIZE=4 \
MAX_MODEL_LEN=32768 \
GPU_MEMORY_UTILIZATION=0.88 \
PORT=8123 \
bash Liquid-CLI/scripts/run_lfm25_vllm_server_clean.sh
```

Training:

```bash
ROOT_DIR=/home/work/.projects/LLM-OS-Models/Terminal \
MODEL_PATH=LLM-OS-Models/LFM2.5-8B-A1B-Terminal-ToolBench-Full-SFT-1Epoch \
VLLM_BASE_URL=http://127.0.0.1:8123/v1 \
TRAIN_GPUS=4,5,6,7 \
NPROC_PER_NODE=4 \
MAX_STEPS=100000 \
MAX_WALL_TIME_HOURS=47.5 \
PROMPTS_PER_RANK=1 \
NUM_GENERATIONS=4 \
ROLLOUT_WORKERS=4 \
MAX_TURNS=12 \
MAX_NEW_TOKENS=768 \
MAX_SEQ_LENGTH=16384 \
WORLD_MODEL_COEFF=0.03 \
LEARNING_RATE=5e-7 \
WARMUP_STEPS=50 \
MAX_GRAD_NORM=0.15 \
SAVE_STEPS=50 \
COMMAND_TIMEOUT=30 \
VERIFIER_TIMEOUT=120 \
MAX_TERMINAL_OUTPUT_CHARS=12000 \
COMMAND_BONUS=0.0 \
FORMAT_PENALTY=0.05 \
REWARD_SUCCESS_BONUS=0.2 \
GRADIENT_CHECKPOINTING=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash Liquid-CLI/scripts/run_lfm_terminal_echo_live_grpo_vllm_no_docker.sh
```

## Evaluation Plan

Do not update the root README leaderboard until checkpoints are evaluated.
Evaluate at least:

- `checkpoint-50`
- `checkpoint-100`
- `checkpoint-200`
- best later checkpoint
- `final_lora`

Use TB2-lite as the fast gate and TerminalBench-2.0 as the final gate.
