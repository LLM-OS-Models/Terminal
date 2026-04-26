#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash liquid_sft/scripts/run_sft_8gpu.sh \
  --config liquid_sft/configs/sft_h200_4gpu_lfm25_1p2b_base.env &
PID_A=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 \
  bash liquid_sft/scripts/run_sft_8gpu.sh \
  --config liquid_sft/configs/sft_h200_4gpu_lfm2_2p6b.env &
PID_B=$!

wait "$PID_A"
wait "$PID_B"
