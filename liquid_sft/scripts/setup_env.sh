#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

if [ ! -d ".liquid-sft-env" ]; then
  uv venv .liquid-sft-env --python 3.12
fi

source .liquid-sft-env/bin/activate

uv pip install --python .liquid-sft-env/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch torchvision torchaudio xformers

uv pip install --python .liquid-sft-env/bin/python \
  huggingface_hub hf_transfer pyyaml sentencepiece protobuf

uv pip install --python .liquid-sft-env/bin/python \
  pandas numpy pyarrow datasets transformers accelerate trl peft bitsandbytes

uv pip install --python .liquid-sft-env/bin/python -e "./unsloth-src[huggingface]"

python - <<'PY'
import torch
print("python ok")
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpus", torch.cuda.device_count())
PY
