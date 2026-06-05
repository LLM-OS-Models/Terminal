import os
from huggingface_hub import snapshot_download
repo = "unsloth/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF"
local_dir = "/home/work/.data/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-GGUF"
token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
path = snapshot_download(
    repo_id=repo,
    allow_patterns=["UD-Q4_K_M/*"],
    local_dir=local_dir,
    token=token,
    max_workers=8,
)
print(path)
