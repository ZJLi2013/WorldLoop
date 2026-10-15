"""Download the w32 assets into the node layout under WL_ROOT.

    python fetch_assets.py <WL_ROOT>          # models and checkpoints (~45 GB)
    python fetch_assets.py <WL_ROOT> --w6     # plus the 677 WM fine-tuning episodes (only to redo WM fine-tuning)
"""
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

root = Path(sys.argv[1])
opts = set(sys.argv[2:])


def get(repo, names, local_dir, repo_type="model"):
    for name in names:
        if not (root / local_dir / name).exists():
            hf_hub_download(repo, name, repo_type=repo_type, local_dir=root / local_dir)
        print("ok", local_dir, name)


get("WorldArena/WorldArena",
    ["fulldata/model.safetensors", "fulldata/metadata.pt", "fulldata/assets/robotwin_clean/norm_stats.json"],
    "vomni/track2/ckpt")
get("Wan-AI/Wan2.2-TI2V-5B",
    ["config.json", "diffusion_pytorch_model.safetensors.index.json"]
    + [f"diffusion_pytorch_model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)],
    "vomni/native/Wan2.2-TI2V-5B")
get("RLinf/RLinf-Wan-LIBERO-Spatial", ["Wan2.2_VAE.pth"], "rlinf/ckpt/RLinf-Wan-LIBERO-Spatial")
get("lidavidsh/bwm-robotwin-adjust-bottle-failure-ft", ["step-750.safetensors"], "vomni/artifacts/w8-pilot")
get("lidavidsh/robotwin-adjust-bottle-t5-success-classifier", ["reward_model.pt", "report.json"],
    "vomni/artifacts/r2-reward-model")
get("google-t5/t5-base", ["config.json", "spiece.model", "tokenizer.json", "model.safetensors"], "vomni/ckpt/t5-base")
# bwm_common.py globs hf/hub/models--BLM-Lab--Boundless-World-Model/snapshots/*/step-12000.safetensors
hf_hub_download("BLM-Lab/Boundless-World-Model", "step-12000.safetensors",
                revision="738a8d3c008e637b8b1b18d5e98a82f6de9c04aa", cache_dir=root / "hf/hub")
print("ok hf/hub step-12000")

resnet = root / "vomni/torch_home/hub/checkpoints/resnet18-f37072fd.pth"
if not resnet.exists():
    resnet.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve("https://download.pytorch.org/models/resnet18-f37072fd.pth", resnet)
print("ok", resnet.relative_to(root))

if "--w6" in opts:
    w6 = root / "vomni/artifacts/w6"
    snapshot_download("lidavidsh/robotwin-adjust-bottle-pi05-rollouts", repo_type="dataset", local_dir=w6,
                      allow_patterns=["episodes/*"])
    # bwm_train.py / fidelity.py mmap the frames uncompressed, as w6_index.py wrote them on the node.
    for npz in sorted((w6 / "episodes").glob("*.npz")):
        out = npz.with_name(npz.stem + "_frames.npy")
        if not out.exists():
            np.save(out, np.load(npz)["frames"])
    print("ok w6", len(list((w6 / "episodes").glob("*_frames.npy"))), "episodes")
