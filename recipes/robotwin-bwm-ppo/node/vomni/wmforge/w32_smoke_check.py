#!/usr/bin/env python3
"""w32 smoke: the critic warmup freezes the policy for training steps 1-4 and the joint update starts at
step 5; the value head reads the VLM prefix (input 2048). Generalises c2_smoke_check.py (steps 1 / 2).

    python w32_smoke_check.py --sft <model.safetensors> --run <smoke checkpoints dir> [--warmup 4]
"""
import argparse
import sys

import torch
from safetensors.torch import load_file


def load(run, step):
    path = f"{run}/global_step_{step}/actor/model_state_dict/full_weights.pt"
    return {k: v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--warmup", type=int, default=4)
    a = ap.parse_args()
    sft = {k: v.float() for k, v in load_file(a.sft).items()}
    by_suffix = {}
    for k in sft:
        by_suffix.setdefault(k.split(".", 1)[-1], k)
    steps = {s: load(a.run, s) for s in range(1, a.warmup + 2)}
    first = steps[1]
    vh = sorted(k for k in first if "value_head" in k)
    in_dim = first[next(k for k in vh if first[k].dim() == 2)].shape[1]
    matched = []
    for k in first:
        if "value_head" in k:
            continue
        sk = k if k in sft else by_suffix.get(k.split(".", 1)[-1])
        if sk is not None and sft[sk].shape == first[k].shape:
            matched.append((k, sk))

    def d_sft(st):
        return max(min((st[k] - sft[sk]).abs().max().item(),
                       (st[k] - sft[sk].to(torch.bfloat16).float()).abs().max().item()) for k, sk in matched)

    warm = {s: d_sft(steps[s]) for s in range(1, a.warmup + 1)}
    joint = max((steps[a.warmup + 1][k] - steps[a.warmup][k]).abs().max().item() for k, _ in matched)
    vh_moved = max((steps[a.warmup][k] - steps[1][k]).abs().max().item() for k in vh)
    finite = all(torch.isfinite(steps[a.warmup + 1][k]).all() for k in vh)
    print(f"value_head params {len(vh)}, input dim {in_dim}; policy params matched to SFT {len(matched)}")
    print("max |step_s - SFT| over policy:", {s: f"{v:.3e}" for s, v in warm.items()})
    print(f"max |step{a.warmup + 1} - step{a.warmup}| over policy = {joint:.3e}; value head moved in warmup = {vh_moved:.3e}; finite={finite}")
    ok = in_dim == 2048 and matched and all(v == 0 for v in warm.values()) and joint > 0 and vh_moved > 0 and finite
    print(f"W32_SMOKE_WEIGHTS {'OK' if ok else 'BAD'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
