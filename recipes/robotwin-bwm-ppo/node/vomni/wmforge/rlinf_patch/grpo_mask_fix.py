#!/usr/bin/env python3
"""Let RLinf's grpo advantage run without a loss mask. The embodied actor only builds loss_mask when
both auto_reset and ignore_terminations are off; the WMForge runs keep both on (fixed 200-step
episodes), so every step is valid and the mask is all ones. GAE already accepts None. Idempotent.

    python grpo_mask_fix.py <tree>/rlinf/algorithms/advantages.py
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "grpo without a loss mask" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)
anchor = (
    "    advantages = grouped_rewards - grouped_reward_mean\n"
    "    advantages = advantages / (grouped_reward_std + 1e-6)\n"
    "\n"
    "    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask\n"
)
assert src.count(anchor) == 1, "grpo anchor not found"
fix = (
    "    advantages = grouped_rewards - grouped_reward_mean\n"
    "    advantages = advantages / (grouped_reward_std + 1e-6)\n"
    "\n"
    "    if loss_mask is None:  # grpo without a loss mask: fixed-length episodes, every step valid\n"
    "        loss_mask = torch.ones(\n"
    "            kwargs[\"n_steps\"], advantages.numel(), dtype=advantages.dtype, device=advantages.device\n"
    "        )\n"
    "    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask\n"
)
open(path, "w", encoding="utf-8").write(src.replace(anchor, fix))
print("PATCHED", path)
