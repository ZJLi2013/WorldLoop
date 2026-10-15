#!/usr/bin/env python3
"""w32: check one rendered config (`--cfg job --resolve` output in train.log) against the frozen recipe.

    python3 w32_cfgcheck.py <train.log> <seed> <max_steps>
"""
import re
import sys

import yaml

log, seed, max_steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
start = next(i for i, line in enumerate(lines) if re.match(r"^[a-z_]+:", line))
cfg = yaml.safe_load("\n".join(line for line in lines[start:] if not line.startswith("RUN_EXIT")))


def get(path):
    c = cfg
    for k in path.split("."):
        if not isinstance(c, dict) or k not in c:
            return "<absent>"
        c = c[k]
    return c


want = {
    "algorithm.bootstrap_type": "none", "actor.model.openpi.value_after_vlm": True,
    "actor.optim.critic_warmup_steps": 8, "actor.optim.value_lr": 1.1e-5, "actor.optim.lr": 5.6e-6,
    "env.train.total_num_envs": 48, "env.train.rollout_epoch": 2, "actor.global_batch_size": 384,
    "actor.micro_batch_size": 16, "algorithm.update_epoch": 2, "algorithm.adv_type": "gae",
    "algorithm.loss_type": "actor_critic", "algorithm.normalize_advantages": True, "algorithm.gamma": 1.0,
    "algorithm.gae_lambda": 0.95, "algorithm.reward_type": "chunk_level", "env.train.ignore_terminations": True,
    "env.train.max_episode_steps": 200, "env.train.use_fixed_reset_state_ids": False,
    "env.train.world_model.native.ckpt": "/data/zhengjli/vomni/artifacts/w8-pilot/step-750.safetensors",
    "env.train.world_model.native.seed_offset": seed * 1000, "actor.seed": seed,
    "runner.max_steps": max_steps, "actor.model.openpi.detach_critic_input": True,
}
bad = {k: (v, get(k)) for k, v in want.items() if get(k) != v}
print("W32_CFG", "OK" if not bad else "BAD", bad)
sys.exit(0 if not bad else 1)
