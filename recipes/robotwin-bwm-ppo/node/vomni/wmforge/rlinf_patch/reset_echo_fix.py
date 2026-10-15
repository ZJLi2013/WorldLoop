#!/usr/bin/env python3
"""Port of RLinf-f22-legacy e8b2ef43 into a tree that lacks it: a restarted world-model episode
echoes the reset pose, not the last command of the episode it replaced. Idempotent.

    python reset_echo_fix.py <tree>/rlinf/envs/sim/world_model/world_model_env.py
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "echoes the reset pose, not the last command" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)
anchor = (
    "        self._reset_metrics(env_idx=None if env_idx is None else target_slots)\n"
    "\n"
    "        # Wrap observation to match libero_env format\n"
)
assert src.count(anchor) == 1, "reset anchor not found"
fix = (
    "        self._reset_metrics(env_idx=None if env_idx is None else target_slots)\n"
    "\n"
    "        # A restarted episode echoes the reset pose, not the last command of the one it replaced.\n"
    "        if env_idx is None:\n"
    "            self._last_policy_action = None\n"
    "        elif self._last_policy_action is not None:\n"
    "            echoed = np.array(self._last_policy_action, dtype=np.float32)\n"
    "            echoed[target_slots] = self._rest_action[: echoed.shape[-1]]\n"
    "            self._last_policy_action = echoed\n"
    "\n"
    "        # Wrap observation to match libero_env format\n"
)
open(path, "w", encoding="utf-8").write(src.replace(anchor, fix))
print("PATCHED", path)
