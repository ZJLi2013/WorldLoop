#!/usr/bin/env python3
"""E1: add each chunk's joint actions (and the joint state) to the RoboTwin label dump.

Edits rlinf/envs/sim/robotwin/robotwin_env.py in place (wt-g only); idempotent.
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "chunk_actions=" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)

anchor_step = "        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(\n            chunk_actions\n        )\n"
assert src.count(anchor_step) == 1, "chunk_step anchor not found"
src = src.replace(
    anchor_step,
    "        self._dump_chunk_actions = np.asarray(chunk_actions, dtype=np.float32).copy()\n" + anchor_step,
)
anchor_dump = "            elapsed_steps=self.elapsed_steps.detach().cpu().numpy(),\n        )\n        self._diagnostic_dump_call = call + 1\n"
assert src.count(anchor_dump) == 1, "dump anchor not found"
src = src.replace(
    anchor_dump,
    "            elapsed_steps=self.elapsed_steps.detach().cpu().numpy(),\n"
    "            chunk_actions=(\n"
    "                getattr(self, \"_dump_chunk_actions\", np.zeros(0, np.float32))\n"
    "                if kind == \"chunk\"\n"
    "                else np.zeros(0, np.float32)\n"
    "            ),\n"
    "            states=(\n"
    "                extracted_obs[\"states\"].detach().cpu().numpy()\n"
    "                if isinstance(extracted_obs.get(\"states\"), torch.Tensor)\n"
    "                else np.zeros(0, np.float32)\n"
    "            ),\n"
    "        )\n"
    "        self._diagnostic_dump_call = call + 1\n",
)
open(path, "w", encoding="utf-8").write(src)
print("PATCHED", path)
