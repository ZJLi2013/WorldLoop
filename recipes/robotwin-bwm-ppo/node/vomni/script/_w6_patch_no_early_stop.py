#!/usr/bin/env python3
"""WMForge w6: let RoboTwin keep simulating after success when RT_NO_EARLY_STOP=1.

Without it, a successful episode stops executing actions at the first success substep and every
later frame is frozen while the commands keep changing. eval_success stays sticky, so reward and
termination are unchanged; only the early returns are gated. Edits envs/_base_task.py in place,
keeps a .orig copy, idempotent; behaviour is unchanged when the variable is unset.
"""
import os
import shutil
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "RT_NO_EARLY_STOP" in src:
    print("ALREADY_PATCHED")
    sys.exit(0)
if not os.path.exists(path + ".orig"):
    shutil.copy2(path, path + ".orig")

GATE = '(os.environ.get("RT_NO_EARLY_STOP") != "1")'
edits = [
    ("        if self.take_action_cnt == self.step_lim or self.eval_success:\n            return\n",
     f"        if self.take_action_cnt == self.step_lim or (self.eval_success and {GATE}):\n            return\n", 1),
    ("            if self.check_success():\n                self.eval_success = True\n                self.get_obs() # update obs\n"
     "                if (self.eval_video_path is not None):\n"
     "                    self.eval_video_ffmpeg.stdin.write(self.now_obs[\"observation\"][\"head_camera\"][\"rgb\"].tobytes())\n"
     "                return\n",
     "            if self.check_success():\n                self.eval_success = True\n                self.get_obs() # update obs\n"
     "                if (self.eval_video_path is not None):\n"
     "                    self.eval_video_ffmpeg.stdin.write(self.now_obs[\"observation\"][\"head_camera\"][\"rgb\"].tobytes())\n"
     f"                if {GATE}:\n                    return\n", 1),
    ("        if getattr(self, \"eval_success\", False):\n            infos[\"success\"] = True\n"
     "            reward = np.array([1], dtype=np.float32)\n            termination = np.array([1], dtype=np.int32)\n"
     "            return reward, termination, truncation, infos\n",
     f"        if getattr(self, \"eval_success\", False) and {GATE}:\n            infos[\"success\"] = True\n"
     "            reward = np.array([1], dtype=np.float32)\n            termination = np.array([1], dtype=np.int32)\n"
     "            return reward, termination, truncation, infos\n", 1),
    ("            if self.check_success():\n                self.eval_success = True\n\n                infos[\"success\"] = True\n"
     "                reward = np.array([1], dtype=np.float32)\n                termination = np.array([1], dtype=np.int32)\n"
     "                return reward, termination, truncation, infos\n",
     "            if self.check_success():\n                self.eval_success = True\n\n                infos[\"success\"] = True\n"
     "                reward = np.array([1], dtype=np.float32)\n                termination = np.array([1], dtype=np.int32)\n"
     f"                if {GATE}:\n                    return reward, termination, truncation, infos\n", 1),
]
for old, new, count in edits:
    assert src.count(old) == count, f"anchor count {src.count(old)} != {count}: {old[:70]!r}"
    src = src.replace(old, new)
open(path, "w", encoding="utf-8").write(src)
print("PATCHED", path)
