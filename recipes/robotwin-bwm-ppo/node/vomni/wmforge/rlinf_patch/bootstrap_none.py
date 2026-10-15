"""WMForge c4: add `algorithm.bootstrap_type: none` to the embodied env worker.

With `ignore_terminations: True` every WM episode ends by truncation at `max_episode_steps`, and the
"standard" bootstrap adds gamma * V(s_T) to the last reward. The episode-level reward is the whole
return, so the target becomes r + V(s_T): at gamma = 1 it has no fixed point and the critic drifts.
`none` keeps the env reward as the full return. Other values behave as before.

    python3 bootstrap_none.py <rlinf tree>
"""
import sys

path = f"{sys.argv[1]}/rlinf/workers/env/env_worker.py"
src = open(path).read()
if 'bootstrap_type == "none"' in src:
    print("already patched", path)
    sys.exit(0)
old = (
    '        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")\n'
    '        if bootstrap_type == "standard":\n'
)
new = (
    '        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")\n'
    '        if bootstrap_type is None or bootstrap_type == "none":\n'
    "            truncated = env_output.truncations[:, -1]\n"
    '            if truncated.any() and not getattr(self, "_bootstrap_none_logged", False):\n'
    "                self._bootstrap_none_logged = True\n"
    "                self.log_info(\n"
    '                    f"bootstrap_type=none: {int(truncated.sum())} truncated envs keep the env reward"\n'
    "                )\n"
    "            return adjusted_rewards\n"
    '        if bootstrap_type == "standard":\n'
)
assert src.count(old) == 1, path
open(path, "w").write(src.replace(old, new))
print("patched", path)
