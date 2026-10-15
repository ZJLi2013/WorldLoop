#!/usr/bin/env python3
"""w32 smoke: pair the env-0 observation dump (RLINF_WM_DUMP, one env worker) with the RLINF_W18_DUMP chunks
of the same episode, and confirm what the policy reads in the WM: the generated frame, the instruction, and a
proprio that echoes its own previous command (the rest action before the first chunk).

    python w32_smoke_inputs.py <wm_dump dir> <w18_dump dir> <out dir>
"""
import glob
import json
import os
import re
import sys

import numpy as np

wm, w18, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
obs = {int(re.search(r"obs_step(\d+)", f).group(1)): np.load(f, allow_pickle=True)
       for f in glob.glob(f"{wm}/obs_step*.npz")}
pids = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{w18}/*.npz")})
calls = sorted(glob.glob(f"{w18}/{pids[0]}_*.npz")) if len(pids) == 1 else []
checks = {"obs_steps": sorted(obs), "w18_workers": len(pids), "w18_calls": len(calls)}
ok = len(pids) == 1 and 0 in obs and len(calls) >= 4
if ok:
    first = [np.load(c, allow_pickle=True) for c in calls[:4]]
    checks["instruction_obs"] = str(obs[0]["task"])
    checks["instruction_w18"] = str(first[0]["instructions"][0])
    checks["reset_state_proprio"] = np.round(obs[0]["state"], 4).tolist()
    echo = []
    for k, z in enumerate(first, start=1):
        step = 50 * k
        if step in obs and z["actions"].size:
            last_cmd = np.asarray(z["actions"][0]).reshape(-1, 14)[-1]
            echo.append(float(np.abs(obs[step]["state"][:14] - last_cmd).max()))
    checks["proprio_vs_previous_command_maxdiff"] = echo
    checks["image_shape"] = list(obs[0]["image"].shape)
    checks["t5_episode_peak_env0"] = float(first[-1]["episode_peak"][0])
    checks["wm_label_env0"] = float(first[-1]["labels"][0])
    ok = (checks["instruction_obs"] == checks["instruction_w18"] and len(echo) >= 3
          and max(echo) < 1e-3 and len(checks["image_shape"]) == 3)
    try:
        from PIL import Image
        for step in sorted(obs)[:5]:
            Image.fromarray(np.asarray(obs[step]["image"]).astype(np.uint8)).save(f"{out}/policy_input_step{step:04d}.png")
    except Exception as e:  # images are evidence, not the gate
        checks["png_error"] = repr(e)
json.dump(checks, open(f"{out}/smoke_inputs.json", "w"), indent=1)
print("W32_SMOKE_INPUTS", "OK" if ok else "BAD", json.dumps(checks))
sys.exit(0 if ok else 1)
