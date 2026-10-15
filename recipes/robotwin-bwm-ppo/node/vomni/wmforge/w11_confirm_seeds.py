#!/usr/bin/env python3
"""WMForge w11: freeze the 144 sealed confirmation seeds for the short G5.

Drawn from RoboTwin's adjust_bottle train_seeds (planner-success seeds), excluding every seed seen
by the BWM data (w6 index), the reward model's data (exp22, r2-reward-data) and the RL init states
(hires48). The draw is the 144 smallest crc32(seed) values, so it is reproducible.
"""
import glob
import json
import zlib

import torch

W = "/data/zhengjli/rlinf/wt-w11/rlinf/envs/sim/robotwin/seeds"
A = "/data/zhengjli/vomni/artifacts"
train = [int(s) for s in json.load(open(f"{W}/train_seeds.json"))["adjust_bottle"]["success_seeds"]]
w6 = {int(r["seed"]) for r in json.load(open(f"{A}/w6/index.json"))["episodes"]}
h48 = {int(s) for s in json.load(open("/data/zhengjli/vomni/exp35/exp36_seeds.json"))["adjust_bottle"]["success_seeds"]}
r2 = set()
for f in glob.glob(f"{A}/r2-reward-data/*.pt"):
    meta = torch.load(f, map_location="cpu", weights_only=False)["metadata"]
    print(f.rsplit("/", 1)[-1], "metadata keys", list(meta))
    for key, val in meta.items():
        if "seed" in key or "reset" in key:
            r2 |= {int(x) for x in torch.as_tensor(val).flatten().tolist()}
expected = sum(s["seeds"] for s in json.load(open(f"{A}/r2-reward-data/manifest.json"))["splits"].values())
print("r2 reward-data seeds", len(r2), "manifest says", expected)
assert len(r2) == expected, "reward-data seeds do not match the manifest; refusing to draw blind"
pool = [s for s in train if s not in w6 | h48 | r2]
pick = sorted(pool, key=lambda s: zlib.crc32(str(s).encode()))[:144]
assert len(pick) == 144, len(pool)
out = f"{A}/w11/confirm_seeds.json"
json.dump({"adjust_bottle": {"task_name": "adjust_bottle", "success_seeds": sorted(pick)}}, open(out, "w"), indent=1)
print("pool", len(pool), "picked", len(pick), "->", out, "overlap with excluded", len(set(pick) & (w6 | h48 | r2)))
