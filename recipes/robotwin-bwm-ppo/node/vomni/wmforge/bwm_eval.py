#!/usr/bin/env python3
"""WMForge A4 / G1-G2: open-loop BWM rollouts of w6 episodes under the frozen serving contract.

Each episode's reset frame and its bridged commands drive 4 serving windows (bwm_common.rollout).
Per episode, <out>/<key>.npz holds the 201 generated frames and <out>/sheet_<key>.png a label sheet
of generated frames 50/100/150/200, named by a hash so policy and outcome stay hidden while labelling;
blind_map_<shard>.jsonl maps the hash back.

    python bwm_eval.py --list <json list of index rows> --shard 0 --nshard 8 --out <dir> [--ckpt <safetensors>]
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import torch
from PIL import Image

import bwm_common as bc

ap = argparse.ArgumentParser()
ap.add_argument("--list", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=bc.BASE_CKPT)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows = json.load(open(a.list))[a.shard::a.nshard]
meta = {"ckpt": a.ckpt, "seed": a.seed, "list_crc": zlib.crc32(open(a.list, "rb").read()),
        "bwm_common_crc": zlib.crc32(open(bc.__file__, "rb").read())}
meta_path = f"{a.out}/meta_{a.shard}.json"
if os.path.exists(meta_path):
    assert json.load(open(meta_path)) == meta, f"{a.out} holds another run: {json.load(open(meta_path))}"
json.dump(meta, open(meta_path, "w"))
pipe = bc.build_pipe(ckpt=a.ckpt)
pipe.eval()
for r in rows:
    key = f"{r['tag']}_s{r['slot']:02d}"
    if os.path.exists(f"{a.out}/{key}.npz"):
        continue
    z = np.load(r["path"])
    act = np.load(r["path"].replace(".npz", "_act.npy"))
    t0 = time.time()
    gen = bc.rollout(pipe, z["frames"][0], act, seed=a.seed)
    np.savez_compressed(f"{a.out}/{key}.npz", frames=gen)
    tiles = [Image.fromarray(gen[t]).resize((320, 240)) for t in (50, 100, 150, 200)]
    sheet = Image.new("RGB", (1280, 240))
    for i, im in enumerate(tiles):
        sheet.paste(im, (320 * i, 0))
    blind = f"{zlib.crc32(f'{a.ckpt}|{key}'.encode()):08x}"
    sheet.save(f"{a.out}/sheet_{blind}.png")
    with open(f"{a.out}/blind_map_{a.shard}.jsonl", "a") as f:
        f.write(json.dumps({"blind": blind, "key": key}) + "\n")
    print("EP", key, "sec %.0f" % (time.time() - t0), "range", int(gen.min()), int(gen.max()), flush=True)
    torch.cuda.empty_cache()
print("DONE", flush=True)
