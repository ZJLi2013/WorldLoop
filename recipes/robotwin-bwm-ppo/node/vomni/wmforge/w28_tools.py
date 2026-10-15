#!/usr/bin/env python3
"""w28: SFT baseline on srdc-20. Reads the per-slot eval dumps written by 03_eval.sh
(RLINF_ROBOTWIN_LABEL_DUMP: <pid>_<call>_reset|chunk.npz; the last chunk's success_once is the episode label).

    python w28_tools.py repeat  <dumpA> <dumpB>
    python w28_tools.py sample  <dump> <confirm_seeds.json> <seed> <out_dir>
    python w28_tools.py analyze <out_dir> <seeds_dir>
"""
import glob
import json
import os
import re
import sys

import numpy as np


def labels(dump):
    calls = {}
    for f in glob.glob(f"{dump}/*.npz"):
        pid, call, kind = re.match(r"(\d+)_(\d+)_(\w+)\.npz", os.path.basename(f)).groups()
        calls.setdefault(pid, []).append((int(call), kind, f))
    out = {}
    for pid, items in calls.items():
        items.sort()
        reset = np.load(next(f for c, k, f in items if k == "reset"), allow_pickle=True)
        last = np.load([f for c, k, f in items if k == "chunk"][-1], allow_pickle=True)
        for seed, ok in zip(reset["reset_state_ids"], last["success_once"]):
            assert int(seed) not in out, f"seed {seed} twice in {dump}"
            out[int(seed)] = bool(ok)
    return out


def repeat(a, b):
    la, lb = labels(a), labels(b)
    same_seeds = set(la) == set(lb)
    agree = sum(la[s] == lb[s] for s in la if s in lb)
    ok = same_seeds and len(la) == 48 and agree == 48
    print("W28_REPEAT", "PASS" if ok else "FAIL", f"seeds={len(la)}/{len(lb)} same={same_seeds} agree={agree}",
          f"rate={np.mean(list(la.values())):.3f}/{np.mean(list(lb.values())):.3f}")
    return ok


def sample(dump, confirm_json, seed, out):
    confirm = {int(s) for s in json.load(open(confirm_json))["adjust_bottle"]["success_seeds"]}
    for f in sorted(glob.glob(f"{dump}/*_000_reset.npz")):
        r = np.load(f, allow_pickle=True)
        ids = [int(x) for x in r["reset_state_ids"]]
        if seed in ids:
            break
    else:
        print("W28_SAMPLE FAIL seed not in dump")
        return False
    s = ids.index(seed)
    pid = os.path.basename(f).split("_")[0]
    chunks = [np.load(c, allow_pickle=True) for c in sorted(glob.glob(f"{dump}/{pid}_*_chunk.npz"))][:4]
    img = r["images"][s]
    states = [r["states"][s]] + [c["states"][s] for c in chunks]
    acts = [c["chunk_actions"][s].shape for c in chunks]
    succ = [bool(c["success"][s]) for c in chunks]
    checks = {
        "seed_in_confirm": seed in confirm,
        "instruction": str(r["instructions"][s]),
        "image_480x640": tuple(img.shape) == (480, 640, 3) and int(img.max()) > int(img.min()),
        "proprio_14_moves": len(states[0]) == 14 and float(np.abs(np.diff(np.array(states), axis=0)).max()) > 1e-3,
        "actions_50x14": all(a == (50, 14) for a in acts) and len(acts) == 4,
        "success_per_chunk": succ,
    }
    ok = (checks["seed_in_confirm"] and bool(checks["instruction"]) and checks["image_480x640"]
          and checks["proprio_14_moves"] and checks["actions_50x14"])
    os.makedirs(out, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img).save(f"{out}/sample_seed{seed}.png")
    except ImportError:
        np.save(f"{out}/sample_seed{seed}_frame0.npy", img)
    checks["proprio_reset"] = [round(float(x), 3) for x in states[0]]
    checks["proprio_chunk1"] = [round(float(x), 3) for x in states[1]]
    json.dump(checks, open(f"{out}/sample_seed{seed}.json", "w"), indent=1)
    print("W28_SAMPLE", "PASS" if ok else "FAIL", json.dumps(checks))
    return ok


def analyze(out, seeds_dir):
    per = {}
    problems = []
    for i in range(3):
        want = {int(s) for s in json.load(open(f"{seeds_dir}/confirm{i}.json"))["adjust_bottle"]["success_seeds"]}
        for es in (101, 202, 303):
            tag = f"sft-e{es}-confirm{i}"
            lab = labels(f"{out}/eval-dump-{tag}")
            if set(lab) != want:
                problems.append(f"{tag}: {len(lab)} seeds, expected {len(want)}")
            for s, ok in lab.items():
                per.setdefault(s, []).append(ok)
    if len(per) != 144 or any(len(v) != 3 for v in per.values()):
        problems.append(f"inits={len(per)} counts={sorted({len(v) for v in per.values()})}")
    means = np.array([np.mean(v) for v in per.values()])
    rng = np.random.default_rng(0)
    boot = np.sort(rng.choice(means, (20000, len(means))).mean(axis=1))
    res = {"success": round(float(means.mean()), 4), "init_bootstrap_90": [round(float(boot[1000]), 4),
           round(float(boot[18999]), 4)], "inits": len(per), "problems": problems,
           "banff_reference": 0.766}
    json.dump(res, open(f"{out}/analysis.json", "w"), indent=1)
    print("W28_RESULT", "OK" if not problems else "INVALID", json.dumps(res))
    return not problems


if __name__ == "__main__":
    cmd = sys.argv[1]
    ok = {"repeat": lambda: repeat(*sys.argv[2:4]),
          "sample": lambda: sample(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]),
          "analyze": lambda: analyze(*sys.argv[2:4])}[cmd]()
    sys.exit(0 if ok else 1)
