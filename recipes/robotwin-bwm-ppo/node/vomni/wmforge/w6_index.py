#!/usr/bin/env python3
"""WMForge w6: check every recorded episode, write its BWM actions, and index it with a seed split.

For each <tag>_s<slot>.npz under --episodes: frame / command counts, chunk-end match against the
rollout dump, success labels; actions [203,14] (bridged eef_abs, frame 0 from the dump's reset joint
state) go to <tag>_s<slot>_act.npy next to it, the frames uncompressed to <tag>_s<slot>_frames.npy.
Split is by initial-state seed, frozen before any training, so every policy and pass of a seed lands on
one side: crc32(seed) % 8 == 0 is val (checkpoint selection), == 4 is final (admission), else train.

    python w6_index.py --episodes /data/zhengjli/vomni/artifacts/w6/episodes --out .../w6/index.json
"""
import argparse
import glob
import json
import os
import re
import zlib

import numpy as np

from bwm_common import bridge, episode_actions

ART = "/data/zhengjli/vomni/artifacts"
ap = argparse.ArgumentParser()
ap.add_argument("--episodes", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--eval-list", default="", help="also write the held-out evaluation set here")
a = ap.parse_args()


def reset_states(tag):
    by_pid = {}
    for f in glob.glob(f"{ART}/w6-dump-{tag}/*.npz"):
        pid, call, kind = re.match(r"(\d+)_(\d+)_(\w+)\.npz", os.path.basename(f)).groups()
        by_pid.setdefault(pid, []).append((int(call), kind, f))
    out = []
    for pid in sorted(by_pid):
        reset = np.load(next(f for c, k, f in sorted(by_pid[pid]) if k == "reset"), allow_pickle=True)
        out.extend(np.asarray(reset["states"], np.float32))
    return out


def split(seed):
    h = zlib.crc32(str(seed).encode()) % 8
    return "val" if h == 0 else "final" if h == 4 else "train"


br = bridge()
cache, rows = {}, []
for path in sorted(glob.glob(f"{a.episodes}/*_s[0-9][0-9].npz")):
    tag, slot = re.match(r"(.+)_s(\d+)\.npz", os.path.basename(path)).groups()
    slot = int(slot)
    if tag not in cache:
        cache[tag] = reset_states(tag)
    try:
        z = dict(np.load(path))
    except Exception:  # still being written by a recorder
        continue
    latched = z["latched"][:201]
    succ = np.flatnonzero(latched)
    seed = int(z["seed"])
    act = episode_actions(br, cache[tag][slot], z["commands"])
    np.save(path.replace(".npz", "_act.npy"), act)
    fpath = path.replace(".npz", "_frames.npy")
    if not os.path.exists(fpath) or os.path.getmtime(fpath) < os.path.getmtime(path):
        np.save(fpath, z["frames"])  # uncompressed, for mmap reads in training
    policy, rep = tag.rsplit("-", 1)
    rows.append({
        "path": path, "tag": tag, "policy": policy, "pass": rep, "slot": slot, "seed": seed,
        "split": split(seed),
        "success": bool(len(succ)), "success_frame": int(succ[0]) if len(succ) else -1,
        "pred_at_200": bool(z["pred"][200]), "rollout_success": bool(z["rollout_success"]),
        "frames": int(z["frames"].shape[0]), "commands": int(z["commands"].shape[0]),
        "chunk_end_mae": [float(x) for x in z["chunk_end_mae"]], "substeps": [int(x) for x in z["substeps"]],
        "act_clipped_frac": float(np.mean(np.abs(act) >= 0.999)),
    })

ok = [r for r in rows if r["frames"] == 203 and r["commands"] == 202]
agree = sum(r["success"] == r["rollout_success"] for r in rows)
mae = np.array([max(r["chunk_end_mae"]) for r in rows]) if rows else np.zeros(1)
summary = {
    "episodes": len(rows), "shape_ok": len(ok), "label_agree_with_rollout": agree,
    "chunk_end_mae_max": {"median": float(np.median(mae)), "p95": float(np.percentile(mae, 95)), "max": float(mae.max())},
    "by_split": {s: {"success": sum(r["success"] for r in rows if r["split"] == s),
                     "failure": sum(not r["success"] for r in rows if r["split"] == s),
                     "seeds": len({r["seed"] for r in rows if r["split"] == s})} for s in ("train", "val", "final")},
    "by_policy": {p: {"success": sum(r["success"] for r in rows if r["policy"] == p),
                      "failure": sum(not r["success"] for r in rows if r["policy"] == p)}
                  for p in sorted({r["policy"] for r in rows})},
    "success_then_false_at_200": sum(r["success"] and not r["pred_at_200"] for r in rows),
}
json.dump({"summary": summary, "episodes": rows}, open(a.out, "w"), indent=1)
print("SUMMARY", json.dumps(summary, indent=1))
if a.eval_list:
    # Per split: every failure, and a stable third of the successes.
    for s in ("val", "final"):
        pick = [r for r in ok if r["split"] == s
                and (not r["success"] or zlib.crc32(f"{r['tag']}_s{r['slot']:02d}".encode()) % 3 == 0)]
        json.dump(pick, open(a.eval_list.replace(".json", f"_{s}.json"), "w"))
        print("EVAL_LIST", s, len(pick), "failures", sum(not r["success"] for r in pick))
