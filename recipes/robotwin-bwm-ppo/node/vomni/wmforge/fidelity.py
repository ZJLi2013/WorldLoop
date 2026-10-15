#!/usr/bin/env python3
"""WMForge G1/G2 (plan r4): pixel fidelity of open-loop BWM rollouts against the simulator frames of
the same commands, reported separately for success and failure episodes.

Per episode and eval dir (bwm_eval.py output, must cover its list.json): PSNR on frames 1-200 per
serving window (1-50, 51-100, 101-150, 151-200) and overall. For failure episodes that share their
initial-state seed with a simulator success (another policy or pass), also the outcome margin
PSNR(gen, own failure) - PSNR(gen, that success) over frames 101-200: positive means the rollout sits
closer to the failure it should draw than to a success from the same start. sim_fail_vs_success_db is
how alike the two simulator videos are; a high value means the margin cannot separate them.

Paired differences against the first eval are averaged per seed first, so repeated episodes of one
initial state count once.

    python fidelity.py --index <w6 index.json> --eval base=<dir> pilot500=<dir> --out <json>
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--eval", nargs="+", required=True, help="name=dir")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = json.load(open(a.index))["episodes"]
by_key = {f"{r['tag']}_s{r['slot']:02d}": r for r in rows}
evals = dict(x.split("=", 1) for x in a.eval)
lists = {n: [f"{r['tag']}_s{r['slot']:02d}" for r in json.load(open(f"{d}/list.json"))] for n, d in evals.items()}
keys = lists[next(iter(evals))]
for n in evals:
    assert lists[n] == keys, f"{n} was run on a different list"


def psnr(x, y):
    mse = np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2, axis=(1, 2, 3))
    return 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-6))


def sim_frames(r):
    return np.load(r["path"].replace(".npz", "_frames.npy"), mmap_mode="r")


succ_by_seed = {}
for r in rows:
    if r["success"]:
        succ_by_seed.setdefault(r["seed"], r)

per = {}
for key in keys:
    r = by_key[key]
    sim = np.asarray(sim_frames(r)[1:201])
    ref = succ_by_seed.get(r["seed"]) if not r["success"] else None
    ref_late = np.asarray(sim_frames(ref)[101:201]) if ref is not None else None
    e = {"success": r["success"], "seed": r["seed"], "policy": r["policy"]}
    if ref_late is not None:
        e["sim_fail_vs_success_db"] = float(psnr(sim[100:], ref_late).mean())
    for name, d in evals.items():
        gen = np.load(f"{d}/{key}.npz")["frames"][1:201]
        p = psnr(gen, sim)
        e[name] = [float(p[i * 50:(i + 1) * 50].mean()) for i in range(4)] + [float(p.mean())]
        if ref_late is not None:
            e[f"{name}/margin"] = float(p[100:].mean() - psnr(gen[100:], ref_late).mean())
    per[key] = e


def seed_mean(vals):
    by = {}
    for s, v in vals:
        by.setdefault(s, []).append(v)
    return np.array([np.mean(v, axis=0) for v in by.values()])


summary = {}
names = list(evals)
for side in ("success", "failure"):
    ks = [k for k in keys if per[k]["success"] == (side == "success")]
    for name in names:
        v = seed_mean([(per[k]["seed"], per[k][name]) for k in ks])
        summary[f"{name}/{side}"] = {"episodes": len(ks), "seeds": len(v), "w1,w2,w3,w4,all": np.round(v.mean(0), 2).tolist()}
        if side == "failure":
            m = [(per[k]["seed"], per[k][f"{name}/margin"]) for k in ks if f"{name}/margin" in per[k]]
            if m:
                mv = seed_mean(m)
                summary[f"{name}/failure_margin"] = {"seeds": len(mv), "mean_db": round(float(mv.mean()), 3),
                                                     "closer_to_failure": int((mv > 0).sum())}
    for other in names[1:]:
        d = seed_mean([(per[k]["seed"], np.subtract(per[k][other], per[k][names[0]])) for k in ks])
        summary[f"{other}-{names[0]}/{side}"] = {
            "seeds": len(d), "mean_db_w1,w2,w3,w4,all": np.round(d.mean(0), 3).tolist(),
            "seeds_better_all": int((d[:, 4] > 0).sum()),
            "late_w3w4_db": round(float(d[:, 2:4].mean()), 3)}
json.dump({"summary": summary, "per_episode": per}, open(a.out, "w"), indent=1)
print(json.dumps(summary, indent=1))
