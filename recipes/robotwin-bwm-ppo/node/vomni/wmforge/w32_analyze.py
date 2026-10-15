#!/usr/bin/env python3
"""w32: per-run paired readout against the w28 SFT labels and the frozen cross-run verdict.

Per run k: for each eval seed, d = run label - SFT label on the same init; average the 3 d per init, then
bootstrap the 144 inits (20 000) for the 90% interval of Delta_k. Across runs: mean, SD, one-sided t, worst.
Verdict (frozen in w29 / w32): UNSTABLE if any Delta < -0.10 or SD >= 0.10; STABLE_GAIN if every Delta > 0 with
lower bound > 0 and SD < 0.07; otherwise UNDECIDED. With fewer than 5 runs the cross-run verdict is PARTIAL.

    python w32_analyze.py <w32 out dir> <w28 out dir> <seeds_dir> <seed> [<seed> ...]
"""
import json
import sys

import numpy as np

from w28_tools import labels

out, w28, seeds_dir = sys.argv[1:4]
run_seeds = [int(s) for s in sys.argv[4:]]
confirm = [{int(s) for s in json.load(open(f"{seeds_dir}/confirm{i}.json"))["adjust_bottle"]["success_seeds"]} for i in range(3)]
sft = {}
for i in range(3):
    for es in (101, 202, 303):
        for s, ok in labels(f"{w28}/eval-dump-sft-e{es}-confirm{i}").items():
            sft[(s, es)] = ok
runs, problems = {}, []
for k in run_seeds:
    per = {}
    for i in range(3):
        for es in (101, 202, 303):
            lab = labels(f"{out}/eval-dump-ppo-s{k}-e{es}-confirm{i}")
            if set(lab) != confirm[i]:
                problems.append(f"s{k} e{es} confirm{i}: {len(lab)} inits")
            for s, ok in lab.items():
                per.setdefault(s, []).append(float(ok) - float(sft[(s, es)]))
    d = np.array([np.mean(v) for v in per.values()])
    rate = float(np.mean([v for i in range(3) for es in (101, 202, 303)
                          for v in labels(f"{out}/eval-dump-ppo-s{k}-e{es}-confirm{i}").values()]))
    boot = np.sort(np.random.default_rng(k).choice(d, (20000, len(d))).mean(axis=1))
    runs[k] = {"success": round(rate, 4), "delta": round(float(d.mean()), 4),
               "delta_90": [round(float(boot[1000]), 4), round(float(boot[18999]), 4)], "inits": len(per)}
deltas = np.array([r["delta"] for r in runs.values()])
summary = {"runs": runs, "problems": problems, "sft_reference": 0.7731}
if len(deltas) >= 2:
    sd = float(deltas.std(ddof=1))
    t = float(deltas.mean() / (sd / np.sqrt(len(deltas)))) if sd > 0 else None
    summary.update({"mean_delta": round(float(deltas.mean()), 4), "sd": round(sd, 4), "t": t,
                    "below_sft": int((deltas < 0).sum()), "worst": round(float(deltas.min()), 4)})
    if problems:
        verdict = "INVALID"
    elif (deltas < -0.10).any() or sd >= 0.10:
        verdict = "UNSTABLE"
    elif len(deltas) < 5:
        verdict = "PARTIAL"
    elif all(r["delta"] > 0 and r["delta_90"][0] > 0 for r in runs.values()) and sd < 0.07:
        verdict = "STABLE_GAIN"
    else:
        verdict = "UNDECIDED"
    summary["verdict"] = verdict
json.dump(summary, open(f"{out}/analysis.json", "w"), indent=1)
print("W32_ANALYSIS", json.dumps(summary))
