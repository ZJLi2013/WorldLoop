#!/usr/bin/env python3
"""WMForge w8: one line per evaluated pilot checkpoint on val, and the pick under the frozen rule
(success paired diff >= -0.3 dB, failure late-window diff > 0, then max failure diff)."""
import glob
import json
import re

A = "/data/zhengjli/vomni/artifacts"
rows = []
for p in sorted(glob.glob(f"{A}/w8-pilot-val-step*-s0/fidelity.json"), key=lambda s: int(re.search(r"step(\d+)", s).group(1))):
    n = re.search(r"step(\d+)", p).group(1)
    s = json.load(open(p))["summary"]
    f, c = s[f"step{n}-base/failure"], s[f"step{n}-base/success"]
    m = s.get(f"step{n}/failure_margin", {})
    rows.append((int(n), f["mean_db_w1,w2,w3,w4,all"][4], f["late_w3w4_db"], f["seeds_better_all"], f["seeds"],
                 c["mean_db_w1,w2,w3,w4,all"][4], c["late_w3w4_db"], c["seeds_better_all"], c["seeds"], m.get("mean_db")))
    print("step %4d  fail %+.2f dB (late %+.2f, %d/%d)  succ %+.2f dB (late %+.2f, %d/%d)  margin %s" % rows[-1])
ok = [r for r in rows if r[5] >= -0.3 and r[2] > 0]
print("PICK", max(ok, key=lambda r: r[1])[0] if ok else None)
