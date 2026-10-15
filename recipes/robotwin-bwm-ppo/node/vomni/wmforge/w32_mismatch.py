#!/usr/bin/env python3
"""w32 per-run mismatch: P(simulator replay fails | WM judged success), as in w18.

    python w32_mismatch.py <e1 dir> <replay dir> <tag> <out json>
"""
import glob
import json
import sys

e1, rep_dir, tag, out = sys.argv[1:5]
rows = {}
for f in glob.glob(f"{rep_dir}/{tag}_*-*.json"):
    for r in json.load(open(f)):
        rows[r["slot"]] = r
man = json.load(open(f"{e1}/manifest.json"))
assert len(rows) == len(man), (len(rows), len(man))
wm = [m for m in man if m["wm_success"]]
fake = [m for m in wm if not rows[m["slot"]]["success"]]
res = {"episodes": len(man), "wm_success": len(wm), "wm_success_sim_fail": len(fake),
       "mismatch": len(fake) / len(wm) if wm else None,
       "sim_replay_success": sum(bool(r["success"]) for r in rows.values()) / len(rows)}
json.dump(res, open(out, "w"), indent=1)
print("W32_MISMATCH", tag, json.dumps(res))
