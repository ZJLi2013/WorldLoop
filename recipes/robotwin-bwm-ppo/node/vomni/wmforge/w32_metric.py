#!/usr/bin/env python3
"""One-line w32 progress for watch.sh: newest run's per-step readings, evals done, per-run sim Delta.

Per step (from the newest train.log): WM-judged success_once, critic explained variance, approx_kl,
clip_fraction; ALERT on a non-finite reading. Per run (analysis.json): simulator Delta vs SFT and its 90% interval.
"""
import glob
import json
import math
import os
import re

O = "/data/zhengjli/vomni/artifacts/w32"
R = "/data/zhengjli/rlinf/results"
KEYS = {"wm": "success_once", "ev": "critic/explained_variance", "kl": "actor/approx_kl",
        "clip": "actor/clip_fraction"}
PAIR = re.compile(r"(?<![\w/])((?:critic|actor)/[\w/]+|success_once)=([-+0-9.eE]+|nan|inf)")

logs = [p for p in glob.glob(f"{R}/w32-ppo-s*/train.log")]
parts = []
if logs:
    log = max(logs, key=os.path.getmtime)
    run = os.path.basename(os.path.dirname(log)).replace("w32-ppo-", "")
    tables = re.split(r"Global Step:\s*(\d+)/\d+", open(log, errors="ignore").read())
    rows = {}
    for i in range(1, len(tables) - 1, 2):
        body = tables[i + 1]
        if "╰" in body:
            rows[int(tables[i])] = dict((k, float(v)) for k, v in PAIR.findall(body[: body.index("╰")]))
    alert = any(not math.isfinite(v) for r in rows.values() for v in r.values())
    wm = " ".join(f"{rows[n].get(KEYS['wm'], float('nan')):.2f}" for n in sorted(rows))
    last = rows[max(rows)] if rows else {}
    tail = " ".join(f"{k}={last[v]:.3g}" for k, v in KEYS.items() if k != "wm" and v in last)
    parts.append(f"{run} step {max(rows) if rows else 0}/14 wm_success[{wm}] {tail}" + (" ALERT_NONFINITE" if alert else ""))
parts.append(f"evals {len(glob.glob(f'{O}/eval-ppo-*.done'))}/45 diag {len(glob.glob(f'{O}/diag-s*.done'))}")
try:
    a = json.load(open(f"{O}/analysis.json"))
    parts.append(" ".join(f"s{k} {r['delta']:+.3f}[{r['delta_90'][0]:+.3f},{r['delta_90'][1]:+.3f}]"
                          for k, r in a["runs"].items()) + f" verdict={a.get('verdict', 'NA')}")
except (OSError, ValueError, KeyError):
    pass
print(" | ".join(parts))
