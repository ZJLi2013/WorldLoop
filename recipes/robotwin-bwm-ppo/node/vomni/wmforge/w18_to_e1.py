#!/usr/bin/env python3
"""w18: turn an RLINF_W18_DUMP directory (one WM episode per env, four 50-step chunks) into the
E1 label-dump layout _w6_record.py replays, plus a manifest of the WM side per replay slot.

Slots follow _w6_record.py's order: pids sorted, then env index. hires48 reset ids map to simulator
seeds through monitor.json (episode i of init_frames_hires48 is success_seeds[i]). The recorded
reset id is checked against the frames: each episode's first generated frame must be closest to
its own init frame among all of them, and the ids must cover 0..N-1 exactly once.

    python w18_to_e1.py --dump <w18-dump> --out <e1-dir> --monitor <monitor.json> --init <init_frames_hires48>
"""
import argparse
import glob
import json
import os
import re

import numpy as np

CHUNK, NCHUNK = 50, 4
ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--monitor", required=True)
ap.add_argument("--init", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
seeds = json.load(open(a.monitor))["adjust_bottle"]["success_seeds"]


def hwc(x):
    x = np.asarray(x)
    while x.ndim > 3:
        x = x[0]
    return x.transpose(1, 2, 0) if x.shape[0] == 3 else x


def load_init(f):
    x = np.load(f, allow_pickle=True)  # init_frames_*: object array holding {"image": [H, W, 3] uint8, ...}
    if x.dtype == object:
        x = x.reshape(-1)[0]["image"]
    return hwc(x).astype(np.float32)


init = [load_init(f) for f in sorted(glob.glob(f"{a.init}/episode_*.npy"))]


def nearest_init(frame):
    f = frame.astype(np.float32)
    if f.shape != init[0].shape:
        ys = np.linspace(0, f.shape[0] - 1, init[0].shape[0]).astype(int)
        xs = np.linspace(0, f.shape[1] - 1, init[0].shape[1]).astype(int)
        f = f[ys][:, xs]
    return [float(np.abs(f - x).mean()) for x in init]


by_pid = {}
for f in glob.glob(f"{a.dump}/*.npz"):
    pid, call = re.match(r"(\d+)_(\d+)\.npz", os.path.basename(f)).groups()
    by_pid.setdefault(pid, []).append((int(call), f))

manifest = []
for pid in sorted(by_pid):
    calls = [np.load(f, allow_pickle=True) for _, f in sorted(by_pid[pid])]
    if not manifest:
        print("DUMP_FIELDS", {k: calls[0][k].shape for k in calls[0].files})
    assert len(calls) == NCHUNK, (pid, len(calls))
    ids = calls[0]["reset_state_ids"].astype(int)
    for k, c in enumerate(calls):
        assert (c["reset_state_ids"].astype(int) == ids).all(), (pid, k)
        assert (c["elapsed_steps"].astype(int) == CHUNK * (k + 1)).all(), (pid, k, c["elapsed_steps"])
        assert c["actions"].shape == (len(ids), CHUNK, 14), (pid, k, c["actions"].shape)
        assert c["bridged_actions"].shape[:2] == (len(ids), CHUNK), (pid, k, c["bridged_actions"].shape)
    wm_label = calls[-1]["labels"] > 0.5
    np.savez_compressed(
        os.path.join(a.out, f"{pid}_0000_reset.npz"),
        reset_state_ids=np.asarray([seeds[i] for i in ids], dtype=np.int64),
        instructions=calls[0]["instructions"],
    )
    for k, c in enumerate(calls):
        np.savez_compressed(
            os.path.join(a.out, f"{pid}_{k + 1:04d}_chunk.npz"),
            chunk_actions=c["actions"].astype(np.float32),
            images=c["images"][:, -1].transpose(0, 2, 3, 1),  # WM frame at the chunk end, [n, H, W, 3]
            success=wm_label if k == NCHUNK - 1 else np.zeros(len(ids), dtype=bool),
        )
    probs = np.concatenate([c["frame_probs"] for c in calls], axis=1)  # [n, 200], frame t+1 per step t
    for i, hid in enumerate(ids):
        d = nearest_init(calls[0]["images"][i, 0].transpose(1, 2, 0))
        manifest.append({
            "slot": len(manifest), "pid": pid, "env": i, "hires": int(hid), "seed": int(seeds[hid]),
            "wm_success": bool(wm_label[i]), "wm_peak": float(calls[-1]["episode_peak"][i]),
            "wm_peak_frame": int(50 + np.argmax(probs[i, 50:])) + 1,
            "instruction": str(calls[0]["instructions"][i]),
            "worker_seed": int(calls[0]["worker_seed"]), "wm_seed_offset": int(calls[0]["wm_seed_offset"]),
            "seed_everything": str(calls[0]["seed_everything"]),
            "init_mae_own": round(d[hid], 3), "init_nearest": int(np.argmin(d)),
        })
hids = sorted(m["hires"] for m in manifest)
assert hids == list(range(len(manifest))), f"reset ids are not 0..{len(manifest) - 1} once each: {hids}"
wrong = [m["slot"] for m in manifest if m["init_nearest"] != m["hires"]]
assert not wrong, f"first WM frame closest to another init frame in slots {wrong}"
json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
print(f"SLOTS {len(manifest)} wm_success={sum(m['wm_success'] for m in manifest)} "
      f"init_mae_own_max={max(m['init_mae_own'] for m in manifest):.2f}")
