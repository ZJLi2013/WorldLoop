#!/usr/bin/env python3
"""WMForge w6: replay a rollout dump through the same 50-command chunk path and record BWM episodes.

Input is an E1-format label dump (one episode per slot: a reset file plus four 50-step chunk files
per env worker pid, chunk_actions patched in by _e1_patch_robotwin_dump.py). Each slot is reset to
its seed and its four chunks are executed with env.chunk_step, as in the rollout, with RoboTwin's
early stop off (RT_NO_EARLY_STOP=1). Pass 1 counts physics substeps per chunk; pass 2 repeats the
replay and grabs the head frame at 50 evenly spaced substeps of each chunk (fixed frame rate, the
last one at the chunk end, where the policy's observation is taken). A final chunk holds the last
command twice for frames 201-202. The chunk-end frames are checked against the rollout's dump images.

    REPO=/data/zhengjli/rlinf/wt-g python _w6_record.py --dump <dir> --tag base-p0 --slots 0 1 --out <dir>

Per slot, <out>/<tag>_s<slot>.npz: frames [203,H,W,3] uint8 (frame 0 = reset, frame 50k+i = i-th
sample of chunk k), commands [202,14] (frame t+1 follows commands[t]), pred [203] (success predicate
at each frame), latched [203] (RoboTwin's sticky eval_success), substeps [5], seed, rollout_success,
chunk_end_mae [4] vs the rollout dump.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

os.environ["RT_NO_EARLY_STOP"] = "1"
REPO = os.environ["REPO"]
sys.path.insert(0, REPO)
EVALDIR = os.path.join(REPO, "evaluations", "robotwin")
CHUNK, NCHUNK, HOLD = 50, 4, 2
ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--slots", type=int, nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cfg", default="rocm_adjust_bottle_pi0_smoke")
ap.add_argument("--failures-only", action="store_true", help="skip slots the rollout scored as success")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

by_pid = {}
for f in glob.glob(f"{a.dump}/*.npz"):
    pid, call, kind = re.match(r"(\d+)_(\d+)_(\w+)\.npz", os.path.basename(f)).groups()
    by_pid.setdefault(pid, []).append((int(call), kind, f))
slot_map = []
for pid in sorted(by_pid):
    calls = sorted(by_pid[pid])
    reset = np.load(next(f for c, k, f in calls if k == "reset"), allow_pickle=True)
    chunks = [np.load(f, allow_pickle=True) for c, k, f in calls if k == "chunk"][:NCHUNK]
    assert len(chunks) == NCHUNK, (pid, len(chunks))
    for i in range(len(reset["reset_state_ids"])):
        slot_map.append((reset, chunks, i))
if a.failures_only:
    a.slots = [s for s in a.slots if not any(bool(c["success"][slot_map[s][2]]) for c in slot_map[s][1])]
    if not a.slots:
        print("DONE no failures", flush=True)
        sys.exit(0)
seeds = [int(slot_map[s][0]["reset_state_ids"][slot_map[s][2]]) for s in a.slots]
acts = np.stack([
    np.stack([np.asarray(c["chunk_actions"][slot_map[s][2]], dtype=np.float32) for c in slot_map[s][1]])
    for s in a.slots
])  # [n, 4, 50, 14]
assert acts.shape[1:] == (NCHUNK, CHUNK, 14), acts.shape
dump_imgs = [[np.asarray(c["images"][slot_map[s][2]]) for c in slot_map[s][1]] for s in a.slots]
rollout_success = [bool(any(bool(c["success"][slot_map[s][2]]) for c in slot_map[s][1])) for s in a.slots]
hold = np.repeat(acts[:, -1, -1:], HOLD, axis=1)  # [n, 2, 14]
chunks_in = [acts[:, k] for k in range(NCHUNK)] + [hold]

import concurrent.futures._base as _futures  # noqa: E402

# VectorEnv waits 120 s per sub-env step; rendering 50 frames inside a chunk takes longer.
_result = _futures.Future.result
_futures.Future.result = lambda self, timeout=None: _result(self, None)

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import open_dict  # noqa: E402

from rlinf.envs import get_env_cls  # noqa: E402

with initialize_config_dir(config_dir=EVALDIR, version_base=None):
    cfg = compose(config_name=a.cfg)
env_cfg = cfg.env.eval
n = len(a.slots)
with open_dict(env_cfg):
    env_cfg.total_num_envs = n
    env_cfg.group_size = 1
    env_cfg.use_fixed_reset_state_ids = False
    env_cfg.video_cfg.save_video = False
    env_cfg.max_episode_steps = 400
    env_cfg.task_config.camera.head_camera_type = "Large_D435"
env = get_env_cls(env_cfg.env_type, env_cfg)(
    cfg=env_cfg, num_envs=n, seed_offset=0, total_num_processes=1, worker_info=None, record_metrics=True)
state = {"count": [0] * n, "targets": [dict() for _ in range(n)], "grab": None}


def head_rgb(task):
    task._update_render()
    task.cameras.update_picture()
    return np.asarray(task.cameras.get_rgb()["head_camera"]["rgb"], dtype=np.uint8)


def hook(tasks):
    # Reset rebuilds the scene, so the substep hook goes on after every reset.
    for j, task in enumerate(tasks):
        if getattr(task.scene.step, "w6_hook", False):
            continue
        inner = task.scene.step

        def step(_inner=inner, _j=j, _task=task):
            _inner()
            state["count"][_j] += 1
            slot = state["targets"][_j].get(state["count"][_j])
            if slot is not None and state["grab"] is not None:
                state["grab"][_j][slot] = (head_rgb(_task), bool(_task.check_success()), bool(_task.eval_success))

        step.w6_hook = True
        task.scene.step = step


def run(sample_counts=None):
    obs, _ = env.reset(env_seeds=seeds)
    tasks = [sub.task for sub in env.venv.envs]
    hook(tasks)
    if sample_counts is not None:
        for j in range(n):
            grabbed[j].append((obs["main_images"][j].numpy().astype(np.uint8), bool(tasks[j].check_success()), False))
    counts = []
    for k, chunk in enumerate(chunks_in):
        state["count"] = [0] * n
        if sample_counts is not None:
            L = chunk.shape[1]
            state["targets"] = [{int(round((i + 1) * sample_counts[j][k] / L)): i for i in range(L)} for j in range(n)]
            state["grab"] = [dict() for _ in range(n)]
        env.chunk_step(chunk)
        counts.append(list(state["count"]))
        if sample_counts is not None:
            for j in range(n):
                if sample_counts[j][k] == 0:  # TOPP plans no substeps for a held command: the scene does not advance
                    grabbed[j].extend([grabbed[j][-1]] * chunk.shape[1])
                    continue
                assert len(state["grab"][j]) == chunk.shape[1], (j, k, len(state["grab"][j]), sample_counts[j][k])
                for i in range(chunk.shape[1]):
                    grabbed[j].append(state["grab"][j][i])
        state["targets"] = [dict() for _ in range(n)]
        state["grab"] = None
    return np.asarray(counts).T  # [n, 5]


grabbed = [[] for _ in range(n)]
counts1 = run()
counts2 = run(sample_counts=counts1)
assert (counts1 == counts2).all(), (counts1, counts2)

rows = []
for j, s in enumerate(a.slots):
    seq = grabbed[j]
    frames = np.stack([x[0] for x in seq])
    pred = np.array([x[1] for x in seq])
    latched = np.array([x[2] for x in seq])
    mae = [float(np.abs(frames[CHUNK * (k + 1)].astype(np.int16) - dump_imgs[j][k].astype(np.int16)).mean())
           for k in range(NCHUNK)]
    succ_frames = np.flatnonzero(latched[: CHUNK * NCHUNK + 1])
    cmds = np.concatenate([acts[j].reshape(-1, 14), hold[j]])
    np.savez_compressed(
        os.path.join(a.out, f"{a.tag}_s{s:02d}.npz"),
        frames=frames, commands=cmds, pred=pred, latched=latched, substeps=counts1[j], seed=seeds[j],
        rollout_success=rollout_success[j], chunk_end_mae=np.asarray(mae), tag=a.tag)
    rows.append({"slot": s, "seed": seeds[j], "frames": len(frames),
                 "success": bool(len(succ_frames)), "success_frame": int(succ_frames[0]) if len(succ_frames) else -1,
                 "pred_at_200": bool(pred[CHUNK * NCHUNK]), "rollout_success": rollout_success[j],
                 "substeps": counts1[j].tolist(), "chunk_end_mae": [round(x, 3) for x in mae]})
    print("ROW", json.dumps(rows[-1]), flush=True)
json.dump(rows, open(os.path.join(a.out, f"{a.tag}_{a.slots[0]:02d}-{a.slots[-1]:02d}.json"), "w"), indent=1)
try:
    env.close()
except Exception:
    pass
print("DONE", flush=True)
