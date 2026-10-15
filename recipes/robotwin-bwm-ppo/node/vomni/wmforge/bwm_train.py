#!/usr/bin/env python3
"""WMForge pilot: fine-tune BWM step-12000 on w6 success + failure episodes (train split only).

Data parallel over torchrun ranks with gradient all-reduce; fp32 master weights and AdamW states,
bf16 forward/backward (bf16 weights cannot absorb 1e-5 updates). Each micro-batch is one 61-frame
window drawn as:
  50% success episodes, any window;
  30% failure episodes, a window whose future reaches past the failure onset - 25 frames;
  20% failure episodes, any window.
Failure onset (frozen before training): the first frame whose commanded gripper on either arm has
closed by more than 20% of its reset opening. A window starts at frame 1 (first serving window)
with probability 1/4, else at a uniform start in [9, 151].

    torchrun --nproc_per_node 8 bwm_train.py --index <w6 index.json> --out <dir> --steps 1000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

import bwm_common as bc

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=bc.BASE_CKPT)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--save-every", type=int, default=250)
ap.add_argument("--val-every", type=int, default=125)
ap.add_argument("--success-only", action="store_true", help="ablation: draw every window from success episodes")
a = ap.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
os.makedirs(a.out, exist_ok=True)


def log(*x):
    # docker exec block-buffers a redirected stdout; train.log is the progress record.
    if rank == 0:
        print(*x, flush=True)
        with open(f"{a.out}/train.log", "a") as f:
            f.write(" ".join(str(v) for v in x) + "\n")


rows = json.load(open(a.index))["episodes"]
train = [r for r in rows if r["split"] == "train" and r["frames"] == 203]
held = [r for r in rows if r["split"] == "val" and r["frames"] == 203]


def onset(r):
    g = np.load(r["path"])["commands"][:200][:, [6, 13]]
    g0 = np.maximum(g[0], 1e-3)
    closed = np.flatnonzero(((g0 - g) / g0 > 0.2).any(axis=1))
    return int(closed[0]) + 1 if len(closed) else 200


succ = [r for r in train if r["success"]]
fail = [r for r in train if not r["success"]]
for r in fail:
    r["onset"] = onset(r)
log("TRAIN success", len(succ), "failure", len(fail), "HELDOUT", len(held),
    "onset median", float(np.median([r["onset"] for r in fail])) if fail else None)
assert succ and fail, "need both outcomes in the train split"

def load(r):
    return (np.load(r["path"].replace(".npz", "_frames.npy"), mmap_mode="r"),
            np.load(r["path"].replace(".npz", "_act.npy")))


def pick(rng):
    u = 0.0 if a.success_only else rng.random()
    if u < 0.5:
        r, lo = succ[rng.integers(len(succ))], 1
    elif u < 0.8:
        r = fail[rng.integers(len(fail))]
        lo = max(1, r["onset"] - 25 - bc.FUT + 1)
    else:
        r, lo = fail[rng.integers(len(fail))], 1
    if lo <= 1 and rng.random() < 0.25:
        start = 1
    else:
        start = int(rng.integers(max(lo, bc.HIST), 152)) if max(lo, bc.HIST) < 152 else 151
    return r, start


def draw(rng):
    r, start = pick(rng)
    frames, act = load(r)
    return bc.window(frames, act, start)


srng = np.random.default_rng(0)
sample = [pick(srng) for _ in range(20000)]
log("SAMPLED first_window %.3f failure_episode %.3f post_onset %.3f" % (
    np.mean([s == 1 for _, s in sample]), np.mean([not r["success"] for r, _ in sample]),
    np.mean([not r["success"] and s + bc.CHUNK - 1 >= r["onset"] for r, s in sample])))


pipe = bc.build_pipe(ckpt=a.ckpt, device=f"cuda:{torch.cuda.current_device()}")
pipe.scheduler.set_timesteps(1000, training=True)
pipe.use_gradient_checkpointing = True
params = list(pipe.dit.parameters()) + list(pipe.action_encoder.parameters())
for p in params:
    p.requires_grad_(True)
# One dry backward finds the parameters the text-free AdaLN path never touches; they stay frozen
# (a zero gradient would still let AdamW's weight decay move them).
pipe.dit.train()
pipe.action_encoder.train()
v, act = draw(np.random.default_rng(5))
bc.flow_loss(pipe, bc.prepare(pipe, v, act, seed=5), 500, seed=5).backward()
used = torch.tensor([p.grad is not None for p in params], device="cuda", dtype=torch.int32)
dist.all_reduce(used, op=dist.ReduceOp.MIN)
frozen = [p for p, u in zip(params, used.tolist()) if not u]
for p in params:
    p.grad = None
for p in frozen:
    p.requires_grad_(False)
params = [p for p, u in zip(params, used.tolist()) if u]
log("TRAINABLE tensors", len(params), "frozen (no gradient)", len(frozen))
master = [p.detach().float().clone().requires_grad_(False) for p in params]
opt = torch.optim.AdamW(master, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.999))
buckets, cur, size = [], [], 0
for i, m in enumerate(master):
    cur.append(i)
    size += m.numel()
    if size >= 256 << 20:  # ~1 GiB of fp32 per all-reduce
        buckets.append(cur)
        cur, size = [], 0
if cur:
    buckets.append(cur)
pipe.dit.train()
pipe.action_encoder.train()

vrng = np.random.default_rng(1234)
val_rows = [r for r in held if not r["success"]][:16] + [r for r in held if r["success"]][:16]
assert any(not r["success"] for r in val_rows) and any(r["success"] for r in val_rows), "val needs both outcomes"
val_wins = [(r["success"], bc.window(*load(r), start=int(vrng.choice(bc.SERVE_STARTS)))) for r in val_rows]


def validate(step):
    pipe.dit.eval()
    pipe.action_encoder.eval()
    mine = val_wins[rank::world]
    out = torch.zeros(2, 2, device="cuda")  # [success?][sum, count]
    with torch.no_grad():
        for k, (ok, (v, act)) in enumerate(mine):
            shared = bc.prepare(pipe, v, act, seed=7 + k)
            l = np.mean([float(bc.flow_loss(pipe, shared, tid, seed=99 + tid)) for tid in (200, 500, 800)])
            out[int(ok), 0] += l
            out[int(ok), 1] += 1
    dist.all_reduce(out)
    pipe.dit.train()
    pipe.action_encoder.train()
    m = (out[:, 0] / out[:, 1].clamp(min=1)).tolist()
    log(f"VAL step {step} failure {m[0]:.4f} success {m[1]:.4f}")
    return m


rng = np.random.default_rng(1000 + rank)
hist = {"val": [], "loss": []}
hist["val"].append([0] + validate(0))
t_start = time.time()
for step in range(1, a.steps + 1):
    lr = a.lr * min(1.0, step / a.warmup)
    for g in opt.param_groups:
        g["lr"] = lr
    tot = 0.0
    for micro in range(a.accum):
        v, act = draw(rng)
        shared = bc.prepare(pipe, v, act, seed=int(rng.integers(1 << 30)))
        loss = bc.flow_loss(pipe, shared, int(rng.integers(0, 1000)), seed=int(rng.integers(1 << 30))) / a.accum
        loss.backward()
        tot += float(loss)
    for bucket in buckets:
        flat = torch.cat([params[i].grad.float().view(-1) for i in bucket])
        dist.all_reduce(flat)
        flat /= world
        off = 0
        for i in bucket:
            n = master[i].numel()
            master[i].grad = flat[off:off + n].view_as(master[i])
            params[i].grad = None
            off += n
    torch.nn.utils.clip_grad_norm_(master, 0.5)
    opt.step()
    opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for p, m in zip(params, master):
            p.copy_(m)
    if step in (1, a.steps):
        probe = torch.stack([master[i].double().sum() for i in (0, len(master) // 2, len(master) - 1)])
        lo, hi = probe.clone(), probe.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        assert torch.equal(lo, hi), f"ranks diverged at step {step}: {lo.tolist()} vs {hi.tolist()}"
        assert all(torch.isfinite(m).all() for m in master[:: max(1, len(master) // 16)]), "non-finite weights"
        log(f"RANK_CHECK step {step} ok")
    lt = torch.tensor([tot], device="cuda")
    dist.all_reduce(lt)
    hist["loss"].append(float(lt) / world)
    if step % 10 == 0:
        el = time.time() - t_start
        log(f"STEP {step} loss {hist['loss'][-1]:.4f} lr {lr:.2e} {el / step:.1f}s/step "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.0f}GiB eta {(a.steps - step) * el / step / 3600:.1f}h")
    if step % a.val_every == 0:
        hist["val"].append([step] + validate(step))
    if step % a.save_every == 0 or step == a.steps:
        if rank == 0:
            bc.save_trainable(pipe, f"{a.out}/step-{step}.safetensors")
            json.dump(hist, open(f"{a.out}/hist.json", "w"))
            log("SAVED", step)
        dist.barrier()
log("DONE")
dist.destroy_process_group()
