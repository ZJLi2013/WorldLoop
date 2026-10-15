#!/usr/bin/env python3
"""Train RoboTwinT5CrossAttn on the R2 splits and report the real-test gate."""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from rlinf.models.embodiment.reward.robotwin_reward_model import (  # noqa: E402
    RoboTwinT5CrossAttnRewardModel,
)

RECALL_TARGET = 0.90
GATE_RECALL = 0.85
GATE_EARLY = 0.10


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def batches(data, batch_size, shuffle, rng):
    index = np.arange(len(data["labels"]))
    if shuffle:
        rng.shuffle(index)
    for start in range(0, len(index), batch_size):
        chunk = index[start : start + batch_size]
        yield (
            data["images"][chunk],
            torch.tensor([data["labels"][i] for i in chunk], dtype=torch.float32),
            [data["instructions"][i] for i in chunk],
        )


@torch.no_grad()
def predict(model, data, device, batch_size=128):
    model.eval()
    probs, losses = [], []
    for images, labels, texts in batches(data, batch_size, False, None):
        out = model(images.to(device), labels=labels.to(device), instructions=texts)
        probs.append(out["probabilities"].float().cpu())
        losses.append(out["loss"].item() * len(labels))
    return torch.cat(probs).numpy(), float(sum(losses) / len(data["labels"]))


def auroc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y, p):
    order = np.argsort(-p)
    y = y[order]
    hits = np.cumsum(y)
    precision = hits / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(y.sum(), 1))


def pick_threshold(y, p):
    candidates = np.unique(p)[::-1]
    for t in candidates:
        pred = p >= t
        recall = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        if recall >= RECALL_TARGET:
            return float(t)
    return float(candidates[-1])


def episode_metrics(data, p, threshold):
    episodes = defaultdict(list)
    meta = data["metadata"]
    for i, key in enumerate(meta["episodes"]):
        episodes[key].append((meta["elapsed"][i], data["labels"][i], p[i] >= threshold, meta["seeds"][i]))
    early, total = [], 0
    per_seed = defaultdict(lambda: [0, 0])
    for frames in episodes.values():
        frames.sort()
        sim = next((k for k, f in enumerate(frames) if f[1]), None)
        model = next((k for k, f in enumerate(frames) if f[2]), None)
        is_early = model is not None and (sim is None or model < sim)
        early.append(is_early)
        per_seed[frames[0][3]][0] += int(is_early)
        per_seed[frames[0][3]][1] += 1
        total += 1
    return float(np.mean(early)), per_seed


def chunk_recall(y, pred):
    return float((pred & (y == 1)).sum() / max((y == 1).sum(), 1))


def bootstrap(data, p, threshold, per_seed, rounds=1000, seed=0):
    rng = np.random.default_rng(seed)
    meta = data["metadata"]
    y = np.asarray(data["labels"])
    pred = p >= threshold
    by_seed = defaultdict(list)
    for i, s in enumerate(meta["seeds"]):
        by_seed[s].append(i)
    seeds = list(by_seed)
    recalls, earlies = [], []
    for _ in range(rounds):
        draw = rng.choice(seeds, size=len(seeds), replace=True)
        idx = np.concatenate([by_seed[s] for s in draw])
        recalls.append(chunk_recall(y[idx], pred[idx]))
        e = sum(per_seed[s][0] for s in draw)
        n = sum(per_seed[s][1] for s in draw)
        earlies.append(e / max(n, 1))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"chunk_recall_ci95": q(recalls), "early_positive_ci95": q(earlies)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--t5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cuda")
    data = {s: load(Path(args.data) / f"{s}.pt") for s in ("train", "val", "test")}
    cfg = OmegaConf.create({"t5_model_name": args.t5, "image_size": [3, 224, 224]})
    model = RoboTwinT5CrossAttnRewardModel(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []
    for epoch in range(args.epochs):
        model.train()
        started, train_loss, n = time.time(), 0.0, 0
        for images, labels, texts in batches(data["train"], args.batch_size, True, rng):
            result = model(images.to(device), labels=labels.to(device), instructions=texts)
            optim.zero_grad()
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            train_loss += result["loss"].item() * len(labels)
            n += len(labels)
        _, val_loss = predict(model, data["val"], device)
        history.append({"epoch": epoch, "train_loss": train_loss / n, "val_loss": val_loss,
                        "seconds": time.time() - started})
        print(json.dumps(history[-1]), flush=True)
        if not np.isfinite(val_loss):
            raise SystemExit("non-finite val loss")
        if val_loss < best - 1e-3:
            best, stale = val_loss, 0
            torch.save(model.state_dict(), out / "reward_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(torch.load(out / "reward_model.pt", map_location=device))
    val_p, _ = predict(model, data["val"], device)
    threshold = pick_threshold(np.asarray(data["val"]["labels"]), val_p)
    test_p, test_loss = predict(model, data["test"], device)
    y = np.asarray(data["test"]["labels"])
    pred = test_p >= threshold
    early, per_seed = episode_metrics(data["test"], test_p, threshold)
    report = {
        "threshold": threshold,
        "threshold_rule": f"highest val threshold with chunk recall >= {RECALL_TARGET}",
        "history": history,
        "test": {
            "loss": test_loss,
            "chunk_recall": chunk_recall(y, pred),
            "chunk_precision": float((pred & (y == 1)).sum() / max(pred.sum(), 1)),
            "episode_early_positive_rate": early,
            "auroc": auroc(y, test_p),
            "average_precision": average_precision(y, test_p),
            "frames": int(len(y)),
            "positives": int(y.sum()),
            **bootstrap(data["test"], test_p, threshold, per_seed),
        },
        "checkpoint_sha256": hashlib.sha256((out / "reward_model.pt").read_bytes()).hexdigest(),
        "t5": args.t5,
    }
    report["real_test_gate"] = {
        "chunk_recall>=0.85": report["test"]["chunk_recall"] >= GATE_RECALL,
        "early_positive<=0.10": early <= GATE_EARLY,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.save(out / "test_probs.npy", test_p)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
