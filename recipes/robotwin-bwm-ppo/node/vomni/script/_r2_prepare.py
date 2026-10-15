#!/usr/bin/env python3
"""npz label dumps -> {images, labels, instructions, ...} splits by reset_state_id."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

KNOWN_RATE = {"base": 0.646, "orig5": 0.632, "ignore5": 0.556, "hist10": 0.444}


def arm_of(text):
    text = text.lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return "unspecified"


def load_policy(tag, root):
    rows = []
    episodes = defaultdict(list)
    for path in sorted(root.glob("*_chunk.npz")):
        pid, call = path.stem.split("_")[:2]
        data = np.load(path, allow_pickle=True)
        for env in range(len(data["success"])):
            elapsed = int(data["elapsed_steps"][env])
            seed = int(data["reset_state_ids"][env])
            key = (tag, pid, env, seed)
            rows.append(
                {
                    "policy": tag,
                    "pid": pid,
                    "env": env,
                    "call": int(call),
                    "seed": seed,
                    "elapsed": elapsed,
                    "label": int(bool(data["success"][env])),
                    "instruction": str(data["instructions"][env]),
                    "image": data["images"][env],
                    "episode": key,
                }
            )
            episodes[key].append(rows[-1])
    return rows, episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", action="append", nargs=2, metavar=("TAG", "DIR"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--rate-tolerance", type=float, default=0.15)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, report, failures = [], {"policies": {}}, []
    for tag, directory in args.dump:
        policy_rows, episodes = load_policy(tag, Path(directory))
        bad = [k for k, v in episodes.items() if len(v) != 4]
        rate = float(np.mean([any(r["label"] for r in v) for v in episodes.values()]))
        report["policies"][tag] = {
            "frames": len(policy_rows),
            "episodes": len(episodes),
            "episodes_without_4_chunks": len(bad),
            "success_once_rate": rate,
            "known_real_rate": KNOWN_RATE.get(tag),
            "image_shape": list(policy_rows[0]["image"].shape) if policy_rows else None,
        }
        if not policy_rows:
            failures.append(f"{tag}: no frames")
        if bad:
            failures.append(f"{tag}: {len(bad)} episodes without 4 chunk frames")
        known = KNOWN_RATE.get(tag)
        if known is not None and abs(rate - known) > args.rate_tolerance:
            failures.append(f"{tag}: success_once {rate:.3f} vs known {known:.3f}")
        rows.extend(policy_rows)

    empty = [r for r in rows if not r["instruction"].strip()]
    if empty:
        failures.append(f"{len(empty)} frames without instruction")
    arms = Counter(arm_of(r["instruction"]) for r in rows)
    report["arm_frames"] = dict(arms)
    if min(arms.get(a, 0) for a in ("left", "right", "unspecified")) == 0:
        failures.append(f"arm coverage incomplete: {dict(arms)}")

    seeds = sorted({r["seed"] for r in rows})
    rng = np.random.default_rng(args.split_seed)
    order = rng.permutation(len(seeds))
    n_train, n_val = int(0.7 * len(seeds)), int(0.15 * len(seeds))
    split_of = {}
    for rank, index in enumerate(order):
        split_of[seeds[index]] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"

    report["splits"] = {}
    for split in ("train", "val", "test"):
        part = [r for r in rows if split_of[r["seed"]] == split]
        payload = {
            "images": torch.from_numpy(np.stack([r["image"] for r in part])),
            "labels": [r["label"] for r in part],
            "instructions": [r["instruction"] for r in part],
            "metadata": {
                "seeds": [r["seed"] for r in part],
                "policies": [r["policy"] for r in part],
                "episodes": ["|".join(map(str, r["episode"])) for r in part],
                "elapsed": [r["elapsed"] for r in part],
            },
        }
        torch.save(payload, out / f"{split}.pt")
        report["splits"][split] = {
            "frames": len(part),
            "positives": int(sum(payload["labels"])),
            "seeds": len({r["seed"] for r in part}),
        }
    overlap = set.intersection(
        *[{r["seed"] for r in rows if split_of[r["seed"]] == s} for s in ("train", "val", "test")]
    )
    report["seed_overlap_across_splits"] = len(overlap)
    report["failures"] = failures
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures or overlap:
        raise SystemExit("R2a checkpoint failed")


if __name__ == "__main__":
    main()
