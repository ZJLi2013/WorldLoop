#!/usr/bin/env python3
"""WMForge w11: package a native BWM checkpoint (diffsynth names, as bwm_train.py saves) as a served
vllm-omni BWM folder.

The served `models/BWM` was converted from step-12000 by an unrecorded script. The name mapping is
recovered by matching tensor values between that folder and step-12000 itself; it must reproduce the
served transformer and action encoder exactly before it is applied to another checkpoint.

    python to_served.py --ckpt <step-N.safetensors> --out <dir>
"""
import argparse
import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

SERVED = "/data/zhengjli/vomni/models/BWM"
BASE = "/data/zhengjli/hf/hub/models--BLM-Lab--Boundless-World-Model/snapshots/738a8d3c008e637b8b1b18d5e98a82f6de9c04aa/step-12000.safetensors"
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def digest(t):
    return hashlib.sha1(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def split(sd):
    dit = {k: v for k, v in sd.items() if not k.startswith("pipe.action_encoder.")}
    ae = {k[len("pipe.action_encoder."):]: v for k, v in sd.items() if k.startswith("pipe.action_encoder.")}
    return dit, ae


served_tr = load_file(f"{SERVED}/transformer/diffusion_pytorch_model.safetensors")
served_ae = load_file(f"{SERVED}/action_encoder/diffusion_pytorch_model.safetensors")
base = load_file(BASE)
base_dit = {k[len("dit."):] if k.startswith("dit.") else k: v for k, v in base.items() if not k.startswith("pipe.action_encoder.")}
base_ae = {k[len("pipe.action_encoder."):]: v for k, v in base.items() if k.startswith("pipe.action_encoder.")}
print("base step-12000 keys", len(base), "dit", len(base_dit), "ae", len(base_ae), "sample", sorted(base)[:3])

by_digest = {}
for k, v in base_dit.items():
    by_digest.setdefault((tuple(v.shape), str(v.dtype), digest(v)), []).append(k)
mapping, missing, ambiguous = {}, [], []
for k, v in served_tr.items():
    hits = by_digest.get((tuple(v.shape), str(v.dtype), digest(v)), [])
    if len(hits) == 1:
        mapping[k] = hits[0]
    elif not hits:
        missing.append(k)
    else:
        ambiguous.append((k, hits))
dit, ae = split(load_file(a.ckpt))
dit = {k[len("dit."):] if k.startswith("dit.") else k: v for k, v in dit.items()}
# step-12000 lacks 4 text-path tensors that the pipeline fills from the Wan base; training keeps
# them frozen, so they are matched against the checkpoint being converted instead.
fallback = {}
for k, v in dit.items():
    fallback.setdefault((tuple(v.shape), str(v.dtype), digest(v)), []).append(k)
still = []
for k in missing:
    v = served_tr[k]
    hits = fallback.get((tuple(v.shape), str(v.dtype), digest(v)), [])
    if len(hits) == 1:
        mapping[k] = hits[0]
        print("matched frozen tensor", k, "->", hits[0])
    else:
        still.append(k)
# The text embedder is off the text-free AdaLN path (no gradient, frozen in training): keep the
# served values. Anything else unmatched still stops the conversion.
TEXT_PATH = "condition_embedder.text_embedder."
keep_served = [k for k in still if k.startswith(TEXT_PATH)]
missing = [k for k in still if not k.startswith(TEXT_PATH)]
print("kept served text-path tensors:", keep_served)
print("served transformer", len(served_tr), "mapped", len(mapping), "missing", len(missing), "ambiguous", len(ambiguous))
if missing:
    print("MISSING sample", missing[:5], {k: (tuple(served_tr[k].shape), str(served_tr[k].dtype)) for k in missing[:5]})
if ambiguous:
    print("AMBIGUOUS sample", ambiguous[:3])
ae_ok = all(k in base_ae and torch.equal(served_ae[k], base_ae[k]) for k in served_ae)
print("action_encoder names equal and values match step-12000:", ae_ok)
if missing or ambiguous or not ae_ok:
    raise SystemExit("mapping is not a pure rename; stop")

out_tr = {k: dit[src].to(served_tr[k].dtype).contiguous() for k, src in mapping.items()}
out_tr.update({k: served_tr[k] for k in keep_served})
assert set(out_tr) == set(served_tr), "converted transformer does not cover the served names"
out_ae = {k: ae[k].to(served_ae[k].dtype).contiguous() for k in served_ae}
changed = sum(not torch.equal(out_tr[k], served_tr[k]) for k in out_tr)
# Self-check: the same mapping applied to step-12000 must give the served transformer back.
rebuilt = {k: base_dit[src].to(served_tr[k].dtype) for k, src in mapping.items() if src in base_dit}
assert all(torch.equal(rebuilt[k], served_tr[k]) for k in rebuilt), "mapping does not rebuild the served weights"
print("mapping rebuilds", len(rebuilt), "served tensors from step-12000 exactly")
print("converted", os.path.basename(a.ckpt), "transformer tensors differing from step-12000:", changed, "of", len(out_tr))

os.makedirs(f"{a.out}/transformer", exist_ok=True)
os.makedirs(f"{a.out}/action_encoder", exist_ok=True)
shutil.copy(f"{SERVED}/model_index.json", a.out)
shutil.copy(f"{SERVED}/transformer/config.json", f"{a.out}/transformer/")
for name in os.listdir(f"{SERVED}/action_encoder"):
    if name.endswith(".json"):
        shutil.copy(f"{SERVED}/action_encoder/{name}", f"{a.out}/action_encoder/")
if not os.path.exists(f"{a.out}/vae"):
    os.symlink(f"{SERVED}/vae", f"{a.out}/vae")
save_file(out_tr, f"{a.out}/transformer/diffusion_pytorch_model.safetensors")
save_file(out_ae, f"{a.out}/action_encoder/diffusion_pytorch_model.safetensors")
json.dump({"source_ckpt": a.ckpt, "mapping_from": BASE, "tensors_changed_vs_base": changed},
          open(f"{a.out}/wmforge_source.json", "w"), indent=1)
print("WROTE", a.out)
