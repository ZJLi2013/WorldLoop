# RoboTwin · BWM · PPO (w32)

**This directory rebuilds the code, environment and inputs of the Stage-2 result:** π0.5 SFT trained with PPO
inside a failure-fine-tuned Boundless World Model (BWM) on RoboTwin `adjust_bottle`, rewarded by a frozen T5
success classifier, beat SFT on 144 held-out simulator inits in 5 / 5 training seeds (mean +9.7 pp, sd 4.0).

| Verified | How |
|---|---|
| The four RLinf trees, RoboTwin and BWM code, and every script come out byte-identical to the w32 node | `setup/materialize.sh` on a scratch root, then a file-by-file diff against the live node layout (2026-10-15); patched RLinf trees also match by `git write-tree` |
| Model and checkpoint hashes | `assets.sha256` was taken on the node; `setup/verify.sh` checks a new machine against it |
| The pipeline runs end to end from this directory | 2026-10-15 on the original collection node, in a fresh root: `materialize.sh`, `verify.sh`, SFT baseline, `w32_driver.sh` pre / smoke / full for SEED=1, all exit 0 (venv and model files linked to the node's copies) |

| Not verified | Why it matters |
|---|---|
| `setup/env.sh` and `setup/fetch_assets.py` on a fresh machine | Written from the node's freeze and the documented manual steps; never executed end to end |
| The w32 gain on another node | That one SEED=1 run gave +0.7 pp [−1.4, +3.0] against its own SFT, below w32's five-seed range [+3.9, +14.4]; node, training randomness or both are not separated |

## Pipeline

All paths below are under `$WL_ROOT`, which plays the role of the node's `/data/zhengjli`.

| Stage | Entry | Tree | Output | Cost (8 GPUs) | Shortcut |
|---|---|---|---|---|---|
| 1 · Data collection (w6) | `vomni/script/_w6_collect.sh` (rollout via `rlinf/script/robotwin/03_eval.sh`, replay via `_w6_record.py`) | `wt-g` | `vomni/artifacts/w6/episodes/` (677 episodes) | hours | HF dataset, see below |
| 2 · Reward (T5 classifier) | `vomni/script/_r2_chain.sh` (`_r2_collect.sh` → `_r2_prepare.py` → `_r2_train.py`) | `wt-r2` | `vomni/artifacts/r2-reward-model/reward_model.pt` | 4 collections + 1-GPU training | HF model |
| 3 · WM fine-tune | `vomni/wmforge/bwm_train.py` (launcher `pilot_run.sh`), pick with `eval_run.sh` + `fidelity.py` | none (native BWM) | `vomni/artifacts/w8-pilot/step-750.safetensors` | 750 steps, 7 GPUs, ~5 h | HF model |
| 4 · SFT baseline (w28) | `vomni/wmforge/w28_driver.sh` | `wt-g` | `vomni/artifacts/w28/eval-dump-sft-*` | ~1 h | none: redo on every new node |
| 5 · PPO in the WM, critic included (w32) | `vomni/wmforge/w32_driver.sh` → `vomni/script/_w11_train.sh` | `wt-w11` (smoke and diagnostic: `wt-w18`) | `rlinf/results/w32-ppo-s<k>/` | ~7.1 h per seed | none |
| 6 · Simulator eval + analysis | inside `w32_driver.sh`: `03_eval.sh` × 9, `w32_analyze.py` | `wt-g` | `vomni/artifacts/w32/analysis.json` | ~1.5 h per seed | none |

The recipe itself (window contract, reward reading, PPO settings, seeds, eval protocol) is spelled out in the
driver headers and in `vomni/wmforge/rlinf_patch/`; the PPO deltas against RLinf defaults are
`value_after_vlm=True`, `critic_warmup_steps=8`, `bootstrap_type=none`, `value_lr=1.1e-5`, `gamma=1.0`.

## Setup on a new machine

Needs an 8-GPU ROCm 6.4 host with Docker, host `python3`, network access to Hugging Face, PyPI, the PyTorch ROCm
index and `gs://big_vision` (openpi fetches the PaliGemma tokenizer at run time; set `OPENPI_DATA_HOME` to a
pre-filled cache to run offline). Roughly 120 GB of disk before any run outputs, about 130 GB more with the
w6 episodes; each saved PPO checkpoint is ~19 GB.

```bash
git clone --recurse-submodules <WorldLoop> && cd WorldLoop/recipes/robotwin-bwm-ppo
export WL_ROOT=/data/$USER                       # containers mount the first path component (/data)

bash setup/materialize.sh                        # RLinf trees + RoboTwin + BWM + scripts, paths rewritten
bash $WL_ROOT/worldloop/setup/env.sh             # image, vkdev container, venv to the w32 freeze, side packages
docker exec vkdev $WL_ROOT/rlinf/venvs/openpi-robotwin/bin/python \
  $WL_ROOT/worldloop/setup/fetch_assets.py $WL_ROOT [--w6] [--r2-data]   # --r2-data needs HF_TOKEN
docker exec vkdev bash -lc "source $WL_ROOT/rlinf/venvs/openpi-robotwin/bin/activate && \
  bash $WL_ROOT/vomni/robotwin_rocm/setup_assets.sh $WL_ROOT/vomni/RoboTwin"   # 30 GB + mesh decimation
bash $WL_ROOT/worldloop/setup/verify.sh
```

## Run

```bash
cd $WL_ROOT/vomni/wmforge
bash w28_driver.sh > $WL_ROOT/vomni/artifacts/w28-run.log 2>&1          # SFT labels on this node
STAGE=pre   bash w32_driver.sh
STAGE=smoke bash w32_driver.sh          # then: touch $WL_ROOT/vomni/artifacts/w32/smoke.passed
STAGE=full  SEED_LIST=1 bash w32_driver.sh
```

Stage 3 from the HF episodes skips re-indexing (`w6_index.py` needs the raw rollout dumps, which were not
kept); `vomni/artifacts/w6/{index.json,eval_list_val.json,eval_list_final.json}` ship here instead:

```bash
cd $WL_ROOT/vomni/wmforge && OUT=$WL_ROOT/vomni/artifacts/w8-pilot && mkdir -p $OUT
GPU=0,1,2,3,4,5,6 bash gpu_run.sh -m torch.distributed.run --nproc_per_node 7 --master_port 29612 \
  bwm_train.py --index $WL_ROOT/vomni/artifacts/w6/index.json --out $OUT --steps 1000 --accum 9 \
  --save-every 250 --val-every 125
```

## Layout

| Path | What |
|---|---|
| `../../third_party/RLinf` | upstream RLinf at `bde6c918` (#1556) |
| `../../third_party/RoboTwin` | RoboTwin `RLinf_support` at `0008ae6` |
| `../../third_party/boundless-world-model` | BWM at `cc0bc74` |
| `patches/rlinf-wt-g.patch` | collection, recording, simulator eval tree |
| `patches/rlinf-wt-w11.patch` | PPO-in-WM tree: native BWM backend, episode-peak T5 reward, reset / seed / bootstrap fixes |
| `patches/rlinf-wt-r2.patch` | T5 classifier model, per-episode label dump for reward data |
| `patches/rlinf-wt-f24.patch` | sim-PPO tree; only needed to retrain the F21 policies that produced part of w6 |
| `patches/robotwin-0008ae6.patch` | `RT_NO_EARLY_STOP` gating and end-pose observation |
| `node/` | every script and small input in its node location (`vomni/script`, `vomni/wmforge`, `rlinf/script/robotwin`, lavapipe shim, seeds, 48 RL init frames, w6 index) |
| `env/` | `pip freeze` of the venv, BWM side packages |
| `assets.sha256` | node hashes of every model file the run loads |

Scripts are kept as they ran, `/data/zhengjli` paths included; `materialize.sh` rewrites them. The local RLinf
branches the patches come from are `wmforge/wt-g-exact`, `wmforge/wt-w11-exact`, `wmforge/wt-r2-exact`.

## Inputs and where they come from

| Input | Source |
|---|---|
| Track 2 π0.5 SFT | HF `WorldArena/WorldArena` `fulldata/` |
| Wan2.2-TI2V-5B DiT | HF `Wan-AI/Wan2.2-TI2V-5B` |
| Wan2.2 VAE | HF `RLinf/RLinf-Wan-LIBERO-Spatial` `Wan2.2_VAE.pth` |
| BWM base `step-12000` | HF `BLM-Lab/Boundless-World-Model` @ `738a8d3c` |
| BWM fine-tuned `step-750` | HF `lidavidsh/bwm-robotwin-adjust-bottle-failure-ft` |
| T5 classifier | HF `lidavidsh/robotwin-adjust-bottle-t5-success-classifier` (`report.json` there names `google-t5/t5-base`; the node copy had a local path) |
| T5 classifier training data | HF dataset `lidavidsh/robotwin-adjust-bottle-t5-reward-data` (private; `train/val/test.pt` + `manifest.json`, the `--data` input of `_r2_train.py`) |
| `t5-base`, ResNet18 | HF `google-t5/t5-base`; `download.pytorch.org` `resnet18-f37072fd.pth` |
| w6 episodes | HF dataset `lidavidsh/robotwin-adjust-bottle-pi05-rollouts` (`.npz` + `_act.npy`; `_frames.npy` is unpacked by `fetch_assets.py --w6`) |
| RoboTwin assets | HF dataset `TianxingChen/RoboTwin2.0`, decimated by `simplify_assets.py` |
| RL init frames, seed lists | in `node/` (`init_frames_hires48/` ↔ `vomni/exp35/exp36_seeds.json` in order; confirm sets in `vomni/artifacts/w11/`) |
| Docker image | `rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4`, id `sha256:d71b40a4…` |

## What cannot be regenerated bit for bit

- **The policies behind stages 1 and 2 are deleted.** w6 used the F21 sim-PPO checkpoints (steps 10/20/30) and
  the reward data used three intermediate RL policies (`f22-orig-step5`, `f22-ignore-step5`,
  `exp16-hires48-step10`). Both datasets survive on HF, so WM fine-tuning and classifier training can be redone
  on the same data; re-running the collection itself gives data of the same kind from new policies.
- **The w32 policies and the w28 SFT dumps are lost** with the node that produced them. Stage 4 has to be
  re-measured anyway: SFT labels are node-specific.
- **`env.sh` rebuilds the venv from a freeze.** Packages that pip resolves differently on another day, and the
  RLinf installer's own steps before the freeze is applied, are outside what was verified.
