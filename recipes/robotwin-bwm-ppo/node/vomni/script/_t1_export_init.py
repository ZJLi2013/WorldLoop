#!/usr/bin/env python3
"""T1: export RoboTwin reset observations as BWM init frames.

The pose format follows init_frames_ep1 -- the set the connected rollout was taken on --
which is RoboTwin's native 16-dim quaternion with the right arm block first. The older
init_frames set from feature4 holds 14-dim sxyz euler with the left block first; feeding
that shape to the served BWM collapses the picture from the second chunk on.
"""
import argparse
import glob
import os

import numpy as np

REPO = os.environ.get("REPO_PATH", "/data/zhengjli/rlinf/wt-f24")
EVALDIR = os.path.join(REPO, "evaluations", "robotwin")


def reference_pose(src):
    """The reset pose and its delta_action twin, read off the reference set."""
    paths = sorted(glob.glob(os.path.join(src, "*.npy")))
    if not paths:
        raise SystemExit(f"no reference episode under {src}")
    item = np.load(paths[0], allow_pickle=True)
    while isinstance(item, np.ndarray) and item.dtype == object and item.size == 1:
        item = item.reshape(-1)[0]
    if isinstance(item, np.ndarray) and item.dtype == object:
        item = item[0]
    return np.asarray(item["init_ee_pose"]), np.asarray(item["delta_action"])


def swap_arm_blocks(endpose16):
    """RoboTwin reports [left, right]; BWM's state_pose puts the right block first."""
    e = np.asarray(endpose16, dtype=np.float32).reshape(-1)
    if e.size != 16:
        raise SystemExit(f"expected a 16-dim endpose, got {e.size}")
    return np.concatenate([e[8:], e[:8]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--out", default="/data/zhengjli/vomni/artifacts/init_frames_n48")
    ap.add_argument("--ref", default="/data/zhengjli/vomni/artifacts/init_frames_ep1")
    ap.add_argument("--cfg", default="rocm_adjust_bottle_pi0_smoke")
    # BWM trains on 480p re-renders and its paper calls the initial frame's fidelity critical
    # for rollout coherence, so the head camera has to be the 640x480 one rather than the
    # 320x240 default the eval config leaves in place.
    ap.add_argument("--head-camera", default="Large_D435")
    # BWM conditions on text. RoboTwin writes a fresh wording per episode, while the model was
    # trained on one fixed phrasing; --instruction overrides every episode with that phrasing.
    ap.add_argument("--instruction", default=os.environ.get("INIT_INSTRUCTION") or None)
    args = ap.parse_args()

    ref_pose, ref_delta = reference_pose(args.ref)
    print(f"reference pose {ref_pose.shape} {np.round(ref_pose, 3).tolist()}")

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=EVALDIR, version_base=None):
        cfg = compose(config_name=args.cfg)
    env_cfg = cfg.env.eval

    from omegaconf import OmegaConf, open_dict

    with open_dict(env_cfg):
        env_cfg.total_num_envs = args.num_envs
        env_cfg.group_size = 1
        env_cfg.use_fixed_reset_state_ids = False
        env_cfg.video_cfg.save_video = False
        if args.head_camera:
            env_cfg.task_config.camera.head_camera_type = args.head_camera
    print(
        OmegaConf.to_yaml(
            {
                "task": env_cfg.task_config.task_name,
                "head_camera_type": env_cfg.task_config.camera.get("head_camera_type"),
            }
        )
    )

    from rlinf.envs import get_env_cls

    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    env = env_cls(
        cfg=env_cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )
    obs, _ = env.reset()

    images = obs["main_images"].cpu().numpy()
    instrs = obs["task_descriptions"]
    endposes = obs["endposes"]
    if endposes is None:
        raise SystemExit("endposes is None -- turn on env.task_config.data_type.endpose")
    endposes = endposes.cpu().numpy()
    print(f"reset -> images {images.shape} {images.dtype}, endposes {endposes.shape}")

    pose0 = swap_arm_blocks(endposes[0])
    delta_err = float(np.abs(pose0 - ref_pose).max()) if pose0.shape == ref_pose.shape else -1.0
    print(f"pose after arm swap {np.round(pose0, 3).tolist()}")
    print(f"max |diff| against {os.path.basename(args.ref)}: {delta_err:.4f}")

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.num_envs):
        img = images[i]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        pose = swap_arm_blocks(endposes[i])
        rec = {
            "image": np.ascontiguousarray(img),
            "init_ee_pose": pose,
            "delta_action": pose.copy() if ref_delta.shape == pose.shape else ref_delta.copy(),
            "instruction": args.instruction or str(instrs[i]),
        }
        p = os.path.join(args.out, f"episode_{i:05d}.npy")
        np.save(p, np.array([rec], dtype=object), allow_pickle=True)
        print(f"  {os.path.basename(p)} image={img.shape} instr={rec['instruction'][:70]}")

    from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper

    ds = NpyTrajectoryDatasetWrapper(args.out)
    print(f"readback: {len(ds)} episodes")


if __name__ == "__main__":
    main()
