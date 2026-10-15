"""Render one RoboTwin observation on lavapipe and print what came back.

    cd /path/to/RoboTwin && python /opt/robotwin_rocm/smoke_test.py

Expected on a working setup (aloha-agilex, adjust_bottle, demo_clean, four
D435 cameras at 320x240, meshes decimated):

    render device: GPU: llvmpipe (LLVM 15.0.7, 256 bits)
    joint_action.vector  (14,)
    endpose  left/right [x,y,z,qw,qx,qy,qz] + gripper
    ~1.5 s per get_obs
"""

import importlib
import os
import sys
import time

import yaml

sys.path.append("./")
import sapien_lavapipe  # noqa: F401  # must precede the RoboTwin import

import sapien
from envs import *  # noqa: F401,F403
from envs._GLOBAL_CONFIGS import CONFIGS_PATH

TASK = os.environ.get("RT_TASK", "adjust_bottle")
CFG = os.environ.get("RT_CFG", "demo_clean")

env = getattr(importlib.import_module(f"envs.{TASK}"), TASK)()

args = yaml.safe_load(open(f"./task_config/{CFG}.yml", encoding="utf-8"))
args["task_name"] = TASK
args["task_config"] = CFG
args["save_path"] = "/tmp/robotwin_smoke"
args["need_plan"] = True
args["collect_data"] = False
args["planner_backend"] = "mplib"  # curobo is CUDA-only

embodiments = yaml.safe_load(open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8"))
robot_file = embodiments[args["embodiment"][0]]["file_path"]
args["left_robot_file"] = args["right_robot_file"] = robot_file
args["dual_arm_embodied"] = True
args["left_embodiment_config"] = args["right_embodiment_config"] = yaml.safe_load(
    open(os.path.join(robot_file, "config.yml"), encoding="utf-8"))
args["embodiment_name"] = str(args["embodiment"][0])

t0 = time.time()
env.setup_demo(now_ep_num=0, seed=0, **args)
print(f"SETUP_OK  {time.time() - t0:.1f}s")
print("render device:", sapien.render.get_device_summary().strip().splitlines()[0])

triangles = sum(len(p.triangles)
                for b in env.scene.render_system.render_bodies
                for s in b.render_shapes
                for p in (getattr(s, "parts", []) or []))
print(f"scene triangles: {triangles}")

obs = env.get_obs()
print("obs keys:", sorted(obs))
for cam, v in obs.get("observation", {}).items():
    if "rgb" in v:
        print(f"  {cam}/rgb {v['rgb'].shape} mean={float(v['rgb'].mean()):.1f}")
print("  joint_action.vector", len(obs["joint_action"]["vector"]))
print("  endpose", sorted(obs["endpose"]))

N = 3
t0 = time.time()
for _ in range(N):
    env.get_obs()
print(f"RENDER {(time.time() - t0) / N:.2f}s per get_obs")

