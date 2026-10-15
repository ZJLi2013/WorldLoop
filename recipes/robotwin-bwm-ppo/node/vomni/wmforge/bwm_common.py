"""WMForge shared pieces for BWM fine-tuning and open-loop evaluation on w6 episodes.

Window contract (plan, 窗口契约): 61 frames = initial frame + 8 history + 52 future. History indices
follow scripts/infer.py with retain_reference_frame (`[0]*9` for the first window, else `[0]` + the
last 8). Serving windows start at frames 1, 51, 101, 151; each takes 50 commands and holds the 50th
for future frames 51-52, which serving discards.

w6 episode npz: frames [203,H,W,3] uint8, commands [202,14] joint14, plus labels. Frame t+1 follows
commands[t]; frame 0 pairs with the reset joint state. Actions are joint14 -> eef_abs through
RobotwinActionBridge, which already normalises with BWM's stat.json p01/p99.
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = "/data/zhengjli/vomni/bwm-repo"
RLINF = "/data/zhengjli/rlinf/wt-g"
for p in (REPO, RLINF):
    if p not in sys.path:
        sys.path.insert(0, p)

DIT = "/data/zhengjli/vomni/native/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json"
VAE = "/data/zhengjli/rlinf/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth"
BASE_CKPT = glob.glob("/data/zhengjli/hf/hub/models--BLM-Lab--Boundless-World-Model/snapshots/*/step-12000.safetensors")[0]
BRIDGE = dict(
    urdf_path="/data/zhengjli/vomni/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf",
    srdf_path="/data/zhengjli/vomni/RoboTwin/assets/embodiments/aloha-agilex/srdf/arx5_description_isaac.srdf",
    stat_path=f"{REPO}/demo/stat.json",
    robot_pose=[0.0, -0.65, 0.0, 0.707, 0.0, 0.0, 0.707],
    swap_arm_blocks=False,
)
H, W = 480, 640
HIST, FUT, CHUNK = 9, 52, 50
NFRAMES = HIST + FUT
N_COND = (HIST - 1) // 4 + 1  # latents covering the initial frame + 8 history frames
SIGMA_H = 0.2083
SERVE_STARTS = (1, 51, 101, 151)


def _history_condition_latents(self, inputs_shared, *, use_history_condition_noise_in_inference=False):
    """BWM's version noises history latents at timesteps[len - 50], i.e. sigma = 1 at 50 inference
    steps, which replaces them with noise. This one applies the serving / training sigma."""
    first = inputs_shared.get("first_frame_latents")
    latents = inputs_shared.get("latents")
    if first is None or latents is None:
        return None, 0
    if first.ndim == 4:
        first = first.unsqueeze(0)
    history_t = min(int(first.shape[2]), int(latents.shape[2]))
    if history_t <= 0:
        return None, 0
    cond = first[:, :, :history_t].clone()
    if use_history_condition_noise_in_inference and history_t > 1:
        # inputs_shared["noise"] aliases latents, whose history slots already hold z; draw our own.
        g = torch.Generator("cpu").manual_seed(int(inputs_shared.get("seed") or 0) + 7919)
        eps = torch.randn(cond[:, :, 1:history_t].shape, generator=g).to(cond.device, cond.dtype)
        cond[:, :, 1:history_t] = (1 - SIGMA_H) * cond[:, :, 1:history_t] + SIGMA_H * eps
    latents[:, :, :history_t] = cond
    return cond, history_t


def build_pipe(ckpt=BASE_CKPT, device="cuda"):
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        from diffsynth.core import ModelConfig
        import wan_video_action.pipelines.wan_video_action as wva
        from wan_video_action.pipelines.wan_video_action import build_wan_video_action_pipeline
        from wan_video_action.utils import resolve_model_path

        wva._prepare_history_condition_latents = _history_condition_latents

        pipe = build_wan_video_action_pipeline(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=[ModelConfig(path=resolve_model_path(DIT)), ModelConfig(path=resolve_model_path(VAE))],
            tokenizer_config=None, ckpt_path=ckpt, action_dim=14, action_mode="adaln")
    finally:
        os.chdir(cwd)
    pipe.use_gradient_checkpointing = False
    pipe.use_gradient_checkpointing_offload = False
    return pipe


def bridge():
    from rlinf.envs.sim.world_model.robotwin_action_bridge import RobotwinActionBridge

    return RobotwinActionBridge(**BRIDGE)


def episode_actions(br, reset_joint14, commands):
    """[203, 14] normalised eef_abs: frame 0 <- reset state, frame t+1 <- commands[t]."""
    joint = np.concatenate([np.asarray(reset_joint14, np.float32)[None], np.asarray(commands, np.float32)])
    return br(joint)


def window_indices(start, T):
    """Frame indices (history + future) of the window whose first future frame is `start`."""
    hist = [0] * HIST if start < HIST else [0] + list(range(start - (HIST - 1), start))
    fut = [min(t, T - 1) for t in range(start, start + FUT)]
    return hist, fut


def window(frames, actions, start, hold_tail=True):
    """Video [1,3,61,H,W] in [-1,1] and actions [1,61,14]. With hold_tail the last two future
    steps repeat the 50th, as serving pads them; their frames repeat the 50th frame too, which is
    what RoboTwin shows for a held command (TOPP plans no substeps for it)."""
    T = len(frames)
    hist, fut = window_indices(start, T)
    idx = hist + fut
    if hold_tail:
        idx[HIST + CHUNK:] = [idx[HIST + CHUNK - 1]] * (FUT - CHUNK)
    v = torch.from_numpy(np.ascontiguousarray(frames[idx])).permute(3, 0, 1, 2).float() / 127.5 - 1.0
    a = torch.from_numpy(np.asarray(actions[idx], np.float32))
    return v.unsqueeze(0), a.unsqueeze(0)


def prepare(pipe, video, action, seed):
    shared = {"input_video": video.to(pipe.device, pipe.torch_dtype), "action": action.to(pipe.device, pipe.torch_dtype),
              "height": H, "width": W, "num_frames": NFRAMES, "num_history_frames": HIST, "num_views": 1,
              "seed": seed, "rand_device": "cpu", "tiled": False, "cfg_scale": 1, "vace_reference_image": None,
              "use_gradient_checkpointing": pipe.use_gradient_checkpointing,
              "use_gradient_checkpointing_offload": False}
    posi, nega = {}, {}
    for unit in pipe.units:
        shared, posi, nega = pipe.unit_runner(unit, pipe, shared, posi, nega)
    return shared


def flow_loss(pipe, shared, timestep_id, seed):
    """Future-only flow matching; initial latent clean, history latents at (1-s) z + s eps."""
    x0 = shared["input_latents"]
    g = torch.Generator("cpu").manual_seed(seed)
    eps = torch.randn(x0.shape, generator=g).to(x0.device, x0.dtype)
    t = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device).unsqueeze(0)
    latents = pipe.scheduler.add_noise(x0, eps, t)
    target = pipe.scheduler.training_target(x0, eps, t)
    cond = shared["first_frame_latents"][:, :, :N_COND].clone()
    assert cond.shape[2] == N_COND, cond.shape
    eh = torch.randn(cond[:, :, 1:].shape, generator=g).to(cond.device, cond.dtype)
    cond[:, :, 1:] = (1 - SIGMA_H) * cond[:, :, 1:] + SIGMA_H * eh
    latents = latents.clone()
    latents[:, :, :N_COND] = cond
    inputs = dict(shared)
    inputs["latents"] = latents
    inputs["fused_condition_latent_frames"] = N_COND
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    pred = pipe.model_fn(**models, **inputs, timestep=t)
    loss = F.mse_loss(pred[:, :, N_COND:].float(), target[:, :, N_COND:].float())
    return loss * pipe.scheduler.training_weight(t).float()


def save_trainable(pipe, path):
    from safetensors.torch import save_file

    sd = {k: v.detach().to("cpu").contiguous() for k, v in pipe.dit.state_dict().items()}
    sd.update({f"pipe.action_encoder.{k}": v.detach().to("cpu").contiguous()
               for k, v in pipe.action_encoder.state_dict().items()})
    save_file(sd, path)


@torch.no_grad()
def generate_window(pipe, cond_frames, window_actions, seed, steps=50):
    """One serving window: 9 condition frames ([3,H,W] in [-1,1]) and 59 actions (9 history +
    50 commands, already bridged) -> the 50 new frames as [3,50,H,W] float in [-1,1] on CPU.
    The last command is held for the 2 frames that fill the latent grid; they are discarded."""
    act = np.asarray(window_actions, np.float32)
    assert act.shape[0] == HIST + CHUNK, act.shape
    act = np.concatenate([act, np.repeat(act[-1:], FUT - CHUNK, axis=0)])
    vid = torch.stack(list(cond_frames), dim=1).unsqueeze(0)  # [1,3,9,H,W]
    out = pipe(input_video=vid.to(pipe.device, pipe.torch_dtype),
               action=torch.from_numpy(act).unsqueeze(0).to(pipe.device, pipe.torch_dtype),
               seed=int(seed), rand_device="cpu", tiled=False, height=H, width=W, num_frames=NFRAMES,
               num_history_frames=HIST, cfg_scale=1.0, num_inference_steps=steps,
               use_history_condition_noise_in_inference=True,
               progress_bar_cmd=lambda it, *a, **k: it, output_type="floatpoint")
    return out[0, :, HIST:HIST + CHUNK].detach().float().cpu()


@torch.no_grad()
def rollout(pipe, frame0, actions, seed=0, steps=50):
    """Open-loop serving rollout: 4 windows of 50 commands from the reset frame.
    frame0 [H,W,3] uint8, actions [>=203,14] -> generated frames [201,H,W,3] uint8 (frame 0 = input)."""
    gen = [torch.from_numpy(np.ascontiguousarray(frame0)).permute(2, 0, 1).float() / 127.5 - 1.0]
    T_act = len(actions)
    for w, start in enumerate(SERVE_STARTS):
        hist, _ = window_indices(start, T_act)
        act_idx = hist + list(range(start, start + CHUNK))
        fut = generate_window(pipe, [gen[i] for i in hist], actions[act_idx], seed + w, steps)
        gen.extend(fut[:, i] for i in range(CHUNK))
    video = torch.stack(gen, dim=0)  # [201,3,H,W]
    return ((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
