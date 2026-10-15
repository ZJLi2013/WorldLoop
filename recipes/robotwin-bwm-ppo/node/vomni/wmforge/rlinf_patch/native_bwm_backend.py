# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend that runs a BWM (Wan2.2-TI2V-5B, action-conditioned) checkpoint in-process.

Each window goes through the same ``generate_window`` the offline open-loop evaluation uses,
with frames kept as tensors end to end, so the env sees exactly what the evaluation scored.
The session semantics follow :class:`RemoteWorldModelBackend`: the reset frame stays in slot 0,
the other 8 condition frames and their actions roll forward after each chunk.
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

import numpy as np
import torch

from rlinf.envs.sim.world_model.backend import FrameQueue

__all__ = ["NativeBwmBackend"]


class NativeBwmBackend:
    """Advances frames on an in-process BWM pipeline.

    Args:
        cfg: The env config; the checkpoint and code location live under
            ``world_model.native``.
        device: Where the pipeline runs and frames land.
    """

    def __init__(self, cfg, device: torch.device):
        native = cfg.world_model.native
        code_dir = native.get("code_dir", "/data/zhengjli/vomni/wmforge")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        import bwm_common

        self._bc = bwm_common
        self.device = device
        self.chunk = int(cfg.chunk)
        self.condition_frame_length = int(cfg.condition_frame_length)
        self.image_size = tuple(cfg.image_size)
        if (self.chunk, self.condition_frame_length) != (bwm_common.CHUNK, bwm_common.HIST):
            raise ValueError(
                f"BWM serves windows of {bwm_common.HIST} + {bwm_common.CHUNK}; "
                f"got {self.condition_frame_length} + {self.chunk}"
            )
        self.steps = int(native.get("num_inference_steps", 50))
        # The env hands every slot seed 0, so this is the only knob that moves the WM noise.
        self.seed_offset = int(native.get("seed_offset", 0))
        self.pipe = bwm_common.build_pipe(ckpt=native.ckpt, device=str(device))
        self.pipe.eval()
        # The policy's commands are bridged by the env; the reset pose is bridged here once.
        init_joint14 = native.get("init_joint14", None)
        self._init_action = None
        if init_joint14 is not None:
            from rlinf.envs.sim.world_model.robotwin_action_bridge import get_bridge

            joint = np.asarray(init_joint14, np.float32)[None]
            self._init_action = torch.from_numpy(get_bridge(cfg.action_bridge)(joint)[0])
        self._sessions: dict[int, dict[str, Any]] = {}

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        task_ids: Sequence[Any],
        seeds: Sequence[int],
    ) -> None:
        for row, (env_id, seed) in enumerate(zip(env_ids, seeds)):
            frames = [f[:, 0].detach().float().cpu() for f in init_frames[row]]  # [3, H, W]
            if self._init_action is not None:
                actions = self._init_action[None].repeat(self.condition_frame_length, 1)
            else:
                actions = init_actions[row].detach().float().cpu().clone()
            self._sessions[int(env_id)] = {
                "seed": int(seed),
                "frames": frames,
                "actions": actions,
                "step_id": 0,
            }

    def close_session(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._sessions.pop(int(env_id), None)

    @torch.no_grad()
    def generate(self, env_ids: Sequence[int], actions: torch.Tensor) -> torch.Tensor:
        missing = [int(i) for i in env_ids if int(i) not in self._sessions]
        if missing:
            raise RuntimeError(f"generate called for env slots without a session: {missing}")
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = actions.detach().float().cpu()
        videos = []
        for row, env_id in enumerate(env_ids):
            s = self._sessions[int(env_id)]
            windowed = torch.cat([s["actions"], actions[row]], dim=0)  # [9 + 50, 14]
            new = self._bc.generate_window(
                self.pipe,
                s["frames"],
                windowed.numpy(),
                self.seed_offset + s["seed"] + s["step_id"],
                self.steps,
            )  # [3, 50, H, W]
            keep = self.condition_frame_length - 1
            s["frames"][1:] = [new[:, i] for i in range(new.shape[1] - keep, new.shape[1])]
            s["actions"][1:] = windowed[-keep:]
            s["step_id"] += 1
            videos.append(new)
        return torch.stack(videos, dim=0).to(self.device)

    def offload(self) -> None:
        for name in ("dit", "vae", "action_encoder"):
            getattr(self.pipe, name).to("cpu")
        torch.cuda.empty_cache()

    def onload(self) -> None:
        for name in ("dit", "vae", "action_encoder"):
            getattr(self.pipe, name).to(self.device)
