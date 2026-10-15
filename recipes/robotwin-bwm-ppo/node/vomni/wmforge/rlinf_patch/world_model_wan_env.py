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

"""The Wan world model as an env: :class:`WorldModelEnv` plus :class:`WanBackend`."""

from __future__ import annotations

from typing import Optional

import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.world_model.backend import WorldModelBackend
from rlinf.envs.sim.world_model.world_model_env import WorldModelEnv

__all__ = ["WanEnv"]

# Scorers that read the task text; the plain ResNet takes frames alone.
TASK_CONDITIONED_REWARDS = (
    "TaskEmbedResnetRewModel",
    "TOPRewardModel",
    "RoboTwinT5ChunkRewardModel",
)


class WanEnv(WorldModelEnv):
    def _build_backend(self) -> WorldModelBackend:
        backend = self.cfg.get("world_model", {}).get("backend", "wan")
        if backend == "remote":
            from rlinf.envs.sim.world_model.remote_backend import (
                RemoteWorldModelBackend,
            )

            return RemoteWorldModelBackend(self.cfg, self._get_runtime_device())
        if backend == "native_bwm":
            from rlinf.envs.sim.world_model.native_bwm_backend import NativeBwmBackend

            return NativeBwmBackend(self.cfg, self._get_runtime_device())
        if backend != "wan":
            raise ValueError(
                f"world_model.backend={backend!r} is not served by this env; "
                "expected wan, remote or native_bwm"
            )
        from rlinf.envs.sim.world_model.wan_backend import WanBackend

        return WanBackend(self.cfg, self._get_runtime_device())

    def _load_reward_model(self):
        # diffsynth ships the Wan pipeline alongside these scorers, so a served backend
        # would otherwise need it installed just to reach the reward model it does use.
        if self.cfg.reward_model.type == "ResnetRewModel":
            from diffsynth.models.reward_model import ResnetRewModel

            return ResnetRewModel(self.cfg.reward_model.from_pretrained)
        elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
            from diffsynth.models.reward_model import TaskEmbedResnetRewModel

            return TaskEmbedResnetRewModel(
                checkpoint_path=self.cfg.reward_model.from_pretrained,
                task_suite_name=self.cfg.task_suite_name,
            )
        elif self.cfg.reward_model.type == "TOPRewardModel":
            from rlinf.models.embodiment.reward.topreward_model import TOPRewardModel

            rm_cfg = OmegaConf.to_container(self.cfg.reward_model, resolve=True)
            rm_cfg.pop("type")
            return TOPRewardModel(
                model_path=rm_cfg.pop("from_pretrained"),
                chunk=self.cfg.chunk,
                **rm_cfg,
            )
        elif self.cfg.reward_model.type == "RoboTwinT5ChunkRewardModel":
            from rlinf.models.embodiment.reward.robotwin_chunk_reward import (
                RoboTwinT5ChunkRewardModel,
            )

            rm_cfg = OmegaConf.to_container(self.cfg.reward_model, resolve=True)
            rm_cfg.pop("type")
            # The base class reads the checkpoint from model_path; the yaml key stays
            # from_pretrained so every scorer in this factory is configured alike.
            rm_cfg["model_path"] = rm_cfg.pop("from_pretrained")
            rm_cfg["chunk"] = self.cfg.chunk
            return RoboTwinT5ChunkRewardModel(OmegaConf.create(rm_cfg))
        raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")

    def _reward_instructions(self) -> Optional[list[str]]:
        if self.cfg.reward_model.type not in TASK_CONDITIONED_REWARDS:
            return None
        # One instruction per scored frame, so each description repeats over its chunk
        instructions = []
        for env_idx in range(self.num_envs):
            instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
        return instructions

    def _infer_next_chunk_rewards(self):
        """``reward_mode: episode_window_max`` scores every frame from ``reward_first_frame`` on,
        keeps each slot's peak and pays one binary reward on the episode's last step; the
        default scores the chunk as the reward model defines."""
        if self.cfg.get("reward_mode", "chunk") != "episode_window_max":
            return super()._infer_next_chunk_rewards()
        num_envs, c, v, t, h, w = self.current_obs.shape
        frames = self.current_obs.permute(0, 3, 1, 2, 4, 5)[:, -self.chunk :]
        frames = frames.reshape(num_envs * self.chunk, 3, v, h, w).squeeze(2).to(self.device)
        if frames.min() < 0:
            frames = (frames + 1.0) / 2.0
        texts = self._reward_instructions()
        probs = []
        for i in range(0, frames.shape[0], 100):
            probs.append(
                self.reward_model.compute_reward(
                    frames[i : i + 100],
                    task_descriptions=None if texts is None else texts[i : i + 100],
                ).float()
            )
        probs = torch.cat(probs).reshape(num_envs, self.chunk).to(self.device)

        elapsed = self.elapsed_steps.to(self.device).long()  # already counts this chunk
        frame_idx = elapsed[:, None] - self.chunk + 1 + torch.arange(self.chunk, device=self.device)
        first = int(self.cfg.get("reward_first_frame", 51))
        probs = torch.where(frame_idx >= first, probs, torch.zeros_like(probs))
        peak = getattr(self, "_episode_peak", None)
        if peak is None or peak.shape[0] != num_envs:
            peak = torch.zeros(num_envs, device=self.device)
        self._episode_peak = torch.maximum(peak.to(self.device), probs.max(dim=1).values)

        rewards = torch.zeros(num_envs, self.chunk, device=self.device)
        last = elapsed >= int(self.cfg.max_episode_steps)
        rewards[:, -1] = ((self._episode_peak >= self.reward_model.threshold) & last).float()
        self.reward_model.last_probs = self._episode_peak.detach()
        return rewards

    def _reset_metrics(self, env_idx=None):
        super()._reset_metrics(env_idx=env_idx)
        peak = getattr(self, "_episode_peak", None)
        if peak is None:
            return
        if env_idx is None:
            peak.zero_()
        else:
            peak[torch.as_tensor(env_idx, device=peak.device, dtype=torch.long)] = 0.0
