"""w18: land each WM chunk under RLINF_W18_DUMP=<dir>: frames, the policy's raw joint14 actions and
the bridged eef actions the WM consumed, per-frame T5 probabilities, the WM verdict, and the random
stream identity. Training is unchanged when the variable is unset.

The existing RLINF_BWM_REWARD_DUMP hook sits on the default reward path, which
``reward_mode: episode_window_max`` overrides, so it never fires in the w11 recipe.

    python w18_dump.py <rlinf-tree>
"""

import sys

W = sys.argv[1]
MARK = "_w18_raw_actions"
TO_NP = (
    "(\n"
    "            actions.detach().float().cpu().numpy()\n"
    "            if torch.is_tensor(actions)\n"
    "            else np.asarray(actions, dtype=np.float32)\n"
    "        )\n"
)


def edit(path, pairs):
    src = open(path).read()
    if MARK in src or "RLINF_W18_DUMP" in src:
        print(f"already patched: {path}")
        return
    for old, new in pairs:
        assert src.count(old) == 1, (path, old[:60], src.count(old))
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"patched: {path}")


edit(f"{W}/rlinf/envs/sim/world_model/world_model_env.py", [(
    "        if self.action_bridge is not None:\n"
    "            actions = self.action_bridge(actions)\n",
    "        self._w18_raw_actions = " + TO_NP +
    "        if self.action_bridge is not None:\n"
    "            actions = self.action_bridge(actions)\n"
    "        self._w18_bridged_actions = " + TO_NP,
)])

edit(f"{W}/rlinf/envs/sim/world_model/world_model_wan_env.py", [
    (
        "        probs = torch.cat(probs).reshape(num_envs, self.chunk).to(self.device)\n",
        "        probs = torch.cat(probs).reshape(num_envs, self.chunk).to(self.device)\n"
        "        frame_probs = probs.detach().float().cpu().numpy()\n",
    ),
    (
        "        self.reward_model.last_probs = self._episode_peak.detach()\n"
        "        return rewards\n",
        "        self.reward_model.last_probs = self._episode_peak.detach()\n"
        "        self._w18_dump(frames, texts, frame_probs, rewards)\n"
        "        return rewards\n"
        "\n"
        "    def _w18_dump(self, frames, texts, frame_probs, rewards):\n"
        "        import os\n"
        "\n"
        "        import numpy as np\n"
        "\n"
        "        root = os.environ.get(\"RLINF_W18_DUMP\")\n"
        "        if not root:\n"
        "            return\n"
        "        os.makedirs(root, exist_ok=True)\n"
        "        call = getattr(self, \"_w18_dump_call\", 0)\n"
        "        n = self.num_envs\n"
        "        imgs = frames.reshape(n, self.chunk, *frames.shape[1:]).clamp(0, 1).mul(255).round()\n"
        "        native = self.cfg.get(\"world_model\", {}).get(\"native\", {})\n"
        "        np.savez_compressed(\n"
        "            os.path.join(root, f\"{os.getpid()}_{call:04d}.npz\"),\n"
        "            images=imgs.to(torch.uint8).cpu().numpy(),  # [n, chunk, 3, H, W]\n"
        "            actions=getattr(self, \"_w18_raw_actions\", np.zeros(0, np.float32)),\n"
        "            bridged_actions=getattr(self, \"_w18_bridged_actions\", np.zeros(0, np.float32)),\n"
        "            frame_probs=frame_probs,  # [n, chunk], before the frame-51 mask\n"
        "            episode_peak=self._episode_peak.detach().float().cpu().numpy(),\n"
        "            labels=rewards[:, -1].detach().float().cpu().numpy(),\n"
        "            reset_state_ids=torch.as_tensor(self.reset_state_ids).cpu().numpy(),\n"
        "            elapsed_steps=torch.as_tensor(self.elapsed_steps).cpu().numpy(),\n"
        "            instructions=np.asarray(\n"
        "                [texts[i * self.chunk] if texts else \"\" for i in range(n)], dtype=object\n"
        "            ),\n"
        "            worker_seed=np.int64(self.seed),\n"
        "            wm_seed_offset=np.int64(native.get(\"seed_offset\", 0) or 0),\n"
        "            seed_everything=np.asarray(os.environ.get(\"RLINF_SEED_EVERYTHING\", \"\")),\n"
        "            call_index=np.int32(call),\n"
        "        )\n"
        "        self._w18_dump_call = call + 1\n",
    ),
])
