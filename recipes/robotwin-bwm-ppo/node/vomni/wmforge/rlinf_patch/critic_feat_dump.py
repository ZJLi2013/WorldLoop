"""WMForge c5: dump the critic input (VLM mean-token features), GAE returns and sample positions.

Active only when RLINF_CRITIC_FEAT_DUMP=<dir> is set. During the first PPO update epoch of every
training step each actor rank saves <dir>/step<N>_rank<R>.pt with the features of its samples in
training order and their index into the flattened [T, B] rollout batch (t = index // B, b = index % B).
Requires critic_epoch_metrics.py (epoch_idx in run_training).

    python3 critic_feat_dump.py <rlinf tree>
"""
import sys

W = sys.argv[1]


def edit(path, marker, pairs):
    src = open(path).read()
    if marker in src:
        print("already patched", path)
        return
    for old, new in pairs:
        assert src.count(old) == 1, (path, old[:80])
        src = src.replace(old, new)
    open(path, "w").write(src)
    print("patched", path)


edit(f"{W}/rlinf/models/embodiment/openpi/openpi_action_model.py", "CRITIC_FEAT_LAST", [
    ("import math\nimport random\n", "import math\nimport os\nimport random\n"),
    (
        "from rlinf.utils.logging import get_logger\n",
        "from rlinf.utils.logging import get_logger\n\nCRITIC_FEAT_LAST: list = []\n",
    ),
    (
        "        prefix_out_value = prefix_out_value.to(dtype=torch.float32)\n"
        "        values_vlm = self.value_head(prefix_out_value)[:, 0]\n",
        "        prefix_out_value = prefix_out_value.to(dtype=torch.float32)\n"
        '        if os.environ.get("RLINF_CRITIC_FEAT_DUMP"):\n'
        "            CRITIC_FEAT_LAST[:] = [prefix_out_value.detach().cpu()]\n"
        "        values_vlm = self.value_head(prefix_out_value)[:, 0]\n",
    ),
])

edit(f"{W}/rlinf/workers/actor/embodied_fsdp_actor_worker.py", "_feat_dump", [
    (
        "        shuffle_id = torch.randperm(rollout_size, generator=g)\n",
        "        shuffle_id = torch.randperm(rollout_size, generator=g)\n"
        "        import os\n"
        "\n"
        '        self._feat_dump = os.environ.get("RLINF_CRITIC_FEAT_DUMP", "")\n'
        "        if self._feat_dump:\n"
        "            self._feat_shuffle = shuffle_id.clone()\n"
        '            self._feat_batch = int(self.rollout_batch["prev_logprobs"].shape[1])\n'
        "            self._feat_rows = []\n",
    ),
    (
        "            epoch_start = {key: len(value) for key, value in metrics.items()}\n",
        "            epoch_start = {key: len(value) for key, value in metrics.items()}\n"
        "            self._feat_collect = epoch_idx == 0\n",
    ),
    (
        "            for key in list(metrics):\n"
        '                if key.startswith("critic/") or key in EV_STAT_KEYS:\n',
        '            if getattr(self, "_feat_collect", False) and self._feat_dump and self._feat_rows:\n'
        "                n = sum(r[0].shape[0] for r in self._feat_rows)\n"
        "                torch.save(\n"
        "                    {\n"
        '                        "feat": torch.cat([r[0] for r in self._feat_rows]).half(),\n'
        '                        "returns": torch.cat([r[1] for r in self._feat_rows]),\n'
        '                        "orig_index": self._feat_shuffle[:n],\n'
        '                        "batch": self._feat_batch,\n'
        "                    },\n"
        '                    f"{self._feat_dump}/step{self.version}_rank{self._rank}.pt",\n'
        "                )\n"
        "                self._feat_rows = []\n"
        "            self._feat_collect = False\n"
        "            for key in list(metrics):\n"
        '                if key.startswith("critic/") or key in EV_STAT_KEYS:\n',
    ),
    (
        "        loss_kwargs = {\n",
        '        if getattr(self, "_feat_collect", False) and self._feat_dump:\n'
        "            from rlinf.models.embodiment.openpi import openpi_action_model as oam\n"
        "\n"
        "            self._feat_rows.append((oam.CRITIC_FEAT_LAST[0], returns.detach().cpu()))\n"
        "\n"
        "        loss_kwargs = {\n",
    ),
])
