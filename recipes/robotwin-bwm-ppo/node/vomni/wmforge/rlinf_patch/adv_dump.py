"""WMForge w17: dump what the truncation bootstrap adds and the advantages the actor computes.

Active only when RLINF_ADV_DUMP=<dir> is set; training is unchanged.
- env worker, standard bootstrap path: per call with truncated envs, <dir>/env_rank<R>_n<N>.pt holds the
  last reward before and after adding gamma * V(s_T), and V(s_T).
- actor, after calculate_adv_and_returns: <dir>/actor_rank<R>_n<N>.pt holds rewards / dones / prev_values
  (/ loss masks) of the rank's rollout batch and the advantages / returns RLinf computed from them.

    python3 adv_dump.py <rlinf tree>
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


edit(f"{W}/rlinf/workers/env/env_worker.py", "RLINF_ADV_DUMP", [(
    "        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values\n"
    "        return adjusted_rewards\n",
    "        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values\n"
    "        import os\n"
    "\n"
    '        dump = os.environ.get("RLINF_ADV_DUMP")\n'
    "        if dump:\n"
    '            self._adv_dump_n = getattr(self, "_adv_dump_n", 0) + 1\n'
    "            torch.save(\n"
    "                {\n"
    '                    "raw_last": rewards[:, -1].detach().float().cpu(),\n'
    '                    "adjusted_last": adjusted_rewards[:, -1].detach().float().cpu(),\n'
    '                    "final_values": final_values.detach().cpu(),\n'
    '                    "truncated": last_step_truncations.detach().cpu(),\n'
    '                    "dtype": str(adjusted_rewards.dtype),\n'
    "                },\n"
    '                f"{dump}/env_rank{self._rank}_n{self._adv_dump_n}.pt",\n'
    "            )\n"
    "        return adjusted_rewards\n",
)])

edit(f"{W}/rlinf/workers/actor/embodied_fsdp_actor_worker.py", "RLINF_ADV_DUMP", [(
    "        advantages_and_returns = calculate_adv_and_returns(**kwargs)\n",
    "        advantages_and_returns = calculate_adv_and_returns(**kwargs)\n"
    "        import os\n"
    "\n"
    '        dump = os.environ.get("RLINF_ADV_DUMP")\n'
    "        if dump:\n"
    '            self._adv_dump_n = getattr(self, "_adv_dump_n", 0) + 1\n'
    '            keep = ("rewards", "dones", "prev_values", "loss_mask", "loss_mask_sum")\n'
    "            record = {\n"
    "                k: self.rollout_batch[k].detach().cpu()\n"
    "                for k in keep\n"
    "                if self.rollout_batch.get(k, None) is not None\n"
    "            }\n"
    "            record.update({k: v.detach().cpu() for k, v in advantages_and_returns.items()})\n"
    '            record["kwargs"] = {\n'
    "                k: kwargs[k]\n"
    '                for k in ("task_type", "adv_type", "num_action_chunks", "gamma", "gae_lambda",\n'
    '                          "group_size", "reward_type", "advantage_mode")\n'
    "            }\n"
    '            torch.save(record, f"{dump}/actor_rank{self._rank}_n{self._adv_dump_n}.pt")\n',
)])
