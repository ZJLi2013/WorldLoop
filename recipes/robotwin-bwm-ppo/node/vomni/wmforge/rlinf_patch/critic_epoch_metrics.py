"""WMForge c3: log critic metrics per PPO update epoch next to the epoch-merged ones.

Adds |V - V_old|, the share of samples where the clipped value loss wins, and per-epoch explained
variance with return / value moments. Metrics only; the loss and the update are unchanged.

    python3 critic_epoch_metrics.py <rlinf tree>
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


edit(f"{W}/rlinf/algorithms/losses.py", "critic/value_delta_abs", [
    (
        "        value_clip_ratio = masked_mean(value_clip_indicator, clip_mask)\n",
        "        value_clip_ratio = masked_mean(value_clip_indicator, clip_mask)\n"
        "\n"
        "    value_delta = (values - prev_values).detach().abs()\n"
        "    clip_wins = (value_loss_clipped > value_loss_original).detach().float()\n"
        "    if loss_mask is None:\n"
        "        value_delta_abs, clip_wins_ratio = value_delta.mean(), clip_wins.mean()\n"
        "    else:\n"
        "        value_delta_abs = masked_mean(value_delta, clip_mask)\n"
        "        clip_wins_ratio = masked_mean(clip_wins, clip_mask)\n",
    ),
    (
        '        "critic/value_clip_ratio": value_clip_ratio.detach(),\n',
        '        "critic/value_clip_ratio": value_clip_ratio.detach(),\n'
        '        "critic/value_delta_abs": value_delta_abs.detach(),\n'
        '        "critic/clip_wins_ratio": clip_wins_ratio.detach(),\n',
    ),
])

edit(f"{W}/rlinf/workers/actor/embodied_fsdp_actor_worker.py", "EV_STAT_KEYS", [
    (
        '        update_epoch = self.cfg.algorithm.get("update_epoch", 1)\n'
        "        for _ in range(update_epoch):\n",
        '        update_epoch = self.cfg.algorithm.get("update_epoch", 1)\n'
        "        from rlinf.utils.metric_utils import (\n"
        "            CRITIC_EXPLAINED_VARIANCE_STAT_KEYS as EV_STAT_KEYS,\n"
        "        )\n"
        "\n"
        "        for epoch_idx in range(update_epoch):\n"
        "            epoch_start = {key: len(value) for key, value in metrics.items()}\n",
    ),
    (
        "                append_to_dict(metrics, data)\n"
        "        # put LR scheduler step here\n",
        "                append_to_dict(metrics, data)\n"
        "            for key in list(metrics):\n"
        '                if key.startswith("critic/") or key in EV_STAT_KEYS:\n'
        "                    new = metrics[key][epoch_start.get(key, 0) :]\n"
        "                    if new:\n"
        '                        metrics[f"_c3/{key}_e{epoch_idx}"] = [float(v) for v in new]\n'
        "        # put LR scheduler step here\n",
    ),
    (
        "            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (\n"
        "                compute_critic_explained_variance_from_stats(reduced_stats).item()\n"
        "            )\n",
        "            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (\n"
        "                compute_critic_explained_variance_from_stats(reduced_stats).item()\n"
        "            )\n"
        "        # Per-epoch copies are averaged like the other metrics; the moments below are\n"
        "        # ratios of the summed statistics, so the common averaging factor cancels.\n"
        "        for epoch_idx in range(update_epoch):\n"
        '            tag = f"_e{epoch_idx}"\n'
        "            for key in [\n"
        '                k for k in mean_metric_dict if k.startswith("_c3/critic/") and k.endswith(tag)\n'
        "            ]:\n"
        '                mean_metric_dict[key[len("_c3/") :]] = mean_metric_dict.pop(key)\n'
        '            stats = [mean_metric_dict.pop(f"_c3/{key}{tag}", None) for key in EV_STAT_KEYS]\n'
        "            if any(v is None for v in stats):\n"
        "                continue\n"
        "            n, rs, rss, es, ess = (float(v) for v in stats)\n"
        "            if n <= 0:\n"
        "                continue\n"
        "            ret_mean, err_mean = rs / n, es / n\n"
        "            ret_var = rss / n - ret_mean * ret_mean\n"
        "            err_var = ess / n - err_mean * err_mean\n"
        '            mean_metric_dict[f"critic/ev{tag}"] = (\n'
        '                1 - err_var / ret_var if ret_var > 1e-12 else float("nan")\n'
        "            )\n"
        '            mean_metric_dict[f"critic/ret_mean{tag}"] = ret_mean\n'
        '            mean_metric_dict[f"critic/ret_var{tag}"] = ret_var\n'
        '            mean_metric_dict[f"critic/v_mean{tag}"] = ret_mean - err_mean\n',
    ),
])
