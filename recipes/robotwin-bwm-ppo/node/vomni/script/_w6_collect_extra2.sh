#!/usr/bin/env bash
# WMForge w6: extra π0.5 base passes over the default adjust_bottle eval seeds (each env.eval.seed
# draws a different 48 of them), where base fails about a third of the time; the hires48 seeds give
# too few failures. Waits for _w6_collect.sh through the same lock.
set -uo pipefail
ROOT=/data/zhengjli/rlinf
ART=/data/zhengjli/vomni/artifacts
CK=/data/zhengjli/vomni/track2/ckpt/fulldata
OUT=$ART/w6
exec 9> "$OUT/collect.lock"
flock 9
export RT_NO_EARLY_STOP=1
for pass in 16 17 18 19 20; do
  tag=base-x$pass
  [ -e "$OUT/$tag.done" ] && continue
  dump=$ART/w6-dump-$tag
  logdir=$ROOT/results/w6-$tag
  docker run --rm -v /data:/data --entrypoint rm busybox -rf "$dump" "$logdir" "$logdir.log"
  EX="++rollout.model.openpi.config_name=pi05_aloha_robotwin"
  EX="$EX ++rollout.model.openpi_data.adapt_to_pi=false"
  EX="$EX ++rollout.model.openpi_data.norm_stats_path=$CK/assets/robotwin_clean"
  EX="$EX ++rollout.model.use_proprio=true"
  EX="$EX ++rollout.model.openpi.action_chunk=50 ++rollout.model.openpi.action_env_dim=14"
  EX="$EX env.eval.task_config.camera.head_camera_type=Large_D435 env.eval.seed=$pass"
  echo "=== ROLLOUT $tag $(date -u +%FT%TZ)"
  REPO=$ROOT/wt-g CKPT="$CK" WRIST=false NIMG=1 \
    ENVS=48 STEPS=200 EPOCH_STEPS=200 PLACEMENT=0-7 NAME="w6_${tag//-/_}" \
    LOGDIR="$logdir" SAVE_VIDEO=False EXTRA="$EX" \
    RLINF_ROBOTWIN_LABEL_DUMP="$dump" \
    bash $ROOT/script/robotwin/03_eval.sh > "$logdir.log" 2>&1
  echo "ROLLOUT_DONE $tag files=$(ls "$dump" 2>/dev/null | wc -l) $(grep -aoE "'eval/success_once': array\([0-9.]+" "$logdir.log" | tail -1) $(date -u +%FT%TZ)"
  [ "$(ls "$dump" 2>/dev/null | wc -l)" -ge 40 ] && touch "$OUT/$tag.done"
done
echo "=== EXTRA2 ROLLOUTS DONE $(date -u +%FT%TZ)"
