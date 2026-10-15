#!/usr/bin/env bash
# WMForge w6: 4 policies x 3 passes over the 48 hires48 seeds (exp36 S1's run, one episode per
# slot), each pass replayed step by step by _w6_record.py into BWM training episodes.
# Rollouts run one after another on GPUs 0-7; each pass's recording runs in the background on CPU.
#
#   nohup bash _w6_collect.sh > /data/zhengjli/vomni/artifacts/w6/collect.log 2>&1 &
set -uo pipefail
ROOT=/data/zhengjli/rlinf
ART=/data/zhengjli/vomni/artifacts
CK=/data/zhengjli/vomni/track2/ckpt/fulldata
F21=$ROOT/results/f21-20step/sim_robotwin_ppo_openpi/checkpoints
OUT=$ART/w6
SCRIPT=/data/zhengjli/vomni/script
mkdir -p "$OUT/episodes"
exec 9> "$OUT/collect.lock"
flock -n 9 || { echo "another _w6_collect.sh is running"; exit 1; }
grep -q CKPT_PT $ROOT/script/robotwin/03_eval.sh || { echo "03_eval.sh lacks CKPT_PT"; exit 1; }
python3 /data/zhengjli/vomni/d0a/_e1_patch_robotwin_dump.py $ROOT/wt-g/rlinf/envs/sim/robotwin/robotwin_env.py
python3 $SCRIPT/_w6_patch_no_early_stop.py /data/zhengjli/vomni/RoboTwin/envs/_base_task.py || exit 1
grep -q RT_NO_EARLY_STOP $ROOT/script/robotwin/03_eval.sh || { echo "03_eval.sh does not pass RT_NO_EARLY_STOP"; exit 1; }
export RT_NO_EARLY_STOP=1

record() {  # dump tag: 24 processes x 2 slots
  local dump=$1 tag=$2
  for s in $(seq 0 2 46); do
    bash $SCRIPT/_w2_vk.sh $SCRIPT/_w6_record.py --dump "$dump" --tag "$tag" --slots $s $((s + 1)) \
      --out "$OUT/episodes" > "$OUT/rec_${tag}_$s.log" 2>&1 &
  done
  wait
  echo "RECORDED $tag $(grep -h '^ROW' $OUT/rec_${tag}_*.log | wc -l) $(date -u +%FT%TZ)"
}

for pol in base step10 step20 step30; do
  for pass in 0 1 2; do
    tag=$pol-p$pass
    [ -e "$OUT/$tag.done" ] && continue
    dump=$ART/w6-dump-$tag
    logdir=$ROOT/results/w6-$tag
    docker run --rm -v /data:/data --entrypoint rm busybox -rf "$dump" "$logdir" "$logdir.log"
    pt=""
    [ "$pol" != base ] && pt=$F21/global_step_${pol#step}/actor/model_state_dict/full_weights.pt
    EX="++rollout.model.openpi.config_name=pi05_aloha_robotwin"
    EX="$EX ++rollout.model.openpi_data.adapt_to_pi=false"
    EX="$EX ++rollout.model.openpi_data.norm_stats_path=$CK/assets/robotwin_clean"
    EX="$EX ++rollout.model.use_proprio=true"
    EX="$EX ++rollout.model.openpi.action_chunk=50 ++rollout.model.openpi.action_env_dim=14"
    EX="$EX env.eval.task_config.camera.head_camera_type=Large_D435"
    EX="$EX env.eval.seeds_path=/data/zhengjli/vomni/exp35/exp36_seeds.json env.eval.seed=$pass"
    echo "=== ROLLOUT $tag ckpt_pt=${pt:-none} $(date -u +%FT%TZ)"
    REPO=$ROOT/wt-g CKPT="$CK" CKPT_PT="$pt" WRIST=false NIMG=1 \
      ENVS=48 STEPS=200 EPOCH_STEPS=200 PLACEMENT=0-7 NAME="w6_${tag//-/_}" \
      LOGDIR="$logdir" SAVE_VIDEO=False EXTRA="$EX" \
      RLINF_ROBOTWIN_LABEL_DUMP="$dump" \
      bash $ROOT/script/robotwin/03_eval.sh > "$logdir.log" 2>&1
    echo "ROLLOUT_DONE $tag files=$(ls "$dump" 2>/dev/null | wc -l) $(grep -aoE "'eval/success_once': array\([0-9.]+" "$logdir.log" | tail -1) $(date -u +%FT%TZ)"
    if [ "$(ls "$dump" 2>/dev/null | wc -l)" -lt 40 ]; then echo "BAD_DUMP $tag"; continue; fi
    (record "$dump" "$tag" && touch "$OUT/$tag.done") &
  done
done
wait
echo "=== ALL DONE $(date -u +%FT%TZ) episodes=$(ls $OUT/episodes/*.npz | wc -l)"
