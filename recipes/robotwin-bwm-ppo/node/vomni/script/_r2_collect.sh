#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/zhengjli/rlinf
WT=$ROOT/wt-r2
CK=/data/zhengjli/vomni/track2/ckpt/fulldata
ART=/data/zhengjli/vomni/artifacts
TAG=${TAG:?set TAG=base|orig5|ignore5|hist10}
EPOCH_STEPS=${EPOCH_STEPS:-2000}
SUFFIX=${SUFFIX:-}

case "$TAG" in
  base) CKPT_PT="" ;;
  orig5) CKPT_PT=$ROOT/results/f22-orig-step5/bwm_robotwin_ppo_openpi/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt ;;
  ignore5) CKPT_PT=$ROOT/results/f22-ignore-step5/bwm_robotwin_ppo_openpi/checkpoints/global_step_5/actor/model_state_dict/full_weights.pt ;;
  hist10) CKPT_PT=$ROOT/results/exp16-hires48-step10/bwm_robotwin_ppo_openpi/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt ;;
  *) echo "unknown TAG=$TAG" >&2; exit 2 ;;
esac
[[ -z "$CKPT_PT" ]] || test -f "$CKPT_PT"

NAME=r2-collect-$TAG$SUFFIX
LOGDIR=$ROOT/results/$NAME
DUMP=$ART/$NAME
test ! -e "$LOGDIR"
test ! -e "$DUMP"

EX="++rollout.model.openpi.config_name=pi05_aloha_robotwin"
EX="$EX ++rollout.model.openpi_data.adapt_to_pi=false"
EX="$EX ++rollout.model.openpi_data.norm_stats_path=$CK/assets/robotwin_clean"
EX="$EX ++rollout.model.use_proprio=true"
EX="$EX ++rollout.model.openpi.action_chunk=50 ++rollout.model.openpi.action_env_dim=14"
EX="$EX ++env.eval.seeds_path=$WT/rlinf/envs/sim/robotwin/seeds/train_seeds.json"

RLINF_ROBOTWIN_LABEL_DUMP="$DUMP" REPO="$WT" \
  CKPT="$CK" CKPT_PT="$CKPT_PT" WRIST=false NIMG=1 \
  ENVS=48 STEPS=200 EPOCH_STEPS="$EPOCH_STEPS" PLACEMENT=0-7 NAME="${NAME//-/_}" \
  LOGDIR="$LOGDIR" SAVE_VIDEO=False EXTRA="$EX" \
  bash "$ROOT/script/robotwin/03_eval.sh" 2>&1 | tee "$LOGDIR.log"
grep -q "RUN_EXIT=0" "$LOGDIR.log"
echo "DUMP_FILES=$(ls "$DUMP" | wc -l)"
