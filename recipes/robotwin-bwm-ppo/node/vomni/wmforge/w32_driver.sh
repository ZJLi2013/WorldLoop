#!/usr/bin/env bash
# WMForge w32: corrected-recipe PPO baseline on srdc-20 (design: rl_stability/w29 + w32 pages).
#   STAGE=pre   bash w32_driver.sh   # tree checks and the eval-tree success-flag path, no GPU
#   STAGE=smoke bash w32_driver.sh   # 5-step smoke on wt-w18, one GPU / one env worker, input evidence
#   STAGE=full  bash w32_driver.sh   # SEED=1..5: config check, 14-step training, 9 evals, mismatch diag, analysis
# Any failed stage stops; .done markers make reruns resume. A watchdog stops the full stage at BUDGET_H.
set -uo pipefail
STAGE=${STAGE:?STAGE}
BUDGET_H=${BUDGET_H:-45}
ROOT=/data/zhengjli/rlinf
R=$ROOT/results
ART=/data/zhengjli/vomni/artifacts
OUT=$ART/w32
SEEDS=$ART/w11/seeds
WMF=/data/zhengjli/vomni/wmforge
S=/data/zhengjli/vomni/script
CK=/data/zhengjli/vomni/track2/ckpt/fulldata
IMG=rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4
VENV=$ROOT/venvs/openpi-robotwin
STEP750=$ART/w8-pilot/step-750.safetensors
CKSUB=bwm_native_step750_ppo_openpi/checkpoints
FIX="++actor.model.openpi.value_after_vlm=True actor.optim.critic_warmup_steps=8 ++algorithm.bootstrap_type=none"
TRAIN_ENV=(CFG=bwm_native_ppo_openpi_t5 CKPT=$CK WM_CKPT=$STEP750 IMG=$IMG VENV=$VENV
           LR=5.6e-6 VLR=1.1e-5)
mkdir -p "$OUT"
exec 9>"$OUT/driver-$STAGE.lock"
flock -n 9 || { echo "ANOTHER_DRIVER_RUNNING"; exit 1; }
log() { echo "$(date +%FT%T) $*"; }
py() {
  docker run --rm --name "w32_py_$$" -v /data:/data -w $WMF --entrypoint bash $IMG \
    -lc "source $VENV/bin/activate && python $*"
}
blobs() {  # tree -> sorted "blob path" list (the trees are copied worktrees, not usable git repos)
  local t=$1
  (cd /tmp && find $t -type f ! -path '*/__pycache__/*' ! -name '*.pyc' ! -path "$t/.git" ! -path "$t/.git/*" -printf '%P\n' \
    | sort | while IFS= read -r f; do echo "$(git hash-object "$t/$f") $f"; done)
}

if [ "$STAGE" = pre ]; then
  blobs $ROOT/wt-w11 > $OUT/blobs_wt-w11.txt
  blobs $ROOT/wt-w18 > $OUT/blobs_wt-w18.txt
  diff <(cut -d' ' -f2- $OUT/blobs_wt-w11.txt) <(cut -d' ' -f2- $OUT/blobs_wt-w18.txt) > $OUT/files_w11_vs_w18.txt
  join -j 2 <(sort -k2 $OUT/blobs_wt-w11.txt) <(sort -k2 $OUT/blobs_wt-w18.txt) | awk '$2 != $3 {print $1}' > $OUT/changed_w11_vs_w18.txt
  echo "W32_PRE wt-w11 files $(wc -l < $OUT/blobs_wt-w11.txt) md5 $(md5sum < $OUT/blobs_wt-w11.txt | cut -c1-12)"
  echo "W32_PRE wt-w18 vs wt-w11: content differs in: $(tr '\n' ' ' < $OUT/changed_w11_vs_w18.txt); file-list diff lines $(wc -l < $OUT/files_w11_vs_w18.txt)"
  echo "== eval tree wt-g: env observation keys handed to the policy"
  grep -n '"main_images"\|"wrist_images"\|"states"\|"task_descriptions"' $ROOT/wt-g/rlinf/envs/sim/robotwin/robotwin_env.py | head -12
  echo "== eval tree wt-g: where success goes"
  grep -n 'success' $ROOT/wt-g/rlinf/envs/sim/robotwin/robotwin_env.py | head -20
  echo "== openpi policy input selection"
  grep -rn 'observation/image\|observation/state\|"prompt"' $ROOT/wt-g/rlinf/models/embodiment/openpi/*.py | head -12
  ls -la $CK/*.safetensors 2>&1 | head -3
  log "=== W32 PRE DONE"
  exit 0
fi

if [ "$STAGE" = smoke ]; then
  SM=$OUT/smoke
  docker run --rm -v /data:/data --entrypoint rm busybox -rf $SM $R/w32-smoke
  mkdir -p $SM
  log "SMOKE start (wt-w18, PLACEMENT=0, 4 envs, 5 steps)"
  env "${TRAIN_ENV[@]}" REPO=$ROOT/wt-w18 PLACEMENT=0 ENVS=4 RE=1 GBS=16 MBS=4 MAX_STEPS=5 SAVE_EVERY=1 SEED=1 \
    NAME=w32_smoke LOGDIR=$R/w32-smoke EXTRA="$FIX" \
    RLINF_W18_DUMP=$SM/w18dump RLINF_WM_TRACE=1 RLINF_WM_DUMP=$SM/wmdump \
    timeout 7200 bash $S/_w11_train.sh > $SM/run.log 2>&1
  log "SMOKE exit=$? $(grep -a RUN_EXIT $R/w32-smoke/train.log | tail -1)"
  grep -a "bootstrap_type=none:" $R/w32-smoke/train.log | head -2 || true
  grep -c "bootstrap_type=none:" $R/w32-smoke/train.log > $SM/bootstrap_lines.txt
  echo "W32_SMOKE_BOOTSTRAP lines=$(cat $SM/bootstrap_lines.txt)"
  grep -aoE "'actor/lr': [0-9.e+-]+" $R/w32-smoke/train.log | head -8
  grep -aE "Global Step|time/step|step_time" $R/w32-smoke/train.log | tail -6
  py w32_smoke_check.py --sft $CK/model.safetensors --run $R/w32-smoke/$CKSUB
  py w32_smoke_inputs.py $SM/wmdump $SM/w18dump $SM/inputs
  log "=== W32 SMOKE DONE"
  exit 0
fi

# ---------------- full ----------------
[ -e $OUT/smoke.passed ] || { echo "SMOKE_NOT_PASSED"; exit 1; }
IDENT=$( (blobs $ROOT/wt-w11; cat "$0" $S/_w11_train.sh; echo "${TRAIN_ENV[*]} $FIX") | md5sum | cut -c1-12)
if [ -e "$OUT/identity.txt" ] && [ "$(cat "$OUT/identity.txt")" != "$IDENT" ]; then
  echo "STALE_RESULTS: $OUT was produced by $(cat "$OUT/identity.txt"), this driver is $IDENT"; exit 1
fi
echo "$IDENT" > "$OUT/identity.txt"
log "=== START w32 full identity=$IDENT budget=${BUDGET_H}h"
(
  exec 9>&-
  sleep $((BUDGET_H * 3600))
  echo "=== BUDGET_EXCEEDED $(date -u +%FT%TZ)"
  for c in $(docker ps --format '{{.Names}}' | grep -E "^w32_"); do docker rm -f "$c"; done
  kill -TERM $$ 2>/dev/null
) &
WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null' EXIT

sim_eval() {  # tag seeds_file ckpt_pt worker_seed
  local tag=$1 seeds=$2 pt=$3 ws=$4
  local dump=$OUT/eval-dump-$tag logdir=$R/w32-eval-$tag
  [ -e "$OUT/eval-$tag.done" ] && return 0
  docker run --rm -v /data:/data --entrypoint rm busybox -rf "$dump" "$logdir" "$logdir.log"
  local EX="++rollout.model.openpi.config_name=pi05_aloha_robotwin ++rollout.model.openpi_data.adapt_to_pi=false"
  EX="$EX ++rollout.model.openpi_data.norm_stats_path=$CK/assets/robotwin_clean ++rollout.model.use_proprio=true"
  EX="$EX ++rollout.model.openpi.action_chunk=50 ++rollout.model.openpi.action_env_dim=14"
  EX="$EX ++rollout.model.openpi.value_after_vlm=True"
  EX="$EX env.eval.task_config.camera.head_camera_type=Large_D435 env.eval.seeds_path=$seeds env.eval.seed=0"
  log "EVAL $tag worker_seed=$ws"
  RLINF_SEED_EVERYTHING=$ws REPO=$ROOT/wt-g CKPT="$CK" CKPT_PT="$pt" WRIST=false NIMG=1 \
    ENVS=48 STEPS=200 EPOCH_STEPS=200 PLACEMENT=0-7 NAME="w32_eval_${tag//-/_}" \
    LOGDIR="$logdir" SAVE_VIDEO=False EXTRA="$EX" RLINF_ROBOTWIN_LABEL_DUMP="$dump" \
    bash $ROOT/script/robotwin/03_eval.sh > "$logdir.log" 2>&1
  local rate exit_line
  rate=$(grep -aoE "'eval/success_once': array\([0-9.]+" "$logdir.log" | tail -1 | grep -oE '[0-9.]+$')
  exit_line=$(grep -a "^RUN_EXIT=" "$logdir.log" | tail -1)
  log "EVAL_DONE $tag success_once=${rate:-NA} $exit_line files=$(ls "$dump" 2>/dev/null | wc -l)"
  [ "$exit_line" = "RUN_EXIT=0" ] && [ -n "${rate:-}" ] || { echo "EVAL_FAILED $tag"; return 1; }
  echo "$rate" > "$OUT/eval-$tag.done"
}

diag() {  # seed ckpt_pt: closed loop in the WM on hires48 0-23, WM successes replayed in the simulator
  local seed=$1 pt=$2 tag=diag-s$1
  [ -e "$OUT/$tag.done" ] && return 0
  local dump=$OUT/$tag-dump logdir=$OUT/$tag-run e1=$OUT/$tag-e1 rep=$OUT/$tag-replay i pids=()
  docker run --rm -v /data:/data --entrypoint rm busybox -rf "$dump" "$logdir" "$e1" "$rep"
  mkdir -p "$rep"
  log "DIAG s$seed"
  env "${TRAIN_ENV[@]}" REPO=$ROOT/wt-w18 PLACEMENT=0-7 SEED=11 ENVS=24 RE=1 GBS=96 MBS=4 MAX_STEPS=1 SAVE_EVERY=999 \
    LR=0 VLR=0 NAME=w32_diag_s$seed LOGDIR="$logdir" RLINF_W18_DUMP="$dump" \
    EXTRA="$FIX env.train.use_fixed_reset_state_ids=True runner.ckpt_path=$pt" \
    timeout 5400 bash $S/_w11_train.sh > "$logdir.log" 2>&1 || { echo "DIAG_WM_FAILED s$seed"; return 1; }
  bash $S/_w2_vk.sh $WMF/w18_to_e1.py --dump "$dump" --out "$e1" --monitor $SEEDS/monitor.json \
    --init $ART/init_frames_hires48 > "$OUT/$tag-convert.log" 2>&1 || { echo "DIAG_CONVERT_FAILED s$seed"; return 1; }
  for ((i = 0; i < 24; i += 2)); do
    ( RT_NO_EARLY_STOP=1 timeout 5400 bash $S/_w2_vk.sh $S/_w6_record.py --dump "$e1" --tag "$tag" \
        --slots $i $((i + 1)) --out "$rep" > "$rep/rec_$i.log" 2>&1 ) &
    pids+=($!)
  done
  wait "${pids[@]}"  # a bare wait would also wait for the budget watchdog
  bash $S/_w2_vk.sh $WMF/w32_mismatch.py "$e1" "$rep" "$tag" "$OUT/$tag.json" | tail -1 \
    && [ -s "$OUT/$tag.json" ] && touch "$OUT/$tag.done"
}

run_seed() {
  local seed=$1 run=$R/w32-ppo-s$1 t0 t1
  local p14=$run/$CKSUB/global_step_14/actor/model_state_dict/full_weights.pt
  if [ ! -e "$OUT/cfg-s$seed.done" ]; then
    env "${TRAIN_ENV[@]}" REPO=$ROOT/wt-w11 PLACEMENT=0-7 ENVS=48 RE=2 GBS=384 MBS=16 MAX_STEPS=14 SAVE_EVERY=14 \
      SEED=$seed NAME=w32_cfg_s$seed LOGDIR=$OUT/cfg-s$seed EXTRA="$FIX --cfg job --resolve" bash $S/_w11_train.sh >/dev/null 2>&1
    py w32_cfgcheck.py $OUT/cfg-s$seed/train.log $seed 14 || { echo "CFG_MISMATCH s$seed"; return 1; }
    touch "$OUT/cfg-s$seed.done"
  fi
  if [ ! -e "$OUT/train-s$seed.done" ]; then
    log "TRAIN s$seed"
    t0=$(date +%s)
    env "${TRAIN_ENV[@]}" REPO=$ROOT/wt-w11 PLACEMENT=0-7 ENVS=48 RE=2 GBS=384 MBS=16 MAX_STEPS=14 SAVE_EVERY=14 \
      SEED=$seed NAME=w32_ppo_s$seed LOGDIR=$run EXTRA="$FIX" bash $S/_w11_train.sh \
      || { echo "TRAIN_FAILED s$seed $(grep -a RUN_EXIT "$run/train.log" | tail -1)"; return 1; }
    [ -e "$p14" ] || { echo "NO_STEP14_CHECKPOINT s$seed"; return 1; }
    t1=$(date +%s)
    echo $((t1 - t0)) > "$OUT/train-s$seed.secs"
    touch "$OUT/train-s$seed.done"
    log "TRAIN_DONE s$seed secs=$((t1 - t0))"
  fi
  for es in 101 202 303; do
    for i in 0 1 2; do
      sim_eval "ppo-s$seed-e$es-confirm$i" "$SEEDS/confirm$i.json" "$p14" "$((es * 100 + i * 10))" || return 1
    done
  done
  diag $seed "$p14" || log "DIAG_SKIPPED s$seed (not part of the verdict)"
}

DONE_SEEDS=""
for seed in ${SEED_LIST:-1 2 3 4 5}; do
  t_run=$(date +%s)
  run_seed $seed || { echo "=== W32 STOPPED s$seed"; exit 1; }
  DONE_SEEDS="$DONE_SEEDS $seed"
  py w32_analyze.py $OUT $ART/w28 $SEEDS $DONE_SEEDS | tail -1
  if [ "$seed" = 1 ] && [ ! -e "$OUT/run1.secs" ]; then
    echo $(( $(date +%s) - t_run )) > "$OUT/run1.secs"
  fi
  if [ "$seed" = 1 ] && [ "$(cat $OUT/run1.secs)" -gt $((9 * 3600)) ] && [ -z "${SEED_LIST:-}" ]; then
    log "RUN1_OVER_9H secs=$(cat $OUT/run1.secs): continuing with seeds 2 3 only, then stop for human"
    for s2 in 2 3; do run_seed $s2 || { echo "=== W32 STOPPED s$s2"; exit 1; }; DONE_SEEDS="$DONE_SEEDS $s2"; done
    py w32_analyze.py $OUT $ART/w28 $SEEDS $DONE_SEEDS | tail -1
    log "=== W32 PARTIAL_3_SEEDS"
    exit 0
  fi
done
log "=== W32 DONE"
