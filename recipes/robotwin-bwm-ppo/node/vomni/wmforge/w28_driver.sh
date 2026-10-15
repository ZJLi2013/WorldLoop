#!/usr/bin/env bash
# WMForge w28: SFT baseline on srdc-20, 144 sealed confirmation seeds x eval seeds 101/202/303, with the
# w14 protocol. Three gates run before the full sweep: eval-tree seeding, a repeated seeded eval with
# identical labels, and a real input sample for confirmation seed 369. Any failure stops the driver.
#   bash w28_driver.sh > /data/zhengjli/vomni/artifacts/w28-run.log 2>&1
set -uo pipefail
ROOT=/data/zhengjli/rlinf
R=$ROOT/results
ART=/data/zhengjli/vomni/artifacts
OUT=$ART/w28
SEEDS=$ART/w11/seeds
WMF=/data/zhengjli/vomni/wmforge
CK=/data/zhengjli/vomni/track2/ckpt/fulldata
IMG=rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4
VENV=$ROOT/venvs/openpi-robotwin
mkdir -p "$OUT"
exec 9>"$OUT/driver.lock"
flock -n 9 || { echo "ANOTHER_DRIVER_RUNNING"; exit 1; }
log() { echo "$(date +%FT%T) $*"; }

# Gate 1: the eval tree seeds rollout workers from RLINF_SEED_EVERYTHING (+ rank) and 03_eval.sh passes it.
grep -q RLINF_SEED_EVERYTHING $ROOT/wt-g/rlinf/workers/rollout/hf/huggingface_worker.py || { echo "W28_GATE1 FAIL wt-g not seedable"; exit 1; }
grep -n "RLINF_SEED_EVERYTHING" $ROOT/wt-g/rlinf/workers/rollout/hf/huggingface_worker.py | head -3
grep -q RLINF_SEED_EVERYTHING $ROOT/script/robotwin/03_eval.sh || { echo "W28_GATE1 FAIL 03_eval.sh does not pass the seed"; exit 1; }
for i in 0 1 2; do [ -s $SEEDS/confirm$i.json ] || { echo "W28_GATE1 FAIL missing confirm$i.json"; exit 1; }; done
log "W28_GATE1 PASS"

py() {
  docker run --rm --name w28_tools -v /data:/data -w $WMF --entrypoint bash $IMG \
    -lc "source $VENV/bin/activate && python $*"
}
sim_eval() {  # tag seeds_file worker_seed (rank is added inside the workers)
  local tag=$1 seeds=$2 es=$3
  local dump=$OUT/eval-dump-$tag logdir=$R/w28-eval-$tag
  [ -e "$OUT/eval-$tag.done" ] && return 0
  docker run --rm -v /data:/data --entrypoint rm busybox -rf "$dump" "$logdir" "$logdir.log"
  local EX="++rollout.model.openpi.config_name=pi05_aloha_robotwin ++rollout.model.openpi_data.adapt_to_pi=false"
  EX="$EX ++rollout.model.openpi_data.norm_stats_path=$CK/assets/robotwin_clean ++rollout.model.use_proprio=true"
  EX="$EX ++rollout.model.openpi.action_chunk=50 ++rollout.model.openpi.action_env_dim=14"
  EX="$EX env.eval.task_config.camera.head_camera_type=Large_D435 env.eval.seeds_path=$seeds env.eval.seed=0"
  log "EVAL $tag worker_seed=$es"
  RLINF_SEED_EVERYTHING=$es REPO=$ROOT/wt-g CKPT="$CK" CKPT_PT="" WRIST=false NIMG=1 \
    ENVS=48 STEPS=200 EPOCH_STEPS=200 PLACEMENT=0-7 NAME="w28_eval_${tag//-/_}" \
    LOGDIR="$logdir" SAVE_VIDEO=False EXTRA="$EX" RLINF_ROBOTWIN_LABEL_DUMP="$dump" \
    bash $ROOT/script/robotwin/03_eval.sh > "$logdir.log" 2>&1
  local rate exit_line
  rate=$(grep -aoE "'eval/success_once': array\([0-9.]+" "$logdir.log" | tail -1 | grep -oE '[0-9.]+$')
  exit_line=$(grep -a "^RUN_EXIT=" "$logdir.log" | tail -1)
  log "EVAL_DONE $tag success_once=${rate:-NA} $exit_line files=$(ls "$dump" 2>/dev/null | wc -l)"
  [ "$exit_line" = "RUN_EXIT=0" ] && [ -n "${rate:-}" ] || { echo "EVAL_FAILED $tag"; return 1; }
  echo "$rate" > "$OUT/eval-$tag.done"
}
wseed() { echo $(( $1 * 100 + $2 * 10 )); }

# Gate 2: the same seeded eval twice gives the same 48 seeds with identical labels.
sim_eval "sft-e101-confirm0" "$SEEDS/confirm0.json" "$(wseed 101 0)" || exit 1
sim_eval "rep-sft-e101-confirm0" "$SEEDS/confirm0.json" "$(wseed 101 0)" || exit 1
py w28_tools.py repeat $OUT/eval-dump-sft-e101-confirm0 $OUT/eval-dump-rep-sft-e101-confirm0 || { echo "=== W28 REPEAT_MISMATCH"; exit 1; }

# Gate 3: the input actually given to the policy for confirmation seed 369.
py w28_tools.py sample $OUT/eval-dump-sft-e101-confirm0 $ART/w11/confirm_seeds.json 369 $OUT/sample || { echo "=== W28 SAMPLE_FAILED"; exit 1; }
log "W28_GATES PASS"

for es in 101 202 303; do
  for i in 0 1 2; do
    sim_eval "sft-e$es-confirm$i" "$SEEDS/confirm$i.json" "$(wseed $es $i)" || { echo "=== W28 STOPPED"; exit 1; }
  done
done
py w28_tools.py analyze $OUT $SEEDS || { echo "=== W28 ANALYSIS_FAILED"; exit 1; }
log "=== W28 DONE"
