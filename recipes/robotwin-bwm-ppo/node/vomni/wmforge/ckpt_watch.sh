#!/usr/bin/env bash
# WMForge w8: once the pilot logs "SAVED N" for step-N.safetensors, generate the val list with it on
# GPUS and compare against the step-12000 baseline (BASE) with fidelity.py. Output dirs are named
# by run, step and generation seed; bwm_eval.py refuses a dir that holds another run.
#
#   RUN=<pilot dir> BASE=<baseline eval dir> LAST=1000 nohup bash ckpt_watch.sh > <log> 2>&1 &
set -uo pipefail
W=/data/zhengjli/vomni/wmforge
A=/data/zhengjli/vomni/artifacts
RUN=${RUN:?RUN}
BASE=${BASE:?BASE}
LAST=${LAST:-1000}
GPUS=${GPUS:-5 6}
SEED=${SEED:-0}
cd $W
while :; do
  for n in $(grep -oE '^SAVED [0-9]+' "$RUN/train.log" 2>/dev/null | awk '{print $2}'); do
    out=$A/$(basename "$RUN")-val-step$n-s$SEED
    [ -e "$out/fidelity.json" ] && continue
    echo "=== EVAL step $n $(date -u +%FT%TZ)"
    REINDEX=0 GPUS="$GPUS" CKPT="$RUN/step-$n.safetensors" SPLIT=val OUT="$out" bash eval_run.sh | tail -1
    GPU=0 bash gpu_run.sh $W/fidelity.py --index $A/w6/index.json --eval base=$BASE "step$n=$out" \
      --out "$out/fidelity.json" 2>&1 | grep -v -i warn | tr -d '\n ' | head -c 3000
    echo
  done
  [ -e "$A/$(basename "$RUN")-val-step$LAST-s$SEED/fidelity.json" ] && break
  sleep 120
done
echo "=== WATCH DONE $(date -u +%FT%TZ)"
