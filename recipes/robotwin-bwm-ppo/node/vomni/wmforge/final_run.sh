#!/usr/bin/env bash
# WMForge w8: after the val watcher finishes, pick the checkpoint by the frozen rule
# (pilot_summary.py) and score only that one on the final split against the final baseline.
#
#   nohup bash final_run.sh > /data/zhengjli/vomni/artifacts/w8-final.log 2>&1 &
set -uo pipefail
W=/data/zhengjli/vomni/wmforge
A=/data/zhengjli/vomni/artifacts
cd $W
until grep -q "WATCH DONE" $A/w8-watch.log 2>/dev/null; do sleep 60; done
python3 $W/pilot_summary.py | tee $A/w8-val-summary.txt
n=$(awk '/^PICK/ {print $2}' $A/w8-val-summary.txt)
[ "$n" = None ] && { echo "NO_CHECKPOINT_PASSES_VAL"; exit 0; }
out=$A/w8-pilot-final-step$n-s0
REINDEX=0 GPUS="0 1 2 3 4 5 6" CKPT=$A/w8-pilot/step-$n.safetensors SPLIT=final OUT=$out bash eval_run.sh | tail -1
GPU=0 bash gpu_run.sh $W/fidelity.py --index $A/w6/index.json --eval base=$A/a4f2-base-s0 "step$n=$out" \
  --out $out/fidelity.json 2>&1 | grep -v -i warn | tr -d '\n ' | head -c 3000
echo
echo "FINAL_DONE step$n $(date -u +%FT%TZ)"
