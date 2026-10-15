#!/usr/bin/env bash
# WMForge w8 pilot: re-index the recorded w6 episodes, then fine-tune step-12000 on GPUs 0-6
# (7 ranks x accum 9 = effective batch 63) for STEPS steps, saving every 250.
#
#   OUT=/data/zhengjli/vomni/artifacts/w8-pilot nohup bash pilot_run.sh > <log> 2>&1 &
set -uo pipefail
W=/data/zhengjli/vomni/wmforge
A=/data/zhengjli/vomni/artifacts
OUT=${OUT:?OUT}
STEPS=${STEPS:-1000}
cd $W
GPU=0 bash gpu_run.sh $W/w6_index.py --episodes $A/w6/episodes --out $A/w6/index.json \
  --eval-list $A/w6/eval_list.json 2>&1 | grep -A40 -E "SUMMARY|EVAL_LIST"
mkdir -p "$OUT"
cp $A/w6/index.json "$OUT/index.json"
GPU=0,1,2,3,4,5,6 bash gpu_run.sh -m torch.distributed.run --nproc_per_node 7 --master_port 29612 \
  $W/bwm_train.py --index "$OUT/index.json" --out "$OUT" --steps $STEPS --accum 9 \
  --save-every 250 --val-every 125
echo "PILOT_EXIT=$?"
