#!/usr/bin/env bash
# WMForge A4 / G1-G2: index the recorded w6 episodes, take one split's evaluation set (every
# failure, successes with crc32(key) % 3 == 0; frozen in w6_index.py), and run bwm_eval.py on
# GPUS. The list is copied into OUT so a directory always belongs to one checkpoint and one list;
# reruns skip episodes already generated.
#
#   CKPT=<safetensors|base> OUT=<dir> SPLIT=val|final bash eval_run.sh
set -uo pipefail
W=/data/zhengjli/vomni/wmforge
A=/data/zhengjli/vomni/artifacts
CKPT=${CKPT:-base}
OUT=${OUT:?OUT}
SPLIT=${SPLIT:-val}
GPUS=${GPUS:-0 1 2 3 4 5 6}  # GPU 7 is held by another project's LLM server
cd $W
if [ "${REINDEX:-1}" = 1 ]; then
  GPU=0 bash gpu_run.sh $W/w6_index.py --episodes $A/w6/episodes --out $A/w6/index.json \
    --eval-list $A/w6/eval_list.json 2>&1 | grep -A40 -E "SUMMARY|EVAL_LIST"
fi
LIST=$A/w6/eval_list_$SPLIT.json
mkdir -p "$OUT"
cp "$LIST" "$OUT/list.json"
EXTRA=""
[ "$CKPT" != base ] && EXTRA="--ckpt $CKPT"
n=$(echo $GPUS | wc -w)
i=0
for g in $GPUS; do
  GPU=$g nohup bash gpu_run.sh $W/bwm_eval.py --list "$OUT/list.json" --shard $i --nshard $n \
    --out "$OUT" $EXTRA > "$OUT/eval_$i.log" 2>&1 < /dev/null &
  i=$((i + 1))
done
wait
echo "GENERATED $(ls "$OUT"/*_s??.npz | wc -l) of $(python3 -c "import json;print(len(json.load(open('$OUT/list.json'))))")"
