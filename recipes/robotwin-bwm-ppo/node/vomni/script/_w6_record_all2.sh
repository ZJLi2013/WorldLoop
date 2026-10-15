#!/usr/bin/env bash
# WMForge w6: record every finished rollout pass with _w6_record.py, 2 slots per process, at most
# JOBS processes at once. Main passes record all slots, base-x* passes only the failures. Slots
# whose npz already exists are skipped, so the script can be rerun until nothing is left.
#
#   JOBS=40 nohup bash _w6_record_all.sh > /data/zhengjli/vomni/artifacts/w6/record.log 2>&1 &
set -uo pipefail
ART=/data/zhengjli/vomni/artifacts
OUT=$ART/w6
EP=$OUT/episodes
S=/data/zhengjli/vomni/script
JOBS=${JOBS:-40}
mkdir -p "$EP" "$OUT/rec"
exec 8> "$OUT/record2.lock"
flock -n 8 || { echo "another _w6_record_all.sh is running"; exit 1; }

while :; do
  launched=0
  for done_mark in "$OUT"/*.done; do
    [ -e "$done_mark" ] || continue
    tag=$(basename "$done_mark" .done)
    extra=""
    case $tag in base-x*) extra=--failures-only ;; esac
    for s in $(seq 0 2 46); do
      log=$OUT/rec/${tag}_$s.log
      [ -e "$EP/${tag}_s$(printf %02d $s).npz" ] && [ -e "$EP/${tag}_s$(printf %02d $((s + 1))).npz" ] && continue
      grep -q '^EXIT=' "$log" 2>/dev/null && continue
      [ -e "$log.running" ] && continue
      while [ "$(ls "$OUT"/rec/*.running 2>/dev/null | wc -l)" -ge "$JOBS" ]; do sleep 20; done
      touch "$log.running"
      ( bash $S/_w2_vk.sh $S/_w6_record.py --dump "$ART/w6-dump-$tag" --tag "$tag" --slots $s $((s + 1)) \
          --out "$EP" $extra > "$log" 2>&1; echo "EXIT=$?" >> "$log"; rm -f "$log.running" ) &
      launched=$((launched + 1))
    done
  done
  running=$(ls "$OUT"/rec/*.running 2>/dev/null | wc -l)
  echo "$(date -u +%FT%TZ) launched=$launched running=$running episodes=$(ls "$EP"/*.npz 2>/dev/null | wc -l)"
  if [ "$launched" -eq 0 ] && [ "$running" -eq 0 ] && grep -q "EXTRA ROLLOUTS DONE" "$OUT/collect_extra.log" 2>/dev/null; then
    break
  fi
  sleep 60
done
wait
echo "=== RECORD ALL DONE $(date -u +%FT%TZ) episodes=$(ls "$EP"/*.npz | wc -l)"
grep -L '^DONE' "$OUT"/rec/*.log | head -20 | sed 's/^/FAILED /'
