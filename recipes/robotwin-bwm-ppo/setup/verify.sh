#!/usr/bin/env bash
# Check the assets against the w32 node's sha256 and that the runtime pieces are in place.
#   WL_ROOT=/data/$USER bash $WL_ROOT/worldloop/setup/verify.sh
set -uo pipefail
WL_ROOT=${WL_ROOT:?set WL_ROOT}
WL_ROOT=${WL_ROOT%/}
fail=0
cd "$WL_ROOT"
# step-12000 sits under a cache path; match it by glob as bwm_common.py does.
sed "s#snapshots/[0-9a-f]*/#snapshots/738a8d3c008e637b8b1b18d5e98a82f6de9c04aa/#" worldloop/assets.sha256 \
  | sha256sum -c --quiet || fail=1
for d in rlinf/wt-g rlinf/wt-w11 rlinf/wt-w18 rlinf/wt-r2 vomni/RoboTwin/assets/embodiments vomni/bwm-repo \
         rlinf/venvs/openpi-robotwin/bin vomni/native/pydeps/diffsynth vomni/artifacts/init_frames_hires48; do
  [ -e "$d" ] || { echo "MISSING $d"; fail=1; }
done
grep -q RT_NO_EARLY_STOP vomni/RoboTwin/envs/_base_task.py || { echo "RoboTwin patch not applied"; fail=1; }
grep -rq RLINF_W18_DUMP rlinf/wt-w18/rlinf/envs/sim/world_model/ \
  || { echo "wt-w18 lacks the dump hook"; fail=1; }
docker inspect vkdev >/dev/null 2>&1 || { echo "vkdev container not running"; fail=1; }
[ "$fail" = 0 ] && echo VERIFY_OK || { echo VERIFY_FAILED; exit 1; }
