#!/usr/bin/env bash
# Rebuild the node layout the w32 run used under $WL_ROOT: four RLinf trees, RoboTwin, BWM code,
# and the scripts, with every /data/zhengjli path rewritten to $WL_ROOT.
#
#   WL_ROOT=/data/$USER bash setup/materialize.sh            # wt-g, wt-w11, wt-w18, wt-r2
#   WL_ROOT=/data/$USER WITH_F24=1 bash setup/materialize.sh # plus wt-f24 (only to retrain F21 for w6)
#
# Containers mount $WL_MOUNT (default: first path component of $WL_ROOT), so $WL_ROOT must live under it.
set -euo pipefail
WL_ROOT=${WL_ROOT:?set WL_ROOT, e.g. /data/$USER}
WL_ROOT=${WL_ROOT%/}
WL_MOUNT=${WL_MOUNT:-/$(echo "$WL_ROOT" | cut -d/ -f2)}
RECIPE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TP=$(cd "$RECIPE/../.." && pwd)/third_party

RLINF_BASE=bde6c918642abf9a4776cb1d5fabcc5087dfe195
ROBOTWIN_BASE=0008ae6
BWM_BASE=cc0bc74
# git write-tree of each patched tree, before the path rewrite; equal ids = byte-identical trees.
declare -A TREE_ID=(
  [wt-g]=dedf924e8223bbb52cfb8c2f5c70f120f4b1667a
  [wt-w11]=df131826301d3438691b9bf12df31eb5d771c98c
  [wt-r2]=3db0e5c0cfbd8a37a89e2f7c4f11ad53e6e80064
  [wt-f24]=67a41532c628d6dcc403c012427028de7dcfb3bf
)
TREES="wt-g wt-w11 wt-r2"
[ "${WITH_F24:-0}" = 1 ] && TREES="$TREES wt-f24"

for sm in RLinf RoboTwin boundless-world-model; do
  [ -e "$TP/$sm/.git" ] || { echo "missing submodule $sm: git submodule update --init"; exit 1; }
done
mkdir -p "$WL_ROOT/rlinf/results" "$WL_ROOT/vomni"

rewrite() {
  grep -rlI --exclude-dir=.git --exclude=.git --exclude-dir=assets -e /data/zhengjli -e '/data:/data' "$@" 2>/dev/null \
    | xargs -r sed -i -e "s#/data/zhengjli#$WL_ROOT#g" -e "s#/data:/data#$WL_MOUNT:$WL_MOUNT#g"
}

for t in $TREES; do
  d=$WL_ROOT/rlinf/$t
  if [ -e "$d" ]; then echo "keep $d"; continue; fi
  git -C "$TP/RLinf" worktree add --detach "$d" "$RLINF_BASE"
  git -C "$d" apply --binary --whitespace=nowarn "$RECIPE/patches/rlinf-$t.patch"
  got=$(git -C "$d" add -A && git -C "$d" write-tree)
  [ "$got" = "${TREE_ID[$t]}" ] || { echo "TREE_MISMATCH $t got $got want ${TREE_ID[$t]}"; exit 1; }
  echo "TREE_OK $t $got"
  rewrite "$d"
done

if [ ! -e "$WL_ROOT/vomni/RoboTwin" ]; then
  git -C "$TP/RoboTwin" worktree add --detach "$WL_ROOT/vomni/RoboTwin" "$ROBOTWIN_BASE"
  git -C "$WL_ROOT/vomni/RoboTwin" apply "$RECIPE/patches/robotwin-0008ae6.patch"
fi
[ -e "$WL_ROOT/vomni/bwm-repo" ] || git -C "$TP/boundless-world-model" worktree add --detach "$WL_ROOT/vomni/bwm-repo" "$BWM_BASE"

cp -r "$RECIPE/node/." "$WL_ROOT/"
rewrite "$WL_ROOT/rlinf/script" "$WL_ROOT/vomni/script" "$WL_ROOT/vomni/wmforge" "$WL_ROOT/vomni/d0a" \
  "$WL_ROOT/vomni/robotwin_rocm" "$WL_ROOT/vomni/artifacts/w6" "$WL_ROOT/vomni/artifacts/w11"
mkdir -p "$WL_ROOT/worldloop"
cp -r "$RECIPE/setup" "$RECIPE/env" "$RECIPE/assets.sha256" "$WL_ROOT/worldloop/"

# wt-w18 = wt-w11 + the per-chunk dump hook; only the smoke and the mismatch diagnostic use it.
if [ ! -e "$WL_ROOT/rlinf/wt-w18" ]; then
  cp -a "$WL_ROOT/rlinf/wt-w11" "$WL_ROOT/rlinf/wt-w18"
  rm -f "$WL_ROOT/rlinf/wt-w18/.git"
  python3 "$WL_ROOT/vomni/wmforge/rlinf_patch/w18_dump.py" "$WL_ROOT/rlinf/wt-w18"
fi

if [ "$WL_ROOT" != /data/zhengjli ]; then
  left=$(grep -rhoI --exclude-dir=.git --exclude=.git --exclude-dir=assets '/data/zhengjli[A-Za-z0-9_./-]*' \
    "$WL_ROOT/rlinf" "$WL_ROOT/vomni" 2>/dev/null | grep -vc "^$WL_ROOT" || true)
  [ "$left" = 0 ] || { echo "UNREWRITTEN_PATHS $left"; exit 1; }
fi
echo "MATERIALIZED $WL_ROOT (mount $WL_MOUNT)"
