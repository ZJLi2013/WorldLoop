#!/bin/bash
# feature25 N0.2: Track 2 ships the OpenPI PyTorch layout (paligemma_with_expert.*), so RLinf's
# own convertor takes it as-is. Runs inside vkdev because the venv's python symlink only
# resolves there, and through `source activate` because the lavapipe ICD lines live in it.
set -uo pipefail

SRC=${SRC:-/data/zhengjli/vomni/track2/ckpt/fulldata}
# /data/zhengjli/ckpt is root-owned; keep converted weights next to the downloaded ones.
OUT=${OUT:-/data/zhengjli/vomni/track2/pi0_rlinf_fulldata}
REPO=${REPO:-/data/zhengjli/rlinf/wt-f24}
VENV=${VENV:-/data/zhengjli/rlinf/venvs/openpi-robotwin}
REF=${REF:-}

mkdir -p "$OUT"
REF_ARG=""
[ -n "$REF" ] && REF_ARG="--reference-model $REF"

docker exec -u "$(id -u):$(id -g)" -e HOME=/tmp vkdev bash -c "
source $VENV/bin/activate
cd $REPO
python -m rlinf.utils.ckpt_convertor.openpi.convert \
  --mode openpi_pytorch_to_openpi_rlinf \
  --input-model       $SRC \
  --input-norm-stats  $SRC/assets/robotwin_clean/norm_stats.json \
  --output-model      $OUT \
  --output-norm-stats $OUT/physical-intelligence/robotwin/norm_stats.json \
  $REF_ARG
"
echo "=== result ==="
ls -la "$OUT" 2>/dev/null
du -sh "$OUT" 2>/dev/null
