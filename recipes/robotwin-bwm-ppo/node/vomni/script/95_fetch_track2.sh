#!/bin/bash
# feature25 N0: fetch WorldArena Track 2 dataset + Pi0 checkpoint.
# Their scripts/download_*.sh assume the pre-1.0 `huggingface-cli download` syntax; the only
# CLI on this node is hub 1.27 inside bwm_serve, so the same four arguments go to `hf download`.
# Run as the calling uid so the extracted tree does not come out root-owned.
set -uo pipefail

DEST=${DEST:-/data/zhengjli/vomni/track2}
VARIANTS=${VARIANTS:-fulldata}
CTR=${CTR:-bwm_serve}

mkdir -p "$DEST"
# bwm_serve runs with HF_HUB_OFFLINE=1, which blocks every resolve call.
DEX="docker exec -u $(id -u):$(id -g) -e HF_HOME=/data/zhengjli/hf -e HOME=/tmp \
  -e HF_HUB_OFFLINE=0 -e HF_HUB_ENABLE_HF_TRANSFER=0 $CTR hf download"

if [ -d "$DEST/dataset" ]; then
  echo "SKIP dataset: $DEST/dataset exists"
else
  echo "=== dataset.tar.gz ==="
  $DEX WorldArena/WorldArena_Robotwin2.0 dataset.tar.gz --repo-type dataset --local-dir "$DEST" \
    && tar -xzf "$DEST/dataset.tar.gz" -C "$DEST" || echo "FAIL dataset"
fi

for v in $VARIANTS; do
  if [ -f "$DEST/ckpt/$v/model.safetensors" ]; then
    echo "SKIP ckpt $v"
    continue
  fi
  echo "=== checkpoint $v ==="
  $DEX WorldArena/WorldArena "$v/model.safetensors" "$v/metadata.pt" \
    "$v/assets/robotwin_clean/norm_stats.json" --local-dir "$DEST/ckpt" || echo "FAIL $v"
done

echo "=== inventory ==="
for d in actions images instructions states; do
  printf '%s\t%s\n' "$d" "$(ls "$DEST/dataset/$d/fixed_scene_task" 2>/dev/null | wc -l)"
done
cat "$DEST"/dataset/instructions/fixed_scene_task/episode1.json 2>/dev/null; echo
du -sh "$DEST"/dataset "$DEST"/ckpt/* 2>/dev/null
