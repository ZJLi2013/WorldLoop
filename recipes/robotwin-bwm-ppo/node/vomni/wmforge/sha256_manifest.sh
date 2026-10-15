#!/usr/bin/env bash
# Usage: sha256_manifest.sh <out-file>; hashes the non-regenerable WMForge assets under /data/zhengjli.
set -euo pipefail
cd /data/zhengjli
P=bwm_native_step750_ppo_openpi/checkpoints/global_step_10/actor/model_state_dict
paths=(
  vomni/artifacts/init_frames_hires48 vomni/artifacts/w11 vomni/artifacts/r2-reward-model
  vomni/ckpt/t5-base models/pi0-tokenizer vomni/artifacts/w8-pilot/step-750.safetensors
  vomni/track2/ckpt/fulldata vomni/models/BWM
)
for r in w11-g5 w16-ppo-s5 w12-s2-step750 w16-ppo-s3 w16-ppo-s4 w11-g5-basewm w12-s2-basewm; do
  paths+=("rlinf/results/$r/$P")
done
find "${paths[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$1.tmp"
mv "$1.tmp" "$1"
echo "MANIFEST_DONE $(wc -l < "$1") files"

