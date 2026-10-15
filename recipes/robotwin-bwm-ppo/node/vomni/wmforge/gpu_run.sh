#!/usr/bin/env bash
# Run a WMForge python script on one GPU inside vkdev with the native BWM stack.
#   GPU=0 bash gpu_run.sh <script.py> [args...]
set -uo pipefail
GPU=${GPU:-0}
docker exec -u "$(id -u):109" -e HOME=/tmp -e USER="$(id -un)" -e HIP_VISIBLE_DEVICES="$GPU" \
  -e PYTHONPATH=/data/zhengjli/vomni/native/pydeps:/data/zhengjli/vomni/bwm-repo \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True vkdev bash -lc "
source /data/zhengjli/rlinf/venvs/openpi-robotwin/bin/activate
python $*" 2>&1 | grep -v -E 'Warning|warn\('
