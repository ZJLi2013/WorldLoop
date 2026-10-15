#!/usr/bin/env bash
# Run a python script from wt-r2 on one GPU, in the same image and venv as the RL runs.
set -euo pipefail

ROOT=/data/zhengjli/rlinf
WT=$ROOT/wt-r2
IMG=rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4
NAME=${NAME:-r2_gpu}

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 32g --network host \
  --security-opt seccomp=unconfined \
  -v /data:/data -w "$WT" \
  -e HF_HUB_OFFLINE=1 -e TORCH_HOME=/data/zhengjli/vomni/torch_home \
  --entrypoint bash "$IMG" -lc "source $ROOT/venvs/openpi-robotwin/bin/activate && export PYTHONPATH=$WT && python $*"
