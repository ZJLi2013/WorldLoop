#!/usr/bin/env bash
# Run a RoboTwin CPU script inside vkdev with 03_eval.sh's environment: bash _w2_vk.sh <script.py> [args...]
set -uo pipefail
REPO=${REPO:-/data/zhengjli/rlinf/wt-g}
RT=/data/zhengjli/vomni/RoboTwin
docker exec -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" -w "$REPO" vkdev bash -lc "
source /data/zhengjli/rlinf/venvs/openpi-robotwin/bin/activate
export REPO=$REPO REPO_PATH=$REPO EMBODIED_PATH=$REPO/examples/embodiment ROBOTWIN_PATH=$RT
export ROBOT_PLATFORM=ALOHA MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export PYTHONPATH=$REPO:$RT:/data/zhengjli/vomni/robotwin_rocm:\${PYTHONPATH:-}
python $*" 2>&1 | grep -v -E 'Warning|warn\('
