#!/usr/bin/env bash
# T1: export RoboTwin reset observations as BWM init frames.
# Same container and exports as 03_eval.sh -- rendering needs the lavapipe shim, which rides
# in on the venv's .pth and is asserted before the run rather than after a 4-minute SIGSEGV.
#
#   NUM_ENVS=2 bash robotwin/04_export_init.sh     # smoke
#   NUM_ENVS=48 bash robotwin/04_export_init.sh
set -u

ROOT=${ROOT:-/data/zhengjli/rlinf}
IMG=${IMG:-rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4}
REPO=${REPO:-$ROOT/wt-f24}
VENV=${VENV:-$ROOT/venvs/openpi-robotwin}
ROBOTWIN_PATH=${ROBOTWIN_PATH:-/data/zhengjli/vomni/RoboTwin}
SCRIPT=${SCRIPT:-/data/zhengjli/vomni/script/_t1_export_init.py}

NUM_ENVS=${NUM_ENVS:-2}
OUT=${OUT:-/data/zhengjli/vomni/artifacts/init_frames_n${NUM_ENVS}}
REF=${REF:-/data/zhengjli/vomni/artifacts/init_frames}
CFG=${CFG:-rocm_adjust_bottle_pi0_smoke}
# Set to BWM's own demo wording to take RoboTwin's per-episode phrasing out of the picture.
INSTRUCTION=${INSTRUCTION:-}
NAME=${NAME:-rlinf_export_init}
TIMEOUT=${TIMEOUT:-3600}

docker rm -f "$NAME" >/dev/null 2>&1
echo "NUM_ENVS=$NUM_ENVS OUT=$OUT CFG=$CFG"

timeout "$TIMEOUT" docker run --rm --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 64g --network host \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  -v /data:/data \
  -w "$REPO" \
  -e HF_HUB_OFFLINE=1 -e HYDRA_FULL_ERROR=1 \
  -e INIT_INSTRUCTION="$INSTRUCTION" \
  --entrypoint bash "$IMG" -lc "
source $VENV/bin/activate
export REPO_PATH=$REPO
export EMBODIED_PATH=$REPO/examples/embodiment
export ROBOTWIN_PATH=$ROBOTWIN_PATH
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO:$ROBOTWIN_PATH:/data/zhengjli/vomni/robotwin_rocm:\${PYTHONPATH:-}
python -c 'import sys, sapien.render; assert \"sapien_lavapipe\" in sys.modules, \"lavapipe shim not imported\"; print(\"lavapipe shim OK\")'
python $SCRIPT --num-envs $NUM_ENVS --out $OUT --ref $REF --cfg $CFG
"
echo "RUN_EXIT=$?"
docker rm -f "$NAME" >/dev/null 2>&1
