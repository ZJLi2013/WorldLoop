#!/usr/bin/env bash
# F21: train pi0.5 with the real RoboTwin simulator as the env and its privileged state as the
# reward. 05_train_bwm.sh minus the world model and the scorer -- no server, no init frames.
#
# Prerequisites are assembled by robotwin/00_setup.sh; run its `check` if this fails early.
#
#   ENVS=48 RE=1 GBS=192 MBS=8 MAX_STEPS=1 bash robotwin/05_train_sim.sh
set -u

ROOT=${ROOT:-/data/zhengjli/rlinf}
IMG=${IMG:-rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4}
REPO=${REPO:-$ROOT/wt-f24}
VENV=${VENV:-$ROOT/venvs/openpi-robotwin}
CFG=${CFG:-sim_robotwin_adjust_bottle_ppo_openpi}
CKPT=${CKPT:-/data/zhengjli/vomni/track2/ckpt/fulldata}
ROBOTWIN_PATH=${ROBOTWIN_PATH:-/data/zhengjli/vomni/RoboTwin}

PLACEMENT=${PLACEMENT:-0-7}
ENVS=${ENVS:-48}
RE=${RE:-1}
GBS=${GBS:-192}
MBS=${MBS:-8}
LR=${LR:-5.6e-6}
MAX_STEPS=${MAX_STEPS:-20}
SAVE_EVERY=${SAVE_EVERY:-10}
NAME=${NAME:-rlinf_sim_robotwin}
LOGDIR=${LOGDIR:-$ROOT/results/f21-sim-$(date +%Y%m%d-%H%M%S)}

mkdir -p "$LOGDIR"
docker rm -f "$NAME" >/dev/null 2>&1

# hydra's override lexer rejects the comma in component_placement, so that one key is derived.
SRC=$REPO/examples/embodiment/config/$CFG.yaml
E2EDIR=$REPO/tests/e2e_tests/embodied
RUNCFG=${CFG}_p${PLACEMENT}
[ -f "$SRC" ] || { echo "NO_SUCH_CFG: $SRC"; exit 1; }
sed -e "s#actor, env, rollout: .*#actor, env, rollout: $PLACEMENT#" "$SRC" > "$E2EDIR/$RUNCFG.yaml"
grep -q "actor, env, rollout: $PLACEMENT" "$E2EDIR/$RUNCFG.yaml" || { echo "BAD_DERIVED_CFG"; exit 1; }

OVERRIDES="actor.model.model_path=$CKPT rollout.model.model_path=$CKPT"
OVERRIDES="$OVERRIDES env.train.rollout_epoch=$RE env.train.total_num_envs=$ENVS"
OVERRIDES="$OVERRIDES actor.global_batch_size=$GBS actor.micro_batch_size=$MBS"
OVERRIDES="$OVERRIDES actor.optim.lr=$LR"
OVERRIDES="$OVERRIDES runner.max_steps=$MAX_STEPS runner.save_interval=$SAVE_EVERY"
OVERRIDES="$OVERRIDES runner.logger.log_path=$LOGDIR"
# The runner takes its step count from max_epochs, so max_steps alone cannot cap a smoke at one.
[ "$MAX_STEPS" = 1 ] && OVERRIDES="$OVERRIDES runner.max_epochs=1"
# Scoring goes through 03_eval.sh instead, so eval envs are never built during training.
OVERRIDES="$OVERRIDES runner.val_check_interval=${VAL_EVERY:--1}"
[ -n "${RESUME:-}" ] && OVERRIDES="$OVERRIDES runner.resume_dir=$RESUME"
OVERRIDES="$OVERRIDES ${EXTRA:-}"

echo "LOGDIR=$LOGDIR"
echo "CFG=$RUNCFG ENVS=$ENVS RE=$RE GBS=$GBS MBS=$MBS MAX_STEPS=$MAX_STEPS SAVE_EVERY=$SAVE_EVERY"
echo "CKPT=$CKPT RESUME=${RESUME:-none}"
echo "CODE=$(git -C "$REPO" rev-parse --short HEAD)"

# A ray actor that dies without a traceback leaves its reason only in its own worker log,
# and --rm takes those with the container. Bind them out.
mkdir -p "$LOGDIR/ray"

docker run --rm --name "$NAME" \
  -v "$LOGDIR/ray":/tmp/ray \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 64g --network host \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  -v /data:/data \
  -w "$REPO" \
  -e HF_HUB_OFFLINE=1 -e HYDRA_FULL_ERROR=1 \
  --entrypoint bash "$IMG" -lc "
# activate, not the interpreter path: this is where the lavapipe ICD exports live.
source $VENV/bin/activate
export REPO_PATH=$REPO
export EMBODIED_PATH=$REPO/examples/embodiment
export ROBOTWIN_PATH=$ROBOTWIN_PATH
export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO:$ROBOTWIN_PATH:/data/zhengjli/vomni/robotwin_rocm:\${PYTHONPATH:-}
# Fail here rather than as a SIGSEGV in an env worker 4 minutes in: _base_task.py:216 turns
# the rt shader and the OIDN denoiser on unconditionally, and lavapipe reports rayTracing=0.
python -c 'import sys, sapien.render; assert \"sapien_lavapipe\" in sys.modules, \"lavapipe shim not imported -- run bash robotwin/00_setup.sh apply\"; print(\"lavapipe shim OK\")'
# Absolute: hydra resolves --config-path against the script's directory, not the cwd.
# DRYRUN prints the merged config and exits, which catches a bad override key in seconds
# instead of after the 10-minute env build.
python examples/embodiment/train_embodied_agent.py \
  --config-path $REPO/tests/e2e_tests/embodied --config-name $RUNCFG \
  ${DRYRUN:+--cfg job --resolve} \
  $OVERRIDES
"
echo "RUN_EXIT=$?"
