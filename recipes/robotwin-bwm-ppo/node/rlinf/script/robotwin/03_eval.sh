#!/usr/bin/env bash
# feature24 B1: score pi0 on real RoboTwin through RLinf's standalone evaluator.
# Two arms, one flag apart -- WRIST=true/NIMG=3 is the three-camera reading (A),
# WRIST=false/NIMG=1 is the single-camera arm the world model can actually match (B).
#
# The environment exports are 05_train_bwm.sh's, minus the world model and the scorer:
# success here is judged by the simulator, so no reward model is built.
#
#   WRIST=false NIMG=1 bash robotwin/03_eval.sh
set -u

ROOT=${ROOT:-/data/zhengjli/rlinf}
IMG=${IMG:-rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4}
REPO=${REPO:-$ROOT/wt-f24}
VENV=${VENV:-$ROOT/venvs/openpi-robotwin}
CFG=${CFG:-rocm_adjust_bottle_pi0_smoke}
# No default: CFG pins config_name to pi0_aloha_robotwin, so a Track 2 pi0.5 checkpoint only
# loads correctly when the caller also overrides it through EXTRA (see _f21_eval144.sh).
CKPT=${CKPT:?set CKPT, and override rollout.model.openpi.config_name to match it}
ROBOTWIN_PATH=${ROBOTWIN_PATH:-/data/zhengjli/vomni/RoboTwin}
# Which arm. num_images_in_input is not descriptive: the value head builds its prefix mask as
# [True]*256*n + [False]*256*(3-n), so leaving it at 3 feeds the critic two blocks of black frames.
WRIST=${WRIST:-false}
NIMG=${NIMG:-1}

PLACEMENT=${PLACEMENT:-0-7}
# ENVS must divide by the rank count: partition_success_seeds hands each rank
# len(success_seeds)//total_num_processes of the 150 adjust_bottle eval seeds, non-overlapping.
ENVS=${ENVS:-48}
STEPS=${STEPS:-200}
# Episodes per slot = EPOCH_STEPS / STEPS, so 48 x 11 = 528 episodes.
EPOCH_STEPS=${EPOCH_STEPS:-2200}
NAME=${NAME:-rlinf_eval_robotwin}
TIMEOUT=${TIMEOUT:-21600}
LOGDIR=${LOGDIR:-$ROOT/results/b1-eval-$(date +%Y%m%d-%H%M%S)}

mkdir -p "$LOGDIR/ray"
docker rm -f "$NAME" >/dev/null 2>&1

EVALDIR=$REPO/evaluations/robotwin
SRC=$EVALDIR/$CFG.yaml
RUNCFG=${CFG}_p${PLACEMENT}
[ -f "$SRC" ] || { echo "NO_SUCH_CFG: $SRC"; exit 1; }
# hydra's override lexer rejects the comma in component_placement, so that one key is derived.
sed -e "s#env,rollout: .*#env,rollout: $PLACEMENT#" "$SRC" > "$EVALDIR/$RUNCFG.yaml"
grep -q "env,rollout: $PLACEMENT" "$EVALDIR/$RUNCFG.yaml" || { echo "BAD_DERIVED_CFG"; exit 1; }

OVERRIDES="rollout.model.model_path=$CKPT rollout.model.openpi.num_images_in_input=$NIMG"
OVERRIDES="$OVERRIDES env.eval.task_config.camera.collect_wrist_camera=$WRIST"
OVERRIDES="$OVERRIDES env.eval.total_num_envs=$ENVS env.eval.group_size=1"
OVERRIDES="$OVERRIDES env.eval.max_episode_steps=$STEPS"
OVERRIDES="$OVERRIDES env.eval.max_steps_per_rollout_epoch=$EPOCH_STEPS"
OVERRIDES="$OVERRIDES env.eval.rollout_epoch=1 runner.max_epochs=1"
# True pins every slot to the seed it started on, so the run scores the same ENVS states
# over and over. False advances through the rank's slice and wraps with a modulo, which is
# what turns 150 eval seeds into 528 episodes.
OVERRIDES="$OVERRIDES env.eval.use_fixed_reset_state_ids=False"
OVERRIDES="$OVERRIDES env.eval.video_cfg.save_video=${SAVE_VIDEO:-False}"
OVERRIDES="$OVERRIDES runner.logger.log_path=$LOGDIR"
OVERRIDES="$OVERRIDES runner.logger.experiment_name=b1_wrist${WRIST}_n${NIMG}"
OVERRIDES="$OVERRIDES ${EXTRA:-}"
[ -n "${CKPT_PT:-}" ] && OVERRIDES="$OVERRIDES runner.ckpt_path=$CKPT_PT"

echo "LOGDIR=$LOGDIR"
echo "CFG=$RUNCFG WRIST=$WRIST NIMG=$NIMG ENVS=$ENVS STEPS=$STEPS EPOCH_STEPS=$EPOCH_STEPS"
echo "EPISODES=$((ENVS * EPOCH_STEPS / STEPS)) CKPT=$CKPT CKPT_PT=${CKPT_PT:-none}"
echo "CODE=$(git -C "$REPO" rev-parse --short HEAD)"

timeout "$TIMEOUT" docker run --rm --name "$NAME" \
  -v "$LOGDIR/ray":/tmp/ray \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 64g --network host \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  -v /data:/data \
  -w "$REPO" \
  -e HF_HUB_OFFLINE=1 -e HYDRA_FULL_ERROR=1 \
  -e RLINF_ROBOTWIN_LABEL_DUMP="${RLINF_ROBOTWIN_LABEL_DUMP:-}" \
  -e RT_NO_EARLY_STOP="${RT_NO_EARLY_STOP:-}" \
  -e RLINF_SEED_EVERYTHING="${RLINF_SEED_EVERYTHING:-}" \
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
# Fail here rather than as a SIGSEGV in an env worker 4 minutes in: _base_task.py:216
# turns the rt shader and the OIDN denoiser on unconditionally, and lavapipe reports
# rayTracing=0. The env workers are ray actors, so the shim cannot be imported from an
# entry point and rides in on the venv's zz_lavapipe.pth -- see robotwin/00_setup.sh.
python -c 'import sys, sapien.render; assert \"sapien_lavapipe\" in sys.modules, \"lavapipe shim not imported -- run bash robotwin/00_setup.sh apply\"; print(\"lavapipe shim OK\")'
# Absolute: hydra resolves --config-path against the script's directory, not the cwd.
python $REPO/evaluations/eval_embodied_agent.py \
  --config-path $EVALDIR --config-name $RUNCFG \
  $OVERRIDES
"
echo "RUN_EXIT=$?"
docker rm -f "$NAME" >/dev/null 2>&1
echo "metrics -> $LOGDIR"
