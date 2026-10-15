#!/usr/bin/env bash
# WMForge w11: PPO for Track 2 pi0.5 inside the in-process step-750 BWM (native_bwm backend,
# episode-level frozen-T5 reward). Derived from robotwin/05_train_bwm.sh without the served-WM
# and Qwen-scorer overrides; env.eval stays off (simulator scoring runs on saved checkpoints).
#
#   ENVS=8 RE=1 GBS=32 MBS=4 MAX_STEPS=1 LR=0 VLR=0 NAME=w11_smoke LOGDIR=<dir> bash _w11_train.sh
set -u
ROOT=/data/zhengjli/rlinf
IMG=${IMG:-rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4}
REPO=${REPO:-$ROOT/wt-w11}
VENV=${VENV:-$ROOT/venvs/openpi-robotwin}
CFG=${CFG:-bwm_native_ppo_openpi_t5}
CKPT=${CKPT:-/data/zhengjli/vomni/track2/ckpt/fulldata}
ROBOTWIN_PATH=/data/zhengjli/vomni/RoboTwin
PLACEMENT=${PLACEMENT:-0-7}
ENVS=${ENVS:-48}
RE=${RE:-2}
GBS=${GBS:-384}
MBS=${MBS:-16}
LR=${LR:-5.6e-6}
VLR=${VLR:-1.1e-4}
MAX_STEPS=${MAX_STEPS:-10}
SAVE_EVERY=${SAVE_EVERY:-5}
NAME=${NAME:-w11_g5}
LOGDIR=${LOGDIR:?LOGDIR}
WM_CKPT=${WM_CKPT:-/data/zhengjli/vomni/artifacts/w8-pilot/step-750.safetensors}

mkdir -p "$LOGDIR/ray"
docker rm -f "$NAME" >/dev/null 2>&1
SRC=$REPO/examples/embodiment/config/$CFG.yaml
E2EDIR=$REPO/tests/e2e_tests/embodied
RUNCFG=${CFG}_p${PLACEMENT}
[ -f "$SRC" ] || { echo "NO_SUCH_CFG: $SRC"; exit 1; }
sed -e "s#actor, env, rollout: .*#actor, env, rollout: $PLACEMENT#" "$SRC" > "$E2EDIR/$RUNCFG.yaml"

OVERRIDES="actor.model.model_path=$CKPT rollout.model.model_path=$CKPT"
OVERRIDES="$OVERRIDES env.train.rollout_epoch=$RE env.train.total_num_envs=$ENVS"
OVERRIDES="$OVERRIDES env.train.world_model.native.ckpt=$WM_CKPT"
OVERRIDES="$OVERRIDES actor.global_batch_size=$GBS actor.micro_batch_size=$MBS"
OVERRIDES="$OVERRIDES actor.optim.lr=$LR actor.optim.value_lr=$VLR"
OVERRIDES="$OVERRIDES runner.max_steps=$MAX_STEPS runner.save_interval=$SAVE_EVERY"
OVERRIDES="$OVERRIDES runner.logger.log_path=$LOGDIR runner.val_check_interval=-1"
[ "$MAX_STEPS" = 1 ] && OVERRIDES="$OVERRIDES runner.max_epochs=1"
# SEED moves all three random streams: PPO shuffle, worker torch RNGs, WM noise. Unset = w11.
# The WM offset is SEED*1000 so window seeds (offset + window index) never overlap across SEEDs.
SEED=${SEED:-}
[ -n "$SEED" ] && OVERRIDES="$OVERRIDES actor.seed=$SEED ++env.train.world_model.native.seed_offset=$((SEED * 1000))"
OVERRIDES="$OVERRIDES ${EXTRA:-}"

echo "LOGDIR=$LOGDIR CFG=$RUNCFG ENVS=$ENVS RE=$RE GBS=$GBS MBS=$MBS LR=$LR VLR=$VLR MAX_STEPS=$MAX_STEPS WM=$WM_CKPT SEED=${SEED:-default}"
echo "CODE=$(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --short | wc -l)"
docker run --rm --name "$NAME" \
  -v "$LOGDIR/ray":/tmp/ray \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size 64g --network host \
  --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
  -v /data:/data -w "$REPO" \
  -e HF_HUB_OFFLINE=1 -e HYDRA_FULL_ERROR=1 -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
  -e RLINF_BWM_REWARD_DUMP="${RLINF_BWM_REWARD_DUMP:-}" -e RLINF_SEED_EVERYTHING="$SEED" \
  -e RLINF_CRITIC_FEAT_DUMP="${RLINF_CRITIC_FEAT_DUMP:-}" -e RLINF_ADV_DUMP="${RLINF_ADV_DUMP:-}" \
  -e RLINF_W18_DUMP="${RLINF_W18_DUMP:-}" \
  -e RLINF_WM_TRACE="${RLINF_WM_TRACE:-}" -e RLINF_WM_DUMP="${RLINF_WM_DUMP:-}" \
  --entrypoint bash "$IMG" -lc "
source $VENV/bin/activate
export REPO_PATH=$REPO EMBODIED_PATH=$REPO/examples/embodiment ROBOTWIN_PATH=$ROBOTWIN_PATH
export ROBOT_PLATFORM=ALOHA MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO:$ROBOTWIN_PATH:/data/zhengjli/vomni/robotwin_rocm:/data/zhengjli/vomni/native/pydeps:/data/zhengjli/vomni/bwm-repo:\${PYTHONPATH:-}
python examples/embodiment/train_embodied_agent.py \
  --config-path $REPO/tests/e2e_tests/embodied --config-name $RUNCFG $OVERRIDES
" > "$LOGDIR/train.log" 2>&1
rc=$?
echo "RUN_EXIT=$rc" >> "$LOGDIR/train.log"
exit $rc
