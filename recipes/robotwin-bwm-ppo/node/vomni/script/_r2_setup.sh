#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/zhengjli/rlinf
SRC=$ROOT/wt-f22
WT=$ROOT/wt-r2
T5=/data/zhengjli/vomni/ckpt/t5-base
TORCH_HUB=/data/zhengjli/vomni/torch_home/hub/checkpoints

git -C "$SRC" diff --quiet
if [[ ! -e "$WT" ]]; then
  git -C "$SRC" worktree add --detach "$WT" HEAD
  git -C "$WT" am -q --keep-cr /tmp/r2_task_reward_model.patch
  git -C "$WT" apply /tmp/f19c_robotwin_label_dump.patch
fi
git -C "$WT" log -1 --oneline
git -C "$WT" status --short

mkdir -p "$T5" "$TORCH_HUB"
for f in config.json spiece.model tokenizer.json model.safetensors; do
  [[ -s "$T5/$f" ]] || curl -fsSL --retry 3 -o "$T5/$f" "https://huggingface.co/google-t5/t5-base/resolve/main/$f"
done
RESNET=$TORCH_HUB/resnet18-f37072fd.pth
[[ -s "$RESNET" ]] || curl -fsSL --retry 3 -o "$RESNET" https://download.pytorch.org/models/resnet18-f37072fd.pth
sha256sum "$T5"/* "$RESNET"

docker exec -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -e T5_MODEL_PATH="$T5" \
  -e TORCH_HOME=/data/zhengjli/vomni/torch_home vkdev bash -c \
  "source $ROOT/venvs/openpi-robotwin/bin/activate && cd $WT && python -m pytest tests/unit_tests/test_robotwin_reward_model.py -q"
