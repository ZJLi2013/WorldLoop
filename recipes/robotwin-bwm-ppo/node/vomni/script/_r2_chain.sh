#!/usr/bin/env bash
# R2 unattended chain: setup -> smoke -> collect x4 -> prepare -> train/test.
# Each stage stops the chain on failure; see exp22 for the checkpoints.
set -euo pipefail

S=/data/zhengjli/vomni/script
ART=/data/zhengjli/vomni/artifacts
ROOT=/data/zhengjli/rlinf
PREP=$ART/r2-reward-data
TRAIN=$ART/r2-reward-model
T5=/data/zhengjli/vomni/ckpt/t5-base
stage() { echo "=== $(date -Is) $*"; }

stage setup
bash $S/_r2_setup.sh

stage smoke base
TAG=base EPOCH_STEPS=200 SUFFIX=-smoke bash $S/_r2_collect.sh
docker exec -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" vkdev bash -c \
  "source $ROOT/venvs/openpi-robotwin/bin/activate && python $S/_r2_prepare.py \
   --dump base $ART/r2-collect-base-smoke --output $ART/r2-smoke-data --rate-tolerance 0.25"

for tag in base orig5 ignore5 hist10; do
  stage collect $tag
  TAG=$tag bash $S/_r2_collect.sh
done

stage prepare
docker exec -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" vkdev bash -c \
  "source $ROOT/venvs/openpi-robotwin/bin/activate && python $S/_r2_prepare.py \
   --dump base $ART/r2-collect-base --dump orig5 $ART/r2-collect-orig5 \
   --dump ignore5 $ART/r2-collect-ignore5 --dump hist10 $ART/r2-collect-hist10 \
   --output $PREP"

stage train
NAME=r2_train bash $S/_r2_gpu.sh $S/_r2_train.py --data $PREP --t5 $T5 --output $TRAIN

stage done
