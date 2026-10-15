#!/usr/bin/env bash
# Image, the vkdev container, the openpi-robotwin venv and BWM's side packages. Run after materialize.sh.
#
#   WL_ROOT=/data/$USER bash $WL_ROOT/worldloop/setup/env.sh
set -euo pipefail
WL_ROOT=${WL_ROOT:?set WL_ROOT}
WL_ROOT=${WL_ROOT%/}
WL_MOUNT=${WL_MOUNT:-/$(echo "$WL_ROOT" | cut -d/ -f2)}
IMG=rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4
IMG_ID=sha256:d71b40a47b4f8b73679e75746c0c1aa2f422f54c041f80251e414bda603a2531

docker image inspect "$IMG" >/dev/null 2>&1 || docker pull "$IMG"
got=$(docker image inspect --format '{{.Id}}' "$IMG")
[ "$got" = "$IMG_ID" ] || echo "WARN image id $got, the w32 run used $IMG_ID"

# Long-lived container the RoboTwin CPU jobs exec into (_w2_vk.sh, _r2_*.sh, 00_setup.sh).
docker inspect vkdev >/dev/null 2>&1 || docker run -d --name vkdev --network host --device /dev/kfd \
  --device /dev/dri --group-add video --security-opt seccomp=unconfined -v "$WL_MOUNT:$WL_MOUNT" "$IMG" sleep infinity

docker exec vkdev bash "$WL_ROOT/worldloop/setup/venv_in_container.sh" "$WL_ROOT"

ROOT=$WL_ROOT/rlinf REPO=$WL_ROOT/rlinf/wt-g bash "$WL_ROOT/rlinf/script/robotwin/00_setup.sh" check
