#!/usr/bin/env bash
# Inside vkdev: build the venv to the w32 freeze, then the source patches and side packages it carried.
set -euo pipefail
WL_ROOT=$1
VENV=$WL_ROOT/rlinf/venvs/openpi-robotwin
PYDEPS=$WL_ROOT/vomni/native/pydeps
SHIM=$WL_ROOT/vomni/robotwin_rocm
ENV_DIR=$WL_ROOT/worldloop/env
ROCM_INDEX=https://download.pytorch.org/whl/rocm6.4

if [ ! -x "$VENV/bin/python" ]; then
  # Exits non-zero at its RoboTwin step on AMD (it looks for a CUDA version); the pinned install finishes it.
  (cd "$WL_ROOT/rlinf/wt-g" && bash requirements/install.sh embodied --model openpi --env robotwin \
    --platform amd --rocm 6.4 --no-flash-attn --no-root --venv "$VENV") || true
fi
uv pip install --python "$VENV/bin/python" --no-deps --index-strategy unsafe-best-match \
  --extra-index-url "$ROCM_INDEX" -r "$ENV_DIR/openpi-robotwin.freeze.txt"

SP=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
# sapien's URDF loader opens files without an encoding.
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SP/sapien/wrapper/urdf_loader.py"
# mplib's screw planner rejects any path that touches a collision; RoboTwin's mplib_screw path then always fails.
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
  "$SP/mplib/planner.py"
# configure_amd() points the venv at RADV, which cannot render on CDNA.
sed -i -e 's#^export AMD_VULKAN_ICD=RADV#unset AMD_VULKAN_ICD#' -e 's#radeon_icd\.x86_64\.json#lvp_icd.x86_64.json#g' \
  "$VENV/bin/activate"
# Ray env workers do not inherit PYTHONPATH; a .pth imports the lavapipe shim in every interpreter.
printf '%s\nimport sapien_lavapipe\n' "$SHIM" > "$SP/zz_lavapipe.pth"

if [ ! -d "$PYDEPS/diffsynth" ]; then
  uv pip install --python "$VENV/bin/python" --target "$PYDEPS" --no-deps --index-strategy unsafe-best-match \
    --extra-index-url "$ROCM_INDEX" diffsynth==2.0.11 ftfy==6.3.1 torchaudio==2.8.0+rocm6.4
fi

grep -q 'encoding="utf-8") as' "$SP/sapien/wrapper/urdf_loader.py"
! grep -q 'or collide or not within_joint_limit' "$SP/mplib/planner.py"
"$VENV/bin/python" -c "import numpy, mplib, sapien; assert numpy.__version__ < '2', numpy.__version__"
PYTHONPATH=$PYDEPS "$VENV/bin/python" -c "import diffsynth, ftfy; print('diffsynth', diffsynth.__file__)"
echo VENV_OK
