#!/usr/bin/env bash
# The openpi + RoboTwin environment on MI GPUs: one place, checkable without rebuilding.
#
# Every step exists because something fails loudly without it, and the reason is on each
# one. Sources: exp19 (venv build, ray/amdsmi, RoboTwin's non-CUDA deps) and
# doc/robotwin_rocm/robotwin_mi_gpu_runtime.md (the lavapipe adaptations).
#
#   bash robotwin/00_setup.sh check   # report only
#   bash robotwin/00_setup.sh apply   # build what is missing, then report
#
# Two things deliberately live outside any checkout: the venv is 25 GB and a worktree is
# disposable, and the scorer's libraries must not be reachable from the training path.
set -u

ACTION=${1:-check}
ROOT=${ROOT:-/data/zhengjli/rlinf}
REPO=${REPO:-$ROOT/wt-f24}
VENV=${VENV:-$ROOT/venvs/openpi-robotwin}
# Built by _setup_f2_venv.sh and shared with the LIBERO arm. Pinned to 4.57.1 there:
# it is the first release with Qwen3-VL, and 5.x segfaults on the first forward here.
LIB_DIR=${LIB_DIR:-$ROOT/venvs/f2-vlm-libs}
SHIM=${SHIM:-/data/zhengjli/vomni/robotwin_rocm}
CONTAINER=${CONTAINER:-vkdev}
LVP=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
# zz_ so site processes it after whatever else is in there.
PTH_NAME=${PTH_NAME:-$VENV/lib/python3.11/site-packages/zz_lavapipe.pth}

dexec() { docker exec "$CONTAINER" bash -lc "$1"; }
say() { printf '%-30s %s\n' "$1" "$2"; }

check() {
  dexec "cat > /tmp/_probe.py <<'PY'
import importlib, os, sys
print('PY', sys.version.split()[0])
for name in ('torch', 'sapien', 'mplib', 'gymnasium', 'ray', 'transformers', 'transforms3d'):
    try:
        m = importlib.import_module(name)
        print(f'{name}={getattr(m, \"__version__\", \"n/a\")}')
    except Exception as exc:
        print(f'{name}=FAIL:{type(exc).__name__}')
try:
    import amdsmi
    amdsmi.amdsmi_init()
    print('devices=%d' % len(amdsmi.amdsmi_get_processor_handles()))
except Exception as exc:
    print('devices=FAIL:%s' % type(exc).__name__)
print('icd=%s' % os.environ.get('VK_ICD_FILENAMES', 'unset'))
PY"

  echo "===== training path ($VENV)"
  if dexec "[ -x $VENV/bin/python ]"; then
    dexec "source $VENV/bin/activate && python /tmp/_probe.py" | sed 's/^/  /'
  else
    say "interpreter" "MISSING -- run apply"
  fi

  echo "===== scorer path (same venv, $LIB_DIR in front)"
  if dexec "[ -d $LIB_DIR ]"; then
    dexec "source $VENV/bin/activate && PYTHONPATH=$LIB_DIR python -c \
      \"import transformers, tokenizers, transformers.models as m; \
        print('transformers=%s tokenizers=%s qwen3_vl=%s' % \
        (transformers.__version__, tokenizers.__version__, hasattr(m, 'qwen3_vl')))\"" \
      | sed 's/^/  /'
  else
    say "lib dir" "MISSING -- apply runs _setup_f2_venv.sh"
  fi

  echo "===== rendering"
  dexec "grep -q lvp_icd $VENV/bin/activate" \
    && say "venv activate ICD" "lavapipe" \
    || say "venv activate ICD" "RADV -- take_picture will fail on CDNA"
  dexec "[ -f $SHIM/sapien_lavapipe.py ]" \
    && say "sapien_lavapipe shim" "$SHIM" \
    || say "sapien_lavapipe shim" "MISSING"
  # Presence is not the property that matters. RoboTwin re-enables the rt shader in every
  # env worker, so the shim has to be *imported* there, and env workers are ray actors that
  # do not reliably inherit PYTHONPATH -- hence the .pth, and hence testing with a bare
  # interpreter and no PYTHONPATH, which is the condition a ray actor actually starts in.
  if dexec "source $VENV/bin/activate && env -u PYTHONPATH python -c \
      \"import sys, sapien.render; sys.exit(0 if 'sapien_lavapipe' in sys.modules else 1)\"" \
      >/dev/null 2>&1; then
    say "shim auto-import" "on, via $PTH_NAME"
  else
    say "shim auto-import" "NOT ACTIVE -- env workers will segfault in take_picture"
  fi
}

apply() {
  if ! dexec "[ -x $VENV/bin/python ]"; then
    echo ">> building the venv (25 GB, tens of minutes)"
    dexec "cd $REPO && bash requirements/install.sh embodied --model openpi --env robotwin \
      --platform amd --rocm 6.4 --no-flash-attn --no-root --venv $VENV"
  fi

  # install_robotwin_env aborts on AMD -- it reads the CUDA version only to set
  # TORCH_CUDA_ARCH_LIST for pytorch3d / warp-lang / curobo, none of which mplib planning
  # needs. These are its remaining, platform-neutral packages.
  echo ">> RoboTwin's non-CUDA dependencies"
  dexec "uv pip install --python $VENV/bin/python \
    mplib==0.2.1 gymnasium==0.29.1 av open3d zarr openai \
    pycollada trimesh transforms3d h5py imageio 2>&1 | tail -2"

  # ray 2.58 dropped the ctypes path into librocm_smi64 and now requires the amdsmi
  # package. Without it the cluster counts 0 accelerators and component_placement fails
  # with a message about node ranks that says nothing about the real cause.
  echo ">> amdsmi for ray"
  dexec "uv pip install --python $VENV/bin/python /opt/rocm/share/amd_smi 2>&1 | tail -1"

  # configure_amd() appends three RADV exports to every AMD venv's activate. RADV claims
  # CDNA support and then fails in take_picture with "cannot create image".
  echo ">> pointing the venv's ICD at lavapipe"
  dexec "sed -i \
    -e 's#^export AMD_VULKAN_ICD=RADV#unset AMD_VULKAN_ICD#' \
    -e 's#radeon_icd\.x86_64\.json#lvp_icd.x86_64.json#g' \
    $VENV/bin/activate"

  # A .pth and not PYTHONPATH: the env workers are ray actors, and the driver's PYTHONPATH
  # does not reach them. site execs any .pth line beginning with "import", after adding the
  # plain path line above it to sys.path, so these two lines put the rt-shader patch into
  # every interpreter that uses this venv no matter who spawned it.
  echo ">> the lavapipe shim as a venv .pth"
  dexec "printf '%s\nimport sapien_lavapipe\n' $SHIM > $PTH_NAME && cat $PTH_NAME"

  if ! dexec "[ -d $LIB_DIR ]"; then
    echo ">> the scorer's libraries"
    bash "$(dirname "${BASH_SOURCE[0]}")/../_setup_f2_venv.sh"
  fi

  echo
  check
}

case "$ACTION" in
  check) check ;;
  apply) apply ;;
  *) echo "usage: $0 [check|apply]"; exit 1 ;;
esac
