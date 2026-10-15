# RoboTwin on AMD Instinct (CDNA) ??? SAPIEN renders on Mesa lavapipe (CPU).
#
# CDNA has no rasterization hardware, and RADV claims depth support it cannot
# back, so GPU rendering is not an option here. Everything below exists to route
# SAPIEN to the software rasterizer instead. See robotwin_mi_gpu_runtime.md next
# to this file; build with this directory as the context:
#
#   docker build -t robotwin-rocm:lavapipe .

FROM rlinf/rlinf:agentic-rlinf0.3-libero-rocm6.4

# the venv's activate script is bash, not sh
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        mesa-vulkan-drivers libvulkan1 vulkan-tools unzip \
    && rm -rf /var/lib/apt/lists/*

# The loader must see lavapipe only. If RADV is visible it enumerates as a
# supported eDiscreteGpu, outscores the software rasterizer, gets picked, and
# then fails when it allocates a depth attachment.
#
# These three variables are exported by the venv's activate script, not by the
# image environment, so ENV cannot override them ??? anyone who activates the venv
# gets the radeon ICD back. Rewrite the script itself.
RUN sed -i \
        -e 's|^export AMD_VULKAN_ICD=.*|unset AMD_VULKAN_ICD|' \
        -e 's|radeon_icd\.x86_64\.json|lvp_icd.x86_64.json|g' \
        /opt/venv/openvla-oft/bin/activate \
    && grep -nE 'VULKAN|VK_ICD|VK_DRIVER' /opt/venv/openvla-oft/bin/activate

# mplib needs --no-deps: it pins sapien==3.0.0b1 and would downgrade the 3.0.1
# the base image ships. Its real requirements are numpy, toppra, transforms3d.
# numpy is pinned because open3d otherwise pulls 2.x over the 1.26.4 that
# torch 2.8.0+rocm6.4 was built against.
RUN . /opt/venv/openvla-oft/bin/activate \
    && python -c "import sapien; assert sapien.__version__ == '3.0.1', sapien.__version__" \
    && pip install --no-cache-dir --no-deps mplib==0.2.1 \
    && pip install --no-cache-dir \
        numpy==1.26.4 toppra==0.6.10 transforms3d==0.4.2 \
        open3d==0.19.0 pycollada==0.9.3 trimesh==4.12.2 \
        gymnasium==0.29.1 h5py imageio zarr \
    && python -c "from mplib.sapien_utils import SapienPlanner" \
    && python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__)"

COPY sapien_lavapipe.py simplify_assets.py smoke_test.py setup_assets.sh /opt/robotwin_rocm/

# activate prepends its own entry, so this stays on the path as a suffix
ENV PYTHONPATH=/opt/robotwin_rocm \
    SAPIEN_RENDER_DEVICE=pci:0

CMD ["bash"]

