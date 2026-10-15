"""Route SAPIEN to the Mesa lavapipe software rasterizer.

Import once, before any SAPIEN scene is created::

    import sapien_lavapipe  # noqa: F401

Covers the two adaptations that live inside SAPIEN. The third one is a keyword
argument the caller has to pass: ``planner_backend="mplib"``, because curobo is
CUDA-only and ``Robot.__init__`` defaults to it.

Setting the Vulkan ICD so that lavapipe is the only visible device is a
prerequisite and happens outside Python; see robotwin_mi_gpu_runtime.md.
"""

import os

import sapien

# svulkan2 scores a software device 0 and then cannot pick it. The pci: hint
# returns 1000 and bypasses the scoring. haosulab/sapien-vulkan-2#10 fixes the
# scoring; set SAPIEN_RENDER_DEVICE="" once a sapien release carries it.
_HINT = os.environ.get("SAPIEN_RENDER_DEVICE", "pci:0")

if _HINT:
    _RenderSystem = sapien.render.RenderSystem

    def _render_system(device=None):
        return _RenderSystem(_HINT if device is None else device)

    sapien.render.RenderSystem = _render_system

# RoboTwin's envs/_base_task.py switches to the "rt" shader unconditionally;
# lavapipe reports rayTracing=0 and the OIDN denoiser needs CUDA.
sapien.render.set_camera_shader_dir = lambda d: None
sapien.render.set_ray_tracing_denoiser = lambda n: None
sapien.render.set_ray_tracing_samples_per_pixel = lambda n: None
sapien.render.set_ray_tracing_path_depth = lambda n: None

