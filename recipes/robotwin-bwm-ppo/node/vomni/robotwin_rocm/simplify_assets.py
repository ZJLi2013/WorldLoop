"""Decimate the aloha-agilex visual meshes that dominate the triangle count.

Software rendering cost tracks triangles, not pixels. 3.82M of the scene's
triangles come from seven meshes that no task manipulates ??? the table alone is
2.09M. Decimating them takes get_obs() from 10.79s to 1.54s with a maximum
per-channel pixel difference of 4/255. Measured in
rlinf/doc/exps/exp18-robotwin-on-rocm.md R5.

    python simplify_assets.py /path/to/RoboTwin

Writes <name>_lowpoly.glb next to each original, rewrites the URDF to point at
them, and keeps the URDF as <urdf>.orig. Re-runnable.
"""

import os
import shutil
import sys

import numpy as np
import open3d as o3d
import trimesh

MESHDIR = "assets/embodiments/aloha-agilex/urdf/aloha_maniskill_sim/meshes"
URDF = "assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"

TARGET_FACES = {
    "box2_Link.dae": 20000,
    "camera_link1.dae": 5000,
    "camera_link2.dae": 5000,
    "d435.dae": 5000,
    "d435_back.dae": 5000,
    "d435_back1.dae": 5000,
    "tracer_base_link.dae": 20000,
}


def decimate(src, target):
    scene = trimesh.load(src, file_type="dae", force="scene")
    before = sum(len(g.faces) for g in scene.geometry.values())
    if before <= target:
        return scene, before, before
    for name, g in list(scene.geometry.items()):
        share = max(64, int(target * len(g.faces) / before))
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(g.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(g.faces, dtype=np.int32)),
        )
        m = m.simplify_quadric_decimation(share)
        new = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False)
        try:
            new.visual = trimesh.visual.ColorVisuals(new, face_colors=g.visual.main_color)
        except Exception:
            pass
        scene.geometry[name] = new
    return scene, before, sum(len(g.faces) for g in scene.geometry.values())


def main(root):
    os.chdir(root)
    renames = {}
    for fn, target in TARGET_FACES.items():
        src = os.path.join(MESHDIR, fn)
        if not os.path.exists(src):
            print(f"skip missing {fn}")
            continue
        out = os.path.join(MESHDIR, fn.replace(".dae", "_lowpoly.glb"))
        scene, before, after = decimate(src, target)
        scene.export(out)
        renames[fn] = os.path.basename(out)
        print(f"{fn:26s} {before:>9d} -> {after:>7d} faces"
              f"   {os.path.getsize(src) >> 20}MB -> {os.path.getsize(out) >> 20}MB")

    if not os.path.exists(URDF + ".orig"):
        shutil.copy2(URDF, URDF + ".orig")
    text = open(URDF + ".orig").read()
    for old, new in renames.items():
        text = text.replace(f"meshes/{old}", f"meshes/{new}")
    open(URDF, "w").write(text)
    print(f"rewrote {URDF} ({len(renames)} meshes); original at {URDF}.orig")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

