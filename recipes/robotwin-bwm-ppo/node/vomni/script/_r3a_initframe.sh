#!/usr/bin/env bash
# What BWM is conditioned on: structure and appearance of the init frames in use.
set -uo pipefail

ROOT=/data/zhengjli/rlinf
ART=/data/zhengjli/vomni/artifacts
OUT=${OUT:-$ART/r3-initframe-check}

docker exec -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" vkdev bash -c "
source $ROOT/venvs/openpi-robotwin/bin/activate
python - <<'PY'
import glob, os
import numpy as np
from PIL import Image

out = '$OUT'
os.makedirs(out, exist_ok=True)


def describe(v, depth=0):
    pad = '  ' * depth
    if isinstance(v, np.ndarray) and v.dtype != object:
        return f'{pad}ndarray {list(v.shape)} {v.dtype} [{v.min()}, {v.max()}]'
    if isinstance(v, np.ndarray):
        return f'{pad}object-array {list(v.shape)}'
    if isinstance(v, dict):
        lines = [f'{pad}dict keys={sorted(v)}']
        for k in sorted(v):
            lines.append(f'{pad}  {k}: ' + describe(v[k], depth + 1).strip())
        return '\n'.join(lines)
    if isinstance(v, (list, tuple)):
        return f'{pad}{type(v).__name__} len={len(v)} first=' + describe(v[0], 0).strip()
    return f'{pad}{type(v).__name__} {str(v)[:60]}'


def biggest_image(v):
    best = None
    stack = [v]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
        elif isinstance(cur, np.ndarray):
            if cur.dtype == object:
                stack.extend(cur.ravel().tolist())
            elif cur.ndim >= 3 and (best is None or cur.size > best.size):
                best = cur
    return best


for name in ('init_frames', 'init_frames_hires2', 'init_frames_hires48'):
    files = sorted(glob.glob(f'$ART/{name}/*.npy'))
    print(f'=== {name}: {len(files)} files ===')
    if not files:
        continue
    raw = np.load(files[0], allow_pickle=True)
    payload = raw.item() if raw.dtype == object and raw.ndim == 0 else raw
    print(describe(payload))

    img = biggest_image(payload)
    if img is None:
        print('  no image array found')
        continue
    while img.ndim > 3:
        img = img[0]
    if img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)
    Image.fromarray(img).save(f'{out}/{name}.png')
    print(f'  saved {out}/{name}.png shape={img.shape}')
PY
"
