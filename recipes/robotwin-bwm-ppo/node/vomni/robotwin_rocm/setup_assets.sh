#!/usr/bin/env bash
# Download RoboTwin's 30 GB asset packs and decimate the high-poly meshes.
#
#   bash setup_assets.sh /path/to/RoboTwin
#
# Re-runnable: huggingface_hub resumes, unzip overwrites, decimation skips
# meshes already at target.
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    allow_patterns=["background_texture.zip", "embodiments.zip", "objects.zip"],
    local_dir="assets",
    repo_type="dataset",
    resume_download=True,
)
PY

cd assets
for z in background_texture.zip embodiments.zip objects.zip; do
    echo "unzip $z"
    unzip -q -o "$z"
done
cd ..

# rewrites the absolute paths baked into task_config/_embodiment_config.yml
python script/update_embodiment_config_path.py

python "$(dirname "$(readlink -f "$0")")/simplify_assets.py" "$(pwd)"

echo "assets ready: $(du -sh assets | cut -f1)"

