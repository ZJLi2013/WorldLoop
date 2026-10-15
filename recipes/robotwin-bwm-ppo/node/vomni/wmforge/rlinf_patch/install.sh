#!/usr/bin/env bash
# WMForge w11: install the native-BWM backend, the episode-level reward and the configs into an
# RLinf tree copied from wt-exp36.
#   bash install.sh /data/zhengjli/rlinf/wt-w11
set -euo pipefail
W=${1:?tree}
P=$(cd "$(dirname "$0")" && pwd)
WM=$W/rlinf/envs/sim/world_model
CFG=$W/examples/embodiment/config
cp "$P/native_bwm_backend.py" "$WM/native_bwm_backend.py"
cp "$P/world_model_wan_env.py" "$WM/world_model_wan_env.py"
cp "$P/bwm_native_robotwin_adjust_bottle_t5.yaml" "$CFG/env/bwm_native_robotwin_adjust_bottle_t5.yaml"
grep -q "def _reset_metrics" "$WM/base_world_env.py" || { echo "base env has no _reset_metrics"; exit 1; }
python3 "$P/reset_echo_fix.py" "$WM/world_model_env.py"
python3 "$P/seed_fix.py" "$W"
python3 "$P/grpo_mask_fix.py" "$W/rlinf/algorithms/advantages.py"
python3 "$P/critic_epoch_metrics.py" "$W"
python3 "$P/bootstrap_none.py" "$W"
python3 "$P/critic_feat_dump.py" "$W"
python3 "$P/adv_dump.py" "$W"
python3 - "$CFG" <<'EOF'
import sys
cfg = sys.argv[1]
src = open(f"{cfg}/bwm_robotwin_adjust_bottle_ppo_openpi_t5.yaml").read()
edits = [
    ("env/bwm_robotwin_adjust_bottle_t5@env.train", "env/bwm_native_robotwin_adjust_bottle_t5@env.train"),
    ('experiment_name: "bwm_robotwin_ppo_openpi"', 'experiment_name: "bwm_native_step750_ppo_openpi"'),
    ("  gamma: 0.99", "  gamma: 1.0"),
]
for old, new in edits:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
open(f"{cfg}/bwm_native_ppo_openpi_t5.yaml", "w").write(src)
print("wrote", f"{cfg}/bwm_native_ppo_openpi_t5.yaml")
EOF
python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]; print('syntax ok')" "$WM/native_bwm_backend.py" "$WM/world_model_wan_env.py"
echo INSTALLED
