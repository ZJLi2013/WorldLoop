#!/usr/bin/env python3
"""Seed the rollout and actor workers from RLINF_SEED_EVERYTHING (+ rank) when it is set.
Without it the embodied path never seeds torch, so pi0.5's flow-SDE noise always starts from
torch's default seed and a "new training seed" only changes the minibatch shuffle. Idempotent.

    python seed_fix.py <tree>
"""
import sys

tree = sys.argv[1]
block = (
    "        import os as _os\n"
    "\n"
    "        if _os.environ.get(\"RLINF_SEED_EVERYTHING\"):\n"
    "            from rlinf.utils.utils import seed_everything\n"
    "\n"
    "            seed_everything(int(_os.environ[\"RLINF_SEED_EVERYTHING\"]) + self._rank)\n"
)
targets = [
    ("rlinf/workers/rollout/hf/huggingface_worker.py",
     "    def init_worker(self):\n",
     "        rollout_model_config = copy.deepcopy(self.model_cfg)\n"),
    ("rlinf/workers/actor/embodied_fsdp_actor_worker.py",
     "        if needed, offload model parameters and optimizer states to CPU.\n        \"\"\"\n",
     "        self.setup_model_and_optimizer()\n"),
]
for rel, head, tail in targets:
    path = f"{tree}/{rel}"
    src = open(path, encoding="utf-8").read()
    if "RLINF_SEED_EVERYTHING" in src:
        print("ALREADY_PATCHED", rel)
        continue
    assert src.count(head + tail) == 1, f"anchor not found in {rel}"
    open(path, "w", encoding="utf-8").write(src.replace(head + tail, head + block + tail))
    print("PATCHED", rel)
