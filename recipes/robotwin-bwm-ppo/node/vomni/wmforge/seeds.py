import json

W = "/data/zhengjli/rlinf/wt-w11/rlinf/envs/sim/robotwin/seeds"
A = "/data/zhengjli/vomni/artifacts"


def flat(x):
    if isinstance(x, dict):
        print("  dict keys", list(x)[:5])
        for k in ("success_seeds", "seeds", "train", "eval"):
            if k in x:
                return list(x[k])
        return [s for v in x.values() for s in (v if isinstance(v, list) else [])]
    return list(x)


tr = json.load(open(f"{W}/train_seeds.json"))["adjust_bottle"]
ev = json.load(open(f"{W}/eval_seeds.json"))["adjust_bottle"]
print("train type", type(tr).__name__)
trs = flat(tr)
print("eval type", type(ev).__name__)
evs = flat(ev)
print("train", len(trs), "eval", len(evs), "sample", trs[:3], evs[:3])
used = {int(r["seed"]) for r in json.load(open(f"{A}/w6/index.json"))["episodes"]}
h48 = json.load(open("/data/zhengjli/vomni/exp35/exp36_seeds.json"))["adjust_bottle"]
print("w6 seeds", len(used), "sample", sorted(used)[:3], "| hires48 type", type(h48).__name__, str(h48)[:120])
print("train ∩ used", len(set(map(int, trs)) & used), "eval ∩ used", len(set(map(int, evs)) & used))
