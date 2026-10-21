"""Summarise results/*.json: mean/sd per (fraction, init) and paired differences."""
import argparse
import glob
import json
import statistics as st

p = argparse.ArgumentParser()
p.add_argument("--dir", default="results")
args = p.parse_args()

runs = {}
for f in glob.glob(f"{args.dir}/*.json"):
    r = json.load(open(f))
    c = r["config"]
    runs[(c["init"], c["fraction"], c["seed"])] = r["final"]


def ms(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else float("nan"))


fracs = sorted({k[1] for k in runs})
print("| fraction | init | n | test IoU | val IoU | Bolivia IoU |")
print("|---|---|---|---|---|---|")
for fr in fracs:
    for init in ["pretrained", "scratch"]:
        seeds = sorted(k[2] for k in runs if k[0] == init and k[1] == fr)
        if not seeds:
            continue
        cells = ["%.3f ± %.3f" % ms([runs[(init, fr, s)][sp]["iou_water"] for s in seeds])
                 for sp in ["test", "valid", "bolivia"]]
        print(f"| {fr:g} | {init} | {len(seeds)} | " + " | ".join(cells) + " |")

print("\nPaired difference (pretrained - scratch, same seed = same label subset):")
for fr in fracs:
    seeds = sorted(s for (i, f, s) in runs if i == "pretrained" and f == fr and ("scratch", fr, s) in runs)
    for sp in ["test", "valid", "bolivia"]:
        d = [runs[("pretrained", fr, s)][sp]["iou_water"] - runs[("scratch", fr, s)][sp]["iou_water"] for s in seeds]
        if d:
            m, sd = ms(d)
            print(f"  f={fr:g} {sp:8s} mean {m:+.3f} sd {sd:.3f}  positive in {sum(x > 0 for x in d)}/{len(d)} seeds")
