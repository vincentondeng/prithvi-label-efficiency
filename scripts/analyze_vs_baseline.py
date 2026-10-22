"""Deep models vs the MNDWI baseline: table and two-panel figure."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dir", default="results")
p.add_argument("--baseline", default="results_baseline/mndwi.json")
p.add_argument("--out", default="figures")
args = p.parse_args()

runs = {}
for f in glob.glob(f"{args.dir}/*.json"):
    r = json.load(open(f))
    c = r["config"]
    runs[(c["init"], round(c["fraction"], 4), c["seed"])] = r["final"]["test"]["iou_water"]
base = {(round(b["fraction"], 4), b["seed"]): b["test"] for b in json.load(open(args.baseline))}
fracs = sorted({k[1] for k in runs})
n_chips = {f: max(1, int(np.floor(f * 252 + 0.5))) for f in fracs}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def ci(d):
    d = np.asarray(d)
    n = len(d)
    half = T975[n - 1] * d.std(ddof=1) / np.sqrt(n)
    return d.mean(), d.mean() - half, d.mean() + half


def vals(init, fr):
    seeds = sorted(s for (i, f, s) in runs if i == init and f == fr)
    return seeds, np.array([runs[(init, fr, s)] for s in seeds])


def gap(init, fr):
    seeds, v = vals(init, fr)
    return v - np.array([base[(fr, s)] for s in seeds])


print("| labels | chips | MNDWI rule (10 seeds) | pretrained - rule (95% t-CI) | scratch - rule (95% t-CI) |")
print("|---|---|---|---|---|")
for fr in fracs:
    bm = np.array([base[(fr, s)] for s in range(10)])
    cells = []
    for init in ["pretrained", "scratch"]:
        d = gap(init, fr)
        m, lo, hi = ci(d)
        cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({int((d > 0).sum())}/{len(d)} above)")
    print(f"| {fr * 100:g}% | {n_chips[fr]} | {bm.mean():.3f} ± {bm.std(ddof=1):.3f} | {cells[0]} | {cells[1]} |")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

col = {"pretrained": "#1b6ca8", "scratch": "#c0572b"}
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
for init in ["pretrained", "scratch"]:
    xs, ms, ss = [], [], []
    for fr in fracs:
        _, v = vals(init, fr)
        ax[0].scatter([n_chips[fr]] * len(v), v, s=9, alpha=0.25, color=col[init])
        xs.append(n_chips[fr]); ms.append(v.mean()); ss.append(v.std(ddof=1))
    xs, ms, ss = map(np.array, (xs, ms, ss))
    ax[0].plot(xs, ms, "-o", color=col[init], label=init)
    ax[0].fill_between(xs, ms - ss, ms + ss, color=col[init], alpha=0.15)
bx = np.array([n_chips[f] for f in fracs])
bms = np.array([np.mean([base[(f, s)] for s in range(10)]) for f in fracs])
bss = np.array([np.std([base[(f, s)] for s in range(10)], ddof=1) for f in fracs])
ax[0].plot(bx, bms, "--s", color="#555555", label="MNDWI rule (fitted on same labels)")
ax[0].fill_between(bx, bms - bss, bms + bss, color="#555555", alpha=0.15)
ax[0].set_xscale("log")
ax[0].set_xticks(bx)
ax[0].set_xticklabels([f"{n_chips[f]}\n({f * 100:g}%)" for f in fracs])
ax[0].xaxis.set_minor_formatter(NullFormatter())
ax[0].set_xlabel("labelled training chips (log scale)")
ax[0].set_ylabel("test water IoU")
ax[0].set_title("Label efficiency vs a one-parameter water index")
ax[0].legend(loc="lower right", fontsize=8)
ax[0].grid(alpha=0.3)

for i, fr in enumerate(fracs):
    for j, init in enumerate(["pretrained", "scratch"]):
        d = gap(init, fr)
        m, lo, hi = ci(d)
        x = i + (-0.13 if j == 0 else 0.13)
        ax[1].scatter(x + (np.arange(len(d)) - (len(d) - 1) / 2) * 0.015, d, s=9, alpha=0.3, color=col[init])
        ax[1].errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=col[init], capsize=3,
                       label=init if i == 0 else None)
ax[1].axhline(0, color="gray", lw=1)
ax[1].set_xticks(range(len(fracs)))
ax[1].set_xticklabels([f"{f * 100:g}%" for f in fracs])
ax[1].set_xlabel("label fraction")
ax[1].set_ylabel("test IoU minus MNDWI rule")
ax[1].set_title("Gain over the rule (mean, 95% t-CI; dots = seeds)")
ax[1].legend(loc="lower right", fontsize=8)
ax[1].grid(alpha=0.3)
fig.tight_layout()
Path(args.out).mkdir(exist_ok=True)
fig.savefig(Path(args.out, "label_efficiency_with_baseline.png"), dpi=200)
fig.savefig(Path(args.out, "label_efficiency_with_baseline.pdf"))
print(f"\nwrote {args.out}/label_efficiency_with_baseline.png and .pdf")
