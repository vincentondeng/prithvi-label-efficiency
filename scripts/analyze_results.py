"""Results table, paired differences with CIs, label-equivalence estimate, and figure."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dir", default="results")
p.add_argument("--out", default="figures")
p.add_argument("--n-boot", type=int, default=10000)
args = p.parse_args()

runs = {}
for f in glob.glob(f"{args.dir}/*.json"):
    r = json.load(open(f))
    c = r["config"]
    runs[(c["init"], c["fraction"], c["seed"])] = r["final"]
fracs = sorted({k[1] for k in runs})
n_chips = {f: max(1, int(np.floor(f * 252 + 0.5))) for f in fracs}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
rng = np.random.default_rng(0)


def vals(init, fr, split="test"):
    seeds = sorted(s for (i, f, s) in runs if i == init and f == fr)
    return seeds, np.array([runs[(init, fr, s)][split]["iou_water"] for s in seeds])


def paired(fr, split="test"):
    sp, vp = vals("pretrained", fr, split)
    ss, vs = vals("scratch", fr, split)
    common = sorted(set(sp) & set(ss))
    d = np.array([runs[("pretrained", fr, s)][split]["iou_water"] - runs[("scratch", fr, s)][split]["iou_water"]
                  for s in common])
    n = len(d)
    t_half = T975[n - 1] * d.std(ddof=1) / np.sqrt(n)
    boot = d[rng.integers(0, n, (args.n_boot, n))].mean(1)
    return d, (d.mean() - t_half, d.mean() + t_half), tuple(np.percentile(boot, [2.5, 97.5]))


Path(args.out).mkdir(exist_ok=True)
lines = []
for split in ["test", "valid", "bolivia"]:
    lines.append(f"\n### {split} water IoU (mean ± sd over seeds)\n")
    lines.append("| labels | chips | seeds | pretrained | scratch | gap (pre - scr) | 95% t-CI | 95% bootstrap CI | pretrained better |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for fr in fracs:
        _, vp = vals("pretrained", fr, split)
        _, vs = vals("scratch", fr, split)
        d, tci, bci = paired(fr, split)
        lines.append(f"| {fr * 100:g}% | {n_chips[fr]} | {len(d)} | {vp.mean():.3f} ± {vp.std(ddof=1):.3f} | "
                     f"{vs.mean():.3f} ± {vs.std(ddof=1):.3f} | {d.mean():+.3f} | [{tci[0]:+.3f}, {tci[1]:+.3f}] | "
                     f"[{bci[0]:+.3f}, {bci[1]:+.3f}] | {int((d > 0).sum())}/{len(d)} |")
table = "\n".join(lines)
Path(args.out, "summary_tables.md").write_text(table + "\n")
test_part = table.split("### valid")[0]
print(test_part)

# label-equivalence: chips scratch would need to match each pretrained mean (log-linear interpolation)
sm = np.array([vals("scratch", f)[1].mean() for f in fracs])
xs = np.log10([n_chips[f] for f in fracs])
order = np.argsort(sm)
print("\nLabel equivalence (rough, interpolating the scratch curve on a log scale):")
for fr in fracs:
    target = vals("pretrained", fr)[1].mean()
    if target > sm.max():
        txt = f"above scratch with all {n_chips[fracs[-1]]} chips"
    elif target <= sm.min():
        txt = f"at or below scratch with {n_chips[fracs[0]]} chips"
    else:
        need = 10 ** np.interp(target, sm[order], xs[order])
        txt = f"scratch needs ~{need:.0f} chips ({need / n_chips[fr]:.1f}x more)"
    print(f"  pretrained with {n_chips[fr]:3d} chips ({fr * 100:g}%): test IoU {target:.3f} -> {txt}")

# figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
except ImportError:
    raise SystemExit("matplotlib missing: conda install -c conda-forge matplotlib")

col = {"pretrained": "#1b6ca8", "scratch": "#c0572b"}
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
for init in ["pretrained", "scratch"]:
    xs_, m_, s_ = [], [], []
    for fr in fracs:
        _, v = vals(init, fr)
        ax[0].scatter([n_chips[fr]] * len(v), v, s=9, alpha=0.25, color=col[init])
        xs_.append(n_chips[fr]); m_.append(v.mean()); s_.append(v.std(ddof=1))
    xs_, m_, s_ = map(np.array, (xs_, m_, s_))
    ax[0].plot(xs_, m_, "-o", color=col[init], label=init)
    ax[0].fill_between(xs_, m_ - s_, m_ + s_, color=col[init], alpha=0.15)
ax[0].set_xscale("log")
ax[0].set_xticks([n_chips[f] for f in fracs])
ax[0].set_xticklabels([f"{n_chips[f]}\n({f * 100:g}%)" for f in fracs])
ax[0].xaxis.set_minor_formatter(NullFormatter())
ax[0].set_xlabel("labelled training chips (log scale)")
ax[0].set_ylabel("test water IoU")
ax[0].set_title("Label efficiency (mean ± 1 sd; dots = seeds)")
ax[0].legend(loc="lower right")
ax[0].grid(alpha=0.3)

for i, fr in enumerate(fracs):
    d, tci, _ = paired(fr)
    ax[1].scatter(i + (np.arange(len(d)) - (len(d) - 1) / 2) * 0.04, d, s=12, alpha=0.4, color="#444444")
    ax[1].errorbar(i, d.mean(), yerr=[[d.mean() - tci[0]], [tci[1] - d.mean()]], fmt="o", color="black", capsize=4)
ax[1].axhline(0, color="gray", lw=1)
ax[1].set_xticks(range(len(fracs)))
ax[1].set_xticklabels([f"{f * 100:g}%" for f in fracs])
ax[1].set_xlabel("label fraction")
ax[1].set_ylabel("test IoU gap (pretrained - scratch)")
ax[1].set_title("Paired gap per seed, mean with 95% t-CI")
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(Path(args.out, "label_efficiency.png"), dpi=200)
fig.savefig(Path(args.out, "label_efficiency.pdf"))
print(f"\nwrote {args.out}/label_efficiency.png, .pdf and summary_tables.md")
