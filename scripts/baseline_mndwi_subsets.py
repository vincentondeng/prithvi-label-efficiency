"""MNDWI threshold baseline using the same label subsets as the deep models."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from lepipe import subsets
from lepipe.data import IGNORE_INDEX, MEAN, STD, Sen1Floods11, read_split

p = argparse.ArgumentParser()
p.add_argument("--splits-dir", default="data/splits")
p.add_argument("--root", default="data/sen1floods11")
p.add_argument("--subsets", default="subsets/label_subsets.json")
p.add_argument("--out", default="results_baseline/mndwi.json")
p.add_argument("--n-seeds", type=int, default=10)
args = p.parse_args()

NB = 2000
edges = np.linspace(-1, 1, NB + 1)


def chip_hist(s):
    img = s["image"].numpy() * STD[:, None, None] + MEAN[:, None, None]
    green, swir1 = img[1], img[4]
    den = green + swir1
    m = np.divide(green - swir1, den, out=np.full_like(den, -1.0), where=den != 0)
    lab = s["label"].numpy()
    valid = lab != IGNORE_INDEX
    return (np.histogram(m[valid & (lab == 1)], bins=edges)[0],
            np.histogram(m[valid & (lab == 0)], bins=edges)[0])


def split_hists(split):
    ds = Sen1Floods11(args.root, read_split(Path(args.splits_dir) / f"flood_{split}_data.csv"))
    out = {}
    for i in range(len(ds)):
        s = ds[i]
        out[s["name"]] = chip_hist(s)
    return out


def agg(hs):
    hs = list(hs)
    return sum(h[0] for h in hs), sum(h[1] for h in hs)


def iou_curve(hw, hl):
    tp = np.cumsum(hw[::-1])[::-1]
    fp = np.cumsum(hl[::-1])[::-1]
    fn = hw.sum() - tp
    return tp / np.maximum(tp + fp + fn, 1)


train = split_hists("train")
curves = {s: iou_curve(*agg(split_hists(s).values())) for s in ["valid", "test", "bolivia"]}

res = []
for fr in subsets.FRACTIONS:
    for seed in range(args.n_seeds):
        names = subsets.get_subset(args.subsets, seed, fr)
        k = int(np.argmax(iou_curve(*agg(train[n] for n in names))))
        res.append({"fraction": fr, "seed": seed, "threshold": float(edges[k]),
                    **{s: float(curves[s][k]) for s in curves}})
Path(args.out).parent.mkdir(exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)

print("MNDWI threshold fitted on the label subset (mean ± sd over seeds)")
print(f"{'labels':>7s} {'chips':>6s} {'threshold':>16s} {'test IoU':>14s} {'valid IoU':>14s} {'Bolivia IoU':>14s}")
for fr in subsets.FRACTIONS:
    r = [x for x in res if x["fraction"] == fr]
    f = lambda key: f"{np.mean([x[key] for x in r]):.3f} ± {np.std([x[key] for x in r], ddof=1):.3f}"
    n = subsets.n_chips(fr, 252)
    print(f"{fr * 100:6g}% {n:6d} {f('threshold'):>16s} {f('test'):>14s} {f('valid'):>14s} {f('bolivia'):>14s}")
