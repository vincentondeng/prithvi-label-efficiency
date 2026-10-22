"""Non-deep baseline: MNDWI threshold, evaluated on the same splits, masking and metric as the models."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from lepipe.data import IGNORE_INDEX, MEAN, STD, Sen1Floods11, read_split

p = argparse.ArgumentParser()
p.add_argument("--splits-dir", default="data/splits")
p.add_argument("--root", default="data/sen1floods11")
args = p.parse_args()

NB = 2000
edges = np.linspace(-1, 1, NB + 1)


def hists(split):
    """Histograms of MNDWI over valid water / valid non-water pixels."""
    ds = Sen1Floods11(args.root, read_split(Path(args.splits_dir) / f"flood_{split}_data.csv"))
    hw, hl = np.zeros(NB, dtype=np.int64), np.zeros(NB, dtype=np.int64)
    for i in range(len(ds)):
        s = ds[i]
        img = s["image"].numpy() * STD[:, None, None] + MEAN[:, None, None]  # back to reflectance units
        green, swir1 = img[1], img[4]
        den = green + swir1
        m = np.divide(green - swir1, den, out=np.full_like(den, -1.0), where=den != 0)
        lab = s["label"].numpy()
        valid = lab != IGNORE_INDEX
        hw += np.histogram(m[valid & (lab == 1)], bins=edges)[0]
        hl += np.histogram(m[valid & (lab == 0)], bins=edges)[0]
    return hw, hl


def iou_curve(hw, hl):
    """Water IoU for every threshold: predict water where MNDWI >= edges[k]."""
    tp = np.cumsum(hw[::-1])[::-1]
    fp = np.cumsum(hl[::-1])[::-1]
    fn = hw.sum() - tp
    return tp / np.maximum(tp + fp + fn, 1)


H = {s: hists(s) for s in ["train", "valid", "test", "bolivia"]}
curves = {s: iou_curve(*H[s]) for s in H}
k_train = int(np.argmax(curves["train"]))
k_zero = NB // 2
print(f"threshold chosen on the 252 training chips: {edges[k_train]:+.3f}")
print(f"{'split':8s} {'IoU @ t=0':>10s} {'IoU @ train-chosen t':>22s} {'oracle best (t)':>18s}")
for s in ["train", "valid", "test", "bolivia"]:
    c = curves[s]
    kb = int(np.argmax(c))
    print(f"{s:8s} {c[k_zero]:10.3f} {c[k_train]:22.3f} {c[kb]:12.3f} ({edges[kb]:+.2f})")
