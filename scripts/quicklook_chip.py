"""Save a 6-panel quicklook of one Sen1Floods11 hand-labelled chip.

Panels: true colour (B04,B03,B02) | false colour (B12,B8A,B04) | hand label |
        MNDWI | MNDWI histograms for labelled water and land | rule vs label (TP/FP/FN).

Usage: python scripts/quicklook_chip.py Ghana_866994 [--tau 0.15] [--root data/sen1floods11] [--out figures/quicklook]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lepipe.data import FILE_BANDS  # noqa: E402  (B02,B03,B04,B8A,B11,B12)

BLUE, GREEN, RED, NIR, SWIR1, SWIR2 = range(6)


def load_chip(root, chip):
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(str(Path(root) / "S2Hand" / f"{chip}_S2Hand.tif"))
    raw = np.stack([ds.GetRasterBand(b).ReadAsArray() for b in FILE_BANDS]).astype(np.float32)
    lab = gdal.Open(str(Path(root) / "LabelHand" / f"{chip}_LabelHand.tif")).ReadAsArray().astype(np.int64)
    return raw, lab


def stretch(rgb, empty):
    """2-98 percentile stretch per band over non-empty pixels."""
    out = np.zeros_like(rgb)
    for i in range(3):
        v = rgb[i][~empty]
        lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
        out[i] = np.clip((rgb[i] - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.moveaxis(out, 0, -1)


p = argparse.ArgumentParser()
p.add_argument("chip")
p.add_argument("--tau", type=float, default=0.15, help="MNDWI threshold (default: global fit from diagnose_events.py)")
p.add_argument("--root", default="data/sen1floods11")
p.add_argument("--out", default="figures/quicklook")
args = p.parse_args()

raw, lab = load_chip(args.root, args.chip)
empty = (raw == 0).all(axis=0)
g, s = raw[GREEN], raw[SWIR1]
den = g + s
mndwi = np.full(g.shape, np.nan, np.float32)
ok = den > 0
mndwi[ok] = (g[ok] - s[ok]) / den[ok]
valid = (lab != -1) & ~empty & ok
pred = mndwi > args.tau

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.ravel()
ax[0].imshow(stretch(raw[[RED, GREEN, BLUE]], empty)); ax[0].set_title("True colour (B04, B03, B02)")
ax[1].imshow(stretch(raw[[SWIR2, NIR, RED]], empty)); ax[1].set_title("False colour (B12, B8A, B04)\nwater dark, vegetation green")

lab_img = np.where(lab == -1, 2, lab).astype(float)
lab_img[empty] = 3
ax[2].imshow(lab_img, cmap=ListedColormap(["#e8dcc0", "#1f5fbf", "#9a9a9a", "#000000"]), vmin=0, vmax=3,
             interpolation="nearest")
ax[2].set_title("Hand label")
ax[2].legend(handles=[Patch(color="#e8dcc0", label="land"), Patch(color="#1f5fbf", label="water"),
                      Patch(color="#9a9a9a", label="no-data"), Patch(color="#000000", label="empty imagery")],
             loc="lower left", fontsize=8, framealpha=0.85)

im = ax[3].imshow(mndwi, cmap="RdBu", vmin=-1, vmax=1); ax[3].set_title("MNDWI (blue = water-like)")
fig.colorbar(im, ax=ax[3], fraction=0.046)

bins = np.linspace(-1, 1, 81)
for cls, col, name in [(1, "#1f5fbf", "labelled water"), (0, "#b08a3c", "labelled land")]:
    v = mndwi[valid & (lab == cls)]
    if v.size:
        ax[4].hist(v, bins=bins, color=col, alpha=0.6, label=f"{name} (n={v.size})", density=True)
ax[4].axvline(args.tau, color="k", ls="--", label=f"tau = {args.tau:+.2f}")
ax[4].set_xlabel("MNDWI"); ax[4].set_ylabel("density"); ax[4].legend(fontsize=8)
ax[4].set_title("MNDWI by label")

cmp_img = np.full(lab.shape, 4, int)  # 4 = not evaluated
cmp_img[valid & pred & (lab == 1)] = 0   # TP
cmp_img[valid & pred & (lab == 0)] = 1   # FP
cmp_img[valid & ~pred & (lab == 1)] = 2  # FN
cmp_img[valid & ~pred & (lab == 0)] = 3  # TN
ax[5].imshow(cmp_img, cmap=ListedColormap(["#1f5fbf", "#e03c31", "#f2b134", "#f2efe6", "#9a9a9a"]),
             vmin=0, vmax=4, interpolation="nearest")
tp, fp, fn = [(cmp_img == k).sum() for k in (0, 1, 2)]
iou = tp / max(tp + fp + fn, 1)
ax[5].set_title(f"Rule vs label (tau={args.tau:+.2f}): IoU {iou:.3f}")
ax[5].legend(handles=[Patch(color="#1f5fbf", label=f"TP {tp}"), Patch(color="#e03c31", label=f"FP {fp}"),
                      Patch(color="#f2b134", label=f"FN {fn}"), Patch(color="#f2efe6", label="TN"),
                      Patch(color="#9a9a9a", label="not evaluated")], loc="lower left", fontsize=8, framealpha=0.85)

for a in (0, 1, 2, 3, 5):
    ax[a].set_xticks([]); ax[a].set_yticks([])
fig.suptitle(f"{args.chip}   (512 x 512 px at 10 m = 5.12 km; tau from global training fit)", fontsize=13)
fig.tight_layout()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
path = out / f"{args.chip}.png"
fig.savefig(path, dpi=110)
print(f"valid px {valid.sum()}, water {int((valid & (lab == 1)).sum())}, land {int((valid & (lab == 0)).sum())}, "
      f"no-data {int((lab == -1).sum())}, empty {int(empty.sum())}")
print(f"MNDWI median: water {np.nanmedian(mndwi[valid & (lab == 1)]) if (valid & (lab == 1)).any() else float('nan'):+.3f}, "
      f"land {np.nanmedian(mndwi[valid & (lab == 0)]) if (valid & (lab == 0)).any() else float('nan'):+.3f}")
print(f"rule IoU {iou:.3f} (TP {tp}, FP {fp}, FN {fn})")
print("wrote", path)
