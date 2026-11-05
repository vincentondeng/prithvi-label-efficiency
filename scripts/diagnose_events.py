"""Per-event and per-chip diagnostics with no model: is an event hard because of the data/labels,
or because of the deep models?

For every event it reports the MNDWI-rule water IoU on the test split, using
  (a) one global threshold fitted on ALL training chips (same idea as the 100% baseline), and
  (b) an 'oracle' threshold fitted on that event's own test chips (diagnostic upper bound only).
For each chip of the chosen event (default Ghana) it lists water / no-data / empty fractions,
the rule's IoU, MNDWI medians for water and land pixels, and mean blue reflectance (haze/cloud hint).

Usage: python scripts/diagnose_events.py [--event Ghana] [--root data/sen1floods11] [--splits-dir data/splits]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lepipe.data import FILE_BANDS, read_split  # noqa: E402

G, S, B = 1, 4, 0  # positions of green (B03), SWIR1 (B11), blue (B02) within FILE_BANDS
EDGES = np.linspace(-1, 1, 401)  # MNDWI histogram bins (width 0.005)
AFRICA = {"Ghana", "Nigeria", "Somalia"}


def load_chip(root, s2_name, label_name):
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(str(Path(root) / "S2Hand" / s2_name))
    raw = np.stack([ds.GetRasterBand(b).ReadAsArray() for b in FILE_BANDS]).astype(np.float32)
    ds = gdal.Open(str(Path(root) / "LabelHand" / label_name))
    lab = ds.ReadAsArray().astype(np.int64)
    return raw, lab


def chip_arrays(raw, lab):
    """Return MNDWI and label over valid pixels (labelled, non-empty, defined index), plus fractions."""
    empty = (raw == 0).all(axis=0)
    g, s = raw[G], raw[S]
    denom = g + s
    defined = denom > 0
    valid = (lab != -1) & ~empty & defined
    m = np.zeros_like(g)
    m[defined] = (g[defined] - s[defined]) / denom[defined]
    n = lab.size
    info = {
        "nodata_frac": float((lab == -1).mean()),
        "empty_frac": float(empty.mean()),
        "valid_px": int(valid.sum()),
        "water_frac": float((lab[valid] == 1).mean()) if valid.any() else float("nan"),
        "blue_mean": float(raw[B][~empty].mean() / 1e4) if (~empty).any() else float("nan"),
    }
    return m[valid], lab[valid], info


def hists(m, y):
    hw, _ = np.histogram(m[y == 1], EDGES)
    hl, _ = np.histogram(m[y == 0], EDGES)
    return hw.astype(np.int64), hl.astype(np.int64)


def iou_curve(hw, hl):
    """IoU of the rule 'MNDWI > tau' for tau at every bin edge (index k: tau = EDGES[k])."""
    # pixels above tau = sum of bins k..end
    tp = np.concatenate([np.cumsum(hw[::-1])[::-1], [0]])
    fp = np.concatenate([np.cumsum(hl[::-1])[::-1], [0]])
    fn = hw.sum() - tp
    return tp / np.maximum(tp + fp + fn, 1)


def iou_at(hw, hl, tau):
    k = int(np.searchsorted(EDGES, tau))
    return float(iou_curve(hw, hl)[k])


p = argparse.ArgumentParser()
p.add_argument("--event", default="Ghana")
p.add_argument("--root", default="data/sen1floods11")
p.add_argument("--splits-dir", default="data/splits")
args = p.parse_args()

# 1) global threshold from all training chips
hw_tr = np.zeros(len(EDGES) - 1, np.int64); hl_tr = np.zeros_like(hw_tr)
for s2, lab in read_split(Path(args.splits_dir) / "flood_train_data.csv"):
    m, y, _ = chip_arrays(*load_chip(args.root, s2, lab))
    a, b = hists(m, y); hw_tr += a; hl_tr += b
curve = iou_curve(hw_tr, hl_tr)
k_star = int(np.argmax(curve)); tau_g = float(EDGES[k_star])
print(f"global MNDWI threshold fitted on all {sum(1 for _ in read_split(Path(args.splits_dir) / 'flood_train_data.csv'))} "
      f"training chips: tau = {tau_g:+.3f} (training IoU {curve[k_star]:.3f})\n")

# 2) per-event test results and per-chip stats
ev_h = defaultdict(lambda: [np.zeros_like(hw_tr), np.zeros_like(hl_tr)])
ev_info = defaultdict(list)
for s2, lab in read_split(Path(args.splits_dir) / "flood_test_data.csv"):
    event = Path(s2).name.split("_")[0]
    m, y, info = chip_arrays(*load_chip(args.root, s2, lab))
    a, b = hists(m, y)
    ev_h[event][0] += a; ev_h[event][1] += b
    info.update({
        "chip": Path(s2).name.replace("_S2Hand.tif", ""),
        "rule_iou": iou_at(a, b, tau_g) if a.sum() else float("nan"),
        "mndwi_water_med": float(np.median(m[y == 1])) if (y == 1).any() else float("nan"),
        "mndwi_land_med": float(np.median(m[y == 0])) if (y == 0).any() else float("nan"),
    })
    ev_info[event].append(info)

print("### MNDWI rule on the test split, per event\n")
print("| event | chips | water frac | no-data frac | rule IoU (global tau) | oracle tau | rule IoU (oracle tau) |")
print("|---|---|---|---|---|---|---|")
for e in sorted(ev_h):
    hw, hl = ev_h[e]
    c = iou_curve(hw, hl); ko = int(np.argmax(c))
    infos = ev_info[e]
    wf = hw.sum() / max(hw.sum() + hl.sum(), 1)
    nd = np.mean([i["nodata_frac"] for i in infos])
    star = " *" if e in AFRICA else ""
    print(f"| {e}{star} | {len(infos)} | {wf:.3f} | {nd:.3f} | {iou_at(hw, hl, tau_g):.3f} | {EDGES[ko]:+.3f} | {c[ko]:.3f} |")
hw_all = sum(v[0] for v in ev_h.values()); hl_all = sum(v[1] for v in ev_h.values())
print(f"| all test | {sum(len(v) for v in ev_info.values())} | | | {iou_at(hw_all, hl_all, tau_g):.3f} | | |")
print("\n'water frac' = water share of valid pixels (pooled); 'oracle tau' is fitted on the event's own test chips "
      "and is a diagnostic upper bound, not a fair baseline.\n")

print(f"### {args.event} test chips\n")
print("| chip | valid px | water frac | no-data frac | empty frac | rule IoU | MNDWI median water | MNDWI median land | mean blue refl. |")
print("|---|---|---|---|---|---|---|---|---|")
for i in sorted(ev_info[args.event], key=lambda r: r["chip"]):
    print(f"| {i['chip']} | {i['valid_px']} | {i['water_frac']:.3f} | {i['nodata_frac']:.3f} | {i['empty_frac']:.3f} | "
          f"{i['rule_iou']:.3f} | {i['mndwi_water_med']:+.3f} | {i['mndwi_land_med']:+.3f} | {i['blue_mean']:.3f} |")
