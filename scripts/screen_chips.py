"""Model-free screen of all hand-labelled chips for label / optical-evidence disagreement.

Per chip it computes:
  water_px, land_px        labelled water / land pixels (no-data and empty imagery excluded)
  nodata_frac, empty_frac  shares of the chip that are no-data (-1) or empty imagery
  cloud_frac               share of non-empty pixels with blue TOA reflectance > 0.2 (rough cloud/haze hint)
  water_invisible          share of labelled-water pixels with MNDWI < 0 (water not visible optically)
  land_waterlike_px        labelled-land pixels with MNDWI > 0.5 (land that looks like open water)
  rule_iou                 MNDWI-rule IoU on the chip at the global training threshold
  suspect_water_px         water_invisible * water_px + land_waterlike_px   (ranking key)
  event_water_share        chip's share of its event's labelled water in its split (its weight in pooled IoU)

Flags (criteria fixed before inspection; a flag means 'optically ambiguous', NOT 'wrong label'):
  F1 invisible water   water_px >= 1000 and water_invisible >= 0.40
  F2 water-like land   land_waterlike_px >= 5000  (>= 0.5 km2)
  F3 cloudy            cloud_frac >= 0.30
  F4 rule fails        water_px >= 5000 and rule_iou < 0.50

Usage: python scripts/screen_chips.py [--root data/sen1floods11] [--splits-dir data/splits] [--top 15]
Writes figures/chip_screen.csv
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lepipe.data import FILE_BANDS, read_split  # noqa: E402

BLUE, GREEN, SWIR1 = 0, 1, 4
EDGES = np.linspace(-1, 1, 401)
AFRICA = {"Ghana", "Nigeria", "Somalia"}
SPLITS = ["train", "valid", "test", "bolivia"]


def load_chip(root, s2_name, label_name):
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(str(Path(root) / "S2Hand" / s2_name))
    raw = np.stack([ds.GetRasterBand(b).ReadAsArray() for b in FILE_BANDS]).astype(np.float32)
    lab = gdal.Open(str(Path(root) / "LabelHand" / label_name)).ReadAsArray().astype(np.int64)
    return raw, lab


def iou_curve(hw, hl):
    tp = np.concatenate([np.cumsum(hw[::-1])[::-1], [0]])
    fp = np.concatenate([np.cumsum(hl[::-1])[::-1], [0]])
    fn = hw.sum() - tp
    return tp / np.maximum(tp + fp + fn, 1)


p = argparse.ArgumentParser()
p.add_argument("--root", default="data/sen1floods11")
p.add_argument("--splits-dir", default="data/splits")
p.add_argument("--top", type=int, default=15)
p.add_argument("--out", default="figures/chip_screen.csv")
args = p.parse_args()

rows, hists = [], {}
for split in SPLITS:
    f = Path(args.splits_dir) / f"flood_{split}_data.csv"
    if not f.exists():
        print(f"(skipping {split}: {f} not found)")
        continue
    for s2, lab_name in read_split(f):
        raw, lab = load_chip(args.root, s2, lab_name)
        empty = (raw == 0).all(axis=0)
        g, s = raw[GREEN], raw[SWIR1]
        den = g + s
        ok = den > 0
        m = np.zeros_like(g)
        m[ok] = (g[ok] - s[ok]) / den[ok]
        valid = (lab != -1) & ~empty & ok
        w, l = valid & (lab == 1), valid & (lab == 0)
        hw, _ = np.histogram(m[w], EDGES)
        hl, _ = np.histogram(m[l], EDGES)
        chip = Path(s2).name.replace("_S2Hand.tif", "")
        hists[chip] = (hw.astype(np.int64), hl.astype(np.int64))
        ne = ~empty
        rows.append({
            "split": split, "event": chip.split("_")[0], "chip": chip,
            "water_px": int(w.sum()), "land_px": int(l.sum()),
            "nodata_frac": round(float((lab == -1).mean()), 4), "empty_frac": round(float(empty.mean()), 4),
            "cloud_frac": round(float((raw[BLUE][ne] > 2000).mean()) if ne.any() else float("nan"), 4),
            "water_invisible": round(float((m[w] < 0).mean()) if w.any() else float("nan"), 4),
            "land_waterlike_px": int((m[l] > 0.5).sum()),
        })

# global threshold from training chips (same fit as diagnose_events.py)
hw_tr = sum(hists[r["chip"]][0] for r in rows if r["split"] == "train")
hl_tr = sum(hists[r["chip"]][1] for r in rows if r["split"] == "train")
curve = iou_curve(hw_tr, hl_tr)
k = int(np.argmax(curve)); tau = float(EDGES[k])
print(f"global threshold from training chips: tau = {tau:+.3f}\n")

water_by_event_split = defaultdict(int)
for r in rows:
    water_by_event_split[(r["split"], r["event"])] += r["water_px"]
for r in rows:
    hw, hl = hists[r["chip"]]
    r["rule_iou"] = round(float(iou_curve(hw, hl)[k]), 4) if hw.sum() else float("nan")
    wi = 0.0 if np.isnan(r["water_invisible"]) else r["water_invisible"]
    r["suspect_water_px"] = int(round(wi * r["water_px"] + r["land_waterlike_px"]))
    tot = water_by_event_split[(r["split"], r["event"])]
    r["event_water_share"] = round(r["water_px"] / tot, 4) if tot else 0.0
    flags = []
    if r["water_px"] >= 1000 and wi >= 0.40: flags.append("F1")
    if r["land_waterlike_px"] >= 5000: flags.append("F2")
    if not np.isnan(r["cloud_frac"]) and r["cloud_frac"] >= 0.30: flags.append("F3")
    if r["water_px"] >= 5000 and not np.isnan(r["rule_iou"]) and r["rule_iou"] < 0.50: flags.append("F4")
    r["flags"] = " ".join(flags)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)

print("### Flag counts by split (chips flagged / chips; share of the split's labelled water in flagged chips)\n")
print("| split | chips | any flag | F1 invisible water | F2 water-like land | F3 cloudy | F4 rule fails | water in flagged chips |")
print("|---|---|---|---|---|---|---|---|")
for split in SPLITS:
    rs = [r for r in rows if r["split"] == split]
    if not rs:
        continue
    wt = sum(r["water_px"] for r in rs)
    wf = sum(r["water_px"] for r in rs if r["flags"])
    c = lambda f: sum(f in r["flags"] for r in rs)
    print(f"| {split} | {len(rs)} | {sum(bool(r['flags']) for r in rs)} | {c('F1')} | {c('F2')} | {c('F3')} | {c('F4')} | "
          f"{wf / max(wt, 1):.0%} |")

print("\n### Test split by event (flagged chips; share of the event's test water in flagged chips)\n")
print("| event | chips | flagged | water in flagged chips |\n|---|---|---|---|")
for e in sorted({r["event"] for r in rows if r["split"] == "test"}):
    rs = [r for r in rows if r["split"] == "test" and r["event"] == e]
    wt = sum(r["water_px"] for r in rs); wf = sum(r["water_px"] for r in rs if r["flags"])
    star = " *" if e in AFRICA else ""
    print(f"| {e}{star} | {len(rs)} | {sum(bool(r['flags']) for r in rs)} | {wf / max(wt, 1):.0%} |")

print(f"\n### Top {args.top} chips by suspect water pixels (all splits)\n")
print("| chip | split | flags | water px | invisible water | water-like land px | cloud frac | rule IoU | share of event water |")
print("|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: -r["suspect_water_px"])[: args.top]:
    print(f"| {r['chip']} | {r['split']} | {r['flags'] or '-'} | {r['water_px']} | {r['water_invisible']:.2f} | "
          f"{r['land_waterlike_px']} | {r['cloud_frac']:.2f} | {r['rule_iou']:.3f} | {r['event_water_share']:.2f} |")
print(f"\nwrote {args.out} ({len(rows)} chips)")
