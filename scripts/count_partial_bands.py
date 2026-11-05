"""Count 'partial-band' pixels: some but not all of the six input bands are exactly zero.
Our current mask (lepipe/data.py, mask_empty) only removes pixels where ALL six bands are zero,
so partial-band pixels (typical at Sentinel-2 swath/detector edges) are fed to the models and scored.

Reports, per split and for the most affected chips: partial-band pixels, how they are labelled
(water / land / no-data), and which bands are zero.

Usage: python scripts/count_partial_bands.py [--root data/sen1floods11] [--splits-dir data/splits] [--top 15]
Writes figures/partial_bands.csv
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lepipe.data import FILE_BANDS, read_split  # noqa: E402

NAMES = ["B02", "B03", "B04", "B8A", "B11", "B12"]
SPLITS = ["train", "valid", "test", "bolivia"]


def load_chip(root, s2_name, label_name):
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(str(Path(root) / "S2Hand" / s2_name))
    raw = np.stack([ds.GetRasterBand(b).ReadAsArray() for b in FILE_BANDS]).astype(np.float32)
    lab = gdal.Open(str(Path(root) / "LabelHand" / label_name)).ReadAsArray().astype(np.int64)
    return raw, lab


p = argparse.ArgumentParser()
p.add_argument("--root", default="data/sen1floods11")
p.add_argument("--splits-dir", default="data/splits")
p.add_argument("--top", type=int, default=15)
p.add_argument("--out", default="figures/partial_bands.csv")
args = p.parse_args()

rows = []
for split in SPLITS:
    f = Path(args.splits_dir) / f"flood_{split}_data.csv"
    if not f.exists():
        print(f"(skipping {split}: {f} not found)"); continue
    for s2, lab_name in read_split(f):
        raw, lab = load_chip(args.root, s2, lab_name)
        zero = raw == 0
        nz = zero.sum(axis=0)
        partial = (nz > 0) & (nz < 6)
        lab_p = lab[partial]
        pattern = Counter("+".join(NAMES[b] for b in range(6) if zero[b, i, j])
                          for i, j in zip(*np.nonzero(partial))) if partial.any() else Counter()
        labelled = (lab != -1) & (nz < 6)
        rows.append({
            "split": split, "chip": Path(s2).name.replace("_S2Hand.tif", ""),
            "partial_px": int(partial.sum()),
            "partial_labelled_water": int((lab_p == 1).sum()),
            "partial_labelled_land": int((lab_p == 0).sum()),
            "partial_nodata": int((lab_p == -1).sum()),
            "labelled_px": int(labelled.sum()),
            "zero_bands_most_common": pattern.most_common(1)[0][0] if pattern else "",
        })

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("### Partial-band pixels by split (counted only where currently scored: labelled water or land)\n")
print("| split | chips | chips affected | scored partial px | as water | as land | share of scored px |")
print("|---|---|---|---|---|---|---|")
for split in SPLITS:
    rs = [r for r in rows if r["split"] == split]
    if not rs: continue
    pw = sum(r["partial_labelled_water"] for r in rs); pl = sum(r["partial_labelled_land"] for r in rs)
    tot = sum(r["labelled_px"] for r in rs)
    aff = sum((r["partial_labelled_water"] + r["partial_labelled_land"]) > 0 for r in rs)
    print(f"| {split} | {len(rs)} | {aff} | {pw + pl} | {pw} | {pl} | {(pw + pl) / max(tot, 1):.3%} |")

print(f"\n### Top {args.top} chips by scored partial-band pixels\n")
print("| chip | split | scored partial px | as water | as land | most common zero bands |")
print("|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: -(r["partial_labelled_water"] + r["partial_labelled_land"]))[: args.top]:
    print(f"| {r['chip']} | {r['split']} | {r['partial_labelled_water'] + r['partial_labelled_land']} | "
          f"{r['partial_labelled_water']} | {r['partial_labelled_land']} | {r['zero_bands_most_common']} |")
print(f"\nwrote {args.out} ({len(rows)} chips)")
