"""Step 4a-2: where are image-empty pixels that the labels call valid?"""
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

zero_pixel = (-MEAN / STD)[:, None, None]
for split in ["train", "valid", "test", "bolivia"]:
    ds = Sen1Floods11(args.root, read_split(Path(args.splits_dir) / f"flood_{split}_data.csv"))
    rows = []
    for i in range(len(ds)):
        s = ds[i]
        img, lab = s["image"].numpy(), s["label"].numpy()
        empty = np.all(np.isclose(img, zero_pixel, atol=1e-4), axis=0) & (lab != IGNORE_INDEX)
        if empty.sum() > 0:
            rows.append((int(empty.sum()), s["name"], int((lab[empty] == 0).sum()), int((lab[empty] == 1).sum())))
    rows.sort(reverse=True)
    print(f"{split}: {len(rows)} chips affected (of {len(ds)})")
    for n, name, n0, n1 in rows[:5]:
        print(f"   {name}: {n} px  (label 0: {n0}, label 1: {n1})")
