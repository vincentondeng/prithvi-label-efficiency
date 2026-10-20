"""Step 4a: exercise the dataset on every chip and print training-relevant statistics."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch.utils.data import DataLoader

from lepipe.data import IGNORE_INDEX, MEAN, STD, Sen1Floods11, read_split

p = argparse.ArgumentParser()
p.add_argument("--splits-dir", default="data/splits")
p.add_argument("--root", default="data/sen1floods11")
args = p.parse_args()

zero_pixel = (-MEAN / STD)[:, None, None]  # what an all-zero raw pixel becomes after normalising
for split in ["train", "valid", "test", "bolivia"]:
    pairs = read_split(Path(args.splits_dir) / f"flood_{split}_data.csv")
    ds = Sen1Floods11(args.root, pairs)
    n_valid = n_water = n_pix = n_nowater_chips = n_zero_but_valid = 0
    vmin, vmax, finite = np.inf, -np.inf, True
    for i in range(len(ds)):
        s = ds[i]
        img, lab = s["image"].numpy(), s["label"].numpy()
        valid = lab != IGNORE_INDEX
        n_pix += lab.size
        n_valid += valid.sum()
        n_water += (lab == 1).sum()
        n_nowater_chips += int((lab == 1).sum() == 0)
        finite &= bool(np.isfinite(img).all())
        vmin, vmax = min(vmin, img.min()), max(vmax, img.max())
        zeros = np.all(np.isclose(img, zero_pixel, atol=1e-4), axis=0)
        n_zero_but_valid += int((zeros & valid).sum())
    print(f"{split:8s} chips={len(ds):3d} valid={100 * n_valid / n_pix:5.1f}% "
          f"water(of valid)={100 * n_water / n_valid:5.1f}% chips_without_water={n_nowater_chips} "
          f"finite={finite} range=[{vmin:.1f},{vmax:.1f}] all-zero-pixels-labelled-valid={n_zero_but_valid}")

ds = Sen1Floods11(args.root, read_split(Path(args.splits_dir) / "flood_train_data.csv"))
batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
print("batch image:", tuple(batch["image"].shape), batch["image"].dtype)
print("batch label:", tuple(batch["label"].shape), batch["label"].dtype,
      "values:", sorted(set(batch["label"].unique().tolist())))
