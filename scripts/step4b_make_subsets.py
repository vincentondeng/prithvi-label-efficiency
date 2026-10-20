"""Step 4b: generate label subsets and report how many contain water."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from lepipe import subsets
from lepipe.data import IGNORE_INDEX, Sen1Floods11, read_split

p = argparse.ArgumentParser()
p.add_argument("--splits-dir", default="data/splits")
p.add_argument("--root", default="data/sen1floods11")
p.add_argument("--out", default="subsets/label_subsets.json")
p.add_argument("--n-seeds", type=int, default=5)
args = p.parse_args()

pairs = read_split(Path(args.splits_dir) / "flood_train_data.csv")
names = [s2 for s2, _ in pairs]
perms = subsets.make_permutations(names, range(args.n_seeds))
subsets.save(args.out, perms, len(names))
print(f"wrote {args.out}: {len(names)} chips, seeds 0..{args.n_seeds - 1}")

# per-chip water statistics (labels after masking)
ds = Sen1Floods11(args.root, pairs)
stats = {}
for i in range(len(ds)):
    s = ds[i]
    lab = s["label"].numpy()
    stats[s["name"]] = (int((lab == 1).sum()), int((lab != IGNORE_INDEX).sum()))

print("\ncell = chips-with-water / chips (water share of valid pixels)")
print("frac   n   " + "  ".join(f"seed{s:<10d}" for s in range(args.n_seeds)))
for frac in subsets.FRACTIONS[:-1]:
    n = subsets.n_chips(frac, len(names))
    cells = []
    for s in range(args.n_seeds):
        chosen = subsets.get_subset(args.out, s, frac)
        w = sum(stats[c][0] for c in chosen)
        v = sum(stats[c][1] for c in chosen)
        hw = sum(stats[c][0] > 0 for c in chosen)
        cells.append(f"{hw:3d}/{n:<3d}({100 * w / v:4.1f}%)")
    print(f"{frac:4.2f} {n:3d}   " + "  ".join(cells))
tot_w = sum(w for w, _ in stats.values())
tot_v = sum(v for _, v in stats.values())
print(f"full {len(names):3d}   {sum(w > 0 for w, _ in stats.values())}/{len(names)} ({100 * tot_w / tot_v:4.1f}%)")
