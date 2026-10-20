"""Sen1Floods11 hand-labeled chips for Prithvi fine-tuning."""
import csv
from pathlib import Path

import numpy as np
import torch
from osgeo import gdal
from torch.utils.data import Dataset

gdal.UseExceptions()

# 1-based file bands: B2 Blue, B3 Green, B4 Red, B8A Narrow NIR, B11 SWIR1, B12 SWIR2
FILE_BANDS = [2, 3, 4, 9, 12, 13]
# Official Prithvi-EO-2.0 pretraining constants (from the model's config.json)
MEAN = np.array([1087, 1342, 1433, 2734, 1958, 1363], dtype=np.float32)
STD = np.array([2248, 2179, 2178, 1850, 1242, 1049], dtype=np.float32)
IGNORE_INDEX = 255  # label value for no-data pixels (originally -1)


def read_split(csv_path):
    """Return a list of (s2_filename, label_filename) from a split CSV.
    The CSVs list Sentinel-1 names; the S2 file has the same name with S2Hand."""
    pairs = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                pairs.append((row[0].strip().replace("S1Hand", "S2Hand"), row[1].strip()))
    return pairs


class Sen1Floods11(Dataset):
    """Returns {'image': (6,H,W) float32 normalised, 'label': (H,W) int64 in {0,1,255}, 'name'}."""

    def __init__(self, root, pairs):
        self.root = Path(root)
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s2_name, label_name = self.pairs[i]
        ds = gdal.Open(str(self.root / "S2Hand" / s2_name))
        img = np.stack([ds.GetRasterBand(b).ReadAsArray() for b in FILE_BANDS]).astype(np.float32)
        ds = None
        img = (img - MEAN[:, None, None]) / STD[:, None, None]

        ds = gdal.Open(str(self.root / "LabelHand" / label_name))
        label = ds.ReadAsArray().astype(np.int64)
        ds = None
        label[label == -1] = IGNORE_INDEX
        return {"image": torch.from_numpy(img), "label": torch.from_numpy(label), "name": s2_name}
