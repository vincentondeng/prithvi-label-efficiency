"""Step 3d: real chip -> official normalisation -> Prithvi encoder."""
import sys
import numpy as np
import torch
from osgeo import gdal
from terratorch.registry import BACKBONE_REGISTRY

gdal.UseExceptions()
NAME = "terratorch_prithvi_eo_v2_100_tl"
ROOT = "data/sen1floods11"
FILE_BANDS = [2, 3, 4, 9, 12, 13]  # 1-based: B2, B3, B4, B8A, B11, B12
MEAN = np.array([1087, 1342, 1433, 2734, 1958, 1363], dtype=np.float32)
STD = np.array([2248, 2179, 2178, 1850, 1242, 1049], dtype=np.float32)

chip = sys.argv[1] if len(sys.argv) > 1 else "Ghana_103272"
img = gdal.Open(f"{ROOT}/S2Hand/{chip}_S2Hand.tif").ReadAsArray().astype(np.float32)
x = img[[b - 1 for b in FILE_BANDS]]
x = (x - MEAN[:, None, None]) / STD[:, None, None]
print(f"{chip}: normalised per-band mean/std:")
for i, n in enumerate(["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]):
    print(f"  {n:6s} mean={x[i].mean():6.2f} std={x[i].std():5.2f}")

model = BACKBONE_REGISTRY.build(NAME, pretrained=True, num_frames=1).eval()
crop = torch.from_numpy(x[:, :224, :224]).reshape(1, 6, 1, 224, 224)
with torch.no_grad():
    out = model(crop)
print("224 crop -> layers:", len(out), "| last shape:", tuple(out[-1].shape),
      "| finite:", bool(torch.isfinite(out[-1]).all()))

try:
    big = BACKBONE_REGISTRY.build(NAME, pretrained=True, num_frames=1, img_size=512).eval()
    full = torch.from_numpy(x).reshape(1, 6, 1, 512, 512)
    with torch.no_grad():
        out = big(full)
    print("512 full chip -> last shape:", tuple(out[-1].shape),
          "| finite:", bool(torch.isfinite(out[-1]).all()))
except Exception as e:
    print("512 build/forward failed:", type(e).__name__, str(e)[:300])
