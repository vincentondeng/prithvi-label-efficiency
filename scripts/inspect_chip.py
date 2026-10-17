"""Print basic facts about one Sen1Floods11 chip and its label."""
import sys
import numpy as np
from osgeo import gdal

name = sys.argv[1] if len(sys.argv) > 1 else "Ghana_103272"
root = "data/sen1floods11"

ds = gdal.Open(f"{root}/S2Hand/{name}_S2Hand.tif")
img = ds.ReadAsArray()  # (bands, H, W)
print(f"image: shape={img.shape} dtype={img.dtype}")
print(f"CRS/pixel size: {ds.GetProjection()[:40]}... / {ds.GetGeoTransform()[1]:.6f} deg")

# 1-based GDAL band numbers: B2 Blue, B3 Green, B4 Red, B8A Narrow NIR, B11 SWIR1, B12 SWIR2
names = {2: "B2 Blue", 3: "B3 Green", 4: "B4 Red", 9: "B8A NIR", 12: "B11 SWIR1", 13: "B12 SWIR2"}
for b, n in names.items():
    band = img[b - 1].astype(float)
    print(f"  {n:10s} (file band {b:2d}): min={band.min():7.0f} mean={band.mean():7.0f} max={band.max():7.0f}")

lab = gdal.Open(f"{root}/LabelHand/{name}_LabelHand.tif").ReadAsArray()
vals, counts = np.unique(lab, return_counts=True)
print(f"label: shape={lab.shape} dtype={lab.dtype}")
for v, c in zip(vals, counts):
    print(f"  value {v:2d}: {c:7d} px ({100 * c / lab.size:5.1f}%)")
