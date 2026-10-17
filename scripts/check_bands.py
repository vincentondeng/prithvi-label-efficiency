"""Sanity check: water pixels should be much darker than land in NIR/SWIR."""
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
name, root = "Bolivia_103757", "data/sen1floods11"
img = gdal.Open(f"{root}/S2Hand/{name}_S2Hand.tif").ReadAsArray().astype(float)
lab = gdal.Open(f"{root}/LabelHand/{name}_LabelHand.tif").ReadAsArray()

names = {2: "B2 Blue", 3: "B3 Green", 4: "B4 Red", 9: "B8A NIR", 12: "B11 SWIR1", 13: "B12 SWIR2"}
print(f"{'band':12s} {'land mean':>10s} {'water mean':>11s} {'water/land':>11s}")
for b, n in names.items():
    land = img[b - 1][lab == 0].mean()
    water = img[b - 1][lab == 1].mean()
    print(f"{n:12s} {land:10.0f} {water:11.0f} {water / land:11.2f}")
