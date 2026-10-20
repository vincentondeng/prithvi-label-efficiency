"""Step 4c-2: is the class token the first or the last token in the encoder output?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F

from lepipe.data import Sen1Floods11, read_split
from lepipe.model import SegModel

root, splits = "data/sen1floods11", "data/splits"
pairs = read_split(f"{splits}/flood_train_data.csv")
idx = [s2 for s2, _ in pairs].index("Ghana_141910_S2Hand.tif")
x = Sen1Floods11(root, pairs)[idx]["image"]

model = SegModel(pretrained=True).eval()
with torch.no_grad():
    feats = model.encoder(x[None].unsqueeze(2))

print("1) similarity to the average patch token (the class token should be the outlier)")
for layer in [2, 5, 8, 11]:
    f = feats[layer][0]            # (1025, 768)
    middle_mean = f[1:-1].mean(0)
    c_first = F.cosine_similarity(f[0], middle_mean, dim=0).item()
    c_last = F.cosine_similarity(f[-1], middle_mean, dim=0).item()
    print(f"   layer {layer:2d}: first token {c_first:6.3f} | last token {c_last:6.3f}")

print("2) spatial alignment: best |correlation| of top-3 feature PCs with block-averaged NIR/SWIR")
grid = 32
img = x.numpy()
blocks = [img[b].reshape(grid, 16, grid, 16).mean((1, 3)).ravel() for b in (3, 4)]  # NIR, SWIR1
f = feats[11][0]
for name, tok in [("class token first (drop first)", f[1:]), ("class token last (drop last)", f[:-1])]:
    t = tok - tok.mean(0)
    _, _, v = torch.pca_lowrank(t, q=3, center=False)
    pcs = (t @ v[:, :3]).numpy()
    best = max(abs(np.corrcoef(pcs[:, k], b)[0, 1]) for k in range(3) for b in blocks)
    print(f"   {name}: {best:.3f}")
