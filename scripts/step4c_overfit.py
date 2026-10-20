"""Step 4c: can the pipeline overfit a tiny batch? (model + loss + metrics smoke test)"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from lepipe.data import IGNORE_INDEX, Sen1Floods11, read_split
from lepipe.metrics import ConfusionMeter
from lepipe.model import SegModel

p = argparse.ArgumentParser()
p.add_argument("--splits-dir", default="data/splits")
p.add_argument("--root", default="data/sen1floods11")
p.add_argument("--device", default="cpu")
p.add_argument("--steps", type=int, default=30)
p.add_argument("--batch-size", type=int, default=2)
p.add_argument("--scratch", action="store_true", help="random-init encoder")
args = p.parse_args()

torch.manual_seed(0)
dev = torch.device(args.device)
ds = Sen1Floods11(args.root, read_split(Path(args.splits_dir) / "flood_train_data.csv"))

chosen = []
for i in range(len(ds)):  # first chips with a clear water share
    lab = ds[i]["label"]
    valid = lab != IGNORE_INDEX
    if valid.sum() > 0 and (lab == 1).sum() / valid.sum() >= 0.05:
        chosen.append(ds[i])
    if len(chosen) == args.batch_size:
        break
x = torch.stack([c["image"] for c in chosen]).to(dev)
y = torch.stack([c["label"] for c in chosen]).to(dev)
print("chips:", [c["name"] for c in chosen], "| x", tuple(x.shape), "| y", tuple(y.shape))

model = SegModel(pretrained=not args.scratch).to(dev)
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | device {dev} | scratch={args.scratch}")
opt = torch.optim.AdamW(model.param_groups(1e-4, 1e-3), weight_decay=0.01)

model.train()
for step in range(1, args.steps + 1):
    t0 = time.time()
    logits = model(x)
    loss = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step == 1 or step % 5 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  ({time.time() - t0:.1f}s/step)  logits {tuple(logits.shape)}")

model.eval()
meter = ConfusionMeter()
with torch.no_grad():
    meter.update(model(x), y)
print("metrics on the same chips:", {k: round(v, 3) for k, v in meter.compute().items()})
