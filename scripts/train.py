"""Train one run: init (pretrained|scratch) x label fraction x seed. Fixed step budget,
final-checkpoint evaluation (no validation-based selection)."""
import argparse
import importlib.metadata as md
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lepipe import subsets
from lepipe.data import IGNORE_INDEX, Sen1Floods11, read_split
from lepipe.metrics import ConfusionMeter
from lepipe.model import SegModel

p = argparse.ArgumentParser()
p.add_argument("--init", choices=["pretrained", "scratch"], required=True)
p.add_argument("--fraction", type=float, default=1.0)
p.add_argument("--seed", type=int, default=0, help="selects the label subset AND the training randomness")
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--lr-encoder", type=float, default=1e-4)
p.add_argument("--lr-decoder", type=float, default=1e-3)
p.add_argument("--weight-decay", type=float, default=0.01)
p.add_argument("--warmup", type=int, default=50)
p.add_argument("--eval-every", type=int, default=250, help="validation monitoring only; 0 = never")
p.add_argument("--max-eval-chips", type=int, default=None, help="debug: limit chips per eval split")
p.add_argument("--splits-dir", default="data/splits")
p.add_argument("--root", default="data/sen1floods11")
p.add_argument("--subsets", default="subsets/label_subsets.json")
p.add_argument("--out-dir", default="results")
p.add_argument("--exclude-events", nargs="*", default=[],
               help="drop these flood events (filename prefixes) from the training pool; their training-split "
                    "chips are evaluated as the extra split 'heldout_train'")
p.add_argument("--head", choices=["decoder", "linear"], default="decoder", help="segmentation head")
p.add_argument("--freeze-encoder", action="store_true", help="train only the head (encoder frozen)")
p.add_argument("--device", default=None)
p.add_argument("--workers", type=int, default=2)
args = p.parse_args()

dev = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available() else "cpu"))
use_amp = dev.type == "cuda"
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ---- data -----------------------------------------------------------------------------
all_train = read_split(Path(args.splits_dir) / "flood_train_data.csv")
by_name = {s2: (s2, lab) for s2, lab in all_train}
chosen = subsets.get_subset(args.subsets, args.seed, args.fraction, exclude_events=args.exclude_events)
assert not any(subsets.event_of(n) in set(args.exclude_events) for n in chosen), "excluded event in training subset"
train_pairs = [by_name[n] for n in chosen]
train_ds = Sen1Floods11(args.root, train_pairs)
gen = torch.Generator()
gen.manual_seed(args.seed)
loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen,
                    drop_last=len(train_ds) >= args.batch_size, num_workers=args.workers,
                    persistent_workers=args.workers > 0)


def batches():
    while True:
        yield from loader


def augment(x, y):
    xs, ys = [], []
    for xi, yi in zip(x, y):
        k = int(torch.randint(0, 4, (1,)))
        xi, yi = torch.rot90(xi, k, (1, 2)), torch.rot90(yi, k, (0, 1))
        if torch.rand(1) < 0.5:
            xi, yi = xi.flip(2), yi.flip(1)
        xs.append(xi)
        ys.append(yi)
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def evaluate(model, split, pairs=None, per_chip=False):
    """Evaluate a split (or an explicit list of pairs). Chips are grouped by flood event (filename prefix);
    per-event confusion matrices are always stored, per-chip matrices when per_chip=True. Split-level metrics
    come from the summed confusion matrix, i.e. over all pixels of all chips."""
    if pairs is None:
        pairs = read_split(Path(args.splits_dir) / f"flood_{split}_data.csv")
    if args.max_eval_chips:
        pairs = pairs[: args.max_eval_chips]
    groups = {}
    for pr in pairs:
        groups.setdefault(subsets.event_of(pr[0]), []).append(pr)
    model.eval()
    total, per_event, chips = ConfusionMeter(), {}, {}
    for event, ev_pairs in sorted(groups.items()):
        dl = DataLoader(Sen1Floods11(args.root, ev_pairs), batch_size=4, shuffle=False,
                        num_workers=args.workers)
        meter = ConfusionMeter()
        for b in dl:
            x, y = b["image"].to(dev), b["label"].to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            meter.update(logits, y)
            if per_chip:
                for i, name in enumerate(b["name"]):
                    cm = ConfusionMeter()
                    cm.update(logits[i:i + 1], y[i:i + 1])
                    chips[Path(name).name.replace("_S2Hand.tif", "")] = cm.cm.tolist()
        total.cm += meter.cm
        per_event[event] = {"n_chips": len(ev_pairs), **meter.compute()}
    model.train()
    out = {**total.compute(), "events": per_event}
    if per_chip:
        out["chips"] = chips
    return out


# ---- model / optimiser ----------------------------------------------------------------
model = SegModel(pretrained=args.init == "pretrained", head=args.head, freeze_encoder=args.freeze_encoder).to(dev)
n_trainable = model.n_trainable()
opt = torch.optim.AdamW(model.param_groups(args.lr_encoder, args.lr_decoder), weight_decay=args.weight_decay)


def lr_lambda(step):
    if step < args.warmup:
        return (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
print(f"init={args.init} fraction={args.fraction} seed={args.seed} train_chips={len(train_ds)} "
      f"device={dev} amp={use_amp} steps={args.steps} bs={args.batch_size} head={args.head} "
      f"frozen={args.freeze_encoder} trainable_params={n_trainable}")

# ---- train ----------------------------------------------------------------------------
model.train()
loss_curve, monitor, window = [], [], []
t0 = time.time()
for step, batch in zip(range(1, args.steps + 1), batches()):
    x, y = augment(batch["image"].to(dev), batch["label"].to(dev))
    if (y != IGNORE_INDEX).sum() == 0:
        continue  # nothing to learn from in this batch
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
        logits = model(x)
    loss = F.cross_entropy(logits.float(), y, ignore_index=IGNORE_INDEX)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    window.append(loss.item())
    if step % 25 == 0 or step == args.steps:
        loss_curve.append((step, float(np.mean(window))))
        window = []
        print(f"step {step:5d}  loss {loss_curve[-1][1]:.4f}  elapsed {time.time() - t0:6.1f}s")
    if args.eval_every and step % args.eval_every == 0 and step != args.steps:
        m = evaluate(model, "valid")
        monitor.append({"step": step, **m})
        print(f"   [monitor] valid iou_water={m['iou_water']:.3f}")
train_seconds = time.time() - t0

# ---- final evaluation (last checkpoint, no selection) ----------------------------------
final = {s: evaluate(model, s, per_chip=True) for s in ["valid", "test", "bolivia"]}
if args.exclude_events:
    held = [pr for pr in all_train if subsets.event_of(pr[0]) in set(args.exclude_events)]
    final["heldout_train"] = evaluate(model, "heldout_train", pairs=held, per_chip=True)
for s, m in final.items():
    print(f"FINAL {s:8s} iou_water={m['iou_water']:.3f} miou={m['miou']:.3f} f1_water={m['f1_water']:.3f}")

result = {
    "config": vars(args), "n_train_chips": len(train_ds), "train_chips": chosen,
    "n_trainable_params": n_trainable,
    "final": final, "monitor_valid": monitor, "loss_curve": loss_curve,
    "train_seconds": train_seconds, "device": str(dev),
    "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
    "versions": {k: md.version(k) for k in ["torch", "terratorch", "torchgeo", "numpy"]},
}
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
path = out / f"{args.init}_f{args.fraction:g}_s{args.seed}.json"
json.dump(result, open(path, "w"), indent=1)
print("wrote", path)
