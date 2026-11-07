"""Frozen-encoder / linear-probe analysis.

Configurations (test water IoU, seeds 0-4, fractions 1/5/10/100%):
  FT-pre   full fine-tuning, pretrained encoder      (results/, main grid)
  FT-scr   full fine-tuning, random encoder          (results/, main grid)
  FD-pre   frozen pretrained encoder + decoder       (results_frozen_decoder/, init=pretrained)
  FD-rnd   frozen random encoder + decoder           (results_frozen_decoder/, init=scratch)
  LP-pre   frozen pretrained encoder + linear head   (results_frozen_linear/, init=pretrained)
  LP-rnd   frozen random encoder + linear head       (results_frozen_linear/, init=scratch)

Paired contrasts (same subset per seed):
  representation value   FD-pre - FD-rnd,  LP-pre - LP-rnd
  value of fine-tuning   FT-pre - FD-pre
  head capacity          FD-pre - LP-pre
  frozen vs scratch      FD-pre - FT-scr,  LP-pre - FT-scr
Note: FT runs come from an earlier batch than the frozen runs (same code path otherwise); contrasts that
mix batches are flagged with (x).

Africa vs rest (Ghana, Nigeria, Somalia) is reported for the frozen runs, which store per-event matrices.

Usage: python scripts/analyze_probe.py [--main results] [--decoder results_frozen_decoder]
                                       [--linear results_frozen_linear] [--out figures/probe_tables.md]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

AFRICA = {"Ghana", "Nigeria", "Somalia"}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
FRACS = [0.01, 0.05, 0.1, 1.0]
SEEDS = range(5)

p = argparse.ArgumentParser()
p.add_argument("--main", default="results")
p.add_argument("--decoder", default="results_frozen_decoder")
p.add_argument("--linear", default="results_frozen_linear")
p.add_argument("--out", default="figures/probe_tables.md")
args = p.parse_args()


def iou(cm):
    cm = np.asarray(cm, float)
    return cm[1, 1] / max(cm[1, 1] + cm[0, 1] + cm[1, 0], 1.0)


def load(d, frozen=None):
    runs = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        j = json.load(open(f)); c = j["config"]
        if frozen is not None and bool(c.get("freeze_encoder", False)) != frozen:
            raise SystemExit(f"{f}: freeze_encoder={c.get('freeze_encoder')} but expected {frozen}")
        runs[(c["init"], round(c["fraction"], 4), c["seed"])] = j
    return runs


M, FD, LP = load(args.main, frozen=False), load(args.decoder, frozen=True), load(args.linear, frozen=True)
if not FD or not LP:
    raise SystemExit("frozen result folders are empty")
for name, R, head in [("decoder", FD, "decoder"), ("linear", LP, "linear")]:
    bad = [k for k, j in R.items() if j["config"].get("head", "decoder") != head]
    if bad:
        raise SystemExit(f"{name} folder contains runs with another head: {bad[:3]}")

CFG = {"FT-pre": (M, "pretrained"), "FT-scr": (M, "scratch"), "FD-pre": (FD, "pretrained"),
       "FD-rnd": (FD, "scratch"), "LP-pre": (LP, "pretrained"), "LP-rnd": (LP, "scratch")}


def vals(cfg, f, metric=lambda j: j["final"]["test"]["iou_water"]):
    R, init = CFG[cfg]
    return {s: metric(R[(init, f, s)]) for s in SEEDS if (init, f, s) in R}


def ms(v):
    x = np.array(list(v.values()), float)
    return "n/a" if len(x) == 0 else f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"


def paired(a, b):
    ss = sorted(set(a) & set(b))
    if len(ss) < 2:
        return "n/a"
    d = np.array([a[s] - b[s] for s in ss]); n = len(d); m = d.mean()
    h = T975.get(n - 1, 1.96) * d.std(ddof=1) / np.sqrt(n)
    return f"{m:+.3f} [{m - h:+.3f}, {m + h:+.3f}] ({int((d > 0).sum())}/{n})"


out = ["# Frozen-encoder and linear-probe analysis\n"]
ntr = {c: next((j["n_trainable_params"] for j in CFG[c][0].values() if "n_trainable_params" in j), None)
       for c in ["FD-pre", "LP-pre"]}
out.append(f"Trainable parameters: frozen + decoder {ntr['FD-pre']:,}; linear probe {ntr['LP-pre']:,} "
           "(full fine-tuning trains the whole encoder as well).\n")

out.append("## Test water IoU (mean ± sd over seeds)\n")
out.append("| labels | " + " | ".join(CFG) + " |\n|---|" + "---|" * len(CFG))
for f in FRACS:
    out.append(f"| {f*100:g}% | " + " | ".join(ms(vals(c, f)) for c in CFG) + " |")

out.append("\n## Paired contrasts (test water IoU; 95% t-CI; seeds where the first is better)\n")
contrasts = [("representation value, decoder head", "FD-pre", "FD-rnd", ""),
             ("representation value, linear head", "LP-pre", "LP-rnd", ""),
             ("value of fine-tuning the encoder", "FT-pre", "FD-pre", " (x)"),
             ("head capacity (decoder vs linear)", "FD-pre", "LP-pre", ""),
             ("frozen pretrained + decoder vs full scratch", "FD-pre", "FT-scr", " (x)"),
             ("linear probe vs full scratch", "LP-pre", "FT-scr", " (x)")]
out.append("| contrast | " + " | ".join(f"{f*100:g}%" for f in FRACS) + " |\n|---|" + "---|" * len(FRACS))
for name, a, b, flag in contrasts:
    out.append(f"| {name}: {a} - {b}{flag} | " + " | ".join(paired(vals(a, f), vals(b, f)) for f in FRACS) + " |")
out.append("\n(x) compares runs from different batches (main grid vs frozen runs).")

out.append("\n## Africa vs rest on the test split (frozen runs; pixel-pooled)\n")
out.append("| labels | config | Africa | rest | gap Africa vs rest |\n|---|---|---|---|---|")
grp = lambda inside: (lambda j: iou(sum((np.array(v["cm"]) for k, v in j["final"]["test"]["events"].items()
                                          if (k in AFRICA) == inside), np.zeros((2, 2)))))
for f in FRACS:
    for c in ["FD-pre", "FD-rnd", "LP-pre", "LP-rnd"]:
        a, r = vals(c, f, grp(True)), vals(c, f, grp(False))
        out.append(f"| {f*100:g}% | {c} | {ms(a)} | {ms(r)} | {np.mean(list(a.values())) - np.mean(list(r.values())):+.3f} |")
out.append("\nPre-training gain on African vs other test chips (frozen + decoder; difference-in-differences):\n")
out.append("| labels | gain Africa | gain rest | difference |\n|---|---|---|---|")
for f in FRACS:
    ga = {s: v - vals("FD-rnd", f, grp(True)).get(s, np.nan) for s, v in vals("FD-pre", f, grp(True)).items()}
    gr = {s: v - vals("FD-rnd", f, grp(False)).get(s, np.nan) for s, v in vals("FD-pre", f, grp(False)).items()}
    out.append(f"| {f*100:g}% | {ms(ga)} | {ms(gr)} | {paired(ga, gr)} |")

text = "\n".join(out)
print(text)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(text + "\n")
print(f"\nwrote {args.out}")
