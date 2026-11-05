"""Africa vs rest breakdown from per-event confusion matrices (results_events/*.json).
Africa = Ghana, Nigeria, Somalia (Sen1Floods11 events in Africa). Pixel-level pooling.
Usage: python scripts/analyze_events.py [--dir results_events] [--split test]"""
import argparse, json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np

AFRICA = {"Ghana", "Nigeria", "Somalia"}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

p = argparse.ArgumentParser()
p.add_argument("--dir", default="results_events")
p.add_argument("--split", default="test", choices=["test", "valid"])
p.add_argument("--out", default="figures/event_tables.md")
args = p.parse_args()


def iou(cm):
    cm = np.asarray(cm, dtype=float)
    tp = cm[1, 1]; fp = cm[0, 1]; fn = cm[1, 0]
    return tp / max(tp + fp + fn, 1.0)


runs = {}  # (init, fraction, seed) -> dict
for f in sorted(glob.glob(f"{args.dir}/*.json")):
    d = json.load(open(f)); c = d["config"]
    ev = d["final"][args.split]["events"]
    cm_all = sum(np.array(v["cm"]) for v in ev.values())
    # sanity: pooled matrix must reproduce the stored split-level IoU
    assert abs(iou(cm_all) - d["final"][args.split]["iou_water"]) < 1e-9, f
    cm_af = sum((np.array(v["cm"]) for k, v in ev.items() if k in AFRICA), np.zeros((2, 2)))
    cm_rest = sum((np.array(v["cm"]) for k, v in ev.items() if k not in AFRICA), np.zeros((2, 2)))
    runs[(c["init"], c["fraction"], c["seed"])] = {
        "all": iou(cm_all), "africa": iou(cm_af), "rest": iou(cm_rest),
        "events": {k: iou(v["cm"]) for k, v in ev.items()},
        "n_chips": {k: v["n_chips"] for k, v in ev.items()},
        "union": {k: float(np.array(v["cm"])[1, 1] + np.array(v["cm"])[0, 1] + np.array(v["cm"])[1, 0])
                  for k, v in ev.items()},
        "train_events": [Path(n).name.split("_")[0] for n in d.get("train_chips", [])],
    }
if not runs:
    raise SystemExit(f"no result files with events found in {args.dir}")

fracs = sorted({k[1] for k in runs})
out = [f"## Africa vs rest, {args.split} water IoU (pixel-pooled; Africa = {', '.join(sorted(AFRICA))})\n"]
n_af = sum(v for k, v in next(iter(runs.values()))["n_chips"].items() if k in AFRICA)
n_all = sum(next(iter(runs.values()))["n_chips"].values())
out.append(f"Chips in this split: {n_af} African, {n_all - n_af} other.\n")


def ms(x):
    x = np.array(x); return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"


def paired(a, b):
    d = np.array(a) - np.array(b); n = len(d)
    m = d.mean(); h = T975.get(n - 1, 1.96) * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return f"{m:+.3f} [{m - h:+.3f}, {m + h:+.3f}] ({int((d > 0).sum())}/{n})"


out.append("| labels | group | pretrained | scratch | gap (pre - scr), 95% t-CI (wins) |\n|---|---|---|---|---|")
for fr in fracs:
    seeds = sorted({k[2] for k in runs if k[1] == fr and ("pretrained", fr, k[2]) in runs and ("scratch", fr, k[2]) in runs})
    for g in ["africa", "rest", "all"]:
        pre = [runs[("pretrained", fr, s)][g] for s in seeds]; scr = [runs[("scratch", fr, s)][g] for s in seeds]
        out.append(f"| {fr*100:g}% (n={len(seeds)}) | {g} | {ms(pre)} | {ms(scr)} | {paired(pre, scr)} |")

out.append("\n### Is the pretrained gap different in Africa than elsewhere? (gap_Africa - gap_rest, paired by seed)\n")
out.append("| labels | gap_Africa - gap_rest, 95% t-CI (seeds where Africa gap larger) |\n|---|---|")
for fr in fracs:
    seeds = sorted({k[2] for k in runs if k[1] == fr})
    dd = [(runs[("pretrained", fr, s)]["africa"] - runs[("scratch", fr, s)]["africa"]) -
          (runs[("pretrained", fr, s)]["rest"] - runs[("scratch", fr, s)]["rest"]) for s in seeds]
    out.append(f"| {fr*100:g}% | {paired(dd, [0]*len(dd))} |")

out.append("\n### Per-event water IoU (mean over seeds; descriptive, few chips per event)\n")
events = sorted(next(iter(runs.values()))["events"])
nchips = next(iter(runs.values()))["n_chips"]
hdr = "| event (chips) | " + " | ".join(f"{i} {fr*100:g}%" for fr in fracs for i in ["pre", "scr"]) + " |"
out.append(hdr); out.append("|---|" + "---|" * (2 * len(fracs)))
for e in events:
    row = []
    for fr in fracs:
        for init in ["pretrained", "scratch"]:
            v = [runs[k]["events"][e] for k in runs if k[0] == init and k[1] == fr]
            row.append(f"{np.mean(v):.3f}")
    star = " *" if e in AFRICA else ""
    out.append(f"| {e}{star} ({nchips[e]}) | " + " | ".join(row) + " |")
out.append("\n`*` = African event.")

out.append("\n### Weight of each African event in the pooled Africa IoU (w_e = union_e / sum of unions; mean over seeds)\n")
out.append("| labels | arm | " + " | ".join(sorted(AFRICA)) + " |\n|---|---|" + "---|" * len(AFRICA))
for fr in fracs:
    for init in ["pretrained", "scratch"]:
        ws = []
        for k in runs:
            if k[0] == init and k[1] == fr:
                u = runs[k]["union"]; tot = sum(u[e] for e in AFRICA if e in u)
                ws.append([u.get(e, 0.0) / tot if tot else float("nan") for e in sorted(AFRICA)])
        ws = np.mean(ws, axis=0)
        out.append(f"| {fr*100:g}% | {init} | " + " | ".join(f"{w:.2f}" for w in ws) + " |")

out.append("\n### African training chips per seed (same subset for both arms) and per-event test IoU\n")
out.append("| labels | seed | train chips: " + " / ".join(sorted(AFRICA)) + " | IoU pre: " + " / ".join(sorted(AFRICA)) + " | IoU scr: " + " / ".join(sorted(AFRICA)) + " |\n|---|---|---|---|---|")
for fr in fracs:
    for sd in sorted({k[2] for k in runs if k[1] == fr}):
        kp, ks = ("pretrained", fr, sd), ("scratch", fr, sd)
        if kp not in runs or ks not in runs:
            continue
        te = runs[kp]["train_events"]
        comp = " / ".join(str(te.count(e)) for e in sorted(AFRICA)) if te else "n/a"
        ip = " / ".join(f"{runs[kp]['events'][e]:.3f}" for e in sorted(AFRICA))
        isr = " / ".join(f"{runs[ks]['events'][e]:.3f}" for e in sorted(AFRICA))
        out.append(f"| {fr*100:g}% | {sd} | {comp} | {ip} | {isr} |")
text = "\n".join(out)
print(text)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(text + "\n")
print("\nwrote", args.out)
