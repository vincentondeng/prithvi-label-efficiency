"""Leave-Africa-out (LOAO) analysis.

Inputs
  results_loao/   trained WITHOUT Ghana/Nigeria/Somalia (stage 5: 10/50/100% of the 196 non-African chips);
                  per-event and per-chip confusion matrices for valid/test/bolivia and 'heldout_train'
                  (the 56 African training-split chips, never trained on).
  results_africa/ trained WITH African chips (stage 1: 10/100% of all 252 chips); per-event matrices.

Tables
  A  Transfer to Africa: pretrained vs scratch on all 97 African chips (heldout_train + African valid + test),
     pooled and per event; non-African test chips for reference.
  B  Value of in-region labels: African TEST chips (21) with vs without African training data, per arm,
     and whether pre-training reduces the penalty of having no African labels.
  C  Sensitivity S1: Table A and the African-test numbers recomputed without chip Ghana_866994
     (identified after seeing results; disclosed as post hoc).
  D  Weight of each event in the pooled Africa-all IoU.

Usage: python scripts/analyze_loao.py [--loao results_loao] [--with-africa results_africa] [--out figures/loao_tables.md]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

AFRICA = ("Ghana", "Nigeria", "Somalia")
EXCLUDE_S1 = "Ghana_866994"
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

p = argparse.ArgumentParser()
p.add_argument("--loao", default="results_loao")
p.add_argument("--with-africa", default="results_africa")
p.add_argument("--out", default="figures/loao_tables.md")
args = p.parse_args()


def iou(cm):
    cm = np.asarray(cm, float)
    return cm[1, 1] / max(cm[1, 1] + cm[0, 1] + cm[1, 0], 1.0)


def union(cm):
    cm = np.asarray(cm, float)
    return cm[1, 1] + cm[0, 1] + cm[1, 0]


def event_of(chip):
    return chip.split("_")[0]


def load(d):
    runs = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        j = json.load(open(f)); c = j["config"]
        runs[(c["init"], round(c["fraction"], 4), c["seed"])] = j
    if not runs:
        raise SystemExit(f"no result files in {d}")
    return runs


def african_chip_cms(run, drop=()):
    """All African chips a LOAO model never trained on: heldout_train + African chips of valid and test."""
    out = {}
    for split in ("heldout_train", "valid", "test"):
        for chip, cm in run["final"][split]["chips"].items():
            if event_of(chip) in AFRICA and chip not in drop:
                out[(split, chip)] = np.array(cm)
    return out


def pooled(cms, events=None):
    sel = [cm for (split, chip), cm in cms.items() if events is None or event_of(chip) in events]
    return sum(sel, np.zeros((2, 2)))


def test_cm_from_events(run, events, include=True):
    ev = run["final"]["test"]["events"]
    return sum((np.array(v["cm"]) for k, v in ev.items() if (k in events) == include), np.zeros((2, 2)))


def test_cm_from_chips(run, events, drop=()):
    ch = run["final"]["test"]["chips"]
    return sum((np.array(cm) for k, cm in ch.items() if event_of(k) in events and k not in drop), np.zeros((2, 2)))


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():.3f} ± {x.std(ddof=1) if len(x) > 1 else 0:.3f}"


def paired(d):
    d = np.asarray(d, float); n = len(d); m = d.mean()
    h = T975.get(n - 1, 1.96) * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return f"{m:+.3f} [{m - h:+.3f}, {m + h:+.3f}] ({int((d > 0).sum())}/{n})"


L = load(args.loao)
W = load(args.with_africa) if Path(args.with_africa).exists() else {}
fracs = sorted({k[1] for k in L})
seeds = lambda R, f: sorted({k[2] for k in R if k[1] == f and ("pretrained", f, k[2]) in R and ("scratch", f, k[2]) in R})
pool_n = {f: L[next(k for k in L if k[1] == f)]["n_train_chips"] for f in fracs}
ex = L[next(iter(L))]
assert not any(event_of(Path(c).name) in AFRICA for c in ex["train_chips"]), "African chip found in LOAO training"
n_af = len(african_chip_cms(ex))
out = [f"# Leave-Africa-out analysis\n",
       f"LOAO training pool: non-African chips only; chips per run by fraction: "
       + ", ".join(f"{f*100:g}% = {pool_n[f]}" for f in fracs) + f". African evaluation chips per run: {n_af}.\n"]

# ---------- A
out.append("## A. Transfer to Africa: pretrained vs scratch, models trained without any African chip\n")
out.append("| labels (chips) | evaluation set | pretrained | scratch | gap pre - scr, 95% t-CI (wins) |\n|---|---|---|---|---|")
for f in fracs:
    ss = seeds(L, f)
    rows = [("Africa, all 97 chips", lambda r: iou(pooled(african_chip_cms(r))))]
    rows += [(f"  {e}", (lambda e: lambda r: iou(pooled(african_chip_cms(r), {e})))(e)) for e in AFRICA]
    rows += [("Africa, test chips only", lambda r: iou(test_cm_from_events(r, AFRICA))),
             ("non-African test chips", lambda r: iou(test_cm_from_events(r, AFRICA, include=False)))]
    for name, fn in rows:
        pre = [fn(L[("pretrained", f, s)]) for s in ss]; scr = [fn(L[("scratch", f, s)]) for s in ss]
        out.append(f"| {f*100:g}% ({pool_n[f]}) | {name} | {ms(pre)} | {ms(scr)} | {paired(np.subtract(pre, scr))} |")

# ---------- B
out.append("\n## B. Value of in-region labels: African TEST chips (21), trained with vs without African chips\n")
if not W:
    out.append(f"(skipped: {args.with_africa} not found)")
else:
    out.append("Penalty = IoU(with Africa) - IoU(without Africa), paired by seed. The training sets also differ in size "
               "(with: 25 / 252 chips; without: 20 / 196), so the penalty mixes 'no African labels' with 'fewer labels'.\n")
    out.append("| labels | arm | with Africa | without Africa | penalty, 95% t-CI (seeds with penalty > 0) |\n|---|---|---|---|---|")
    pen = {}
    for f in [x for x in fracs if any(k[1] == x for k in W)]:
        ss = sorted(set(seeds(L, f)) & set(seeds(W, f)))
        for arm in ("pretrained", "scratch"):
            w = [iou(test_cm_from_events(W[(arm, f, s)], AFRICA)) for s in ss]
            wo = [iou(test_cm_from_events(L[(arm, f, s)], AFRICA)) for s in ss]
            pen[(arm, f)] = np.subtract(w, wo)
            out.append(f"| {f*100:g}% | {arm} | {ms(w)} | {ms(wo)} | {paired(pen[(arm, f)])} |")
    out.append("\nDoes pre-training reduce the penalty? (penalty_scratch - penalty_pretrained; > 0 means yes)\n")
    out.append("| labels | difference, 95% t-CI (seeds where pre-training reduces the penalty) |\n|---|---|")
    for f in sorted({k[1] for k in pen}):
        out.append(f"| {f*100:g}% | {paired(pen[('scratch', f)] - pen[('pretrained', f)])} |")

# ---------- C
out.append(f"\n## C. Sensitivity S1: without chip {EXCLUDE_S1} (post hoc; see quicklook figure)\n")
out.append("| labels | evaluation set | pretrained | scratch | gap pre - scr (wins) | gap with all chips |\n|---|---|---|---|---|---|")
for f in fracs:
    ss = seeds(L, f)
    for name, fn, fn_all in [
        ("Africa, all chips", lambda r: iou(pooled(african_chip_cms(r, drop={EXCLUDE_S1}))), lambda r: iou(pooled(african_chip_cms(r)))),
        ("  Ghana", lambda r: iou(pooled(african_chip_cms(r, drop={EXCLUDE_S1}), {"Ghana"})), lambda r: iou(pooled(african_chip_cms(r), {"Ghana"}))),
        ("Africa, test chips", lambda r: iou(test_cm_from_chips(r, AFRICA, drop={EXCLUDE_S1})), lambda r: iou(test_cm_from_chips(r, AFRICA))),
    ]:
        pre = [fn(L[("pretrained", f, s)]) for s in ss]; scr = [fn(L[("scratch", f, s)]) for s in ss]
        g_all = np.mean([fn_all(L[("pretrained", f, s)]) - fn_all(L[("scratch", f, s)]) for s in ss])
        out.append(f"| {f*100:g}% | {name} | {ms(pre)} | {ms(scr)} | {paired(np.subtract(pre, scr))} | {g_all:+.3f} |")

# ---------- D
out.append("\n## D. Weight of each event in the pooled 'Africa, all 97 chips' IoU (union share; mean over runs)\n")
out.append("| labels | arm | " + " | ".join(AFRICA) + " |\n|---|---|" + "---|" * len(AFRICA))
for f in fracs:
    for arm in ("pretrained", "scratch"):
        ws = []
        for s in seeds(L, f):
            cms = african_chip_cms(L[(arm, f, s)])
            u = [union(pooled(cms, {e})) for e in AFRICA]
            ws.append(np.array(u) / max(sum(u), 1))
        out.append(f"| {f*100:g}% | {arm} | " + " | ".join(f"{w:.2f}" for w in np.mean(ws, axis=0)) + " |")
cms = african_chip_cms(ex)
out.append(f"\nGhana_866994 share of Ghana's labelled water over the 97 chips: "
           f"{sum(cm[1].sum() for (s, c), cm in cms.items() if c == EXCLUDE_S1) / max(sum(cm[1].sum() for (s, c), cm in cms.items() if event_of(c) == 'Ghana'), 1):.2f}")

text = "\n".join(out)
print(text)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(text + "\n")
print(f"\nwrote {args.out}")
