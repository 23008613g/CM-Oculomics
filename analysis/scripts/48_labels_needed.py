# -*- coding: utf-8 -*-
"""
48_labels_needed.py — what does local validation actually cost?

WHY: the paper concludes that a model's performance on a new cohort cannot be anticipated
without labels, so a deployer has to measure it. That conclusion is only useful if it comes
with a price. This script answers the price in the unit a deployer cares about: how many
labelled target images are needed before the estimate of external AUC is worth acting on.

For each cohort we draw labelled subsets of increasing size WITHOUT replacement from the
cohort itself, which is exactly the situation of a site that labels part of its own data, and
record the spread of the resulting AUC estimates around the value obtained with the whole
cohort. The reported requirement is the smallest subset size whose 95% interval half-width
falls to 0.05, and to 0.10 for a looser tolerance.

USAGE  python scripts/48_labels_needed.py
OUTPUT results/external2/labels_needed.json, Figure_labels_needed.png
"""
import os, sys, json
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "external2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SETS = [("ddr", "DDR (China)"), ("jsiec", "JSIEC (China)"), ("deepdrid", "DeepDRiD (China)"),
        ("aptos", "APTOS (India)"), ("idrid", "IDRiD (India)"),
        ("messidor2", "Messidor-2 (France)"), ("eyepacs", "EyePACS (USA)")]
KINDS = ["dinov2_l", "retfound"]
NAME = {"dinov2_l": "DINOv2", "retfound": "RETFound"}
C = {"dinov2_l": "#D55E00", "retfound": "#0072B2"}
SEEDS = 5   # raised from 3: seeds 3-4 were added for both backbones so that
            # the one degenerate RETFound run does not carry 1/3 of the weight
DRAWS = 600
GRID = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200]
TOL = (0.05, 0.10)


def ensemble(kind, s):
    ps = [np.load(os.path.join(OUT, f"pred_{kind}_seed{k}_{s}.npz")) for k in range(SEEDS)]
    return ps[0]["ys"], np.mean([p["pos"] for p in ps], 0)


def halfwidth(y, p, n, rng, metric="auc"):
    """Spread of the estimate a site would obtain from n of its own labelled images.

    Balanced accuracy is reported alongside AUC because it is the quantity the published
    estimators target, so the cost of measuring can be compared with the error of estimating
    on the same scale."""
    idx = np.arange(len(y))
    vals = []
    for _ in range(DRAWS):
        sel = rng.choice(idx, n, replace=False)
        if len(np.unique(y[sel])) < 2:
            continue
        vals.append(roc_auc_score(y[sel], p[sel]) if metric == "auc"
                    else balanced_accuracy_score(y[sel], (p[sel] > 0.5).astype(int)))
    if len(vals) < 50:
        return np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float((hi - lo) / 2)


res = {}
print("=" * 78)
print("HOW MANY LABELLED TARGET IMAGES DOES LOCAL VALIDATION NEED?")
print("=" * 78)
print("95% interval half-width of the external AUC estimate, by number of labelled images")
print()
for kind in KINDS:
    rows = []
    print(f"{NAME[kind]}")
    print(f"{'cohort':<22}{'n avail':>8}" + "".join(f"{g:>7}" for g in GRID[:8]))
    for s, label in SETS:
        y, p = ensemble(kind, s)
        rng = np.random.RandomState(0)
        hw, hwb = {}, {}
        for n in GRID:
            if n > len(y):
                continue
            hw[n] = halfwidth(y, p, n, rng, "auc")
            hwb[n] = halfwidth(y, p, n, rng, "bacc")
        need = {}
        for t in TOL:
            ok = [n for n in sorted(hw) if np.isfinite(hw[n]) and hw[n] <= t]
            need[t] = int(ok[0]) if ok else None
        needb = {}
        for t in TOL:
            ok = [n for n in sorted(hwb) if np.isfinite(hwb[n]) and hwb[n] <= t]
            needb[t] = int(ok[0]) if ok else None
        rows.append(dict(cohort=s, label=label, n_available=int(len(y)),
                         full_auc=float(roc_auc_score(y, p)),
                         halfwidth={str(k): (None if not np.isfinite(v) else v)
                                    for k, v in hw.items()},
                         halfwidth_bacc={str(k): (None if not np.isfinite(v) else v)
                                         for k, v in hwb.items()},
                         need_005=need[0.05], need_010=need[0.10],
                         need_bacc_005=needb[0.05], need_bacc_010=needb[0.10]))
        cells = "".join(f"{hw[g]:>7.3f}" if g in hw and np.isfinite(hw[g]) else f"{'-':>7}"
                        for g in GRID[:8])
        print(f"{label:<22}{len(y):>8}{cells}")
    res[kind] = rows
    r5 = [r["need_005"] for r in rows if r["need_005"]]
    r10 = [r["need_010"] for r in rows if r["need_010"]]
    print(f"  reach half-width 0.10 with {min(r10)}-{max(r10)} labelled images "
          f"(median {int(np.median(r10))}), in {len(r10)}/{len(rows)} cohorts")
    if r5:
        print(f"  reach half-width 0.05 with {min(r5)}-{max(r5)} labelled images "
              f"(median {int(np.median(r5))}), in {len(r5)}/{len(rows)} cohorts")
    else:
        print("  half-width 0.05 was not reached in any cohort at the sizes available")
    print()

nb = [r["need_bacc_010"] for k in KINDS for r in res[k] if r["need_bacc_010"]]
nb5 = [r["need_bacc_005"] for k in KINDS for r in res[k] if r["need_bacc_005"]]
_est = json.load(open(os.path.join(OUT, "aline_atc_doc.json")))["summary"]
_maes = [v["mae"] for k, v in _est.items() if isinstance(v, dict)]
print("BALANCED ACCURACY (the quantity the published estimators target, MAE %.3f-%.3f):"
      % (min(_maes), max(_maes)))
print("  measured to +/-0.10 with %d-%d labels (median %d), in %d/14 cohort-models"
      % (min(nb), max(nb), int(np.median(nb)), len(nb)))
if nb5:
    print("  measured to +/-0.05 with %d-%d labels (median %d), in %d/14"
          % (min(nb5), max(nb5), int(np.median(nb5)), len(nb5)))
print()
allneed10 = [r["need_010"] for k in KINDS for r in res[k] if r["need_010"]]
allneed5 = [r["need_005"] for k in KINDS for r in res[k] if r["need_005"]]
summary = dict(n_cohort_models=len([1 for k in KINDS for _ in res[k]]),
               reached_010=len(allneed10), reached_005=len(allneed5),
               median_010=int(np.median(allneed10)) if allneed10 else None,
               range_010=[int(min(allneed10)), int(max(allneed10))] if allneed10 else None,
               median_005=int(np.median(allneed5)) if allneed5 else None,
               range_005=[int(min(allneed5)), int(max(allneed5))] if allneed5 else None,
               bacc_median_010=int(np.median(nb)) if nb else None,
               bacc_range_010=[int(min(nb)), int(max(nb))] if nb else None,
               bacc_median_005=int(np.median(nb5)) if nb5 else None,
               bacc_range_005=[int(min(nb5)), int(max(nb5))] if nb5 else None)
print("OVERALL:", json.dumps(summary))
json.dump(dict(per_backbone=res, summary=summary, draws=DRAWS, grid=GRID),
          open(os.path.join(OUT, "labels_needed.json"), "w"), indent=2)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
for a, kind in zip(ax, KINDS):
    for r in res[kind]:
        xs = sorted(int(k) for k, v in r["halfwidth"].items() if v is not None)
        ys = [r["halfwidth"][str(x)] for x in xs]
        a.plot(xs, ys, marker="o", ms=3.5, lw=1.4, alpha=0.85, label=r["label"].split(" (")[0])
    for t, ls in zip(TOL, (":", "--")):
        a.axhline(t, color="k", lw=1.1, ls=ls)
        a.text(1250, t + 0.004, f"±{t:.2f}", ha="right", fontsize=8.5)
    a.set_xscale("log")
    a.set_xlabel("labelled target images")
    a.set_title(NAME[kind], fontsize=11)
ax[0].set_ylabel("95% interval half-width of external AUC")
ax[0].set_ylim(0, 0.32)
ax[0].legend(fontsize=8, ncol=2, frameon=False)
fig.suptitle("The price of local validation: precision of the external AUC estimate "
             "against the number of labelled target images", fontsize=12, y=1.02)
fig.tight_layout()
fn = os.path.join(OUT, "Figure_labels_needed.png")
fig.savefig(fn, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nsaved {fn}")
