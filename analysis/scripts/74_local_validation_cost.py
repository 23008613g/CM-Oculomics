# -*- coding: utf-8 -*-
"""
74_local_validation_cost.py - what does it cost a site to measure the model on its own patients?

WHY: scripts/48 drew labelled subsets WITHOUT replacement from finite, class-balanced public
cohorts. That design shrinks the interval as a subset approaches the whole cohort (a 78-image
cohort is fully known at 78 images) and assumes a site whose patients are half positive. All three
blind reviewers said the resulting "50 images" understates the cost. Here a site is modelled as a
superpopulation with the cohort's score distributions and a chosen prevalence:

  * n labelled images; the number of positives k ~ Binomial(n, prevalence);
  * positives and negatives are drawn WITH replacement from the cohort's positive and negative
    scores (five-seed ensemble, as in the paper);
  * balanced accuracy at the 0.5 threshold is then exactly (Bin(k, se)/k + Bin(n-k, sp)/(n-k))/2,
    with se and sp the cohort's sensitivity and specificity; AUC is computed from the drawn scores.

A draw with no positive or no negative gives no estimate and counts as a miss. The requirement is
the smallest n at which the estimate falls within +/-delta of the cohort value in at least 95% of
draws. Prevalences: 50% (the balanced public subsets), 20%, 10% and 8.8% (the development cohort).

USAGE   python scripts/74_local_validation_cost.py
OUTPUT  results/external2/local_validation_cost.json, Figure_local_validation_cost.png,
        Figure_local_validation_cost_auc.png
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(ROOT, "results", "external2")
COH = ["ddr", "jsiec", "deepdrid", "aptos", "idrid", "messidor2", "eyepacs"]
KINDS = {"dinov2_l": "DINOv2", "retfound": "RETFound"}
PREV = [0.5, 0.2, 0.1, 0.088]
GRID = np.unique(np.round(np.logspace(np.log10(20), np.log10(4000), 40)).astype(int))
DRAWS_BA, DRAWS_AUC, TOL = 4000, 600, (0.10, 0.05)
rng = np.random.RandomState(0)


def ens(k, c):
    z = [np.load(os.path.join(OUT, "pred_%s_seed%d_%s.npz" % (k, s, c))) for s in range(5)]
    return z[0]["ys"].astype(int), np.mean([x["pos"] for x in z], 0)


def ba_err(se, sp, ba, n, prev):
    k = rng.binomial(n, prev, DRAWS_BA)
    ok = (k > 0) & (k < n)
    kk, nn = np.maximum(k, 1), np.maximum(n - k, 1)
    est = (rng.binomial(kk, se) / kk + rng.binomial(nn, sp) / nn) / 2
    err = np.where(ok, np.abs(est - ba), np.inf)
    return err


def auc_err(pos, neg, auc, n, prev):
    errs = []
    for _ in range(DRAWS_AUC):
        k = rng.binomial(n, prev)
        if k == 0 or k == n:
            errs.append(np.inf); continue
        a = rng.choice(pos, k, replace=True); b = np.sort(rng.choice(neg, n - k, replace=True))
        lo = np.searchsorted(b, a, "left"); hi = np.searchsorted(b, a, "right")
        errs.append(abs((lo + 0.5 * (hi - lo)).sum() / (k * (n - k)) - auc))
    return np.array(errs)


def need(errfun, tol):
    for n in GRID:
        if np.mean(errfun(n) <= tol) >= 0.95:
            return int(n)
    return None


res = {"prevalences": PREV, "grid": GRID.tolist(), "rows": []}
curves = {}   # (metric, kind, prev) -> list over cohorts of 95th-percentile error per n
for k, kn in KINDS.items():
    for c in COH:
        y, p = ens(k, c)
        pos, neg = p[y == 1], p[y == 0]
        se, sp = float((pos > 0.5).mean()), float((neg <= 0.5).mean()); ba = (se + sp) / 2
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p))
        row = {"backbone": kn, "cohort": c, "ba": round(ba, 3), "auc": round(auc, 3), "need": {}}
        for prev in PREV:
            fb = lambda n, prev=prev: ba_err(se, sp, ba, n, prev)
            fa = lambda n, prev=prev: auc_err(pos, neg, auc, n, prev)
            row["need"][str(prev)] = {"ba_0.10": need(fb, 0.10), "ba_0.05": need(fb, 0.05),
                                      "auc_0.10": need(fa, 0.10), "auc_0.05": need(fa, 0.05)}
            curves.setdefault(("ba", kn, prev), []).append([float(np.percentile(fb(n), 95)) for n in GRID])
            curves.setdefault(("auc", kn, prev), []).append([float(np.percentile(fa(n), 95)) for n in GRID])
        res["rows"].append(row)
        print("%-8s %-10s BA %.3f AUC %.3f  " % (kn, c, ba, auc) +
              "  ".join("p=%s: %s/%s" % (pv, row["need"][str(pv)]["ba_0.10"], row["need"][str(pv)]["ba_0.05"]) for pv in PREV))

summ = {}
for prev in PREV:
    for key in ("ba_0.10", "ba_0.05", "auc_0.10", "auc_0.05"):
        v = [r["need"][str(prev)][key] for r in res["rows"]]
        got = [x for x in v if x is not None]
        summ["%s|%s" % (prev, key)] = {"median_images": int(np.median(got)) if got else None,
                                       "range_images": [int(min(got)), int(max(got))] if got else None,
                                       "median_positives": round(float(np.median(got)) * prev, 1) if got else None,
                                       "reached": "%d/%d" % (len(got), len(v))}
res["summary"] = summ
# labelled images at which the 95th-percentile error of balanced accuracy (median over the 14
# cohort-backbone combinations) first falls below 0.18, the MAE of the best label-free estimator
beat = {}
for prev in PREV:
    Mb = np.array(curves[("ba", "DINOv2", prev)] + curves[("ba", "RETFound", prev)])
    med = np.median(Mb, 0)
    beat[str(prev)] = int(GRID[np.argmax(med <= 0.18)]) if (med <= 0.18).any() else None
res["images_to_beat_best_estimator_mae_0.18"] = beat
print("images needed for the 95th-percentile BA error to fall below 0.18:", beat)
json.dump(res, open(os.path.join(OUT, "local_validation_cost.json"), "w"), indent=2)
print()
for kk, v in summ.items():
    print("%-16s median %s images (range %s), ~%s positives, reached %s"
          % (kk, v["median_images"], v["range_images"], v["median_positives"], v["reached"]))


COLS = {0.5: "#999999", 0.2: "#56B4E9", 0.1: "#0072B2", 0.088: "#D55E00"}


def lab(prev):
    return "%g%% intolerant" % (100 * prev) + (" (development cohort)" if prev == 0.088 else
                                               " (balanced public subsets)" if prev == 0.5 else "")


def panel(a, metric, kn, xmode, title):
    for prev in PREV:
        M = np.array(curves[(metric, kn, prev)]); M[~np.isfinite(M)] = np.nan
        x = GRID * (prev if xmode == "pos" else 1)
        a.plot(x, np.nanmedian(M, 0), color=COLS[prev], lw=2.2, label=lab(prev))
        a.fill_between(x, np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0), color=COLS[prev], alpha=0.15)
    a.axhline(0.10, ls="--", color="k", lw=1); a.axhline(0.05, ls=":", color="k", lw=1)
    if metric == "ba":
        a.axhline(0.18, ls="-.", color="#A07800", lw=1.2)
        a.text(a.get_xlim()[1] if False else (x.max()), 0.187, "MAE of the best label-free estimator (0.18)",
               ha="right", fontsize=8.5, color="#A07800")
    a.set_xscale("log"); a.set_ylim(0, 0.35)
    a.set_xlabel("labelled images from the site" if xmode == "img" else "intolerant eyes among them (expected)")
    a.set_title(title, fontsize=10.5)


import warnings; warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
# main figure: DINOv2 balanced accuracy, by images and by intolerant eyes
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
panel(ax[0], "ba", "DINOv2", "img", "(a) by labelled images")
panel(ax[1], "ba", "DINOv2", "pos", "(b) by intolerant eyes among them")
ax[0].set_ylabel("95th-percentile error of balanced accuracy"); ax[0].legend(fontsize=8.5, loc="upper right")
fig.suptitle("DINOv2 ensemble, median over the seven public cohorts (band: interquartile range)", fontsize=11, y=1.01)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_local_validation_cost.png"), dpi=300, bbox_inches="tight"); plt.close()
# supplementary: RETFound balanced accuracy, AUC for both backbones
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
panel(ax[0], "ba", "RETFound", "img", "(a) RETFound, balanced accuracy")
panel(ax[1], "auc", "DINOv2", "img", "(b) DINOv2, AUC")
panel(ax[2], "auc", "RETFound", "img", "(c) RETFound, AUC")
ax[0].set_ylabel("95th-percentile error"); ax[0].legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_local_validation_cost_auc.png"), dpi=300, bbox_inches="tight"); plt.close()
print("saved figures")
