# -*- coding: utf-8 -*-
"""
44_agreement_confidence.py — the two label-free signals a reviewer will ask about.

WHY: 42/43 tested three distributional divergence measures. The established literature on
predicting out-of-distribution performance from unlabelled data does not rely on those; it
relies on (i) agreement between independently trained models ("agreement-on-the-line",
Baek et al. 2022) and (ii) predictive confidence. Both are computable from the cached
predictions with no GPU, so the claim that label-free signals do not anticipate transfer here
should be tested against them too, or it is a straw man.

Both signals are label-free: agreement needs only the predicted classes of several models,
confidence only the predicted probabilities.

OUTPUT  results/external2/agreement_confidence.json  and  Figure_agreement.png
"""
import os, json, itertools
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "external2")
SETS = [("ddr", "DDR (China)"), ("jsiec", "JSIEC (China)"), ("deepdrid", "DeepDRiD (China)"),
        ("aptos", "APTOS (India)"), ("idrid", "IDRiD (India)"),
        ("messidor2", "Messidor-2 (France)"), ("eyepacs", "EyePACS (USA)")]
KINDS = ["dinov2_l", "retfound"]
SEEDS = 5   # raised from 3: seeds 3-4 were added for both backbones so that
            # the one degenerate RETFound run does not carry 1/3 of the weight
NAME = {"dinov2_l": "DINOv2", "retfound": "RETFound"}
C = {"dinov2_l": "#D55E00", "retfound": "#0072B2"}

meta = json.load(open(os.path.join(OUT, "meta_summary.json")))
obs = {r["key"]: {"dinov2_l": r["dino"], "retfound": r["ret"]} for r in meta["rows"]}

res = {}
for kind in KINDS:
    rows = []
    for s, label in SETS:
        probs = [np.load(os.path.join(OUT, f"pred_{kind}_seed{k}_{s}.npz"))["pos"]
                 for k in range(SEEDS)]
        preds = [(p > 0.5).astype(int) for p in probs]
        # (i) agreement: fraction of images on which two independently seeded models agree
        agr = float(np.mean([(a == b).mean() for a, b in itertools.combinations(preds, 2)]))
        # (ii) confidence: mean distance of the ensemble probability from the decision boundary
        ens = np.mean(probs, 0)
        conf = float(np.mean(np.maximum(ens, 1 - ens)))
        rows.append(dict(key=s, label=label, n=len(ens), external_auc=obs[s][kind],
                         agreement=agr, confidence=conf))
    corr = {}
    for m in ("agreement", "confidence"):
        rho, p = stats.spearmanr([r[m] for r in rows], [r["external_auc"] for r in rows])
        corr[m] = dict(spearman_rho=float(rho), p=float(p), n=len(rows))
    res[kind] = dict(rows=rows, correlations=corr)

print("=" * 78)
print("LABEL-FREE AGREEMENT AND CONFIDENCE VERSUS OBSERVED EXTERNAL AUC")
print("=" * 78)
for kind in KINDS:
    print(f"\n{NAME[kind]}")
    print(f"{'cohort':<22}{'n':>6}{'extAUC':>9}{'agreement':>12}{'confidence':>12}")
    for r in res[kind]["rows"]:
        print(f"{r['label']:<22}{r['n']:>6}{r['external_auc']:>9.3f}"
              f"{r['agreement']:>12.3f}{r['confidence']:>12.3f}")
    for m, c in res[kind]["correlations"].items():
        print(f"  Spearman {m:<11} vs external AUC: rho={c['spearman_rho']:+.3f} "
              f"p={c['p']:.3f} (n={c['n']})")

json.dump(res, open(os.path.join(OUT, "agreement_confidence.json"), "w"), indent=2)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.7))
for a, (m, xlabel, tag) in zip(ax, [("agreement", "between-seed agreement", "(a)"),
                                    ("confidence", "mean predictive confidence", "(b)")]):
    subs = []
    for kind in KINDS:
        rows = res[kind]["rows"]
        a.scatter([r[m] for r in rows], [r["external_auc"] for r in rows],
                  s=30 + 80 * np.array([r["n"] for r in rows]) / max(r["n"] for r in rows),
                  facecolor=C[kind], edgecolor="k", linewidth=0.5, alpha=0.85,
                  label=NAME[kind], zorder=3)
        c = res[kind]["correlations"][m]
        subs.append(f"{NAME[kind]} $\\rho$={c['spearman_rho']:+.2f} (p={c['p']:.2f})")
    if tag == "(a)":
        for r in res["dinov2_l"]["rows"]:
            a.annotate(r["key"].upper(), (r[m], r["external_auc"]), textcoords="offset points",
                       xytext=(8, -3), fontsize=7.5, color="#444")
    a.set_xlabel(xlabel + "   (label-free)")
    a.set_ylabel("observed external AUC" if tag == "(a)" else "")
    a.set_ylim(0.64, 1.02); a.margins(x=0.15)
    a.set_title(f"{tag} {xlabel}\n" + "   ".join(subs), fontsize=9.5)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, -0.07))
fig.tight_layout()
fn = os.path.join(OUT, "Figure_agreement.png")
fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close()
print(f"\nsaved {fn}")
