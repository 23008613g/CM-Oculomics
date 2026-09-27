# -*- coding: utf-8 -*-
"""
64_per_seed_table.py - one row per (cohort, backbone, seed): the supplementary data table
behind Sections 2.3-2.4, Fig. 11, Fig. 12 and Table S2.

WHY: reviewers could not reconcile the text with the figures because every figure was built
from summaries. This table lists, for each of the ten external models on each of the seven
cohorts, the observed discrimination and thresholded performance, and the four estimators'
outputs, so any summary in the paper can be recomputed from it. The ten in-distribution anchor
rows (held-out test split) are included because every estimator is anchored on them. The
cohort-level label-free signals are on a second sheet.

Everything is read from the cached prediction and result files; nothing is retrained.

USAGE   python scripts/64_per_seed_table.py
OUTPUT  results/external2/per_seed_table.xlsx and per_seed_table.csv
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(ROOT, "results", "external2")
SETS = [("ddr", "DDR", "China"), ("jsiec", "JSIEC", "China"), ("deepdrid", "DeepDRiD", "China"),
        ("aptos", "APTOS", "India"), ("idrid", "IDRiD", "India"),
        ("messidor2", "Messidor-2", "France"), ("eyepacs", "EyePACS", "USA")]
KINDS = [("dinov2_l", "DINOv2"), ("retfound", "RETFound")]
SEEDS = 5
THR = 0.5          # the threshold used by the estimators (scripts/46)
METH = ["ALine-D", "ALine-S", "DoC", "ATC"]


def thresholded(y, p):
    yhat = (p > THR).astype(int)       # strictly above, exactly as scripts/46 pred()
    se = float((yhat[y == 1] == 1).mean()); sp = float((yhat[y == 0] == 0).mean())
    return se, sp, (se + sp) / 2


est = pd.DataFrame(json.load(open(os.path.join(OUT, "aline_atc_doc.json")))["rows"])
rows = []
# in-distribution anchor rows
z = np.load(os.path.join(OUT, "pred_ID_test.npz")); y = z["ys"].astype(int)
for kind, kname in KINDS:
    for s in range(SEEDS):
        p = z["%s_seed%d" % (kind, s)]
        se, sp, ba = thresholded(y, p)
        rows.append(dict(cohort="In-house held-out test (anchor)", country="China", role="anchor",
                         backbone=kname, seed=s, n=len(y), n_pdr=int(y.sum()),
                         auc=roc_auc_score(y, p), balanced_accuracy=ba, sensitivity=se,
                         specificity=sp))
# external rows
for key, name, country in SETS:
    for kind, kname in KINDS:
        for s in range(SEEDS):
            f = np.load(os.path.join(OUT, "pred_%s_seed%d_%s.npz" % (kind, s, key)))
            y, p = f["ys"].astype(int), f["pos"]
            se, sp, ba = thresholded(y, p)
            e = est[(est.cohort == key) & (est.model == "%s_seed%d" % (kind, s))]
            assert len(e) == 1, (key, kind, s)
            e = e.iloc[0]
            assert abs(e["true"] - ba) < 1e-9, "balanced accuracy disagrees with scripts/46"
            rec = dict(cohort=name, country=country, role="external target", backbone=kname,
                       seed=s, n=len(y), n_pdr=int(y.sum()), auc=roc_auc_score(y, p),
                       balanced_accuracy=ba, sensitivity=se, specificity=sp)
            rec.update({"est_" + m: float(e[m]) for m in METH})
            rows.append(rec)
df = pd.DataFrame(rows)
for c in ["auc", "balanced_accuracy", "sensitivity", "specificity"] + ["est_" + m for m in METH]:
    df[c] = df[c].round(4)

# cohort-level sheet: ensemble AUC, seed spread, label-free signals
meta = {r["key"]: r for r in json.load(open(os.path.join(OUT, "meta_summary.json")))["rows"]}
shift = json.load(open(os.path.join(OUT, "shift_metrics.json")))
agree = json.load(open(os.path.join(OUT, "agreement_confidence.json")))
crow = []
for key, name, country in SETS:
    for kind, kname in KINDS:
        sh = {r["key"]: r for r in shift[kind]["rows"]}[key]
        ag = {r["key"]: r for r in agree[kind]["rows"]}[key]
        g = df[(df.cohort == name) & (df.backbone == kname)]
        crow.append(dict(cohort=name, country=country, backbone=kname, n=int(g.n.iloc[0]),
                         n_pdr=int(g.n_pdr.iloc[0]),
                         ensemble_auc=round(meta[key]["dino" if kind == "dinov2_l" else "ret"], 4),
                         seed_auc_mean=round(g.auc.mean(), 4), seed_auc_sd=round(g.auc.std(ddof=1), 4),
                         seed_ba_mean=round(g.balanced_accuracy.mean(), 4),
                         frechet=round(sh["frechet"], 3), mmd2=round(sh["mmd2"], 4),
                         domain_classifier_auc=round(sh["dom_auc"], 4),
                         between_seed_agreement=round(ag["agreement"], 4),
                         mean_confidence=round(ag["confidence"], 4)))
cdf = pd.DataFrame(crow)

readme = pd.DataFrame({"column": [
    "cohort / country / role", "backbone, seed", "n, n_pdr", "auc",
    "balanced_accuracy, sensitivity, specificity", "est_ALine-D ... est_ATC",
    "ensemble_auc (cohort sheet)", "seed_auc_mean, seed_auc_sd", "frechet, mmd2, domain_classifier_auc",
    "between_seed_agreement, mean_confidence"],
    "meaning": [
    "Evaluation set. 'anchor' = in-house held-out test split (140 images, 11 PDR), disjoint at "
    "patient level from all training data; 'external target' = class-balanced public subset.",
    "Pretraining family and training seed. Each model was trained on cross-validation folds 1-4 "
    "with early stopping on fold 0; no fine-tuning on any external cohort.",
    "Images and PDR-positive images in the evaluation set.",
    "Area under the ROC curve of that single model.",
    "At a fixed threshold of 0.5 on the predicted probability (the threshold used by the "
    "estimators).",
    "Label-free estimate of balanced accuracy by each published estimator (scripts/46). DoC is "
    "unbounded and can exceed 1.",
    "AUC of the mean-probability ensemble of the five seeds (the values quoted in the text).",
    "Mean and SD over the five single-seed AUCs.",
    "Distributional divergence between development and target embeddings, averaged over seeds "
    "(scripts/42).",
    "Mean pairwise agreement of the five seeds at 0.5, and mean max(p, 1-p) of the ensemble "
    "(scripts/44)."]})

xl = os.path.join(OUT, "per_seed_table.xlsx")
with pd.ExcelWriter(xl, engine="openpyxl") as w:
    readme.to_excel(w, sheet_name="README", index=False)
    df.to_excel(w, sheet_name="per_model", index=False)
    cdf.to_excel(w, sheet_name="per_cohort", index=False)
    # which data trained each model and which results it produced (scripts/69)
    pv = os.path.join(ROOT, "results", "model_provenance.csv")
    if os.path.exists(pv):
        pd.read_csv(pv).to_excel(w, sheet_name="model_provenance", index=False)
df.to_csv(os.path.join(OUT, "per_seed_table.csv"), index=False)
print("per_model rows: %d (anchor %d, external %d)" % (len(df), int((df.role == "anchor").sum()),
                                                        int((df.role != "anchor").sum())))
print(cdf[["cohort", "backbone", "ensemble_auc", "seed_auc_mean", "seed_auc_sd", "seed_ba_mean"]]
      .to_string(index=False))
print("saved", xl)
