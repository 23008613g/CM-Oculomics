# -*- coding: utf-8 -*-
"""
61_heldout_test.py - performance on the held-out in-house test split.

WHY: the development cohort was split at the patient level into a five-fold cross-validation
set (805 images) and an independent held-out test set (140 images from 69 patients, 11 PDR)
that no model saw during training or early stopping. The manuscript reports only the
cross-validation estimate as its internal result. The held-out split is the cleanest internal
check available, so its numbers are computed here, with intervals that reflect how few
positives it contains.

The models are the ten used throughout the transfer analysis (two backbones x five seeds),
each trained on folds 1-4 of the cross-validation set with early stopping on fold 0; the test
split is disjoint from all of them at the patient level.

USAGE   python scripts/61_heldout_test.py
OUTPUT  results/heldout_test.json
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
PRED = os.path.join(ROOT, "results", "external2", "pred_ID_test.npz")
SPLITS = os.path.join(ROOT, "data_anon", "splits.csv")
NBOOT = 2000


def boot(y, p, seed=0):
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [roc_auc_score(y[b], p[b]) for b in
         (np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
          for _ in range(NBOOT))]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


z = np.load(PRED)
y = z["ys"].astype(int)
sp = pd.read_csv(SPLITS).dropna(subset=["grade"])
te = sp[sp.split == "test"].reset_index(drop=True)
assert len(te) == len(y), "test rows do not line up with the cached predictions"
assert (te.grade.eq("PDR").astype(int).values == y).all(), "label order mismatch"

out = {"n_images": int(len(y)), "n_pdr": int(y.sum()),
       "n_patients": int(te.anon_id.nunique()),
       "n_patients_pdr": int(te.groupby("anon_id").grade.apply(lambda g: (g == "PDR").any()).sum()),
       "models": {}}
print("HELD-OUT TEST SPLIT  %d images (%d PDR), %d patients (%d intolerant)"
      % (out["n_images"], out["n_pdr"], out["n_patients"], out["n_patients_pdr"]))

for kind in ("dinov2_l", "retfound"):
    per = [float(roc_auc_score(y, z["%s_seed%d" % (kind, s)])) for s in range(5)]
    ens = np.mean([z["%s_seed%d" % (kind, s)] for s in range(5)], 0)
    te2 = te.assign(p=ens, y=y)
    g = te2.groupby("anon_id").agg(p=("p", "mean"), y=("y", "max"))
    rec = {"per_seed_auc": [round(a, 4) for a in per],
           "per_seed_mean": round(float(np.mean(per)), 4),
           "per_seed_sd": round(float(np.std(per, ddof=1)), 4),
           "ensemble_auc": round(float(roc_auc_score(y, ens)), 4),
           "ensemble_auc_ci95": boot(y, ens),
           "ensemble_patient_auc": round(float(roc_auc_score(g.y, g.p)), 4),
           "ensemble_patient_auc_ci95": boot(g.y.values, g.p.values)}
    # sensitivity / specificity of the ensemble at the operating point fixed in cross-validation
    # (Youden 0.618 of the canonical run) and at 0.5; the split played no part in choosing either
    for thr in (0.618, 0.5):
        yhat = (ens >= thr).astype(int)
        rec["at_%.3f" % thr] = {"sensitivity": round(float((yhat[y == 1] == 1).mean()), 3),
                                "specificity": round(float((yhat[y == 0] == 0).mean()), 3),
                                "tp": int(((yhat == 1) & (y == 1)).sum()),
                                "fp": int(((yhat == 1) & (y == 0)).sum())}
    out["models"][kind] = rec
    print("\n%s" % kind)
    print("  per seed        %s" % "  ".join("%.3f" % a for a in per))
    print("  mean +- sd      %.3f +- %.3f" % (rec["per_seed_mean"], rec["per_seed_sd"]))
    print("  5-seed ensemble image   %.3f  95%% CI %s" % (rec["ensemble_auc"], rec["ensemble_auc_ci95"]))
    print("  5-seed ensemble patient %.3f  95%% CI %s"
          % (rec["ensemble_patient_auc"], rec["ensemble_patient_auc_ci95"]))

json.dump(out, open(os.path.join(ROOT, "results", "heldout_test.json"), "w"), indent=2)
print("\nsaved results/heldout_test.json")
