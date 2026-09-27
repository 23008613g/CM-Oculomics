# -*- coding: utf-8 -*-
"""
56_headline_metrics.py — recompute the headline performance metrics of Section 2.1.

WHY: results/dino_experiment/metrics_dinov2.json holds every number the paper reports for
its primary result (image- and patient-level AUC with confidence intervals, balanced
accuracy, sensitivity, specificity, PR-AUC, macro-F1, kappa and the operating threshold).
It was written in June, before the overlay masking, and no script under scripts/ recreates
it, so the retraining queue left it behind. This script rebuilds it from the canonical
out-of-fold predictions using the conventions stated in the Methods.

Conventions, matching the manuscript:
  * canonical run  = oof_dinov2_l_res224_frac1.0_main.csv (the first of the three repeats)
  * patient level  = mean predicted probability per patient, label = max over that patient's
                     eyes (a patient counts as intolerant if any eye progressed to PDR)
  * threshold      = Youden-optimal on the pooled out-of-fold predictions, chosen separately
                     at image and at patient level
  * intervals      = stratified bootstrap, 2,000 replicates, percentile method

USAGE   python scripts/56_headline_metrics.py
OUTPUT  results/dino_experiment/metrics_dinov2.json  (backup: *_preMask.json)
"""
import os, sys, json, shutil
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             balanced_accuracy_score, accuracy_score, f1_score,
                             cohen_kappa_score, confusion_matrix)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "dino_experiment")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
CANON = os.path.join(D, "oof_dinov2_l_res224_frac1.0_main.csv")
DST = os.path.join(D, "metrics_dinov2.json")
NBOOT = 2000


def youden(y, p):
    fpr, tpr, t = roc_curve(y, p)
    return float(t[np.argmax(tpr - fpr)])


def point_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(AUC=float(roc_auc_score(y, p)),
                PR_AUC=float(average_precision_score(y, p)),
                balanced_acc=float(balanced_accuracy_score(y, yh)),
                sensitivity=float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                specificity=float(tn / (tn + fp)) if (tn + fp) else float("nan"),
                accuracy=float(accuracy_score(y, yh)),
                macroF1=float(f1_score(y, yh, average="macro")),
                kappa=float(cohen_kappa_score(y, yh)))


def boot_ci(y, p, thr, keys, seed=0):
    """Stratified bootstrap: resample positives and negatives separately."""
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    acc = {k: [] for k in keys}
    for _ in range(NBOOT):
        b = np.concatenate([rng.choice(pos, len(pos), replace=True),
                            rng.choice(neg, len(neg), replace=True)])
        yb, pb = y[b], p[b]
        if len(np.unique(yb)) < 2:
            continue
        m = point_metrics(yb, pb, thr)
        for k in keys:
            acc[k].append(m[k])
    return {k: [round(float(np.percentile(v, 2.5)), 3),
                round(float(np.percentile(v, 97.5)), 3)] for k, v in acc.items()}


def block(y, p, extra):
    thr = youden(y, p)
    m = point_metrics(y, p, thr)
    keys = ["AUC", "PR_AUC", "balanced_acc", "sensitivity", "specificity"]
    out = dict(extra)
    out["threshold"] = round(thr, 3)
    out.update({k: round(v, 3) for k, v in m.items()})
    out["CI95"] = boot_ci(y, p, thr, keys)
    return out


def main():
    if os.path.exists(DST):
        bak = DST.replace(".json", "_preMask.json")
        if not os.path.exists(bak):
            shutil.copy2(DST, bak); print("backed up -> %s" % os.path.basename(bak))
    d = pd.read_csv(CANON)
    print("canonical run: %s  (%d rows)" % (os.path.basename(CANON), len(d)))

    yi, pi = d["y"].values.astype(int), d["prob"].values
    g = d.groupby("anon_id").agg(p=("prob", "mean"), y=("y", "max"))
    yp, pp = g["y"].values.astype(int), g["p"].values

    res = {
        "per_image": block(yi, pi, dict(n=int(len(yi)), n_intolerant=int(yi.sum()))),
        "per_patient": block(yp, pp, dict(n_patients=int(len(yp)),
                                          n_intolerant_patients=int(yp.sum()))),
        "_source": {"oof_csv": os.path.basename(CANON),
                    "patient_rule": "mean probability per patient; label = max over eyes",
                    "threshold_rule": "Youden-optimal on the pooled out-of-fold predictions",
                    "bootstrap": "stratified, %d replicates, percentile interval" % NBOOT},
    }
    json.dump(res, open(DST, "w"), indent=1)

    for lvl in ("per_image", "per_patient"):
        r = res[lvl]
        print("\n%s" % lvl.upper())
        print("  n=%s  intolerant=%s  threshold=%.3f"
              % (r.get("n", r.get("n_patients")),
                 r.get("n_intolerant", r.get("n_intolerant_patients")), r["threshold"]))
        for k in ("AUC", "PR_AUC", "balanced_acc", "sensitivity", "specificity"):
            print("  %-13s %.3f  95%% CI [%.3f, %.3f]" % (k, r[k], *r["CI95"][k]))
        for k in ("accuracy", "macroF1", "kappa"):
            print("  %-13s %.3f" % (k, r[k]))
    print("\nsaved %s" % DST)


if __name__ == "__main__":
    main()
