# -*- coding: utf-8 -*-
"""
75_estimator_baselines.py - a label-free baseline for the performance estimators.

WHY: Section 2.4 compared the four published estimators with a constant equal to the mean
observed accuracy. That constant needs the target labels, so it is a reference, not an
alternative a site could use (all three blind reviewers said so). The natural label-free
baseline is to assume nothing changes: predict each model's in-distribution (anchor) balanced
accuracy for every cohort. An estimator is only useful if it beats that. This script scores
the no-shift baseline on the same 70 model-cohort pairs, and reports how well each estimator
RANKS the pairs (Spearman), separately from its level error.

USAGE   python scripts/75_estimator_baselines.py
OUTPUT  results/external2/estimator_baselines.json
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(ROOT, "results", "external2")
J = json.load(open(os.path.join(OUT, "aline_atc_doc.json")))
df = pd.DataFrame(J["rows"])
acc_id = J["id_anchor"]["balanced_acc"]
df["No-shift"] = df.model.map(acc_id)
df["Mean observed (uses labels)"] = df.true.mean()
METH = ["ALine-D", "ALine-S", "DoC", "ATC", "No-shift", "Mean observed (uses labels)"]
out = {"n_pairs": int(len(df)), "methods": {}}
print("%-28s %6s %7s %6s %9s" % ("method", "MAE", "bias", "worst", "Spearman"))
for m in METH:
    e = df[m] - df.true
    rho = stats.spearmanr(df[m], df.true)[0] if df[m].nunique() > 1 else float("nan")
    out["methods"][m] = {"mae": round(float(e.abs().mean()), 3), "bias": round(float(e.mean()), 3),
                         "worst": round(float(e.abs().max()), 3),
                         "spearman_with_true": None if not np.isfinite(rho) else round(float(rho), 2)}
    r = out["methods"][m]
    print("%-28s %6.3f %+7.3f %6.3f %9s" % (m, r["mae"], r["bias"], r["worst"],
                                            "n/a" if r["spearman_with_true"] is None else "%+.2f" % r["spearman_with_true"]))
json.dump(out, open(os.path.join(OUT, "estimator_baselines.json"), "w"), indent=2)
print("saved results/external2/estimator_baselines.json")
