# -*- coding: utf-8 -*-
"""
66_text_numbers.py - the numbers quoted in the 2026-09-27 text revision that no earlier script
wrote to disk.

WHY: the pre-submission review (project/review_2026-09-27/) led to text that states the
calibration intercept, where the decision curve crosses zero, how many intolerant eyes the
selective-prediction curve keeps, how collinear the vascular measures are, and how the seed
spread differs by cohort. Each was first checked by hand; this script computes them from the
same result files the figures use, so every one is traceable.

USAGE   python scripts/66_text_numbers.py
OUTPUT  results/text_numbers.json
"""
import os, sys, json
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join(ROOT, "results")
DE = os.path.join(R, "dino_experiment")
out = {}

# ---- calibration and decision curve on the canonical pooled out-of-fold predictions
d = pd.read_csv(os.path.join(DE, "oof_dinov2_l_res224_frac1.0_main.csv"))
y, p = d.y.values.astype(int), np.clip(d.prob.values, 1e-6, 1 - 1e-6)
lp = np.log(p / (1 - p)); n = len(y); prev = y.mean()
cal = sm.GLM(y, np.ones(n), family=sm.families.Binomial(), offset=lp).fit()
slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
out["calibration"] = {
    "n": int(n), "prevalence": round(float(prev), 4), "mean_predicted": round(float(p.mean()), 4),
    "observed_over_expected": round(float(prev / p.mean()), 3),
    "intercept_calibration_in_the_large": round(float(cal.params[0]), 3),
    "intercept_ci95": [round(float(x), 3) for x in cal.conf_int()[0]],
    "slope": round(float(slope.params[1]), 3),
    "slope_ci95": [round(float(x), 3) for x in slope.conf_int()[1]]}


def nb(t):
    tp = ((p >= t) & (y == 1)).sum(); fp = ((p >= t) & (y == 0)).sum()
    return tp / n - fp / n * t / (1 - t)


grid = np.round(np.arange(0.01, 0.991, 0.01), 2)
nbs = np.array([nb(t) for t in grid])
first_neg = float(grid[np.argmax(nbs < 0)]) if (nbs < 0).any() else None
out["decision_curve"] = {"net_benefit_at_0.618": round(float(nb(0.618)), 4),
                         "net_benefit_at_0.5": round(float(nb(0.5)), 4),
                         "first_threshold_with_negative_net_benefit": first_neg,
                         "net_benefit_treat_none": 0.0,
                         "npv_treat_all_tolerant": round(float(1 - prev), 4)}

# ---- image-level AUC with a patient-clustered bootstrap (images of one patient resampled together)
from sklearn.metrics import roc_auc_score
rng = np.random.RandomState(0)
pts = d.anon_id.unique(); idx = {k: np.where(d.anon_id.values == k)[0] for k in pts}
vals = []
for _ in range(2000):
    ii = np.concatenate([idx[k] for k in rng.choice(pts, len(pts), True)])
    if y[ii].min() != y[ii].max():
        vals.append(roc_auc_score(y[ii], p[ii]))
out["clustered_bootstrap"] = {"image_auc": round(float(roc_auc_score(y, p)), 4),
                              "ci95_patient_clustered": [round(float(x), 3) for x in np.percentile(vals, [2.5, 97.5])],
                              "replicates": len(vals)}

# ---- selective prediction: intolerant eyes kept at each coverage (held-out test, scripts/31)
cur = np.load(os.path.join(DE, "dino_uncertainty_curve.npz"))
ys, sd = cur["ys"], cur["std_p"]; o = np.argsort(sd)
out["selective_prediction"] = {
    "set": "held-out test split, deployed model, MC-dropout 30 passes, threshold 0.5",
    "n": int(len(ys)), "n_intolerant": int(ys.sum()),
    "rows": [{"coverage": round(float(c), 1), "kept": int(max(1, int(len(o) * c))),
              "intolerant_kept": int(ys[o[:max(1, int(len(o) * c))]].sum()),
              "balanced_accuracy": round(float(b), 3)} for c, b in zip(cur["coverage"], cur["bacc"])]}

# ---- vascular measures: collinearity and association with the model score (Fig. 6b data)
bio = pd.read_csv(os.path.join(R, "imaging_biomarkers.csv"))
m = bio.merge(d[["anon_image", "prob"]], on="anon_image", how="inner")
feats = ["vessel_skeleton", "vessel_density", "fractal_dim", "dark_lesion_area"]
out["biomarker_spearman"] = {
    "n_images": int(len(m)),
    "skeleton_vs_density": round(float(stats.spearmanr(m.vessel_skeleton, m.vessel_density)[0]), 3),
    "with_model_score": {f: round(float(stats.spearmanr(m[f], m.prob)[0]), 3) for f in feats}}

# ---- multivariable model size (scripts/57)
s26 = None
for enc in ("utf-8", "gbk"):
    try:
        s26 = json.load(open(os.path.join(R, "section26_numbers.json"), encoding=enc)); break
    except UnicodeDecodeError:
        continue
mv = s26["multivariable_logit"]
out["multivariable"] = {"n": mv["n"], "n_predictors": len(mv["terms"]),
                        "events": int(y.sum()),
                        "events_per_predictor": round(y.sum() / len(mv["terms"]), 1)}

# ---- seed spread by cohort (scripts/40)
meta = json.load(open(os.path.join(R, "external2", "meta_summary.json")))
ratio = {r["key"]: round(r["ret_seed_sd"] / r["dino_seed_sd"], 1) for r in meta["rows"]}
out["seed_sd_ratio_retfound_over_dinov2"] = ratio
out["by_population"] = {k: {"china_mean": round(v["china_mean"], 3), "other_mean": round(v["other_mean"], 3),
                            "gap": round(v["gap"], 3), "separated": v["separated"]}
                        for k, v in meta["by_population"].items()}

json.dump(out, open(os.path.join(R, "text_numbers.json"), "w", encoding="utf-8"), indent=2)
print(json.dumps(out, indent=1))
