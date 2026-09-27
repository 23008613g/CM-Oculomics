# -*- coding: utf-8 -*-
"""
57_section26_numbers.py - every quantity Section 2.6 reports, written to a results file.

WHY: the calibration errors, decision-curve summary, risk strata, subgroup AUCs, screening
metrics and laboratory correlations quoted in Section 2.6 were only ever produced inside the
figure script, so there was no results file to check the manuscript against. After
retraining they all have to be recomputed anyway, so they are computed here once and saved.

Conventions follow the figure code they replace: 10-bin expected calibration error, Youden
operating point on the pooled out-of-fold predictions, risk bands at probability 0.2 and 0.5,
and Spearman correlations against the routine laboratory panel.

USAGE   python scripts/57_section26_numbers.py
OUTPUT  results/section26_numbers.json
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); DE = os.path.join(R, "dino_experiment")
DATA = os.path.join(ROOT, "data_anon")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

OOF = pd.read_csv(os.path.join(DE, "oof_dinov2_l_res224_frac1.0_main.csv"))
OOF["y"] = OOF.y.astype(int)
RET = pd.read_csv(os.path.join(DE, "oof_retfound_res224_frac1.0_rep1.csv"))
RET["y"] = RET.y.astype(int)
BIO = pd.read_csv(os.path.join(R, "imaging_biomarkers.csv"))
# the OOF table already carries anon_id, so take only the syndrome column from the manifest
META = pd.read_csv(os.path.join(DATA, "manifest.csv"))[["anon_image", "tcm_syndrome"]]
PTS = pd.read_csv(os.path.join(DATA, "patients_anon.csv"), encoding="utf-8-sig")
CLIN = ["年龄", "糖尿病年限（年）", "BMI",
        "HbA1c", "LDL", "HDL", "Triglyceride"]
SEX = "性别"
for c in CLIN:
    PTS[c] = pd.to_numeric(PTS[c], errors="coerce")


def youden(y, p):
    fpr, tpr, t = roc_curve(y, p)
    return float(t[np.argmax(tpr - fpr)])


def ece(p, y, n=10):
    bins = np.linspace(0, 1, n + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n - 1)
    e = 0.0
    for b in range(n):
        m = idx == b
        if m.sum():
            e += m.sum() / len(y) * abs((y[m] == 1).mean() - p[m].mean())
    return float(e)


def ece_ci(p, y, nboot=2000, seed=0):
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(nboot):
        b = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        v.append(ece(p[b], y[b]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


def cal_slope(p, y):
    """Logistic recalibration slope of the outcome on the logit. Below 1 = over-confident."""
    q = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q))
    m = sm.Logit(y, sm.add_constant(z)).fit(disp=0)
    par = m.params
    return float(par.iloc[1] if hasattr(par, "iloc") else par[1])


def patient_level(df):
    g = df.groupby("anon_id").agg(p=("prob", "mean"), y=("y", "max"))
    return g["y"].values.astype(int), g["p"].values


out = {}

# ---- calibration -------------------------------------------------------------
yi, pi = OOF.y.values, OOF.prob.values
yp, pp = patient_level(OOF)
yri, pri = RET.y.values, RET.prob.values
yrp, prp = patient_level(RET)
out["calibration"] = {
    "dinov2_image_ece": round(ece(pi, yi), 3), "dinov2_image_ece_ci": ece_ci(pi, yi),
    "dinov2_patient_ece": round(ece(pp, yp), 3), "dinov2_patient_ece_ci": ece_ci(pp, yp),
    "retfound_image_ece": round(ece(pri, yri), 3),
    "retfound_patient_ece": round(ece(prp, yrp), 3),
    "dinov2_calibration_slope": round(cal_slope(pi, yi), 2),
    "retfound_calibration_slope": round(cal_slope(pri, yri), 2),
}
q = np.clip(pi, 1e-6, 1 - 1e-6)
z = np.log(q / (1 - q))
GRID = np.arange(0.20, 5.001, 0.01)


def nll(T):
    pt = 1 / (1 + np.exp(-z / T))
    pt = np.clip(pt, 1e-9, 1 - 1e-9)
    return float(-np.mean(yi * np.log(pt) + (1 - yi) * np.log(1 - pt)))


# Temperature is fitted the conventional way, by negative log-likelihood. Picking the T that
# minimises ECE on the same predictions would be tuning on the evaluation metric, so that
# value is recorded separately and only as a lower bound on what any T could achieve.
T_nll = float(GRID[int(np.argmin([nll(T) for T in GRID]))])
eces = [ece(1 / (1 + np.exp(-z / T)), yi) for T in GRID]
T_ece = float(GRID[int(np.argmin(eces))])
out["calibration"].update({
    "temperature_fitted_by_nll": round(T_nll, 2),
    "ece_after_nll_temperature": round(ece(1 / (1 + np.exp(-z / T_nll)), yi), 3),
    "temperature_minimising_ece": round(T_ece, 2),
    "ece_at_that_temperature": round(min(eces), 3),
    "ece_optimum_is_interior_to_grid": bool(GRID[0] < T_ece < GRID[-1]),
    "grid": [float(GRID[0]), float(GRID[-1])],
})

# ---- screening metrics at the Youden point -----------------------------------
thr = youden(yi, pi)
yh = (pi >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yi, yh).ravel()
out["screening"] = {
    "threshold_youden": round(thr, 3),
    "sensitivity": round(tp / (tp + fn), 3), "specificity": round(tn / (tn + fp), 3),
    "ppv": round(tp / (tp + fp), 3), "npv": round(tn / (tn + fn), 3),
    "brier": round(float(np.mean((pi - yi) ** 2)), 3),
}
npvs = []
for t in np.arange(0.3, 0.71, 0.05):
    yh2 = (pi >= t).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(yi, yh2).ravel()
    npvs.append(tn2 / (tn2 + fn2))
out["screening"]["npv_range_thr_0.3_to_0.7"] = [round(min(npvs), 3), round(max(npvs), 3)]

# ---- risk stratification ------------------------------------------------------
d = OOF.copy()
d["risk"] = pd.cut(d.prob, [-.01, 0.2, 0.5, 1.01], labels=["Low", "Medium", "High"])
rate = d.groupby("risk", observed=True).y.mean() * 100
cnt = d.groupby("risk", observed=True).y.size()
out["risk_strata"] = {str(k): {"rate_pct": round(float(v), 1), "n": int(cnt[k])}
                      for k, v in rate.items()}

# ---- multivariable logistic regression ---------------------------------------
dd = OOF.merge(BIO, on="anon_image", how="left").merge(META, on="anon_image", how="left")
dd = dd.merge(PTS[["anon_id", SEX] + CLIN], on="anon_id", how="left")
bio_feats = ["vessel_density", "vessel_skeleton", "fractal_dim",
             "dark_lesion_area", "bright_lesion_area"]
feats = bio_feats + CLIN
sub = dd.dropna(subset=feats + ["y"]).copy()
X = (sub[feats] - sub[feats].mean()) / sub[feats].std()
m = sm.Logit(sub["y"], sm.add_constant(X)).fit(disp=0)
ci = np.exp(m.conf_int())
out["multivariable_logit"] = {"n": int(len(sub)), "terms": {}}
for f in feats:
    out["multivariable_logit"]["terms"][f] = {
        "OR": round(float(np.exp(m.params[f])), 2),
        "ci": [round(float(ci.loc[f, 0]), 2), round(float(ci.loc[f, 1]), 2)],
        "p": float(m.pvalues[f])}

# ---- subgroups ----------------------------------------------------------------
dd["sex"] = dd[SEX].map({"男": "Male", "女": "Female"})
dd["age_grp"] = np.where(dd[CLIN[0]] >= 65, "Age>=65", "Age<65")
dd["dur_grp"] = np.where(dd[CLIN[1]] >= 7, "Dur>=7y", "Dur<7y")
sg = {}
for col in ["sex", "age_grp", "dur_grp", "tcm_syndrome"]:
    for v in sorted(x for x in dd[col].dropna().unique()):
        s = dd[dd[col] == v]
        if s.y.nunique() == 2:
            sg[str(v)] = {"n": int(len(s)), "auc": round(float(roc_auc_score(s.y, s.prob)), 3)}
out["subgroup_auc"] = sg

# ---- risk score versus routine laboratory indicators --------------------------
g = OOF.groupby("anon_id").agg(prob=("prob", "mean")).reset_index()
mm = g.merge(PTS[["anon_id"] + CLIN], on="anon_id", how="left")
labs = {}
for c in CLIN:
    s = mm.dropna(subset=[c, "prob"])
    r, pv = stats.spearmanr(s.prob, s[c])
    labs[c] = {"spearman_r": round(float(r), 3), "p": round(float(pv), 4), "n": int(len(s))}
out["risk_vs_labs"] = labs
out["risk_vs_labs_max_abs_r"] = round(max(abs(v["spearman_r"]) for v in labs.values()), 3)

json.dump(out, open(os.path.join(R, "section26_numbers.json"), "w"), indent=2, ensure_ascii=False)

c = out["calibration"]
print("CALIBRATION")
print("  DINOv2   image ECE %.3f %s | patient ECE %.3f %s | slope %.2f"
      % (c["dinov2_image_ece"], c["dinov2_image_ece_ci"], c["dinov2_patient_ece"],
         c["dinov2_patient_ece_ci"], c["dinov2_calibration_slope"]))
print("  RETFound image ECE %.3f | patient ECE %.3f | slope %.2f"
      % (c["retfound_image_ece"], c["retfound_patient_ece"], c["retfound_calibration_slope"]))
print("  temperature fitted by NLL: T=%.2f -> ECE %.3f (uncalibrated %.3f)"
      % (c["temperature_fitted_by_nll"], c["ece_after_nll_temperature"], c["dinov2_image_ece"]))
print("  best possible T for ECE alone: T=%.2f -> %.3f (interior to grid: %s)"
      % (c["temperature_minimising_ece"], c["ece_at_that_temperature"],
         c["ece_optimum_is_interior_to_grid"]))
s = out["screening"]
print("\nSCREENING (Youden thr=%.3f)" % s["threshold_youden"])
print("  sens %.3f  spec %.3f  PPV %.3f  NPV %.3f  Brier %.3f  NPV across thr 0.3-0.7 %s"
      % (s["sensitivity"], s["specificity"], s["ppv"], s["npv"], s["brier"],
         s["npv_range_thr_0.3_to_0.7"]))
print("\nRISK STRATA")
for k, v in out["risk_strata"].items():
    print("  %-7s %5.1f%%  (n=%d)" % (k, v["rate_pct"], v["n"]))
print("\nTOP MULTIVARIABLE TERMS (by p)")
for f, v in sorted(out["multivariable_logit"]["terms"].items(), key=lambda kv: kv[1]["p"])[:4]:
    print("  %-22s OR %7.2f  95%% CI [%.2f, %.2f]  p=%.2g" % (f, v["OR"], v["ci"][0], v["ci"][1], v["p"]))
print("\nSUBGROUP AUC range %.3f to %.3f over %d subgroups"
      % (min(v["auc"] for v in sg.values()), max(v["auc"] for v in sg.values()), len(sg)))
print("RISK vs LABS: max |Spearman r| = %.3f" % out["risk_vs_labs_max_abs_r"])
print("\nsaved results/section26_numbers.json")
