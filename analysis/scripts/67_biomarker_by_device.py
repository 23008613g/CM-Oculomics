# -*- coding: utf-8 -*-
"""
67_biomarker_by_device.py - are the handcrafted vascular biomarkers measuring the eye or the
camera?

WHY: the development photographs come from three camera set-ups whose intolerance rates differ
(scripts/60). The deep model survives stratification by device (AUC 0.962 within VISUCAM), but
the handcrafted vessel measures are computed from raw pixel statistics and depend on
resolution, contrast and field of view. Fig. 4b shows them bimodal within the tolerant class.
This script asks, for each biomarker:
  * among TOLERANT eyes only (so the outcome cannot explain it), how well does the biomarker
    separate VISUCAM from the other devices?
  * is its association with intolerance still there WITHIN VISUCAM, the only device with enough
    intolerant eyes?
  * what happens to the multivariable odds ratios of Section 2.6 when device is added, or when
    the model is fitted within VISUCAM only? (Same specification as scripts/57.)

USAGE   python scripts/67_biomarker_by_device.py
OUTPUT  results/biomarker_by_device.json
"""
import os, sys, json
import numpy as np, pandas as pd
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join(ROOT, "results"); DATA = os.path.join(ROOT, "data_anon")
DEVICE = {(2400, 2040): "VISUCAM 200", (720, 576): "Topcon + frame grabber",
          (2592, 1728): "Canon digital back"}
FEATS = ["vessel_skeleton", "vessel_density", "fractal_dim", "dark_lesion_area"]

bio = pd.read_csv(os.path.join(R, "imaging_biomarkers.csv"))
bio["device"] = bio.anon_image.map(
    lambda f: DEVICE.get(Image.open(os.path.join(DATA, "images_with_overlays", f)).size, "other"))
bio["y"] = (bio.label == "PDR").astype(int)
vis = bio[bio.device == "VISUCAM 200"]
out = {"n_images": int(len(bio)), "n_visucam": int(len(vis)), "n_visucam_pdr": int(vis.y.sum()),
       "univariate": {}}


def auc_p(df, f):
    a = roc_auc_score(df.y, df[f]); a = max(a, 1 - a)
    p = stats.mannwhitneyu(df[df.y == 0][f].dropna(), df[df.y == 1][f].dropna()).pvalue
    return round(float(a), 3), float(p)


tol = bio[bio.y == 0]
print("%-18s %-15s %-15s %s" % ("biomarker", "pooled AUC, p", "VISUCAM AUC, p", "device sep. (tolerant)"))
for f in FEATS:
    pa, pp = auc_p(bio, f); va, vp = auc_p(vis, f)
    dev = roc_auc_score((tol.device == "VISUCAM 200").astype(int), tol[f]); dev = max(dev, 1 - dev)
    med = tol.groupby("device")[f].median().round(3).to_dict()
    out["univariate"][f] = {"pooled_auc": pa, "pooled_p": pp, "visucam_auc": va, "visucam_p": vp,
                            "tolerant_device_separation_auc": round(float(dev), 3),
                            "tolerant_median_by_device": med}
    print("%-18s %.3f, %.1e   %.3f, %.1e   %.3f" % (f, pa, pp, va, vp, dev))

# ---- multivariable model, same specification as scripts/57
OOF = pd.read_csv(os.path.join(R, "dino_experiment", "oof_dinov2_l_res224_frac1.0_main.csv"))
PTS = pd.read_csv(os.path.join(DATA, "patients_anon.csv"), encoding="utf-8-sig")
CLIN = ["年龄", "糖尿病年限（年）", "BMI", "HbA1c", "LDL", "HDL", "Triglyceride"]
for c in CLIN:
    PTS[c] = pd.to_numeric(PTS[c], errors="coerce")
dd = OOF.merge(bio.drop(columns=["y"]), on="anon_image", how="left").merge(
    PTS[["anon_id"] + CLIN], on="anon_id", how="left")
dd["y"] = dd.y.astype(int)
bio_feats = ["vessel_density", "vessel_skeleton", "fractal_dim", "dark_lesion_area", "bright_lesion_area"]
feats = bio_feats + CLIN


def fit(df, device_terms=False):
    sub = df.dropna(subset=feats + ["y"]).copy()
    X = (sub[feats] - sub[feats].mean()) / sub[feats].std()
    if device_terms:
        X = pd.concat([X, pd.get_dummies(sub.device, drop_first=True, dtype=float)], axis=1)
    m = sm.Logit(sub["y"], sm.add_constant(X)).fit(disp=0, method="bfgs", maxiter=2000)
    ci = np.exp(m.conf_int())
    return {"n": int(len(sub)), "events": int(sub.y.sum()),
            "skeleton_OR": round(float(np.exp(m.params["vessel_skeleton"])), 1),
            "skeleton_ci": [round(float(ci.loc["vessel_skeleton", 0]), 1), round(float(ci.loc["vessel_skeleton", 1]), 1)],
            "skeleton_p": float(m.pvalues["vessel_skeleton"]),
            "density_OR": round(float(np.exp(m.params["vessel_density"])), 2)}


out["multivariable"] = {"pooled_as_reported": fit(dd), "pooled_plus_device": fit(dd, True),
                        "within_visucam": fit(dd[dd.device == "VISUCAM 200"])}
for k, v in out["multivariable"].items():
    print("%-20s n=%d events=%d  skeleton OR %.1f (%.1f-%.1f) p=%.1e  density OR %.2f"
          % (k, v["n"], v["events"], v["skeleton_OR"], *v["skeleton_ci"], v["skeleton_p"], v["density_OR"]))

json.dump(out, open(os.path.join(R, "biomarker_by_device.json"), "w", encoding="utf-8"), indent=2)
print("saved results/biomarker_by_device.json")
