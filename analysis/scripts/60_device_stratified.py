# -*- coding: utf-8 -*-
"""
60_device_stratified.py - is the model's discrimination explained by acquisition device?

WHY: the development photographs come from three acquisition set-ups, identifiable by native
resolution, and the outcome rate differs sharply between them (roughly 12.6% PDR on the
VISUCAM 200, 2.4% on the Topcon frame-grabber set-up, 0% on the Canon digital back). A model
could therefore score well partly by recognising the device. This script measures how much of
the reported discrimination survives when device is held fixed or adjusted for.

Estimands, all on the canonical pooled out-of-fold predictions (the run reported in Section 2.1):
  * AUC within each device stratum (where both classes occur);
  * AUC of a device-only score, i.e. each image scored by its device's PDR rate, with the rate
    estimated only from the OTHER folds so the score is itself out-of-fold;
  * a logistic model of the outcome on the model's logit plus device indicators, to ask whether
    the model carries signal once device is accounted for.

USAGE   python scripts/60_device_stratified.py
OUTPUT  results/device_stratified.json
"""
import os, sys, json
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ORIG = os.path.join(ROOT, "data_anon", "images_with_overlays")   # native resolution
OOF = os.path.join(ROOT, "results", "dino_experiment", "oof_dinov2_l_res224_frac1.0_main.csv")
DEVICE = {"2400x2040": "VISUCAM 200", "720x576": "Topcon + frame grabber",
          "2592x1728": "Canon digital back"}
NBOOT = 2000


def boot_auc(y, p, seed=0):
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(NBOOT):
        b = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        v.append(roc_auc_score(y[b], p[b]))
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]


d = pd.read_csv(OOF)
d["y"] = d["y"].astype(int)
d["device"] = d["anon_image"].map(
    lambda f: DEVICE.get("%dx%d" % Image.open(os.path.join(ORIG, f)).size, "other"))
out = {"source": os.path.basename(OOF), "n_images": int(len(d)), "strata": {}}

print("DEVICE COMPOSITION OF THE CROSS-VALIDATION SET (n = %d)" % len(d))
comp = d.groupby("device").agg(images=("y", "size"), pdr=("y", "sum"),
                              patients=("anon_id", "nunique"))
comp["pdr_rate"] = comp.pdr / comp.images
print(comp.to_string())

print("\nMODEL AUC WITHIN EACH DEVICE")
for dev, g in d.groupby("device"):
    rec = {"images": int(len(g)), "pdr": int(g.y.sum()), "patients": int(g.anon_id.nunique())}
    if g.y.nunique() == 2:
        rec["auc"] = round(float(roc_auc_score(g.y, g.prob)), 4)
        rec["auc_ci95"] = boot_auc(g.y.values, g.prob.values)
        print("  %-24s n=%4d  PDR=%3d  AUC %.4f  95%% CI %s"
              % (dev, rec["images"], rec["pdr"], rec["auc"], rec["auc_ci95"]))
    else:
        rec["auc"] = None
        print("  %-24s n=%4d  PDR=%3d  AUC undefined (one class only)"
              % (dev, rec["images"], rec["pdr"]))
    out["strata"][dev] = rec

pooled = roc_auc_score(d.y, d.prob)
out["pooled_auc"] = round(float(pooled), 4)
print("  %-24s n=%4d  PDR=%3d  AUC %.4f" % ("all devices (pooled)", len(d), d.y.sum(), pooled))

# device-only score, out-of-fold: rate estimated on the other folds only
dev_score = np.zeros(len(d))
for k in sorted(d.fold.unique()):
    tr, va = d.fold != k, d.fold == k
    rate = d[tr].groupby("device").y.mean()
    dev_score[va.values] = d[va].device.map(rate).fillna(d[tr].y.mean()).values
auc_dev = roc_auc_score(d.y, dev_score)
out["device_only_auc"] = round(float(auc_dev), 4)
print("\nDEVICE ALONE AS A PREDICTOR (out-of-fold device PDR rate)")
print("  AUC %.4f" % auc_dev)

# does the model carry signal once device is adjusted for?
logit = np.log(np.clip(d.prob, 1e-6, 1 - 1e-6) / (1 - np.clip(d.prob, 1e-6, 1 - 1e-6)))
X = pd.get_dummies(d.device, drop_first=True, dtype=float)
X["model_logit"] = logit.values
X = sm.add_constant(X)
try:
    m = sm.Logit(d.y, X).fit(disp=0, method="bfgs", maxiter=500)
    coef, pv = float(m.params["model_logit"]), float(m.pvalues["model_logit"])
    out["adjusted_model_logit"] = {"coef": round(coef, 3), "p": pv}
    print("\nMODEL SIGNAL ADJUSTED FOR DEVICE (logistic regression)")
    print("  model logit coefficient %.3f, p = %.2g" % (coef, pv))
except Exception as e:
    out["adjusted_model_logit"] = {"error": str(e)}
    print("\nadjusted model failed: %s" % e)

# the headline, restricted to the one device where both classes are well represented
vis = d[d.device == "VISUCAM 200"]
g = vis.groupby("anon_id").agg(p=("prob", "mean"), y=("y", "max"))
out["visucam_patient_auc"] = round(float(roc_auc_score(g.y, g.p)), 4)
print("\nVISUCAM ONLY, PATIENT LEVEL: AUC %.4f (n=%d patients, %d intolerant)"
      % (out["visucam_patient_auc"], len(g), int(g.y.sum())))

json.dump(out, open(os.path.join(ROOT, "results", "device_stratified.json"), "w"), indent=2)
print("\nsaved results/device_stratified.json")
