# -*- coding: utf-8 -*-
"""
53_backbone_table.py — the internal backbone comparison, recomputed after retraining
on the overlay-masked images.

Every number the manuscript reports for Fig. 3a and Fig. 4a is produced here, so the
figure script no longer has to carry them hardcoded. Three repeats exist for the two
foundation models; the three ImageNet baselines were run once, and that is stated in the
output rather than hidden by averaging different numbers of runs together.

USAGE   python scripts/53_backbone_table.py
OUTPUT  results/dino_experiment/backbone_table_postMask.json
"""
import os, sys, json, glob
import numpy as np, pandas as pd, importlib.util
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "dino_experiment")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

spec = importlib.util.spec_from_file_location("delong06", os.path.join(ROOT, "scripts", "06_delong.py"))
DL = importlib.util.module_from_spec(spec); sys.modules["delong06"] = DL; spec.loader.exec_module(DL)

# backbone -> the OOF files that are repeats of the SAME protocol
RUNS = {
    "DINOv2 (primary)": sorted(glob.glob(os.path.join(D, "oof_dinov2_l_res224_frac1.0_rep*.csv"))),
    "RETFound":         sorted(glob.glob(os.path.join(D, "oof_retfound_res224_frac1.0_rep*.csv"))),
    "Swin-V2":          [os.path.join(D, "oof_swinv2_res256_frac1.0_main.csv")],
    "ViT-B":            [os.path.join(D, "oof_vit_b_res224_frac1.0_main.csv")],
    "ResNet-50":        [os.path.join(D, "oof_resnet50_res224_frac1.0_main.csv")],
}


def load(f):
    d = pd.read_csv(f)
    g = d.groupby("anon_id").agg(p=("prob", "mean"), y=("y", "max"))
    return d, g


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.RandomState(seed); idx = np.arange(len(y)); a = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2: continue
        a.append(roc_auc_score(y[b], p[b]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


out = {}
print("=" * 96)
print("INTERNAL BACKBONE COMPARISON — pooled out-of-fold, retrained on masked images")
print("=" * 96)
print("%-18s %3s | %-28s | %-28s" % ("backbone", "n", "image AUC", "patient AUC"))
for name, files in RUNS.items():
    if not all(os.path.exists(f) for f in files):
        print("%-18s  -- missing files, skipped" % name); continue
    iaucs, paucs = [], []
    for f in files:
        d, g = load(f)
        iaucs.append(roc_auc_score(d.y, d.prob)); paucs.append(roc_auc_score(g.y, g.p))
    d0, g0 = load(files[0])
    ci_i = boot_ci(d0.y.values, d0.prob.values)
    ci_p = boot_ci(g0.y.values, g0.p.values)
    n = len(files)
    if n > 1:
        si = "%.4f +- %.4f" % (np.mean(iaucs), np.std(iaucs, ddof=1))
        sp = "%.4f +- %.4f" % (np.mean(paucs), np.std(paucs, ddof=1))
    else:
        si = "%.4f (single run)" % iaucs[0]; sp = "%.4f (single run)" % paucs[0]
    print("%-18s %3d | %-28s | %-28s" % (name, n, si, sp))
    out[name] = {"n_repeats": n,
                 "image_aucs": [round(x, 4) for x in iaucs],
                 "patient_aucs": [round(x, 4) for x in paucs],
                 "image_mean": round(float(np.mean(iaucs)), 4),
                 "patient_mean": round(float(np.mean(paucs)), 4),
                 "image_sd": round(float(np.std(iaucs, ddof=1)), 4) if n > 1 else None,
                 "patient_sd": round(float(np.std(paucs, ddof=1)), 4) if n > 1 else None,
                 "run1_image_ci95": [round(x, 4) for x in ci_i],
                 "run1_patient_ci95": [round(x, 4) for x in ci_p],
                 "files": [os.path.basename(f) for f in files]}

# ---- DINOv2 vs RETFound, DeLong per matched repeat (never on a mixed ensemble) ----
print()
print("DINOv2 vs RETFound, DeLong on each matched repeat (same OOF rows, paired):")
pairs = []
for i, (fa, fb) in enumerate(zip(RUNS["DINOv2 (primary)"], RUNS["RETFound"]), 1):
    da, _ = load(fa); db, _ = load(fb)
    m = da[["anon_image", "y", "prob"]].merge(db[["anon_image", "prob"]], on="anon_image",
                                              suffixes=("_dino", "_ret"))
    a1, a2, z, p = DL.delong_test(m.y.values.astype(int), m.prob_dino.values, m.prob_ret.values)
    pairs.append({"repeat": i, "dinov2": round(float(a1), 4), "retfound": round(float(a2), 4),
                  "delta": round(float(a1 - a2), 4), "p": float(p)})
    print("   repeat %d: DINOv2 %.4f  RETFound %.4f  delta %+.4f  p = %.3f"
          % (i, a1, a2, a1 - a2, p))
out["delong_dinov2_vs_retfound_per_repeat"] = pairs
dd = [x["delta"] for x in pairs]
print("   mean delta %+.4f (range %+.4f to %+.4f); every repeat p >= %.3f"
      % (np.mean(dd), min(dd), max(dd), min(x["p"] for x in pairs)))

json.dump(out, open(os.path.join(D, "backbone_table_postMask.json"), "w"), indent=2)
print("\nsaved results/dino_experiment/backbone_table_postMask.json")
