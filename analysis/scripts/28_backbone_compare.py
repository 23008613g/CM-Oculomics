# -*- coding: utf-8 -*-
"""Compare RETFound vs DINOv2-L: pooled/patient AUC+CI, DeLong (internal+external)."""
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PP = os.path.join(ROOT, "results", "perpatient", "oof_predictions.csv")
DINO = os.path.join(ROOT, "results", "dino_experiment", "oof_dinov2_l_res224_frac1.0_main.csv")
EXT_RET = os.path.join(ROOT, "results", "external_valpred.npz")
EXT_DINO = os.path.join(ROOT, "results", "dino_experiment", "ddr_dinov2_l_res224.npz")
rng = np.random.default_rng(42)


# ---- fast DeLong for two correlated ROC AUCs (same samples) ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong(y, p1, p2):
    y = np.asarray(y); p1 = np.asarray(p1); p2 = np.asarray(p2)
    pos = y == 1; neg = y == 0
    m, n = pos.sum(), neg.sum()
    preds = np.vstack([p1, p2])
    aucs, v01, v10 = [], [], []
    for k in range(2):
        X = preds[k][pos]; Y = preds[k][neg]
        tx = _midrank(X); ty = _midrank(Y); tz = _midrank(np.concatenate([X, Y]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc)
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    v01 = np.array(v01); v10 = np.array(v10)
    s01 = np.cov(v01); s10 = np.cov(v10)
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    from scipy import stats
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], aucs[0] - aucs[1], p


def boot_auc_ci(y, p, n=2000):
    y = np.asarray(y); p = np.asarray(p); a = []
    for _ in range(n):
        b = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[b])) < 2: continue
        a.append(roc_auc_score(y[b], p[b]))
    return roc_auc_score(y, p), np.percentile(a, [2.5, 97.5])


def patient_level(df):
    g = df.groupby("anon_id").agg(y=("y", "max"), prob=("prob", "mean")).reset_index()
    return g["y"].values, g["prob"].values


# ---- load & align internal OOF on same images ----
ret = pd.read_csv(PP); dino = pd.read_csv(DINO)
m = ret[["anon_id", "anon_image", "y", "prob"]].merge(
    dino[["anon_image", "prob"]], on="anon_image", suffixes=("_ret", "_dino"))
print(f"internal aligned images: {len(m)} (RET {len(ret)}, DINO {len(dino)})")
y = m["y"].values

# image-level pooled
a_ret, ci_ret = boot_auc_ci(y, m["prob_ret"].values)
a_dino, ci_dino = boot_auc_ci(y, m["prob_dino"].values)
ar, ad, d, pdl = delong(y, m["prob_ret"].values, m["prob_dino"].values)
print(f"\n[INTERNAL image-level] RETFound {a_ret:.3f} {ci_ret.round(3)} | DINOv2 {a_dino:.3f} {ci_dino.round(3)}")
print(f"  DeLong RET-DINO ΔAUC={d:+.3f} p={pdl:.3f}")

# patient-level
yr, pr = patient_level(m.rename(columns={"prob_ret": "prob"}))
yd, pd_ = patient_level(m.rename(columns={"prob_dino": "prob"}))
pa_ret, pci_ret = boot_auc_ci(yr, pr); pa_dino, pci_dino = boot_auc_ci(yd, pd_)
par, pad, pd2, ppdl = delong(yr, pr, pd_)
print(f"\n[INTERNAL patient-level] RETFound {pa_ret:.3f} {pci_ret.round(3)} | DINOv2 {pa_dino:.3f} {pci_dino.round(3)}")
print(f"  DeLong RET-DINO ΔAUC={pd2:+.3f} p={ppdl:.3f}")

# ---- external DDR ----
er = np.load(EXT_RET); ed = np.load(EXT_DINO)
# both scored on DDR subset (same order = subset_labels.csv)
yr_e, pr_e = er["ys"], er["pos"]; yd_e, pd_e = ed["ys"], ed["pos"]
ea_ret, eci_ret = boot_auc_ci(yr_e, pr_e); ea_dino, eci_dino = boot_auc_ci(yd_e, pd_e)
if len(yr_e) == len(yd_e) and (yr_e == yd_e).all():
    ear, ead, ed2, epdl = delong(yr_e, pr_e, pd_e)
    ext_delong = f"ΔAUC={ed2:+.3f} p={epdl:.3f}"
else:
    ext_delong = "labels misaligned — DeLong skipped"
print(f"\n[EXTERNAL DDR] RETFound {ea_ret:.3f} {eci_ret.round(3)} | DINOv2 {ea_dino:.3f} {eci_dino.round(3)}")
print(f"  DeLong external: {ext_delong}")

# ---- bal-acc/sens/spec at 0.5 for DINOv2 internal (for table) ----
def opstats(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    return balanced_accuracy_score(y, pred), sens, spec, average_precision_score(y, p)

bd = opstats(y, m["prob_dino"].values)
br = opstats(y, m["prob_ret"].values)

# ---- write CSV ----
rows = [
    dict(backbone="RETFound (ViT-L/16)", internal_pooled_auc=round(a_ret,3),
         internal_ci=f"[{ci_ret[0]:.3f},{ci_ret[1]:.3f}]", patient_auc=round(pa_ret,3),
         patient_ci=f"[{pci_ret[0]:.3f},{pci_ret[1]:.3f}]", external_ddr_auc=round(ea_ret,3),
         external_ci=f"[{eci_ret[0]:.3f},{eci_ret[1]:.3f}]", bal_acc=round(br[0],3),
         sens=round(br[1],3), spec=round(br[2],3), pr_auc=round(br[3],3),
         delong_p_vs_retfound_internal="—", delong_p_vs_retfound_external="—"),
    dict(backbone="DINOv2 (ViT-L/14)", internal_pooled_auc=round(a_dino,3),
         internal_ci=f"[{ci_dino[0]:.3f},{ci_dino[1]:.3f}]", patient_auc=round(pa_dino,3),
         patient_ci=f"[{pci_dino[0]:.3f},{pci_dino[1]:.3f}]", external_ddr_auc=round(ea_dino,3),
         external_ci=f"[{eci_dino[0]:.3f},{eci_dino[1]:.3f}]", bal_acc=round(bd[0],3),
         sens=round(bd[1],3), spec=round(bd[2],3), pr_auc=round(bd[3],3),
         delong_p_vs_retfound_internal=round(pdl,3),
         delong_p_vs_retfound_external=(round(epdl,3) if 'epdl' in dir() else "NA")),
]
out = os.path.join(ROOT, "results", "dino_experiment", "backbone_comparison_v32.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nsaved {out}")
