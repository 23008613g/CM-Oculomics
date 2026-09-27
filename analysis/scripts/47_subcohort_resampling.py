# -*- coding: utf-8 -*-
"""
47_subcohort_resampling.py — break the n=7 ceiling on the correlation analysis.

WHY: relating a label-free signal to observed performance across seven cohorts gives seven
points, and a reviewer will say that is too few to conclude anything. The cohorts differ
twenty-fold in size, so the large ones can be partitioned into DISJOINT subsets of the size of
the smallest cohort. Each subset yields its own shift measurement and its own AUC, raising the
number of points to roughly forty while keeping every comparison at a common n.

Subsets drawn from one cohort are not independent of each other, so this is reported as a
sensitivity analysis: the pooled correlation is given alongside the within-cohort correlations,
which ask the sharper question of whether shift tracks performance even among subsets of the
same population.

USAGE  python scripts/47_subcohort_resampling.py
OUTPUT results/external2/subcohort.json, Figure_subcohort.png

2026-09-27: panel titles count 40 disjoint subsets x 5 seeds per backbone instead of calling
the 400 measurements "400 disjoint subsets".
"""
import os, sys, json
import numpy as np
from scipy import stats, linalg
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "external2")
EMB = os.path.join(OUT, "emb")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SETS = [("ddr", "DDR (China)"), ("jsiec", "JSIEC (China)"), ("deepdrid", "DeepDRiD (China)"),
        ("aptos", "APTOS (India)"), ("idrid", "IDRiD (India)"),
        ("messidor2", "Messidor-2 (France)"), ("eyepacs", "EyePACS (USA)")]
CHINA = {"ddr", "jsiec", "deepdrid"}
KINDS = ["dinov2_l", "retfound"]
NAME = {"dinov2_l": "DINOv2", "retfound": "RETFound"}
C = {"dinov2_l": "#D55E00", "retfound": "#0072B2"}
SEEDS = 5   # raised from 3: seeds 3-4 were added for both backbones so that
            # the one degenerate RETFound run does not carry 1/3 of the weight
NPC = 16
NSUB = 70          # the size of the smallest cohort


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cm, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cm))


def mmd2_rbf(a, b):
    z = np.vstack([a, b])
    d2 = distance.squareform(distance.pdist(z, "sqeuclidean"))
    med = np.median(d2[d2 > 0])
    k = np.exp(-d2 / med)
    n, m = len(a), len(b)
    kxx = (k[:n, :n].sum() - np.trace(k[:n, :n])) / (n * (n - 1))
    kyy = (k[n:, n:].sum() - np.trace(k[n:, n:])) / (m * (m - 1))
    return float(kxx + kyy - 2 * k[:n, n:].mean())


rows = []
rng = np.random.RandomState(0)
for kind in KINDS:
    for seed in range(SEEDS):
        fdev = np.load(os.path.join(EMB, f"{kind}_seed{seed}_DEV.npy"))
        pca = PCA(NPC, random_state=0).fit(fdev)
        zdev = pca.transform(fdev)
        for s, label in SETS:
            z = pca.transform(np.load(os.path.join(EMB, f"{kind}_seed{seed}_{s}.npy")))
            d = np.load(os.path.join(OUT, f"pred_{kind}_seed{seed}_{s}.npz"))
            y, p = d["ys"], d["pos"]
            idx = rng.permutation(len(y))
            nchunk = max(1, len(y) // NSUB)
            for c in range(nchunk):
                sel = idx[c * NSUB:(c + 1) * NSUB] if nchunk > 1 else idx
                if len(np.unique(y[sel])) < 2:
                    continue
                dev = zdev[rng.choice(len(zdev), len(sel), replace=False)]
                rows.append(dict(
                    cohort=s, label=label, backbone=kind, seed=seed, chunk=c, n=int(len(sel)),
                    auc=float(roc_auc_score(y[sel], p[sel])),
                    frechet=frechet(dev, z[sel]), mmd2=mmd2_rbf(dev, z[sel]),
                    china=s in CHINA))

print("=" * 78)
print("SUB-COHORT RESAMPLING  (disjoint subsets of n=%d within each cohort)" % NSUB)
print("=" * 78)
per_seed_sets = len(rows) // (len(KINDS) * SEEDS)
print(f"{len(rows)} subset measurements  ({per_seed_sets} disjoint subsets "
      f"x {len(KINDS)} backbones x {SEEDS} seeds)")
print()
print(f"{'cohort':<22}{'subsets':>9}{'AUC mean':>10}{'AUC sd':>9}")
for s, label in SETS:
    g = [r for r in rows if r["cohort"] == s and r["backbone"] == "dinov2_l" and r["seed"] == 0]
    if g:
        a = np.array([r["auc"] for r in g])
        print(f"{label:<22}{len(g):>9}{a.mean():>10.3f}{a.std(ddof=1) if len(a) > 1 else 0:>9.3f}")

res = {}
print()
print("Spearman correlation of shift with subset AUC")
for kind in KINDS:
    g = [r for r in rows if r["backbone"] == kind]
    res[kind] = {}
    for m in ("frechet", "mmd2"):
        rho, p = stats.spearmanr([r[m] for r in g], [r["auc"] for r in g])
        # within-cohort: does shift track AUC among subsets of the SAME population?
        wr = []
        for s, _ in SETS:
            gg = [r for r in g if r["cohort"] == s]
            if len(gg) >= 5:
                w, _ = stats.spearmanr([r[m] for r in gg], [r["auc"] for r in gg])
                if np.isfinite(w):
                    wr.append(w)
        res[kind][m] = dict(pooled_rho=float(rho), pooled_p=float(p), n=len(g),
                            within_cohort_rho=[float(x) for x in wr],
                            within_cohort_mean=float(np.mean(wr)) if wr else None)
        wm = res[kind][m]["within_cohort_mean"]
        print(f"  {NAME[kind]:<9} {m:<9} pooled rho={rho:+.3f} p={p:.3f} (n={len(g)})"
              + (f"   within-cohort mean rho={wm:+.3f} ({len(wr)} cohorts)" if wr else ""))

json.dump(dict(rows=rows, correlations=res, n_subsets=len(rows), subset_size=NSUB),
          open(os.path.join(OUT, "subcohort.json"), "w"), indent=2)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.9))
for a, m, lab in ((ax[0], "frechet", "Fréchet distance"),
                  (ax[1], "mmd2", "MMD$^2$ (RBF kernel)")):
    for kind in KINDS:
        g = [r for r in rows if r["backbone"] == kind]
        a.scatter([r[m] for r in g], [r["auc"] for r in g], s=16, alpha=0.55,
                  facecolor=C[kind], edgecolor="none", label=NAME[kind])
    sub = "   ".join(f"{NAME[k]} $\\rho$={res[k][m]['pooled_rho']:+.2f}" for k in KINDS)
    a.set_xlabel(lab + "   (larger = more shift)")
    a.set_ylabel("subset AUC (n=%d)" % NSUB)
    # Count units honestly: len(rows) is subsets x backbones x seeds, not independent subsets.
    a.set_title(f"({'ab'[m == 'mmd2']}) {lab}\n{per_seed_sets} disjoint subsets x {SEEDS} seeds "
                f"per backbone:  {sub}", fontsize=10)
ax[0].legend(fontsize=9, loc="lower right")
fig.suptitle("Across %d size-matched subsets the association is weak, and its sign differs "
             "between backbones" % per_seed_sets, fontsize=12, y=1.02)
fig.tight_layout()
fn = os.path.join(OUT, "Figure_subcohort.png")
fig.savefig(fn, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nsaved {fn}")
