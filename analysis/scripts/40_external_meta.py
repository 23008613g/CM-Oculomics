# -*- coding: utf-8 -*-
"""
40_external_meta.py — pool the five external datasets into one picture.

Produces a forest plot of the per-dataset DINOv2-minus-RETFound AUC difference, an
inverse-variance pooled estimate, and a seed-stability comparison (SD of AUC across the
five training seeds, per backbone per dataset).

2026-09-27: the (b) reference line reads the held-out-test AUC of the same ensembles from
results/heldout_test.json (was a hardcoded 0.938); the (c) axis label follows SEEDS; the (b)
title no longer asserts that transfer splits "by population".
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "external2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SETS = [("ddr", "DDR (China)"), ("jsiec", "JSIEC (China, Shantou)"),
        ("deepdrid", "DeepDRiD (China, multi-centre)"),
        ("aptos", "APTOS (India, Aravind)"),
        ("idrid", "IDRiD (India, Nanded)"), ("messidor2", "Messidor-2 (France)"),
        ("eyepacs", "EyePACS (USA)")]
SEEDS = 5   # raised from 3: seeds 3-4 were added for both backbones so that
            # the one degenerate RETFound run does not carry 1/3 of the weight
C = {"dinov2_l": "#D55E00", "retfound": "#0072B2"}


def load(kind, s):
    """per-seed AUCs and the mean-probability ensemble."""
    aucs, probs, ys = [], [], None
    for k in range(SEEDS):
        f = os.path.join(OUT, f"pred_{kind}_seed{k}_{s}.npz")
        z = np.load(f); ys = z["ys"]; probs.append(z["pos"])
        aucs.append(roc_auc_score(z["ys"], z["pos"]))
    return np.array(aucs), ys, np.mean(probs, 0)


def boot_delta(y, p1, p2, n=4000, seed=0):
    """bootstrap CI for the paired AUC difference."""
    rng = np.random.RandomState(seed); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        d.append(roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p2[b]))
    d = np.array(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), d.std()


rows, res = [], {}
for s, label in SETS:
    a1, y, p1 = load("dinov2_l", s)
    a2, _, p2 = load("retfound", s)
    d, lo, hi, se = boot_delta(y, p1, p2)
    rows.append(dict(key=s, label=label, n=len(y), npos=int(y.sum()),
                     dino=roc_auc_score(y, p1), ret=roc_auc_score(y, p2),
                     delta=d, lo=lo, hi=hi, se=max(se, 1e-6),
                     dino_seed_sd=a1.std(ddof=1), ret_seed_sd=a2.std(ddof=1),
                     dino_seeds=a1.tolist(), ret_seeds=a2.tolist()))

# inverse-variance (fixed-effect) pooling of the AUC difference
w = np.array([1 / r["se"] ** 2 for r in rows])
dl = np.array([r["delta"] for r in rows])
pool = float((w * dl).sum() / w.sum())
pool_se = float(np.sqrt(1 / w.sum()))
# heterogeneity
Q = float((w * (dl - pool) ** 2).sum())
I2 = max(0.0, (Q - (len(rows) - 1)) / Q) * 100 if Q > 0 else 0.0
# sample-size weighted mean (simple, reported alongside)
nn = np.array([r["n"] for r in rows], float)
wmean = float((nn * dl).sum() / nn.sum())

print("=" * 78)
print("SEVEN-COHORT EXTERNAL COMPARISON  (DINOv2 minus RETFound, zero fine-tuning)")
print("=" * 78)
print(f"{'dataset':<26}{'n':>6}{'DINOv2':>9}{'RETF':>8}{'dAUC':>9}   95% CI")
for r in rows:
    print(f"{r['label']:<26}{r['n']:>6}{r['dino']:>9.3f}{r['ret']:>8.3f}"
          f"{r['delta']:>+9.3f}   [{r['lo']:+.3f}, {r['hi']:+.3f}]")
print("-" * 78)
print(f"{'POOLED (inverse-variance)':<26}{int(nn.sum()):>6}{'':>9}{'':>8}{pool:>+9.3f}"
      f"   [{pool - 1.96 * pool_se:+.3f}, {pool + 1.96 * pool_se:+.3f}]")
print(f"{'sample-size weighted mean':<26}{'':>6}{'':>9}{'':>8}{wmean:>+9.3f}")
print(f"heterogeneity: Q={Q:.1f}, I^2={I2:.0f}%")
print()
print("SEED STABILITY (SD of AUC over %d training seeds; lower = more reproducible)" % SEEDS)
print(f"{'dataset':<26}{'DINOv2':>9}{'RETFound':>10}")
for r in rows:
    print(f"{r['label']:<26}{r['dino_seed_sd']:>9.3f}{r['ret_seed_sd']:>10.3f}")
print(f"{'MEAN':<26}{np.mean([r['dino_seed_sd'] for r in rows]):>9.3f}"
      f"{np.mean([r['ret_seed_sd'] for r in rows]):>10.3f}")

# ---- does the training population explain transfer better than the backbone does? ----
# the development cohort is Chinese (Longhua Hospital, Shanghai)
CHINA = {"ddr", "jsiec", "deepdrid"}
from scipy import stats as _st
grp = {}
for bk in ("dino", "ret"):
    cn = [r[bk] for r in rows if r["key"] in CHINA]
    ot = [r[bk] for r in rows if r["key"] not in CHINA]
    u, p = _st.mannwhitneyu(cn, ot, alternative="greater")
    grp[bk] = dict(china=cn, other=ot, china_mean=float(np.mean(cn)),
                   other_mean=float(np.mean(ot)), gap=float(np.mean(cn) - np.mean(ot)),
                   separated=bool(min(cn) > max(ot)), mannwhitney_p=float(p))
print()
print("TRANSFER BY POPULATION  (development cohort is Chinese: Longhua Hospital, Shanghai)")
for bk, nm in (("dino", "DINOv2"), ("ret", "RETFound")):
    g = grp[bk]
    print(f"  {nm:<9} Chinese cohorts {g['china_mean']:.3f}  "
          f"non-Chinese {g['other_mean']:.3f}   gap {g['gap']:+.3f}   "
          f"{'PERFECT SEPARATION' if g['separated'] else 'overlapping'}   "
          f"Mann-Whitney one-sided p={g['mannwhitney_p']:.3f}")
print(f"  (external AUC spread across the 7 cohorts, DINOv2: "
      f"{min(r['dino'] for r in rows):.3f} to {max(r['dino'] for r in rows):.3f})")

json.dump(dict(rows=rows, pooled=pool, pooled_se=pool_se, I2=I2,
               size_weighted=wmean, by_population=grp),
          open(os.path.join(OUT, "meta_summary.json"), "w"), indent=2)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6),
                       gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

# (a) forest plot
a = ax[0]
ys = np.arange(len(rows))[::-1]
for yv, r in zip(ys, rows):
    col = C["dinov2_l"] if r["delta"] > 0 else C["retfound"]
    a.plot([r["lo"], r["hi"]], [yv, yv], color=col, lw=2.2)
    a.scatter([r["delta"]], [yv], s=28 + 90 * r["n"] / max(nn), color=col, zorder=3)
a.axvline(0, color="k", lw=1, ls="--")
a.axvspan(pool - 1.96 * pool_se, pool + 1.96 * pool_se, color="grey", alpha=0.18)
a.axvline(pool, color="k", lw=2)
a.set_yticks(ys)
a.set_yticklabels([f"{r['label']}\nn={r['n']}" for r in rows], fontsize=9)
a.set_xlabel("$\\Delta$AUC  (DINOv2 $-$ RETFound)")
a.set_title(f"(a) Per-dataset difference\nfixed-effect pooled {pool:+.3f} "
            f"[{pool - 1.96 * pool_se:+.3f}, {pool + 1.96 * pool_se:+.3f}], "
            f"I$^2$={I2:.0f}%", fontsize=10)
a.text(0.02, 0.03, "← favours RETFound   |   favours DINOv2 →",
       transform=a.transAxes, fontsize=8, color="#555")

# (b) absolute AUC per dataset
a = ax[1]
x = np.arange(len(rows))
a.bar(x - 0.2, [r["dino"] for r in rows], 0.4, color=C["dinov2_l"], label="DINOv2")
a.bar(x + 0.2, [r["ret"] for r in rows], 0.4, color=C["retfound"], label="RETFound")
# Reference line: the SAME five-seed ensembles scored on the in-house held-out test split
# (scripts/61_heldout_test.py). These are exactly the models evaluated externally, so the gap
# to each bar is a like-for-like transfer loss. The cross-validation estimate (0.952) comes
# from different models and its early stopping used the evaluated fold (see Methods).
_ht = json.load(open(os.path.join(ROOT, "results", "heldout_test.json")))["models"]
_ref = {k: _ht[k]["ensemble_auc"] for k in ("dinov2_l", "retfound")}
a.axhline(_ref["dinov2_l"], color="k", ls=":", lw=1.4,
          label="in-house held-out test\n(DINOv2 %.3f, RETFound %.3f)" % (_ref["dinov2_l"], _ref["retfound"]))
a.set_xticks(x); a.set_xticklabels([r["key"].upper() for r in rows], fontsize=9, rotation=20,
                                   ha="right")
# headroom above the tallest bar so neither the legend nor the group labels sit on the data
a.set_ylim(0.5, 1.12); a.set_ylabel("External AUC")
a.legend(fontsize=7.5, loc="upper right", ncol=1, frameon=True, framealpha=0.95)
# shade the three Chinese cohorts: the development cohort is Chinese too
nchina = sum(1 for r in rows if r["key"] in CHINA)
a.axvspan(-0.5, nchina - 0.5, color="#F0E442", alpha=0.25, zorder=0)
# group captions go BELOW the tick labels (x in data units, y in axes fraction)
from matplotlib.transforms import blended_transform_factory
tf = blended_transform_factory(a.transData, a.transAxes)
a.text((nchina - 1) / 2, -0.30, "Chinese cohorts\n(as development)", ha="center", va="top",
       fontsize=8, transform=tf)
a.text((nchina + len(rows) - 1) / 2, -0.30, "non-Chinese cohorts", ha="center", va="top",
       fontsize=8, transform=tf)
# The title states only what was measured. Cohort origin is confounded with grading protocol,
# camera and image quality, and the separation is complete for DINOv2 only.
a.set_title("(b) External AUC varies more across cohorts than between backbones\n"
            f"Chinese vs non-Chinese mean: DINOv2 {grp['dino']['china_mean']:.3f} vs "
            f"{grp['dino']['other_mean']:.3f}, RETFound {grp['ret']['china_mean']:.3f} vs "
            f"{grp['ret']['other_mean']:.3f}", fontsize=9)

# (c) seed stability
a = ax[2]
a.bar(x - 0.2, [r["dino_seed_sd"] for r in rows], 0.4, color=C["dinov2_l"], label="DINOv2")
a.bar(x + 0.2, [r["ret_seed_sd"] for r in rows], 0.4, color=C["retfound"], label="RETFound")
a.set_xticks(x); a.set_xticklabels([r["key"].upper() for r in rows], fontsize=9, rotation=20,
                                   ha="right")
a.set_ylabel("SD of AUC across %d seeds" % SEEDS); a.legend(fontsize=9)
a.set_title("(c) Reproducibility\nlower = more stable across training seeds", fontsize=10)

fig.tight_layout()
fn = os.path.join(OUT, "Figure_external_meta.png")
fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close()
print(f"\nsaved {fn}")
