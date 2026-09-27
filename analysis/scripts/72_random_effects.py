# -*- coding: utf-8 -*-
"""
72_random_effects.py - how much of external performance is cohort, how much is backbone?

WHY: Section 2.3 pooled the per-cohort DINOv2-minus-RETFound AUC difference with fixed-effect
inverse-variance weights although I^2 was 82%, and set the 0.140 gap between cohort groups beside
that pooled difference as if the two were comparable. All three blind reviewers asked for (i) a
random-effects pooled estimate with a prediction interval and (ii) a decomposition of the
variation in external AUC into cohort, backbone, cohort-by-backbone and seed, with and without
the one RETFound run that collapsed. This script provides both from the cached predictions.

  * DerSimonian-Laird random effects on the per-cohort ensemble differences (bootstrap SEs
    from scripts/40), with a 95% prediction interval (t, k-2 df).
  * Variance components from the balanced two-way layout of per-seed AUCs (7 cohorts x
    2 backbones x 5 seeds), cohort random and backbone fixed (expected mean squares); a
    percentile interval for each share by resampling cohorts (2,000 replicates).
  * The same with the collapsed seed index removed for both backbones (balanced, 4 seeds).

USAGE   python scripts/72_random_effects.py
OUTPUT  results/external2/random_effects.json
"""
import os, sys, json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(ROOT, "results", "external2")
COH = ["ddr", "jsiec", "deepdrid", "aptos", "idrid", "messidor2", "eyepacs"]
KINDS = ["dinov2_l", "retfound"]
SEEDS = 5


def pred(k, s, c):
    z = np.load(os.path.join(OUT, "pred_%s_seed%d_%s.npz" % (k, s, c)))
    return z["ys"].astype(int), z["pos"]


def dl(d, se):
    """DerSimonian-Laird random-effects pooling with a prediction interval."""
    d, v = np.asarray(d), np.asarray(se) ** 2
    w = 1 / v; fe = (w * d).sum() / w.sum()
    Q = float((w * (d - fe) ** 2).sum()); k = len(d)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (v + tau2); re = float((ws * d).sum() / ws.sum()); se_re = float(np.sqrt(1 / ws.sum()))
    t = stats.t.ppf(0.975, k - 2); half = t * np.sqrt(tau2 + se_re ** 2)
    return dict(fixed=round(float(fe), 4), random=round(re, 4),
                ci95=[round(re - 1.96 * se_re, 4), round(re + 1.96 * se_re, 4)],
                prediction_interval=[round(re - half, 4), round(re + half, 4)],
                tau2=round(tau2, 5), I2=round(max(0.0, (Q - (k - 1)) / Q) * 100, 1) if Q > 0 else 0.0,
                k=k)


def boot_delta(y, p1, p2, n=4000, seed=0):        # as scripts/40
    rng = np.random.RandomState(seed); idx = np.arange(len(y)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        d.append(roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p2[b]))
    return float(np.mean(d)), float(np.std(d))


# per-seed AUC table A[cohort, backbone, seed]
A = np.zeros((len(COH), 2, SEEDS))
for i, c in enumerate(COH):
    for j, k in enumerate(KINDS):
        for s in range(SEEDS):
            y, p = pred(k, s, c); A[i, j, s] = roc_auc_score(y, p)
collapsed = int(np.argmin(A[:, 1, :].mean(0)))       # RETFound seed with the lowest mean AUC
print("collapsed RETFound seed:", collapsed, " mean AUC per RETFound seed:", np.round(A[:, 1, :].mean(0), 3))


def pooled(seeds_ret):
    d, se = [], []
    for c in COH:
        y, p1 = pred("dinov2_l", 0, c)
        p1 = np.mean([pred("dinov2_l", s, c)[1] for s in range(SEEDS)], 0)
        p2 = np.mean([pred("retfound", s, c)[1] for s in seeds_ret], 0)
        m, s_ = boot_delta(y, p1, p2); d.append(m); se.append(max(s_, 1e-6))
    return dict(per_cohort=[round(x, 4) for x in d], **dl(d, se))


def components(X):
    """Balanced cohort (random) x backbone (fixed) x replicate layout; EMS estimators."""
    a, b, n = X.shape
    gm = X.mean(); mc = X.mean((1, 2)); mb = X.mean((0, 2)); mcb = X.mean(2)
    SSc = b * n * ((mc - gm) ** 2).sum(); SSb = a * n * ((mb - gm) ** 2).sum()
    SScb = n * ((mcb - mc[:, None] - mb[None, :] + gm) ** 2).sum(); SSe = ((X - mcb[:, :, None]) ** 2).sum()
    MSc, MSb, MScb, MSe = SSc / (a - 1), SSb / (b - 1), SScb / ((a - 1) * (b - 1)), SSe / (a * b * (n - 1))
    comp = {"cohort": max(0.0, (MSc - MSe) / (b * n)),          # cohort random, main effect
            "backbone": max(0.0, (MSb - MScb) / (a * n)),       # fixed effect, as a variance
            "cohort_x_backbone": max(0.0, (MScb - MSe) / n),
            "seed_residual": MSe}
    tot = sum(comp.values())
    return {k: v / tot for k, v in comp.items()}, comp


def with_ci(X, reps=2000, seed=0):
    share, comp = components(X)
    rng = np.random.RandomState(seed); boot = {k: [] for k in share}
    for _ in range(reps):
        idx = rng.choice(X.shape[0], X.shape[0], replace=True)
        s, _ = components(X[idx])
        for k in s:
            boot[k].append(s[k])
    return {k: {"share": round(share[k], 3),
                "ci95": [round(float(np.percentile(boot[k], 2.5)), 3), round(float(np.percentile(boot[k], 97.5)), 3)],
                "variance": round(comp[k], 6)} for k in share}


keep = [s for s in range(SEEDS) if s != collapsed]
out = {"collapsed_retfound_seed": collapsed,
       "pooled_delta_auc_all_seeds": pooled(range(SEEDS)),
       "pooled_delta_auc_without_collapsed_run": pooled(keep),
       "variance_components_all_seeds": with_ci(A),
       "variance_components_without_collapsed_seed_index": with_ci(A[:, :, keep]),
       "retfound_seed_sd_without_collapsed": {c: round(float(A[i, 1, keep].std(ddof=1)), 3) for i, c in enumerate(COH)},
       "dinov2_seed_sd": {c: round(float(A[i, 0, :].std(ddof=1)), 3) for i, c in enumerate(COH)},
       "logit_scale_components_all_seeds": {k: round(v, 3) for k, v in components(np.log(A / (1 - A)))[0].items()}}
json.dump(out, open(os.path.join(OUT, "random_effects.json"), "w"), indent=2)
for k in ("pooled_delta_auc_all_seeds", "pooled_delta_auc_without_collapsed_run"):
    r = out[k]; print("%-42s fixed %+.3f | random %+.3f [%+.3f, %+.3f] | PI [%+.3f, %+.3f] | I2 %.0f%%"
                      % (k, r["fixed"], r["random"], *r["ci95"], *r["prediction_interval"], r["I2"]))
for k in ("variance_components_all_seeds", "variance_components_without_collapsed_seed_index"):
    print(k); [print("   %-18s %.3f  [%.3f, %.3f]" % (n, v["share"], *v["ci95"])) for n, v in out[k].items()]
print("logit scale:", out["logit_scale_components_all_seeds"])
print("RETFound seed SD without collapsed run:", out["retfound_seed_sd_without_collapsed"])
print("DINOv2 seed SD:", out["dinov2_seed_sd"])
