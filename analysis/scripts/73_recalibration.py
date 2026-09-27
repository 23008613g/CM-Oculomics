# -*- coding: utf-8 -*-
"""
73_recalibration.py - recalibrate the risk scores, then redo the decision curve.

WHY: the model was trained with inverse-frequency class weights, which is equivalent to
training under a 50% class prior. Its outputs therefore overstate absolute risk (mean predicted
0.218 against an observed 0.088; scripts/66), and the decision curve on those outputs was
negative at the 0.62 operating point. A decision curve only means something on probabilities
that are calibrated in the large. Two recalibrations are compared, neither of which uses the
predictions it is applied to:
  * prior correction, closed form, no fitting: logit(p) + log(pi / (1 - pi)), with pi the
    intolerant fraction of the training folds of the model that produced the prediction;
  * cross-fitted logistic recalibration: for fold k, y ~ logit(p) fitted on the out-of-fold
    predictions of the other four folds, applied to fold k.
Both are monotone, so discrimination is unchanged. The prior correction is then applied, with
the prevalence of the models' training folds, to the five-seed ensemble on the held-out test
split, which played no part in any fitting.

Decision-curve thresholds of 5-20% are reported: a range around the 8.8% base rate in which a
clinician might act (for example, arrange closer review). The range was chosen for this revision,
not before the original analysis.

USAGE   python scripts/73_recalibration.py
OUTPUT  results/recalibration.json, results/dino_experiment/oof_recalibrated.csv (local only)
"""
import os, sys, json
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join(ROOT, "results"); DE = os.path.join(R, "dino_experiment")
EPS = 1e-6
TH = [0.05, 0.10, 0.15, 0.20]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def calib(y, p):
    lp = logit(p); n = len(y)
    off = sm.GLM(y, np.ones(n), family=sm.families.Binomial(), offset=lp).fit()
    sl = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean()) for b in range(10) if (bins == b).any())
    return {"mean_predicted": round(float(p.mean()), 3), "observed": round(float(y.mean()), 3),
            "O_over_E": round(float(y.mean() / p.mean()), 2),
            "intercept": round(float(off.params[0]), 2), "intercept_ci95": [round(float(x), 2) for x in off.conf_int()[0]],
            "slope": round(float(sl.params[1]), 2), "slope_ci95": [round(float(x), 2) for x in sl.conf_int()[1]],
            "ece": round(float(ece), 3), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "auc": round(float(roc_auc_score(y, p)), 4)}


def nb(y, p, t):
    n = len(y); tp = ((p >= t) & (y == 1)).sum(); fp = ((p >= t) & (y == 0)).sum()
    return float(tp / n - fp / n * t / (1 - t))


def dca(y, p):
    grid = np.round(np.arange(0.01, 0.60, 0.01), 2)
    prev = y.mean()
    rows = [(t, nb(y, p, t), prev - (1 - prev) * t / (1 - t)) for t in grid]
    better = [t for t, m, a in rows if m > max(0.0, a) + 1e-12]
    return {"at": {str(t): {"model": round(nb(y, p, t), 4),
                            "treat_all": round(float(prev - (1 - prev) * t / (1 - t)), 4),
                            "sens": round(float((p[y == 1] >= t).mean()), 3),
                            "spec": round(float((p[y == 0] < t).mean()), 3)} for t in TH},
            "range_above_both_defaults": [min(better), max(better)] if better else None,
            "curve": [[float(t), round(m, 5), round(a, 5)] for t, m, a in rows]}


d = pd.read_csv(os.path.join(DE, "oof_dinov2_l_res224_frac1.0_main.csv"))
y, p, fold = d.y.values.astype(int), d.prob.values, d.fold.values
p_prior, p_log = np.zeros_like(p), np.zeros_like(p)
for k in np.unique(fold):
    tr, te = fold != k, fold == k
    pi = y[tr].mean()                                   # training prevalence of fold k's model
    p_prior[te] = expit(logit(p[te]) + np.log(pi / (1 - pi)))
    m = sm.GLM(y[tr], sm.add_constant(logit(p[tr])), family=sm.families.Binomial()).fit()
    p_log[te] = m.predict(sm.add_constant(logit(p[te]), has_constant="add"))
d.assign(prob_prior=p_prior, prob_logistic=p_log).to_csv(os.path.join(DE, "oof_recalibrated.csv"), index=False)

out = {"cross_validation": {"n": int(len(y)), "prevalence": round(float(y.mean()), 4)}}
for name, q in (("original", p), ("prior_corrected", p_prior), ("logistic_crossfit", p_log)):
    out["cross_validation"][name] = {"calibration": calib(y, q), "decision_curve": dca(y, q)}

# held-out test split, five-seed ensemble, prior-corrected with the training-fold prevalence
sp = pd.read_csv(os.path.join(ROOT, "data_anon", "splits.csv"))
tv = sp[sp.split == "train_val"]; tr14 = tv[tv.fold != 0]
pi14 = float((tr14.grade == "PDR").mean())
z = np.load(os.path.join(R, "external2", "pred_ID_test.npz")); yt = z["ys"].astype(int)
ens = np.mean([z["dinov2_l_seed%d" % s] for s in range(5)], 0)
ens_c = expit(logit(ens) + np.log(pi14 / (1 - pi14)))
out["held_out_test"] = {"n": int(len(yt)), "n_intolerant": int(yt.sum()), "training_prevalence": round(pi14, 4),
                        "original": {"calibration": calib(yt, ens), "decision_curve": dca(yt, ens)},
                        "prior_corrected": {"calibration": calib(yt, ens_c), "decision_curve": dca(yt, ens_c)}}
json.dump(out, open(os.path.join(R, "recalibration.json"), "w"), indent=2)

for part in ("cross_validation", "held_out_test"):
    print("==", part)
    for name, v in out[part].items():
        if not isinstance(v, dict) or "calibration" not in v:
            continue
        c, dc = v["calibration"], v["decision_curve"]
        print("  %-18s mean %.3f vs obs %.3f  O/E %.2f  int %+.2f %s  slope %.2f  ECE %.3f  Brier %.4f  NB>defaults %s"
              % (name, c["mean_predicted"], c["observed"], c["O_over_E"], c["intercept"], c["intercept_ci95"],
                 c["slope"], c["ece"], c["brier"], dc["range_above_both_defaults"]))
        print("      " + "  ".join("t=%s NB %+.4f (all %+.4f) se %.2f sp %.2f" % (t, a["model"], a["treat_all"], a["sens"], a["spec"])
                                   for t, a in dc["at"].items()))
