# -*- coding: utf-8 -*-
"""
77_same_task_estimators.py - the four published estimators, and the no-shift baseline, in the
same-task control of scripts/76.

WHY: see scripts/76. This script applies exactly the estimators of scripts/46 (ALine-D, ALine-S
with the slope fitted on agreements alone, difference of confidences, average thresholded
confidence) and the no-shift baseline of scripts/75, to models trained and anchored on the SAME
task (PDR versus NPDR, DDR), in two target settings:
  * "balanced": the class-balanced public subsets of the paper (prior shift from the anchor's
    ~15% to 50%, no task change);
  * "natural": the full APTOS, EyePACS, Messidor-2 and IDRiD cohorts at their own prevalence
    (neither task change nor designed prior shift).
It prints the paper's setting (task change + prior shift, scripts/46 and 75) alongside, so the
three settings can be compared on one scale.

USAGE   python scripts/77_same_task_estimators.py
OUTPUT  results/same_task/estimators.json, Figure_same_task.png
"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
OUT = os.environ.get("SAME_TASK_DIR", os.path.join(ROOT, "results", "same_task"))   # override: self-test, or the released public_outputs/same_task
KINDS, SEEDS = ["dinov2_l", "retfound"], range(5)
MODELS = ["%s_seed%d" % (k, s) for k in KINDS for s in SEEDS]
SETTINGS = {"balanced": (os.environ["SAME_TASK_BAL"].split(",") if os.environ.get("SAME_TASK_BAL")   # self-test only
                         else ["jsiec", "deepdrid", "aptos", "idrid", "messidor2", "eyepacs"]),
            "natural": ["aptos", "eyepacs", "messidor2", "idrid"]}
METH = ["ALine-D", "ALine-S", "DoC", "ATC", "No-shift"]
EPS = 1e-6


def load(m, s):
    z = np.load(os.path.join(OUT, "pred_%s_%s.npz" % (m, s)))
    return z["ys"].astype(int), z["pos"]


conf = lambda p: np.maximum(p, 1 - p)
pred = lambda p: (p > 0.5).astype(int)
probit = lambda p: norm.ppf(np.clip(np.asarray(p, float), EPS, 1 - EPS))


def atc_threshold(c_id, acc_id):
    return float(np.quantile(c_id, max(0.0, min(1.0, 1.0 - acc_id))))


missing = [(m, s) for m in MODELS for s in ["anchor"] + ["bal_" + c for c in SETTINGS["balanced"]] + ["nat_" + c for c in SETTINGS["natural"]]
           if not os.path.exists(os.path.join(OUT, "pred_%s_%s.npz" % (m, s)))]
if missing:
    raise SystemExit("%d prediction files missing, e.g. %s; is scripts/76 finished?" % (len(missing), missing[:3]))

y_id = load(MODELS[0], "anchor")[0]
P_id = {m: load(m, "anchor")[1] for m in MODELS}
acc_id = {m: balanced_accuracy_score(y_id, pred(P_id[m])) for m in MODELS}
conf_id = {m: float(conf(P_id[m]).mean()) for m in MODELS}
agr_id = {(a, b): float((pred(P_id[a]) == pred(P_id[b])).mean()) for a, b in itertools.combinations(MODELS, 2)}
out = {"anchor": {"n": int(len(y_id)), "n_pdr": int(y_id.sum()),
                  "auc_per_model": {m: round(float(roc_auc_score(y_id, P_id[m])), 3) for m in MODELS},
                  "balanced_acc": {m: round(acc_id[m], 3) for m in MODELS}}, "settings": {}}
print("anchor: n=%d, %d PDR; balanced accuracy %s" % (len(y_id), y_id.sum(),
      " ".join("%.2f" % acc_id[m] for m in MODELS)))

rows_all = {}
for setting, cohorts in SETTINGS.items():
    pre = "bal_" if setting == "balanced" else "nat_"
    rows, per_cohort = [], {}
    for c in cohorts:
        y_o = load(MODELS[0], pre + c)[0]
        Po = {m: load(m, pre + c)[1] for m in MODELS}
        acc_o = {m: balanced_accuracy_score(y_o, pred(Po[m])) for m in MODELS}
        conf_o = {m: float(conf(Po[m]).mean()) for m in MODELS}
        agr_o = {(a, b): float((pred(Po[a]) == pred(Po[b])).mean()) for a, b in itertools.combinations(MODELS, 2)}
        x = probit([agr_id[p] for p in agr_id]); yv = probit([agr_o[p] for p in agr_id])
        a_s, b_s = np.polyfit(x, yv, 1) if np.std(x) > 1e-9 else (np.nan, np.nan)
        ens = {k: np.mean([Po["%s_seed%d" % (k, s)] for s in SEEDS], 0) for k in KINDS}
        per_cohort[c] = {"n": int(len(y_o)), "prevalence": round(float(y_o.mean()), 3),
                         "ensemble_auc": {k: round(float(roc_auc_score(y_o, ens[k])), 3) for k in KINDS}}
        for m in MODELS:
            pairs = [p for p in agr_id if m in p]
            est = {"ALine-D": acc_id[m] - float(np.mean([agr_id[p] - agr_o[p] for p in pairs])),
                   "ALine-S": float(norm.cdf(a_s * probit(acc_id[m]) + b_s)) if np.isfinite(a_s) else np.nan,
                   "DoC": acc_id[m] - (conf_id[m] - conf_o[m]),
                   "ATC": float((conf(Po[m]) > atc_threshold(conf(P_id[m]), acc_id[m])).mean()),
                   "No-shift": acc_id[m]}
            rows.append(dict(cohort=c, model=m, true=acc_o[m], **est))
    df = pd.DataFrame(rows); rows_all[setting] = df
    summ = {}
    for mth in METH:
        e = df[mth] - df.true
        rho = stats.spearmanr(df[mth], df.true)[0]
        summ[mth] = {"mae": round(float(e.abs().mean()), 3), "bias": round(float(e.mean()), 3),
                     "worst": round(float(e.abs().max()), 3), "spearman": round(float(rho), 2)}
    summ["Mean observed (uses labels)"] = {"mae": round(float((df.true.mean() - df.true).abs().mean()), 3)}
    for c in cohorts:                                  # per-cohort means over the ten models
        g = df[df.cohort == c]
        per_cohort[c]["mean_true"] = round(float(g.true.mean()), 3)
        per_cohort[c]["mean_estimate"] = {mth: round(float(g[mth].mean()), 3) for mth in METH}
    out["settings"][setting] = {"cohorts": per_cohort, "n_pairs": int(len(df)), "summary": summ,
                                "true_range": [round(float(df.true.min()), 3), round(float(df.true.max()), 3)]}

# mechanism: sensitivity and specificity at 0.5 (range over the ten models), and the two label-free
# inputs of the estimators (mean confidence, mean pairwise agreement over the 45 model pairs)
mech = {}
for s in ["anchor"] + ["bal_" + c for c in SETTINGS["balanced"]] + ["nat_" + c for c in SETTINGS["natural"]]:
    y = load(MODELS[0], s)[0]; P = {m: load(m, s)[1] for m in MODELS}
    se = [float(pred(P[m])[y == 1].mean()) for m in MODELS]
    sp = [float(1 - pred(P[m])[y == 0].mean()) for m in MODELS]
    mech[s] = {"sensitivity_range": [round(min(se), 3), round(max(se), 3)],
               "specificity_range": [round(min(sp), 3), round(max(sp), 3)],
               "mean_confidence": round(float(np.mean([conf(P[m]).mean() for m in MODELS])), 3),
               "mean_pairwise_agreement": round(float(np.mean([(pred(P[a]) == pred(P[b])).mean()
                                                               for a, b in itertools.combinations(MODELS, 2)])), 3)}
out["mechanism_at_0.5"] = mech
paper = json.load(open(os.path.join(ROOT, "results", "external2", "estimator_baselines.json")))["methods"]
out["paper_setting_task_change_plus_prior_shift"] = paper
json.dump(out, open(os.path.join(OUT, "estimators.json"), "w"), indent=2)
# one row per model, cohort and setting (Data S1, sheet same_task_control)
pd.concat([df.assign(setting=st) for st, df in rows_all.items()])[["setting", "cohort", "model", "true"] + METH] \
    .rename(columns={"true": "balanced_accuracy"}).round(4) \
    .to_csv(os.path.join(OUT, "same_task_rows.csv"), index=False)

print("\n%-28s %-24s %-24s %-24s" % ("MAE / bias / Spearman", "paper (task+prior shift)", "same task, balanced", "same task, natural"))
for mth in METH:
    p = paper[mth if mth != "No-shift" else "No-shift"]
    cells = ["%.3f / %+.3f / %+.2f" % (p["mae"], p["bias"], p["spearman_with_true"] or 0)]
    for st in ("balanced", "natural"):
        s = out["settings"][st]["summary"][mth]
        cells.append("%.3f / %+.3f / %+.2f" % (s["mae"], s["bias"], s["spearman"]))
    print("%-28s %-24s %-24s %-24s" % (mth, *cells))
for st in ("balanced", "natural"):
    print("%s: constant (uses labels) MAE %.3f; true balanced accuracy %s" %
          (st, out["settings"][st]["summary"]["Mean observed (uses labels)"]["mae"], out["settings"][st]["true_range"]))
    for c, v in out["settings"][st]["cohorts"].items():
        print("   %-10s n=%5d prev %.3f  ensemble AUC DINOv2 %.3f RETFound %.3f  mean true BA %.3f  estimates %s"
              % (c, v["n"], v["prevalence"], v["ensemble_auc"]["dinov2_l"], v["ensemble_auc"]["retfound"],
                 v["mean_true"], " ".join("%s %.3f" % (k, e) for k, e in v["mean_estimate"].items())))

# figure: estimated versus true, two settings x five methods; sized so that it stays legible after
# the package's downscaling to 1200 px on the long side (scripts/65)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 5, figsize=(12.5, 5.9), sharex=True, sharey=True)
col = {"dinov2_l": "#D55E00", "retfound": "#0072B2"}
for i, st in enumerate(("balanced", "natural")):
    df = rows_all[st]
    for j, mth in enumerate(METH):
        a = ax[i, j]
        for k in KINDS:
            g = df[df.model.str.startswith(k)]
            a.scatter(g.true, g[mth], s=14, color=col[k], edgecolor="k", lw=0.25, alpha=0.8,
                      label={"dinov2_l": "DINOv2", "retfound": "RETFound"}[k])
        a.plot([0.3, 1.05], [0.3, 1.05], "k--", lw=0.9); a.set_xlim(0.4, 1.02); a.set_ylim(0.4, 1.02)
        a.set_xticks([0.4, 0.6, 0.8, 1.0]); a.set_yticks([0.4, 0.6, 0.8, 1.0])
        s = out["settings"][st]["summary"][mth]
        a.set_title("(%s) %s\nMAE %.3f, bias %+.3f, ρ %+.2f" % ("abcdefghij"[i * 5 + j], mth, s["mae"], s["bias"], s["spearman"]),
                    fontsize=8.5)
        if j == 0:
            a.set_ylabel(("Class-balanced targets" if st == "balanced" else
                          "Natural prevalence") + "\nestimated balanced accuracy", fontsize=8.5)
        if i == 1:
            a.set_xlabel("achieved balanced accuracy", fontsize=8.5)
ax[0, 0].legend(fontsize=7.5, loc="lower right", handletextpad=0.2, borderaxespad=0.2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Figure_same_task.png"), dpi=300, bbox_inches="tight")
print("saved %s" % os.path.join(OUT, "estimators.json"))
