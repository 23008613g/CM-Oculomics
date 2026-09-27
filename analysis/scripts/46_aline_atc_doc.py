# -*- coding: utf-8 -*-
"""
46_aline_atc_doc.py — run the actual methods the OOD-performance-prediction literature uses.

WHY: 44 correlated raw agreement and confidence with external AUC. That is not what the
literature does. The published estimators anchor on in-distribution behaviour and produce a
NUMBER: an estimate of the model's accuracy on the unlabelled target cohort. This script
implements three of them and reports how far the estimates are from the truth.

  ALine-D   agreement-on-the-line, difference form (Baek et al., NeurIPS 2022)
            acc_ood_hat = acc_id - (agreement_id - agreement_ood)
  ALine-S   same, with the drop rescaled by the slope fitted across models
  DoC       difference of confidences (Guillory et al., ICCV 2021)
            acc_ood_hat = acc_id - (conf_id - conf_ood)
  ATC       average thresholded confidence (Garg et al., ICLR 2022)
            pick t so that the ID fraction above t equals acc_id; estimate = OOD fraction above t

The in-distribution anchor is the held-out in-house test split (140 images, 11 PDR), so ID
balanced accuracy rests on 11 positives. That fragility is part of the finding and is reported.

USAGE  python scripts/46_aline_atc_doc.py
OUTPUT results/external2/aline_atc_doc.json, Figure_estimators.png

2026-09-27: figure axis now contains every estimate (one DoC estimate exceeds 1.0), panels are
lettered (a)-(d), and the decision threshold (0.5) is stated on the axis.
"""
import os, sys, json, itertools, importlib.util
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(ROOT, "results", "external2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES = 224
SEEDS = 5   # raised from 3: seeds 3-4 were added for both backbones so that
            # the one degenerate RETFound run does not carry 1/3 of the weight
KINDS = ["dinov2_l", "retfound"]
NAME = {"dinov2_l": "DINOv2", "retfound": "RETFound"}
C = {"dinov2_l": "#D55E00", "retfound": "#0072B2"}
SETS = [("ddr", "DDR (China)"), ("jsiec", "JSIEC (China)"), ("deepdrid", "DeepDRiD (China)"),
        ("aptos", "APTOS (India)"), ("idrid", "IDRiD (India)"),
        ("messidor2", "Messidor-2 (France)"), ("eyepacs", "EyePACS (USA)")]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EXP = _load(os.path.join(ROOT, "scripts", "27_dino_experiment.py"), "exp27")

# 27_dino_experiment.py builds DataLoaders with num_workers=4; because it is imported
# dynamically, Windows spawn workers cannot unpickle its Dataset class. Same fix as script 38.
_ORIG_DATALOADER = EXP.DataLoader


def _dataloader_no_workers(*a, **kw):
    kw["num_workers"] = 0
    return _ORIG_DATALOADER(*a, **kw)


EXP.DataLoader = _dataloader_no_workers


# ------------------------------------------------------------------ in-distribution anchor
def id_probs():
    """Predictions of every checkpoint on the held-out in-house test split, cached."""
    fn = os.path.join(OUT, "pred_ID_test.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return z["ys"], {k: z[k] for k in z.files if k != "ys"}
    sp = pd.read_csv(os.path.join(ROOT, "data_anon", "splits.csv")).dropna(subset=["grade"])
    te = sp[sp.split == "test"].copy()
    ds = EXP.DS(te, False, RES)
    ys = np.array([EXP.GRADE_CLASSES.index(g) for g in te["grade"]])
    out = {}
    for kind in KINDS:
        for s in range(SEEDS):
            ck = os.path.join(OUT, f"ckpt_{kind}_seed{s}.pth")
            model = EXP.make_model(kind, RES).to(DEVICE)
            model.load_state_dict(torch.load(ck, map_location=DEVICE))
            out[f"{kind}_seed{s}"] = EXP.infer(model, ds, RES)
            print(f"[ID] {kind}_seed{s} scored on {len(ds)} held-out images", flush=True)
            del model
            torch.cuda.empty_cache()
    np.savez(fn, ys=ys, **out)
    return ys, out


def conf(p):
    return np.maximum(p, 1 - p)


def pred(p):
    return (p > 0.5).astype(int)


# ------------------------------------------------------------------ estimators
EPS = 1e-6


def probit(p):
    """Inverse standard-normal CDF, the scale on which accuracy-on-the-line is linear."""
    from scipy.stats import norm
    return norm.ppf(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))


def inv_probit(z):
    from scipy.stats import norm
    return norm.cdf(z)


def atc_threshold(c_id, acc_id):
    """t such that the ID fraction of confidences above t equals ID accuracy."""
    return float(np.quantile(c_id, max(0.0, min(1.0, 1.0 - acc_id))))


def main():
    y_id, P_id = id_probs()
    models = [f"{k}_seed{s}" for k in KINDS for s in range(SEEDS)]

    # ID quantities. Balanced accuracy, because the in-house test split is 129:11.
    acc_id = {m: balanced_accuracy_score(y_id, pred(P_id[m])) for m in models}
    conf_id = {m: float(conf(P_id[m]).mean()) for m in models}
    agr_id = {(a, b): float((pred(P_id[a]) == pred(P_id[b])).mean())
              for a, b in itertools.combinations(models, 2)}
    print("\nIN-DISTRIBUTION ANCHOR (held-out in-house test, n=%d, %d PDR)"
          % (len(y_id), int(y_id.sum())))
    for m in models:
        print(f"  {m:<18} balanced acc {acc_id[m]:.3f}   mean confidence {conf_id[m]:.3f}")

    rows = []
    for s, label in SETS:
        Po = {m: np.load(os.path.join(OUT, f"pred_{m}_{s}.npz"))["pos"] for m in models}
        y_o = np.load(os.path.join(OUT, f"pred_{models[0]}_{s}.npz"))["ys"]
        acc_o = {m: balanced_accuracy_score(y_o, pred(Po[m])) for m in models}
        conf_o = {m: float(conf(Po[m]).mean()) for m in models}
        agr_o = {(a, b): float((pred(Po[a]) == pred(Po[b])).mean())
                 for a, b in itertools.combinations(models, 2)}

        # ALine-S. Agreement-on-the-line (Baek et al., NeurIPS 2022) states that the line
        # relating in-distribution to out-of-distribution AGREEMENT has the same slope and
        # intercept as the line relating in-distribution to out-of-distribution ACCURACY.
        # The slope is therefore fitted on agreements alone, in probit space, and then
        # applied to each model's in-distribution accuracy. An earlier version of this
        # script fitted the slope on acc_o, the target cohort's TRUE accuracy, which needs
        # target labels and so is not a label-free estimator at all; that is fixed here.
        x_agr = np.array([agr_id[p] for p in agr_id])
        y_agr = np.array([agr_o[p] for p in agr_id])
        ok = np.isfinite(x_agr) & np.isfinite(y_agr)
        if ok.sum() >= 2:
            px, py = probit(x_agr[ok]), probit(y_agr[ok])
            if np.std(px) > 1e-9:
                a_s, b_s = np.polyfit(px, py, 1)
            else:
                a_s, b_s = np.nan, np.nan
        else:
            a_s, b_s = np.nan, np.nan

        for m in models:
            pairs = [p for p in agr_id if m in p]
            d_agr = float(np.mean([agr_id[p] - agr_o[p] for p in pairs]))
            est = {
                "ALine-D": acc_id[m] - d_agr,
                "ALine-S": (float(inv_probit(a_s * probit(acc_id[m]) + b_s))
                            if np.isfinite(a_s) and np.isfinite(b_s) else np.nan),
                "DoC": acc_id[m] - (conf_id[m] - conf_o[m]),
                "ATC": float((conf(Po[m]) > atc_threshold(conf(P_id[m]), acc_id[m])).mean()),
            }
            rows.append(dict(cohort=s, label=label, model=m, backbone=m.rsplit("_seed", 1)[0],
                             true=acc_o[m], **est))

    df = pd.DataFrame(rows)
    METH = ["ALine-D", "ALine-S", "DoC", "ATC"]
    print("\n" + "=" * 78)
    print("ESTIMATED versus TRUE balanced accuracy on each external cohort")
    print("=" * 78)
    print(f"{'cohort':<22}{'true':>8}" + "".join(f"{m:>10}" for m in METH))
    for s, label in SETS:
        g = df[df.cohort == s]
        print(f"{label:<22}{g['true'].mean():>8.3f}"
              + "".join(f"{g[m].mean():>10.3f}" for m in METH))
    print("-" * 78)
    summary = {}
    for m in METH:
        mae = float((df[m] - df["true"]).abs().mean())
        bias = float((df[m] - df["true"]).mean())
        worst = float((df[m] - df["true"]).abs().max())
        summary[m] = dict(mae=mae, bias=bias, worst=worst)
        print(f"{m:<22} MAE {mae:.3f}   bias {bias:+.3f}   worst-case error {worst:.3f}")
    print("\nfor scale: true balanced accuracy ranges %.3f to %.3f across cohorts"
          % (df['true'].min(), df['true'].max()))
    naive = float((df["true"].mean() - df["true"]).abs().mean())
    print("a constant predictor (always guess the mean) would have MAE %.3f" % naive)
    summary["_constant_baseline_mae"] = naive

    json.dump(dict(rows=rows, summary=summary,
                   id_anchor=dict(n=int(len(y_id)), n_pos=int(y_id.sum()),
                                  balanced_acc={m: acc_id[m] for m in models})),
              open(os.path.join(OUT, "aline_atc_doc.json"), "w"), indent=2)

    # ------------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.6), sharey=True, sharex=True)
    # The axis must contain every estimate: DoC is unbounded and one estimate exceeds 1.0.
    # A fixed upper limit of 1.02 once hid such a point; now it is shown, above a dotted line
    # marking the largest attainable balanced accuracy.
    lo = 0.35
    hi = max(1.02, float(np.nanmax(df[METH].values)) + 0.03)
    for a, m, tag in zip(ax, METH, "abcd"):
        for kind in KINDS:
            g = df[df.backbone == kind]
            a.scatter(g["true"], g[m], s=26, facecolor=C[kind], edgecolor="k",
                      linewidth=0.4, alpha=0.8, label=NAME[kind], zorder=3)
        a.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        a.axhline(1.0, color="#888", ls=":", lw=1)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_xlabel("true balanced accuracy (threshold 0.5)")
        n_out = int((df[m] > 1.0).sum())
        a.set_title(f"({tag}) {m}\nMAE {summary[m]['mae']:.3f}, bias {summary[m]['bias']:+.3f}, "
                    f"worst {summary[m]['worst']:.3f}"
                    + (f"\n{n_out} estimate{'s' if n_out > 1 else ''} > 1 (impossible)" if n_out else ""),
                    fontsize=9.5)
    ax[0].set_ylabel("estimated balanced accuracy")
    ax[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Published unsupervised performance estimators applied to the seven cohorts: "
                 "estimates fall off the identity line", fontsize=12, y=1.02)
    fig.tight_layout()
    fn = os.path.join(OUT, "Figure_estimators.png")
    fig.savefig(fn, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {fn}")


if __name__ == "__main__":
    main()
