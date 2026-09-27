# -*- coding: utf-8 -*-
"""
32_figures_dinov2.py — regenerate the DINOv2-primary data figures (v33) into paper2/.
Overwrites only data figures (Fig3,4,5,7,8,9,10,11,13); leaves nanobanana Fig1/Fig12 and
backbone-independent Fig2/Fig6 untouched. Reads ONLY DINOv2 artifacts + handcrafted-feature
files (backbone-independent). Okabe-Ito palette, 300 DPI, editable-text where relevant.
"""
import os, sys, json, glob
import numpy as np, pandas as pd
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
                             confusion_matrix, balanced_accuracy_score)
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); DATA = os.path.join(ROOT, "data_anon")
DE = os.path.join(R, "dino_experiment"); OUTF = os.path.join(R, "figures", "paper2")
GCAM = os.path.join(R, "figures", "gradcam_dino")
os.makedirs(OUTF, exist_ok=True)
plt.rcParams.update({"font.size": 16, "axes.titlesize": 14, "axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 12, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlepad": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
                     "svg.fonttype": "none"})
C = {"main": "#0072B2", "accent": "#D55E00", "ok": "#009E73", "grey": "#999999",
     "purple": "#CC79A7", "orange": "#E69F00", "sky": "#56B4E9", "yellow": "#F0E442"}
def plabel(ax, l, x=-0.085, y=1.20): ax.text(x, y, l, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")

# ---- data ----
OOF = pd.read_csv(os.path.join(DE, "oof_dinov2_l_res224_frac1.0_main.csv")); OOF["y"] = OOF.y.astype(int)
EXT = np.load(os.path.join(DE, "ddr_dinov2_l_res224.npz"))
EXT_RET = np.load(os.path.join(R, "external_valpred.npz"))
META = pd.read_csv(os.path.join(DATA, "manifest.csv"))[["anon_image", "tcm_syndrome"]]
PTS = pd.read_csv(os.path.join(DATA, "patients_anon.csv"), encoding="utf-8-sig")
BIO = pd.read_csv(os.path.join(R, "imaging_biomarkers.csv"))
MC = pd.read_csv(os.path.join(R, "method_comparison.csv"))
BT = json.load(open(os.path.join(DE, "backbone_table_postMask.json"), encoding="utf-8"))
# Fig 4a bars come from BT. The CI printed on Fig 3a comes from MET (scripts/56, stratified
# bootstrap), the same file the manuscript text quotes; BT (scripts/53) uses a different,
# unstratified bootstrap and gave 0.922-0.975 against the text's 0.921-0.976.
MET = json.load(open(os.path.join(DE, "metrics_dinov2.json"), encoding="utf-8"))
def _json_any(fn):   # scripts/57 wrote this file in the console code page (GBK) on Windows
    for enc in ("utf-8", "gbk"):
        try:
            return json.load(open(fn, encoding=enc))
        except UnicodeDecodeError:
            continue
    raise
S26 = _json_any(os.path.join(R, "section26_numbers.json"))
BB_ORDER = ["DINOv2 (primary)", "RETFound", "Swin-V2", "ViT-B", "ResNet-50"]


def youden(y, p): fpr, tpr, t = roc_curve(y, p); return t[np.argmax(tpr - fpr)]


# ============ Fig 3 — performance ============
def fig3():
    fig = plt.figure(figsize=(15.5, 4.8)); gs = GridSpec(1, 4, figure=fig, wspace=0.40)
    y, p = OOF.y.values, OOF.prob.values
    # a ROC per fold + mean
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "a")
    for k, g in OOF.groupby("fold"):
        fpr, tpr, _ = roc_curve(g.y, g.prob); ax.plot(fpr, tpr, color=C["sky"], lw=0.9, alpha=0.5)
    yy0, pp0 = OOF.y.values, OOF.prob.values
    fpr, tpr, _ = roc_curve(yy0, pp0)
    ax.plot(fpr, tpr, color=C["main"], lw=2.6, label=f"Pooled AUC {roc_auc_score(yy0,pp0):.3f}\n(95% CI {MET['per_image']['CI95']['AUC'][0]:.3f}–{MET['per_image']['CI95']['AUC'][1]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity"); ax.set_title("ROC (pooled out-of-fold)"); ax.legend(loc="lower right")
    # b PR per fold
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "b")
    for k, g in OOF.groupby("fold"):
        pr, rc, _ = precision_recall_curve(g.y, g.prob); ax.plot(rc, pr, color=C["ok"], lw=0.9, alpha=0.6)
    pr, rc, _ = precision_recall_curve(y, p)
    ax.plot(rc, pr, color=C["accent"], lw=2.4, label=f"PR-AUC {average_precision_score(y,p):.3f}")
    ax.axhline(y.mean(), ls="--", color=C["grey"], label=f"Prevalence {y.mean():.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–recall")
    # the upper half of a PR panel is occupied by the per-fold curves, so the legend goes
    # bottom-left, which is empty for every fold
    ax.legend(loc="lower left", framealpha=0.95, borderpad=0.6)
    # c confusion at pooled Youden
    ax = fig.add_subplot(gs[0, 2]); plabel(ax, "c")
    thr = youden(y, p); cm = confusion_matrix(y, (p >= thr).astype(int))
    im = ax.imshow(cm, cmap="Blues"); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Tol.", "Intol."]); ax.set_yticklabels(["Tol.", "Intol."])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion (τ={thr:.2f})"); ax.grid(False)
    # d sens/spec vs threshold
    ax = fig.add_subplot(gs[0, 3]); plabel(ax, "d")
    ts = np.linspace(0.01, 0.99, 99); sens = [((p >= t) & (y == 1)).sum() / (y == 1).sum() for t in ts]
    spec = [((p < t) & (y == 0)).sum() / (y == 0).sum() for t in ts]
    ax.plot(ts, sens, color=C["accent"], lw=2, label="Sensitivity"); ax.plot(ts, spec, color=C["main"], lw=2, label="Specificity")
    ax.axvline(thr, ls="--", color=C["ok"], label=f"Youden τ={thr:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Value"); ax.set_title("Operating point"); ax.legend(loc="lower center")
    fig.savefig(os.path.join(OUTF, "Figure3_performance.png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Fig3 done")


# ============ Fig 4 — comparison + external (DINOv2 vs RETFound) ============
def fig4():
    fig = plt.figure(figsize=(15.5, 4.9)); gs = GridSpec(1, 3, figure=fig, wspace=0.42)
    # a backbone bars — POOLED out-of-fold AUC + bootstrap 95% CI (one convention for all 5; v34)
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "a")
    names = ["DINOv2 (primary)", "RETFound", "Swin-V2", "ViT-B", "ResNet-50"]
    # first training run of each backbone: one convention for all five. The two
    # foundation models also have three repeats, whose mean is reported in the text.
    aucs = [BT[n]["image_aucs"][0] for n in BB_ORDER]
    cilo = [BT[n]["run1_image_ci95"][0] for n in BB_ORDER]
    cihi = [BT[n]["run1_image_ci95"][1] for n in BB_ORDER]
    yerr = [[a - l for a, l in zip(aucs, cilo)], [h - a for a, h in zip(aucs, cihi)]]
    cols = [C["accent"], C["main"], C["sky"], C["sky"], C["grey"]]
    bars = ax.bar(range(5), aucs, yerr=yerr, color=cols, edgecolor="k", capsize=4)
    for b, a, h in zip(bars, aucs, cihi): ax.text(b.get_x() + b.get_width() / 2, h + 0.006, f"{a:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_xticklabels(names, fontsize=10.5, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.82, 1.0)
    # The DeLong note used to sit in a box over the top of the axis and hid the y label;
    # it is now the second line of the title.
    _dl = BT["delong_dinov2_vs_retfound_per_repeat"]
    ax.set_ylabel("AUC (pooled OOF, 95% CI)")
    ax.set_title("Backbone comparison (first run of each)\nDINOv2 vs RETFound: DeLong p = %.2f–%.2f "
                 "over %d repeats" % (min(x["p"] for x in _dl), max(x["p"] for x in _dl), len(_dl)),
                 fontsize=10.5)
    # b deep vs traditional
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "b")
    # best traditional classifier per feature group, read from method_comparison.csv
    _mc = MC[MC.model != "RETFound (ours)"]
    _pick = {"Clinical only": "Clinical only", "Imaging biomarkers": "Imaging biomarkers",
             "Biomarkers + clinical": "Biomarkers + Clinical"}
    best = {}
    for _lab, _pref in _pick.items():
        _g = _mc[_mc.features.str.startswith(_pref)]
        best[_lab] = float(_g.AUC_mean.max())
    best["Deep (DINOv2)"] = float(roc_auc_score(OOF.y, OOF.prob))
    cc = [C["grey"], C["sky"], C["orange"], C["accent"]]
    bars = ax.bar(range(4), list(best.values()), color=cc, edgecolor="k")
    for b, v in zip(bars, best.values()): ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(list(best.keys()), fontsize=10.5, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC"); ax.set_title("Deep vs handcrafted")
    # c external ROC: DINOv2 vs RETFound
    ax = fig.add_subplot(gs[0, 2]); plabel(ax, "c")
    # five-seed mean-probability ensembles, the same quantity Section 2.3 and Fig. 11
    # report; single deploy models would let one unlucky RETFound run set the gap
    def _ens(kind):
        zs = [np.load(os.path.join(R, "external2", "pred_%s_seed%d_ddr.npz" % (kind, k)))
              for k in range(5)]
        return zs[0]["ys"].astype(int), np.mean([z["pos"] for z in zs], 0)
    yd, pd_ = _ens("dinov2_l"); yr, pr_ = _ens("retfound")
    f1, t1, _ = roc_curve(yd, pd_); f2, t2, _ = roc_curve(yr, pr_)
    ax.plot(f1, t1, color=C["accent"], lw=2.6, label=f"DINOv2  {roc_auc_score(yd,pd_):.3f}")
    ax.plot(f2, t2, color=C["main"], lw=2.2, ls="-", label=f"RETFound  {roc_auc_score(yr,pr_):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("External validation (DDR)"); ax.legend(loc="lower right", title="AUC")
    _dl = json.load(open(os.path.join(R, "external2", "external2_summary.json")))
    _p = _dl["ensemble"]["ddr"]["delong_p"]
    ax.annotate("DeLong p = %.0e" % _p, xy=(0.30, 0.42), fontsize=11,
                color=C["accent"], fontweight="bold")
    fig.savefig(os.path.join(OUTF, "Figure4_comparison.png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Fig4 done")


# ============ Fig 5 — Grad-CAM grid + biomarker violins ============
def fig5():
    order = ["PDR", "NPDR"]; style = {"PDR": ("INTOLERANT (PDR)", "#c23a36"), "NPDR": ("TOLERANT (NPDR)", "#15936a")}
    RET = os.path.join(R, "figures", "gradcam")  # RETFound CAMs (scripts/05), same eyes
    # Explicit one-eye-per-patient selection: 7 DISTINCT patients per group, every shown eye
    # classified correctly by BOTH backbones. Previously the PDR block reused P0274_OS twice
    # (_01 and _02 = same eye of the same patient); the unused 8th test PDR patient (P0504) is
    # excluded because DINOv2 misclassifies it, so it cannot appear under the "correct for both"
    # premise. NPDR trimmed to 7 to keep the two blocks symmetric.
    # 2026-09-27: the eyes are no longer hand-picked. The old list put six VISUCAM images in the
    # intolerant row and seven Topcon images in the tolerant row, so the rows differed by camera.
    # scripts/70 selects them by rule (held-out test, VISUCAM only, one per patient, correct by
    # both models, equal numbers drawn at random) and draws their tiles.
    _sel = json.load(open(os.path.join(R, "fig4a_eyes.json"), encoding="utf-8"))["eyes"]
    eyes = {g: ["%s_%s" % (g, e["image"]) for e in _sel[g]] for g in ("PDR", "NPDR")}
    ncol = max(len(eyes["PDR"]), len(eyes["NPDR"])); rowlabs = ["Original", "RETFound\nGrad-CAM", "DINOv2\nGrad-CAM"]
    def srcpaths(g, e):  # e = e.g. 'PDR_P0030_OD_01.jpg'
        return [os.path.join(GCAM, f"orig_{e}.png"), os.path.join(RET, f"cam_{e}.png"), os.path.join(GCAM, f"cam_{e}.png")]
    fig = plt.figure(figsize=(2.0 * ncol, 18.5))
    gs = GridSpec(10, ncol, figure=fig, height_ratios=[0.16, 1, 1, 1, 0.16, 1, 1, 1, 0.55, 2.4], hspace=0.10, wspace=0.05)
    rb = {"PDR": 0, "NPDR": 4}  # group header-band row index
    for g in order:
        lab, col = style[g]; cax = fig.add_subplot(gs[rb[g], :]); cax.axis("off")
        cax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=cax.transAxes, color=col, alpha=0.92))
        cax.text(0.5, 0.5, lab, transform=cax.transAxes, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        for j in range(ncol):
            paths = srcpaths(g, eyes[g][j]) if j < len(eyes[g]) else [None, None, None]
            for r in range(3):
                ax = fig.add_subplot(gs[rb[g] + 1 + r, j])
                if paths[r] and os.path.exists(paths[r]): ax.imshow(Image.open(paths[r]))
                ax.set_xticks([]); ax.set_yticks([])
                if j < len(eyes[g]):
                    for s in ax.spines.values(): s.set_visible(True); s.set_color(col); s.set_linewidth(2)
                if j == 0: ax.set_ylabel(rowlabs[r], fontsize=10, fontweight="bold")
    # biomarker violins (backbone-independent) — own 1x4 sub-grid so the image grid can be 7 cols
    vgs = gs[9, :].subgridspec(1, 4, wspace=0.55)
    bm = BIO.copy(); bm["yb"] = (bm.label == "PDR").astype(int)
    feats = ["vessel_skeleton", "vessel_density", "fractal_dim", "dark_lesion_area"]
    titles = ["Skeleton length", "Vessel density", "Fractal dimension", "Dark-lesion area"]
    for i, (f, t) in enumerate(zip(feats, titles)):
        ax = fig.add_subplot(vgs[0, i])
        g0 = bm[bm.yb == 0][f].dropna().values; g1 = bm[bm.yb == 1][f].dropna().values
        parts = ax.violinplot([g0, g1], showextrema=False)
        for pc, cc in zip(parts["bodies"], [C["main"], C["accent"]]): pc.set_facecolor(cc); pc.set_alpha(0.45)
        for xi, (gg, cc) in enumerate(zip([g0, g1], [C["main"], C["accent"]]), 1):
            ax.scatter(np.random.normal(xi, 0.05, len(gg)), gg, s=5, color=cc, alpha=0.25, zorder=2)
            ax.plot(xi, gg.mean(), "^", color="k", ms=8, zorder=4)
        u, pmw = stats.mannwhitneyu(g0, g1); sub = bm.dropna(subset=[f]); auc = roc_auc_score(sub.yb, sub[f])
        auc = max(auc, 1 - auc)
        lo, hi = np.percentile(np.concatenate([g0, g1]), [1, 99]); ax.set_ylim(lo, hi)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Tol.", "Intol."]); ax.set_title(t, fontsize=12, pad=24)
        ax.text(0.5, 1.02, f"p={pmw:.1e}, AUC={auc:.2f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
        plabel(ax, "bcde"[i], x=-0.18, y=1.18)   # legend refers to panels b-e
    fig.suptitle("Grad-CAM, RETFound vs DINOv2, on the same held-out VISUCAM eyes\n(warm colors drive the prediction)", fontsize=12.5, y=0.912)
    fig.text(0.085, 0.902, "a", fontsize=18, fontweight="bold")
    fig.savefig(os.path.join(OUTF, "Figure5_interpretability.png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Fig5 done")


# ============ Fig 7 — trustworthy (selective + calibration) ============
def ece_points(p, y, n=10):
    bins = np.linspace(0, 1, n + 1); idx = np.clip(np.digitize(p, bins) - 1, 0, n - 1); xs, ys = [], []; e = 0
    for b in range(n):
        m = idx == b
        if m.sum(): xs.append(p[m].mean()); ys.append((y[m] == 1).mean()); e += m.sum() / len(y) * abs((y[m] == 1).mean() - p[m].mean())
    return xs, ys, e

def fig7():
    cur = np.load(os.path.join(DE, "dino_uncertainty_curve.npz"))
    fig = plt.figure(figsize=(11, 4.6)); gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "a")
    ax.plot(cur["coverage"] * 100, cur["bacc"], "o-", color=C["ok"], lw=2)
    # The curve is computed on the held-out test split (scripts/31), not on the pooled OOF
    # predictions, so it is labelled as such; at each coverage the number of intolerant eyes
    # kept is printed, because at 8% prevalence the most certain half holds only a few.
    _ys, _sd = cur["ys"], cur["std_p"]; _o = np.argsort(_sd)
    for cv, ba in zip(cur["coverage"], cur["bacc"]):
        _k = int(_ys[_o[:max(1, int(len(_o) * cv))]].sum())
        ax.annotate(f"{_k}", (cv * 100, ba), textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=8, color="#555")
    ax.set_xlabel("Coverage % (most-certain cases kept)"); ax.set_ylabel("Balanced accuracy")
    ax.set_title("Selective prediction (MC-dropout)\nheld-out test, n = %d, %d intolerant; "
                 "numbers = intolerant kept" % (len(_ys), int(_ys.sum())), fontsize=10)
    ax.set_ylim(0.5, 1.05)
    # calibration on pooled OOF before/after temperature
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "b")
    y, p = OOF.y.values, np.clip(OOF.prob.values, 1e-6, 1 - 1e-6)
    # T is the NLL-fitted temperature reported in the text (scripts/57), not a literal
    z = np.log(p / (1 - p)); T = S26["calibration"]["temperature_fitted_by_nll"]; pa = 1 / (1 + np.exp(-z / T))
    xb, yb, eb = ece_points(p, y); xa, ya, ea = ece_points(pa, y)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(xb, yb, "o-", color=C["accent"], label=f"Before (ECE={eb:.3f})")
    ax.plot(xa, ya, "s-", color=C["ok"], label=f"After T={T:.2f} (ECE={ea:.3f})")
    # 2026-09-27: prior correction for the class-weighted training (scripts/73), which fixes the
    # calibration-in-the-large that temperature scaling cannot
    _rc = pd.read_csv(os.path.join(DE, "oof_recalibrated.csv"))
    xr, yr, er = ece_points(_rc.prob_prior.values, _rc.y.values)
    ax.plot(xr, yr, "D-", color=C["main"], label=f"Prior-corrected (ECE={er:.3f})")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency"); ax.set_title("Calibration (pooled OOF)"); ax.legend(loc="upper left")
    fig.savefig(os.path.join(OUTF, "Figure7_trustworthy.png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Fig7 done")


# ============ Fig 8 — clinical ============
def fig8():
    import statsmodels.api as sm
    fig = plt.figure(figsize=(16.5, 11.4)); gs = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.42)
    # data
    d = OOF.merge(BIO, on="anon_image", how="left").merge(META, on="anon_image", how="left")
    clin = ["年龄", "糖尿病年限（年）", "BMI", "HbA1c", "LDL", "HDL", "Triglyceride"]
    for c in clin: PTS[c] = pd.to_numeric(PTS[c], errors="coerce")
    d = d.merge(PTS[["anon_id", "性别"] + clin], on="anon_id", how="left")
    # a multivariable OR (handcrafted; backbone-independent)
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "a")
    bio_feats = ["vessel_density", "vessel_skeleton", "fractal_dim", "dark_lesion_area", "bright_lesion_area"]
    feats = bio_feats + clin; sub = d.dropna(subset=feats + ["y"]).copy()
    X = (sub[feats] - sub[feats].mean()) / sub[feats].std(); m = sm.Logit(sub["y"], sm.add_constant(X)).fit(disp=0)
    OR = np.exp(m.params); ci = np.exp(m.conf_int()); tab = pd.DataFrame({"f": OR.index, "OR": OR.values, "lo": ci[0].values, "hi": ci[1].values, "p": m.pvalues.values})
    tab = tab[tab.f != "const"].sort_values("p").iloc[::-1]
    nm = {"vessel_skeleton": "Vessel skeleton", "vessel_density": "Vessel density", "fractal_dim": "Fractal dim",
          "dark_lesion_area": "Dark lesion", "bright_lesion_area": "Bright lesion", "年龄": "Age", "糖尿病年限（年）": "DM duration",
          "BMI": "BMI", "HbA1c": "HbA1c", "LDL": "LDL", "HDL": "HDL", "Triglyceride": "Triglyceride"}
    yy = range(len(tab))
    ax.errorbar(tab.OR, yy, xerr=[tab.OR - tab.lo, tab.hi - tab.OR], fmt="o", color=C["main"], capsize=3)
    ax.axvline(1, ls="--", color=C["accent"]); ax.set_yticks(list(yy)); ax.set_yticklabels([nm.get(f, f) for f in tab.f], fontsize=12)
    ax.set_xscale("log"); ax.set_xlabel("Adjusted OR (per 1 SD, 95% CI)"); ax.set_title("Multivariable logistic regression")
    # b DCA
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "b")
    y = d.dropna(subset=["prob"]).y.values; p = d.dropna(subset=["prob"]).prob.values; n = len(y)
    ts = np.linspace(0.01, 0.6, 60); nbm = []; nba = []
    for pt in ts:
        tp = ((p >= pt) & (y == 1)).sum(); fp = ((p >= pt) & (y == 0)).sum(); nbm.append(tp / n - fp / n * (pt / (1 - pt)))
        nba.append((y == 1).sum() / n - (y == 0).sum() / n * (pt / (1 - pt)))
    # 2026-09-27: the same decision curve on prior-corrected probabilities (scripts/73)
    _rc = pd.read_csv(os.path.join(DE, "oof_recalibrated.csv")); yc, pc = _rc.y.values, _rc.prob_prior.values
    nbc = [((pc >= pt) & (yc == 1)).sum() / n - ((pc >= pt) & (yc == 0)).sum() / n * (pt / (1 - pt)) for pt in ts]
    ax.plot(ts, nbm, color=C["ok"], lw=2, ls="--", label="Model, uncalibrated")
    ax.plot(ts, nbc, color=C["main"], lw=2.2, label="Model, prior-corrected")
    ax.plot(ts, nba, color=C["grey"], lw=1.5, label="Treat all")
    ax.axhline(0, color="k", lw=1, label="Treat none"); ax.set_ylim(-0.05, y.mean() + 0.05)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.set_title("Decision curve analysis"); ax.legend()
    # c risk stratification
    ax = fig.add_subplot(gs[1, 0]); plabel(ax, "c")
    dd = d.dropna(subset=["prob"]).copy(); dd["risk"] = pd.cut(dd.prob, [-.01, 0.2, 0.5, 1.01], labels=["Low", "Medium", "High"])
    rate = dd.groupby("risk").y.mean() * 100; cnt = dd.groupby("risk").y.size()
    bars = ax.bar(rate.index.astype(str), rate.values, color=[C["ok"], C["purple"], C["accent"]], edgecolor="k")
    for b, r, c in zip(bars, rate.values, cnt.values): ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.1f}%\n(n={c})", ha="center", fontsize=12)
    ax.set_ylabel("Observed intolerance rate (%)"); ax.set_title("Risk stratification")
    # d subgroup
    ax = fig.add_subplot(gs[1, 1]); plabel(ax, "d")
    dd["sex"] = dd["性别"].map({"男": "Male", "女": "Female"}); dd["age"] = np.where(dd["年龄"] >= 65, "Age≥65", "Age<65")
    dd["dur"] = np.where(dd["糖尿病年限（年）"] >= 7, "Dur≥7y", "Dur<7y")
    syn = {"痰瘀阻滞证": "Phlegm-stasis", "阴虚夹瘀证": "Yin-def+stasis", "气阴两虚证": "Qi-Yin-def", "脾肾两虚证": "Spleen-Kidney-def"}
    groups = [("Overall", dd)]
    for col in ["sex", "age", "dur"]:
        for v in sorted(dd[col].dropna().unique()): groups.append((v, dd[dd[col] == v]))
    for s in sorted(dd.tcm_syndrome.dropna().unique()):
        sub2 = dd[dd.tcm_syndrome == s]
        if sub2.y.nunique() == 2: groups.append((syn.get(s, s), sub2))
    pairs = [(f"{nm2} (n={len(g)})", roc_auc_score(g.y, g.prob)) for nm2, g in groups if g.y.nunique() == 2 and len(g) >= 20]
    pairs.sort(key=lambda x: x[1]); names = [a for a, _ in pairs]; av = [b for _, b in pairs]
    for i, a in enumerate(av):
        col = C["accent"] if a == max(av) else C["main"]
        ax.hlines(i, 0.5, a, color=col, lw=3); ax.plot(a, i, "o", color=col, ms=10, mec="white", mew=1.2)
        ax.text(a + 0.006, i, f"{a:.3f}", va="center", fontsize=11, fontweight="bold", color=col)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=11); ax.set_xlim(0.5, 1.04)
    ax.axvline(0.5, ls="--", color=C["accent"], alpha=0.5); ax.set_xlabel("AUC"); ax.set_title("Subgroup performance"); ax.grid(axis="y", alpha=0)
    fig.savefig(os.path.join(OUTF, "Figure8_clinical.png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Fig8 done")


# ============ Fig 9 — screening / speed / bio-corr ============
def fig9():
    fig = plt.figure(figsize=(16, 5.4)); gs = GridSpec(1, 3, figure=fig, wspace=0.36, top=0.84)
    y, p = OOF.y.values, OOF.prob.values; thr = youden(y, p); pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp); ppv, npv = tp / (tp + fp), tn / (tn + fn)
    brier = np.mean((p - y) ** 2)
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "a")
    vals = [sens, spec, ppv, npv]; bars = ax.bar(["Sens", "Spec", "PPV", "NPV"], vals, color=[C["accent"], C["main"], C["purple"], C["ok"]], edgecolor="k")
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Value"); ax.set_title(f"At the Youden point\n(Brier = {brier:.3f}, threshold-free)", pad=10)
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "b")
    ax.bar(["CFP + DINOv2\n(this work)"], [18.7 / 1000], color=C["ok"], edgecolor="k", width=0.5)
    ax.text(0, 18.7 / 1000 + 0.001, "0.019 s/image", ha="center", fontweight="bold")
    ax.set_ylabel("Inference time per image (s)"); ax.set_title("Inference speed", pad=10); ax.set_ylim(0, 0.05)
    ax = fig.add_subplot(gs[0, 2]); plabel(ax, "c")
    g = OOF.groupby("anon_id").agg(prob=("prob", "mean")).reset_index()
    clin = ["HbA1c", "LDL", "HDL", "Triglyceride", "BMI", "糖尿病年限（年）", "年龄"]
    for c in clin: PTS[c] = pd.to_numeric(PTS[c], errors="coerce")
    mm = g.merge(PTS[["anon_id"] + clin], on="anon_id", how="left")
    en = {"糖尿病年限（年）": "DM duration", "年龄": "Age"}; out = []
    for c in clin:
        s = mm.dropna(subset=[c, "prob"]); r, pv = stats.spearmanr(s.prob, s[c]); out.append((en.get(c, c), r, pv))
    out.sort(key=lambda x: abs(x[1])); feats = [o[0] for o in out]; rs = [o[1] for o in out]
    cols = [C["accent"] if o[2] < 0.05 else C["grey"] for o in out]
    ax.barh(range(len(feats)), rs, color=cols, edgecolor="k"); ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=12)
    ax.axvline(0, color="k", lw=1); ax.set_xlabel("Spearman r (risk vs indicator)"); ax.set_title("Prediction vs labs")
    fig.savefig(os.path.join(OUTF, "Figure9_screening.png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Fig9 done")


# ============ Fig 10 — robustness (label-eff overlay + quality) ============
def fig10():
    fig = plt.figure(figsize=(15.5, 4.8)); gs = GridSpec(1, 4, figure=fig, wspace=0.40)
    # a label efficiency DINOv2 vs RETFound
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "a")
    fr = [10, 25, 50, 100]
    def _le(kind, res):
        out = []
        for _f in ("0.1", "0.25", "0.5", "1.0"):
            _d = pd.read_csv(os.path.join(DE, "oof_%s_res%d_frac%s_labeleff.csv" % (kind, res, _f)))
            out.append(roc_auc_score(_d.y, _d.prob))
        return out
    dino = _le("dinov2_l", 224); ret = _le("retfound", 224)
    ax.plot(fr, dino, "o-", color=C["accent"], lw=2, label="DINOv2"); ax.plot(fr, ret, "s--", color=C["main"], lw=2, label="RETFound")
    ax.set_xlabel("Training labels used (%)"); ax.set_ylabel("Pooled AUC"); ax.set_title("Label efficiency"); ax.legend(loc="lower right"); ax.set_ylim(0.78, 0.98)
    # b quality score distribution
    q = pd.read_csv(os.path.join(R, "image_quality.csv"))
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "b")
    ax.hist(q.quality_score, bins=30, color=C["sky"], edgecolor="k", alpha=0.8)
    lo, hi = q.quality_score.quantile([0.33, 0.66]); ax.axvline(lo, ls="--", color=C["grey"]); ax.axvline(hi, ls="--", color=C["grey"])
    ax.set_xlabel("Composite quality score"); ax.set_ylabel("Images"); ax.set_title("Quality distribution")
    # c AUC by tier
    ax = fig.add_subplot(gs[0, 2]); plabel(ax, "c")
    d = OOF.merge(q[["anon_image", "quality_tier"]], on="anon_image")
    tiers = ["Low", "Medium", "High"]; ta = [roc_auc_score(d[d.quality_tier == t].y, d[d.quality_tier == t].prob) for t in tiers]
    bars = ax.bar(tiers, ta, color=[C["grey"], C["sky"], C["ok"]], edgecolor="k")
    for b, v in zip(bars, ta): ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    # tier sizes and positives, so the reader can judge how much each bar rests on
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"{t}\nn={int((d.quality_tier == t).sum())}\n{int(d[d.quality_tier == t].y.sum())} PDR"
                        for t in tiers], fontsize=10)
    ax.set_ylim(0.7, 1.0); ax.set_ylabel("AUC"); ax.set_title("AUC by tier")
    # d exclusion robustness
    ax = fig.add_subplot(gs[0, 3]); plabel(ax, "d")
    dq = OOF.merge(q[["anon_image", "quality_score"]], on="anon_image"); excl = [0, 10, 20, 30]; av = []
    for e in excl:
        thr = dq.quality_score.quantile(e / 100); s = dq[dq.quality_score >= thr]; av.append(roc_auc_score(s.y, s.prob))
    ax.plot(excl, av, "o-", color=C["accent"], lw=2); ax.set_xlabel("Images excluded (%)"); ax.set_ylabel("AUC")
    ax.set_title("Excluding lowest-quality images"); ax.set_ylim(0.9, 0.98)
    fig.savefig(os.path.join(OUTF, "Figure10_robustness.png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Fig10 done")


# ============ Fig 11 — rigour & fairness ============
def fig11():
    fig = plt.figure(figsize=(15, 10.4)); gs = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.32)
    rng = np.random.default_rng(42)
    # a deep vs strongest handcrafted
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "a")
    d = OOF.merge(BIO[["anon_image", "vessel_skeleton"]], on="anon_image"); skel = np.zeros(len(d))
    for k in range(5):
        tr, te = d[d.fold != k], d[d.fold == k]; sc = StandardScaler().fit(tr[["vessel_skeleton"]])
        mdl = LogisticRegression(max_iter=1000).fit(sc.transform(tr[["vessel_skeleton"]]), tr.y)
        skel[(d.fold == k).values] = mdl.predict_proba(sc.transform(te[["vessel_skeleton"]]))[:, 1]
    a_deep = roc_auc_score(d.y, d.prob); a_skel = roc_auc_score(d.y, skel)
    diffs = []
    for _ in range(2000):
        b = rng.choice(len(d), len(d), True)
        if len(np.unique(d.y.values[b])) < 2: continue
        diffs.append(roc_auc_score(d.y.values[b], d.prob.values[b]) - roc_auc_score(d.y.values[b], skel[b]))
    dlo, dhi = np.percentile(diffs, [2.5, 97.5]); dd = a_deep - a_skel
    bars = ax.bar(["Deep\n(DINOv2)", "Skeleton\n(LR)"], [a_deep, a_skel], color=[C["accent"], C["grey"]], edgecolor="k")
    for b, v in zip(bars, [a_deep, a_skel]): ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC"); ax.set_title(f"Deep vs best single feature\nΔAUC={dd:.3f} [{dlo:.3f},{dhi:.3f}], p<0.001")
    global FIG11A
    FIG11A = (a_deep, a_skel, dd, dlo, dhi)
    # b internal vs external
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "b")
    yi, pi = OOF.y.values, OOF.prob.values; ye, pe = EXT["ys"].astype(int), EXT["pos"]
    def bci(y, p):
        a = roc_auc_score(y, p); bs = [roc_auc_score(y[b], p[b]) for b in (rng.choice(len(y), len(y), True) for _ in range(2000)) if len(np.unique(y[b])) > 1]
        return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    ai, lo_i, hi_i = bci(yi, pi); ae, lo_e, hi_e = bci(ye, pe)
    ax.bar([0, 1], [ai, ae], yerr=[[ai - lo_i, ae - lo_e], [hi_i - ai, hi_e - ae]], color=[C["main"], C["accent"]], edgecolor="k", capsize=5)
    for x, v in zip([0, 1], [ai, ae]): ax.text(x, v + 0.03, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Internal\n(pooled OOF)", "External\n(DDR)"]); ax.set_ylim(0.7, 1.02)
    ax.axhline(0.5, ls="--", color=C["grey"]); ax.set_ylabel("AUC"); ax.set_title("Internal vs external (95% CI)")
    # c subgroup AUC + CI
    ax = fig.add_subplot(gs[1, 0]); plabel(ax, "c")
    d2 = OOF.merge(META, on="anon_image", how="left").merge(PTS[["anon_id", "性别", "年龄"]], on="anon_id", how="left")
    d2["sex"] = d2["性别"].map({"男": "Male", "女": "Female"}); d2["age"] = np.where(pd.to_numeric(d2["年龄"], errors="coerce") >= 65, "Age≥65", "Age<65")
    syn = {"痰瘀阻滞证": "Phlegm-stasis", "阴虚夹瘀证": "Yin-def+stasis", "气阴两虚证": "Qi-Yin-def", "脾肾两虚证": "Spleen-Kidney-def"}
    groups = [("Overall", d2)]
    for col in ["sex", "age"]:
        for v in sorted(d2[col].dropna().unique()): groups.append((v, d2[d2[col] == v]))
    for s in sorted(d2.tcm_syndrome.dropna().unique()):
        sub = d2[d2.tcm_syndrome == s]
        if sub.y.nunique() == 2: groups.append((syn.get(s, s), sub))
    rows = []
    for nm2, g in groups:
        if g.y.nunique() == 2 and len(g) >= 20: rows.append((nm2,) + bci(g.y.values, g.prob.values))
    rows.sort(key=lambda x: x[1]); yy = range(len(rows))
    for i, (nm2, a, lo, hi) in zip(yy, rows):
        ax.plot(a, i, "o", color=C["main"], ms=9); ax.hlines(i, lo, hi, color=C["main"], lw=2.5)
    ax.set_yticks(list(yy)); ax.set_yticklabels([r[0] for r in rows], fontsize=11); ax.axvline(0.5, ls="--", color=C["grey"])
    ax.set_xlim(0.5, 1.03); ax.set_xlabel("AUC (95% CI)"); ax.set_title("Subgroup fairness"); ax.grid(axis="y", alpha=0)
    # d operating points
    ax = fig.add_subplot(gs[1, 1]); plabel(ax, "d")
    ts = np.linspace(0.05, 0.95, 50); sens = []; spec = []; npv = []
    for t in ts:
        pred = (pi >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yi, pred, labels=[0, 1]).ravel()
        sens.append(tp / (tp + fn) if (tp + fn) else np.nan); spec.append(tn / (tn + fp) if (tn + fp) else np.nan)
        npv.append(tn / (tn + fn) if (tn + fn) else np.nan)
    ax.plot(ts, sens, color=C["accent"], lw=2, label="Sensitivity"); ax.plot(ts, spec, color=C["main"], lw=2, label="Specificity")
    ax.plot(ts, npv, color=C["ok"], lw=2, label="NPV"); ax.axvline(youden(yi, pi), ls="--", color=C["grey"], label="Youden")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Value"); ax.set_title("Operating-point analysis")
    # sensitivity falls into the lower right, so the legend goes bottom-left, where the
    # lowest curve (specificity) has not yet risen
    ax.legend(loc="lower left", fontsize=11, framealpha=0.95, borderpad=0.6)
    fig.savefig(os.path.join(OUTF, "Figure11_rigour.png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Fig11 done | Fig11a:", FIG11A)


# ============ Fig 13 — feature space ============
def fig13():
    emb = np.load(os.path.join(DE, "dino_tsne_emb.npz")); X, y = emb["X"], emb["y"]
    fig = plt.figure(figsize=(12, 5.2)); gs = GridSpec(1, 2, figure=fig, wspace=0.28, width_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "a", x=-0.1, y=1.06)
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X)
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=14, c=C["main"], alpha=0.6, label="Tolerant (NPDR)", edgecolors="none")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=22, c=C["accent"], alpha=0.85, label="Intolerant (PDR)", marker="^", edgecolors="none")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_title("DINOv2 feature space\n(tolerant vs intolerant)"); ax.legend(loc="best"); ax.grid(alpha=0.15)
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "b", x=-0.1, y=1.06)
    d = BIO.merge(OOF[["anon_image", "prob"]], on="anon_image", how="inner")
    cols = ["vessel_skeleton", "vessel_density", "fractal_dim", "dark_lesion_area", "bright_lesion_area", "contrast", "prob"]
    nice = ["Skeleton", "Density", "Fractal", "Dark-lesion", "Bright-lesion", "Contrast", "Risk score"]
    corr = d[cols].corr(method="spearman").values; im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(nice))); ax.set_xticklabels(nice, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(len(nice))); ax.set_yticklabels(nice, fontsize=12)
    for i in range(len(nice)):
        for j in range(len(nice)): ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=10, color="white" if abs(corr[i, j]) > 0.5 else "#333")
    ax.set_title("Spearman correlation:\nbiomarkers & risk score"); ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ρ")
    fig.savefig(os.path.join(OUTF, "Figure13_featurespace.png"), dpi=300, bbox_inches="tight"); plt.close()
    print("Fig13 done")


# ============ Fig 14 — Grad-CAM consistency: RETFound vs DINOv2 on the SAME eyes ============
def fig14():
    RET = os.path.join(R, "figures", "gradcam")        # RETFound CAMs (scripts/05)
    pdr = ["P0030_OD_01.jpg", "P0144_OD_01.jpg", "P0287_OD_01.jpg", "P0433_OD_01.jpg"]
    npdr = ["P0011_OD_01.jpg", "P0083_OD_01.jpg", "P0122_OD_01.jpg", "P0194_OD_01.jpg"]
    cols = [("PDR", e) for e in pdr] + [("NPDR", e) for e in npdr]
    rows = [("Original", lambda g, e: os.path.join(GCAM, f"orig_{g}_{e}.png")),
            ("RETFound\nGrad-CAM", lambda g, e: os.path.join(RET, f"cam_{g}_{e}.png")),
            ("DINOv2\nGrad-CAM", lambda g, e: os.path.join(GCAM, f"cam_{g}_{e}.png"))]
    bcol = {"PDR": "#c23a36", "NPDR": "#15936a"}
    fig = plt.figure(figsize=(2.0 * len(cols), 6.6))
    gs = GridSpec(4, len(cols), figure=fig, height_ratios=[0.20, 1, 1, 1], hspace=0.06, wspace=0.05)
    # header bands
    for span, lab, g in [(range(0, 4), "INTOLERANT (PDR)", "PDR"), (range(4, 8), "TOLERANT (NPDR)", "NPDR")]:
        cax = fig.add_subplot(gs[0, span.start:span.stop]); cax.axis("off")
        cax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=cax.transAxes, color=bcol[g], alpha=0.92))
        cax.text(0.5, 0.5, lab, transform=cax.transAxes, ha="center", va="center", color="white", fontsize=13, fontweight="bold")
    for ri, (rlab, fn) in enumerate(rows, start=1):
        for ci, (g, e) in enumerate(cols):
            ax = fig.add_subplot(gs[ri, ci]); p = fn(g, e)
            if os.path.exists(p): ax.imshow(Image.open(p))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(True); s.set_color(bcol[g]); s.set_linewidth(2)
            if ci == 0: ax.set_ylabel(rlab, fontsize=11, fontweight="bold")
    fig.suptitle("Grad-CAM consistency between backbones: RETFound and DINOv2 attend to the same lesion/vascular regions",
                 fontsize=12.5, y=0.97)
    fig.savefig(os.path.join(OUTF, "Figure14_gradcam_compare.png"), dpi=300, bbox_inches="tight"); plt.close(); print("Fig14 done")


# ============ Fig 2 — cohort + task landscape (backbone-independent except intolerance AUC) ============
def fig2():
    fig = plt.figure(figsize=(11, 3.8)); gs = GridSpec(1, 2, figure=fig, wspace=0.85)
    ax = fig.add_subplot(gs[0, 0]); plabel(ax, "a", x=-0.12, y=1.06)
    labs = ["Tolerant (NPDR)", "Intolerant (PDR)"]; cnt = [893, 95]; cols = [C["main"], C["accent"]]
    for i, (vv, col) in enumerate(zip(cnt, cols)):
        ax.hlines(i, 0, vv, color=col, lw=3, zorder=2); ax.plot(vv, i, "o", color=col, ms=13, mec="white", mew=1.4, zorder=3)
        ax.text(vv, i + 0.18, str(vv), ha="center", va="bottom", fontweight="bold", color=col, fontsize=16)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labs); ax.set_xlim(0, 1050); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Fundus images (n)"); ax.set_title("Cohort: 489 patients / 988 images"); ax.grid(axis="y", alpha=0)
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "b", x=-0.12, y=1.06)
    t = ["Blood → syndrome", "Fundus → syndrome", "Fundus → intolerance"]
    # the two syndrome AUCs come from the syndrome analysis (Supplementary Note S1);
    # the third is this paper's canonical pooled out-of-fold run
    # blood -> syndrome from 54_syndrome_from_blood.py; fundus -> syndrome from the
    # five-fold RETFound run on the masked images. Both were literals here before.
    _blood = json.load(open(os.path.join(R, "syndrome_from_blood.json"),
                            encoding="utf-8"))["blood_metabolic_bmi"]["macro_auc_ovr"]
    _syn = json.load(open(os.path.join(R, "retfound_syndrome", "cv_summary.json"),
                          encoding="utf-8"))["macro_auc"]
    _fundus = float(str(_syn).split()[0])          # stored as "mean +- sd"
    v = [_blood, _fundus, float(roc_auc_score(OOF.y, OOF.prob))]
    tcol = [C["sky"], C["sky"], C["ok"]]
    for i, (val, col) in enumerate(zip(v, tcol)):
        ax.hlines(i, 0.5, val, color=col, lw=3, zorder=2); ax.plot(val, i, "o", color=col, ms=13, mec="white", mew=1.3, zorder=3)
        ax.text(val + 0.012, i, f"{val:.3f}", va="center", fontweight="bold", fontsize=15, color=col)
    ax.axvline(0.5, ls="--", color=C["accent"], alpha=0.6); ax.set_yticks([0, 1, 2]); ax.set_yticklabels(t, fontsize=15)
    ax.set_xlim(0.5, 1.03); ax.set_xlabel("AUC"); ax.set_title("Imaging predicts intolerance,\nnot TCM syndrome"); ax.grid(axis="y", alpha=0)
    fig.savefig(os.path.join(OUTF, "Figure2_cohort.png"), bbox_inches="tight", dpi=300); plt.close(); print("Fig2 done")


# ============ Fig 6 — TCM syndrome digital characterization (rebuilt for font consistency) ============
def fig6():
    prof = pd.read_csv(os.path.join(R, "tab_tcm_profile.csv")).set_index("Unnamed: 0")
    order = ["痰瘀阻滞证", "阴虚夹瘀证", "气阴两虚证", "脾肾两虚证"]
    en = {"痰瘀阻滞证": "Phlegm-stasis", "阴虚夹瘀证": "Yin-def+stasis", "气阴两虚证": "Qi-Yin def", "脾肾两虚证": "Spleen-Kidney def"}
    prof = prof.loc[order]
    cols = [C["main"], C["sky"], C["orange"], C["accent"]]
    fig = plt.figure(figsize=(13.5, 5.6)); gs = GridSpec(1, 2, figure=fig, wspace=0.42, width_ratios=[1, 1.1])
    # (a) radar — normalized objective profiles
    metrics = ["Age", "Duration", "Male%", "Hypertension%", "Smoking%", "Intolerance%"]
    nice = ["Age", "DM dur.", "Male %", "HTN %", "Smoking %", "Intol. %"]
    norm = prof[metrics].copy()
    for c in metrics:
        v = norm[c]; norm[c] = (v - v.min()) / (v.max() - v.min() + 1e-9)
    N = len(metrics); ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); ang += ang[:1]
    ax = fig.add_subplot(gs[0, 0], polar=True)
    for i, s in enumerate(order):
        vals = norm.loc[s, metrics].tolist(); vals += vals[:1]
        ax.plot(ang, vals, color=cols[i], lw=2, label=en[s]); ax.fill(ang, vals, color=cols[i], alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(nice, fontsize=13); ax.set_yticklabels([]); ax.set_ylim(0, 1)
    ax.tick_params(pad=8); ax.set_title("Objective syndrome profiles (normalized)", pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=11, frameon=False)
    ax.text(-0.12, 1.16, "a", transform=ax.transAxes, fontsize=17, fontweight="bold")
    # (b) excess→deficiency progression (twin axis: years vs %)
    ax = fig.add_subplot(gs[0, 1]); plabel(ax, "b"); ax2 = ax.twinx(); ax2.grid(False)
    x = list(range(len(order)))
    l1, = ax.plot(x, prof["Age"], "o-", color=C["main"], lw=2, label="Age (yr)")
    l2, = ax.plot(x, prof["Duration"], "s-", color=C["ok"], lw=2, label="DM duration (yr)")
    l3, = ax2.plot(x, prof["Hypertension%"] * 100, "^-", color=C["accent"], lw=2, label="Hypertension (%)")
    l4, = ax2.plot(x, prof["Intolerance%"] * 100, "d--", color=C["purple"], lw=2, label="Intolerance (%)")
    ax.set_xticks(x); ax.set_xticklabels([en[o] for o in order], rotation=18, ha="right", fontsize=11)
    ax.set_ylabel("Years"); ax2.set_ylabel("Percent (%)"); ax.set_title("Excess → deficiency progression")
    ax.legend(handles=[l1, l2, l3, l4], fontsize=10.5, loc="center left")
    fig.savefig(os.path.join(OUTF, "Figure6_tcm.png"), dpi=300, bbox_inches="tight"); plt.close(); print("Fig6 done")


if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8(); fig9(); fig10(); fig11(); fig13()
    print("ALL DINOv2 figures regenerated into paper2/")
