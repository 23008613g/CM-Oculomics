# -*- coding: utf-8 -*-
"""
45_shift_all_figure.py — one figure covering every label-free signal tested.

Combines the three distributional divergence measures (42) with the two signals the
out-of-distribution-performance-prediction literature actually uses, model agreement and
predictive confidence (44), so that the negative result is not a straw man.
"""
import os, json
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "external2")
KINDS = ["dinov2_l", "retfound"]
NAME = {"dinov2_l": "DINOv2", "retfound": "RETFound"}
C = {"dinov2_l": "#D55E00", "retfound": "#0072B2"}
CHINA = {"ddr", "jsiec", "deepdrid"}

shift = json.load(open(os.path.join(OUT, "shift_metrics.json")))
agree = json.load(open(os.path.join(OUT, "agreement_confidence.json")))

# merge the two result files on cohort key
DATA = {}
for kind in KINDS:
    a = {r["key"]: r for r in agree[kind]["rows"]}
    DATA[kind] = [dict(r, **{k: a[r["key"]][k] for k in ("agreement", "confidence")})
                  for r in shift[kind]["rows"]]

PANELS = [("frechet", "Fréchet distance", "(a)"),
          ("mmd2", "MMD$^2$ (RBF kernel)", "(b)"),
          ("dom_auc", "domain-classifier AUC", "(c)"),
          ("agreement", "between-seed agreement", "(d)"),
          ("confidence", "mean predictive confidence", "(e)")]

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
ax = axes.ravel()

rho_table = {}
for a, (key, xlabel, tag) in zip(ax, PANELS):
    subs = []
    for kind in KINDS:
        rows = DATA[kind]
        x = [r[key] for r in rows]
        y = [r["external_auc"] for r in rows]
        rho, p = stats.spearmanr(x, y)
        rho_table[(kind, key)] = (rho, p)
        subs.append(f"{NAME[kind]} $\\rho$={rho:+.2f}")
        a.scatter(x, y, s=28 + 75 * np.array([r["n"] for r in rows]) / max(r["n"] for r in rows),
                  facecolor=C[kind], edgecolor="k", linewidth=0.5, alpha=0.85,
                  label=NAME[kind], zorder=3)
    if tag == "(a)":
        for r in DATA["dinov2_l"]:
            a.annotate(r["key"].upper(), (r[key], r["external_auc"]),
                       textcoords="offset points", xytext=(7, -3), fontsize=7, color="#444")
    a.set_xlabel(xlabel, fontsize=9.5)
    a.set_ylabel("observed external AUC", fontsize=9.5)
    a.set_ylim(0.64, 1.02); a.margins(x=0.15)
    a.set_title(f"{tag} {xlabel}\n" + "   ".join(subs), fontsize=9)

# (f) every correlation on one scale
a = ax[5]
labels = [p[1].replace("$^2$", "²").replace(" (RBF kernel)", "") for p in PANELS]
xs = np.arange(len(PANELS))
# Each signal is signed so that + means the direction a usable predictor must show: the three
# divergences should FALL as AUC rises (more shift, worse transfer), agreement and confidence
# should RISE with AUC. An earlier version plotted raw rho with the "usable" band at rho <= -0.7,
# which is the right side only for the divergences.
EXPECT = {"frechet": -1, "mmd2": -1, "dom_auc": -1, "agreement": +1, "confidence": +1}
for off, kind in zip((-0.19, 0.19), KINDS):
    a.bar(xs + off, [EXPECT[p[0]] * rho_table[(kind, p[0])][0] for p in PANELS], 0.38,
          color=C[kind], edgecolor="k", linewidth=0.6, label=NAME[kind])
a.axhline(0, color="k", lw=1)
a.axhspan(0.7, 1, color="#2E7D32", alpha=0.12)
a.text(len(PANELS) - 0.5, 0.85, "a usable predictor\nwould sit here", ha="right", va="center",
       fontsize=8, color="#2E7D32")
a.set_xticks(xs)
a.set_xticklabels(labels, fontsize=8, rotation=22, ha="right", rotation_mode="anchor")
a.set_ylim(-1, 1)
a.set_ylabel(r"Spearman $\rho$, signed so that +" "\n" "= direction a predictor must show", fontsize=9)
# Summary computed from the ten raw correlations (was a hardcoded "all |rho| < 0.4, all
# p > 0.38", true only of an earlier three-seed analysis). Ten tests: Bonferroni alpha = 0.005.
_r = [v[0] for v in rho_table.values()]; _p = [v[1] for v in rho_table.values()]
_nsig = sum(p < 0.05 / len(_p) for p in _p)
_best = max(EXPECT[k] * v[0] for (kk, k), v in rho_table.items())
a.set_title("(f) None of the five signals is a usable predictor\n"
            r"best signed $\rho$ = %+.2f; raw $\rho$ from %+.2f to %+.2f" "\n"
            "smallest p = %.3f; %d of %d significant after Bonferroni"
            % (_best, min(_r), max(_r), min(_p), _nsig, len(_p)), fontsize=8.5)
a.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.83))

fig.suptitle("No label-free signal anticipates how well the model transfers to a new cohort",
             fontsize=12.5, y=0.995)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, -0.025))
fig.tight_layout(rect=[0, 0.01, 1, 0.975])
fn = os.path.join(OUT, "Figure_shift_all.png")
fig.savefig(fn, dpi=300, bbox_inches="tight"); plt.close()

print("Spearman rho, every label-free signal vs observed external AUC (n=7 cohorts)")
print(f"{'signal':<28}{'DINOv2':>18}{'RETFound':>18}")
for key, lab, _ in PANELS:
    d = rho_table[("dinov2_l", key)]; r = rho_table[("retfound", key)]
    print(f"{lab.replace('$^2$','2'):<28}{d[0]:>+10.3f} (p={d[1]:.2f}){r[0]:>+10.3f} (p={r[1]:.2f})")
print("\nconfidently wrong — the cohort each backbone does WORST on:")
for kind in KINDS:
    rows = DATA[kind]
    w = min(rows, key=lambda r: r["external_auc"])
    ra = sorted(rows, key=lambda r: -r["agreement"]).index(w) + 1
    print(f"  {NAME[kind]:<9} worst cohort {w['label']:<22} AUC {w['external_auc']:.3f}, "
          f"agreement {w['agreement']:.3f} (rank {ra} of 7), confidence {w['confidence']:.3f}")
print(f"\nsaved {fn}")
