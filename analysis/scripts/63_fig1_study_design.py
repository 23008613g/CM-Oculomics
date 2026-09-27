# -*- coding: utf-8 -*-
"""
63_fig1_study_design.py - Figure 1, the study design, drawn from what was actually done.

WHY: the previous Figure 1 was a raster schematic of a multimodal fusion model (image +
clinical + TCM-syndrome branches) with "AUC 0.938" and "Sensitivity >= 0.93" printed on it.
The reported model is image-only, fusion added nothing (Note S1), and both numbers came from
a superseded analysis. All three blind reviewers flagged it. This version shows the design
only (cohort, split, model, the two evaluation arms, the estimation study, the release) and
deliberately prints no performance numbers, so it cannot go stale when a result changes.
Every count on it is read from the data files, not typed in.

USAGE   python scripts/63_fig1_study_design.py
OUTPUT  results/figures/paper2/Figure1_study_design.png (+ .pdf)
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
DATA = os.path.join(ROOT, "data_anon")
OUTF = os.path.join(ROOT, "results", "figures", "paper2")

# ---------------------------------------------------------------- counts from the data
man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
sp = pd.read_csv(os.path.join(DATA, "splits.csv"))
cv, te = sp[sp.split == "train_val"], sp[sp.split == "test"]
meta = json.load(open(os.path.join(ROOT, "results", "external2", "meta_summary.json")))
n_ext = sum(r["n"] for r in meta["rows"])
N = dict(pat=man.anon_id.nunique(), img=len(man), an_img=len(sp), an_pat=sp.anon_id.nunique(),
         cv_img=len(cv), cv_pat=cv.anon_id.nunique(), cv_pos=int((cv.grade == "PDR").sum()),
         te_img=len(te), te_pat=te.anon_id.nunique(), te_pos=int((te.grade == "PDR").sum()),
         ext=n_ext, ncoh=len(meta["rows"]))
print(N)

# ---------------------------------------------------------------- drawing helpers
COL = {"data": ("#E8F1FA", "#2F6DA3"), "model": ("#FDEBD9", "#C0561B"),
       "int": ("#E7F4EC", "#2E7D4F"), "ext": ("#F1E9F7", "#6A3D9A"),
       "est": ("#FFF6D6", "#A07800"), "rel": ("#EFEFEF", "#555555")}
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(13.0, 5.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 56); ax.axis("off")


def box(x, y, w, h, kind, title, lines, tsize=11.5, lsize=9.6):
    fc, ec = COL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=tsize,
            fontweight="bold", color=ec)
    ax.text(x + 1.4, y + h - 5.4, "\n".join(lines), ha="left", va="top", fontsize=lsize,
            color="#222", linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color="#444", shrinkA=0, shrinkB=0))


# column 1: development cohort and split
box(1, 22, 29, 32, "data", "Development cohort",
    ["Single centre, retrospective,",
     "photographs 2021-2024",
     "%d patients, %d photographs" % (N["pat"], N["img"]),
     "three camera set-ups",
     "",
     "Label per eye: progression to PDR",
     "during an anti-VEGF course",
     "(intolerant) vs remained NPDR",
     "(tolerant)",
     "",
     "Analysed (clinical record linked):",
     "%d images, %d patients" % (N["an_img"], N["an_pat"])])
box(1, 11.5, 29, 8.5, "data", "5-fold cross-validation",
    ["%d images, %d patients, %d PDR" % (N["cv_img"], N["cv_pat"], N["cv_pos"])], tsize=10.5)
box(1, 1, 29, 8.5, "data", "Held-out test (patient-disjoint)",
    ["%d images, %d patients, %d PDR" % (N["te_img"], N["te_pat"], N["te_pos"])], tsize=10.5)

# column 2: model
box(36, 22, 24, 32, "model", "Image-only model",
    ["Fine-tuned end to end, 224 px",
     "",
     "DINOv2 ViT-L/14",
     "  primary; natural-image",
     "  self-supervised pretraining",
     "RETFound ViT-L/16",
     "  comparator; retinal",
     "  pretraining",
     "",
     "5 training seeds per backbone",
     "Clinical / TCM fusion: Note S1"])

# column 3: internal and external arms
box(66, 30, 30, 24, "int", "Internal evaluation",
    ["Pooled out-of-fold predictions",
     "Held-out test split",
     "Calibration, decision curve,",
     "subgroups, acquisition device",
     "Grad-CAM, vascular measures"])
box(66, 1, 30, 26, "ext", "External transfer, no fine-tuning",
    ["%d public cohorts, 3 continents:" % N["ncoh"],
     "DDR, JSIEC, DeepDRiD (China)",
     "APTOS, IDRiD (India)",
     "Messidor-2 (France), EyePACS (USA)",
     "Proxy task: PDR vs NPDR",
     "Class-balanced subsets, %s images" % format(N["ext"], ",")])

# column 4: estimation study and release
box(102, 17, 27, 37, "est", "Can transfer be predicted\nwithout target labels?",
    ["",
     "5 label-free shift signals",
     "4 published estimators",
     "  (ALine-D, ALine-S, DoC, ATC)",
     "  anchored on the held-out test",
     "",
     "Compared with the observed",
     "performance of each cohort",
     "",
     "Labels needed for local",
     "validation"], tsize=10.5)
box(102, 1, 27, 13, "rel", "Open release",
    ["Code and weights (Apache-2.0)",
     "Zenodo archive, web demo"], tsize=10.5)

# arrows
arrow(30.8, 38, 35.2, 38)                 # cohort -> model
arrow(15.5, 22, 15.5, 20.4)               # cohort -> CV
arrow(60.8, 42, 65.2, 42)                 # model -> internal
arrow(60.8, 30, 65.2, 16)                 # model -> external
arrow(96.8, 16, 101.2, 30)                # external -> estimation
arrow(96.8, 42, 101.2, 42)                # internal (anchor) -> estimation
ax.text(99, 44.2, "anchor", ha="center", fontsize=8.5, color="#555")
ax.text(99, 19.5, "targets", ha="center", fontsize=8.5, color="#555", rotation=0)

os.makedirs(OUTF, exist_ok=True)
fn = os.path.join(OUTF, "Figure1_study_design.png")
fig.savefig(fn, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(fn.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
plt.close()
print("saved", fn)
