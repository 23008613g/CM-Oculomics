# -*- coding: utf-8 -*-
"""
71_figS1_framework.py - Figure S1, redrawn without management recommendations.

WHY: the previous Figure S1 (an image-generated pathway) ended in two management boxes: "high
risk + deficiency pattern: prioritize early escalation (laser / surgery)" and "low risk + excess
pattern: standard anti-VEGF, less-frequent monitoring". The data do not support either: the
endpoint is a retrospective proxy, syndrome added no predictive value and intolerance rates did
not differ by syndrome (Note S1), and the main text states that the negative predictive value
does not establish that monitoring can safely be reduced. A blind reviewer flagged the conflict.
This version keeps the two-axis idea as what it is, a hypothesis for prospective study, and
makes no recommendation.

USAGE   python scripts/71_figS1_framework.py
OUTPUT  results/figures/paper2/FigureS1_framework.png
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "figures", "paper2", "FigureS1_framework.png")
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(11, 7.0)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 110); ax.set_ylim(0, 70); ax.axis("off")
C = {"in": ("#E8F1FA", "#2F6DA3"), "img": ("#FDEBD9", "#C0561B"), "tcm": ("#E7F4EC", "#2E7D4F"),
     "mid": ("#F1E9F7", "#6A3D9A"), "q": ("#FFF6D6", "#A07800")}


def box(x, y, w, h, k, title, body, ts=12, bs=10):
    fc, ec = C[k]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=ts, fontweight="bold", color=ec)
    ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=bs, color="#222", linespacing=1.35)


def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#444"))


ax.text(55, 68.5, "Two axes that could be combined in a prospective study (hypothesis only)",
        ha="center", va="top", fontsize=14, fontweight="bold")
box(37, 51, 36, 11, "in", "Inputs", "one color fundus photograph\n+ routine clinical data")
box(3, 31, 40, 15, "img", "Imaging axis",
    "DINOv2 risk score for progression to PDR\nduring an anti-VEGF course\n(research output; not calibrated)")
box(67, 31, 40, 15, "tcm", "Constitutional axis",
    "syndrome characterized from clinical\nand constitutional variables\n(Supplementary Note S1)")
box(37, 17.5, 36, 10, "mid", "Combined profile", "imaging score + syndrome position")
box(6, 0.8, 98, 13, "q", "Question for prospective evaluation",
    "Does the combination stratify real per-injection outcomes better than either axis alone?\n"
    "In this cohort syndrome added no predictive value and intolerance rates did not differ by syndrome;\n"
    "no management recommendation is made.", ts=11, bs=9.5)
arrow((49, 51), (30, 46.6)); arrow((61, 51), (80, 46.6))
arrow((23, 31), (40, 26)); arrow((87, 31), (70, 26)); arrow((55, 17.5), (55, 14.4))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
print("saved", OUT)
