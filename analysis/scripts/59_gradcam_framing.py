# -*- coding: utf-8 -*-
"""
59_gradcam_framing.py - how much of a Grad-CAM map falls outside the retina, and why.

WHY: a reader running the released demo on an image with wide dark borders sees warm spots
in the corners, outside the fundus. This script establishes what causes that, so the paper
can state it accurately rather than leaving a reviewer to wonder.

The explanation is the framing of the image, not the model using the border. Grad-CAM on a
vision transformer is computed on a 16x16 token grid; self-attention is global, so tokens
covering the dark surround still carry class-relevant information and receive non-zero
gradients, which the visualisation then plots at their spatial position. The smaller the
fundus disc is within the frame, the more such tokens there are.

Three measurements:
  1. across development and DDR images, disc coverage against the share of displayed heat
     that lies outside the disc;
  2. the same image cropped to its disc, which changes the framing and nothing else;
  3. the sixteen eyes shown in Figure 4a, to confirm the published figure is unaffected.

USAGE   python scripts/59_gradcam_framing.py
OUTPUT  results/gradcam_framing.json
"""
import os, sys, glob, json
import numpy as np
import torch
from PIL import Image
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(ROOT, "release_repo")
sys.path.insert(0, REL)
os.chdir(REL)                      # app.py resolves its weights relative to the repo
os.environ.setdefault("WEIGHTS_PATH", "weights/dino_deploy.pth")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import app as APP                                            # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget   # noqa: E402

DEV = os.path.join(ROOT, "data_anon", "images")
DDR = os.path.join(ROOT, "external_data", "ddr")
FIG4_EYES = os.path.join(ROOT, "results", "figs7_eyes.txt")


def disc(pil, size=224):
    """The illuminated fundus, on the same 224 grid the CAM is drawn on."""
    return np.array(pil.resize((size, size)).convert("L")).astype(float) > 18


def measure(path_or_img):
    pil = (Image.open(path_or_img).convert("RGB")
           if isinstance(path_or_img, str) else path_or_img)
    x = APP._tf(pil).unsqueeze(0).to(APP.DEVICE)
    with torch.no_grad():
        p = torch.softmax(APP._model(x), 1)[0]
    raw = APP._cam(input_tensor=x,
                   targets=[ClassifierOutputTarget(int(p.argmax()))])[0]
    m = disc(pil)
    g = raw - raw.min()
    g = g / (g.max() + 1e-8)                 # the stretch the app applies before display
    return dict(coverage=float(m.mean()),
                outside=float(g[~m].sum() / g.sum()),
                prob=float(p[1]))


out = {"note": ("Grad-CAM computed as in the released application: last transformer block, "
                "16x16 token grid, explained class = the predicted class, per-image min-max "
                "stretch before display.")}

# ---- 1. disc coverage against out-of-retina heat, two cohorts ----------------------
print("1. framing versus out-of-retina heat")
groups, rows = {}, []
for tag, files in (("development", sorted(glob.glob(os.path.join(DEV, "*.jpg")))[:12]),
                   ("DDR", sorted(glob.glob(os.path.join(DDR, "**", "*.jpg"),
                                            recursive=True))[:12])):
    vals = [measure(f) for f in files]
    groups[tag] = dict(n=len(vals),
                       mean_coverage=round(float(np.mean([v["coverage"] for v in vals])), 3),
                       mean_outside=round(float(np.mean([v["outside"] for v in vals])), 3))
    rows += vals
    print("   %-12s n=%d  disc %.0f%% of frame  ->  %.0f%% of heat outside"
          % (tag, len(vals), 100 * groups[tag]["mean_coverage"],
             100 * groups[tag]["mean_outside"]))
rho, pv = stats.spearmanr([r["coverage"] for r in rows], [r["outside"] for r in rows])
out["cohorts"] = groups
out["correlation"] = dict(spearman_rho=round(float(rho), 2), p=float(pv), n=len(rows))
print("   Spearman(coverage, outside) = %+.2f, p = %.1g, n = %d" % (rho, pv, len(rows)))

# ---- 2. the same image, re-framed ---------------------------------------------------
print("\n2. the same image cropped to its disc")
ex = os.path.join(REL, "assets", "example_fundus_uncropped.jpg")
if not os.path.exists(ex):                 # the shipped example is already the cropped one
    ex = os.path.join(ROOT, "results", "figS7", "example_fundus_uncropped_original.jpg")
pil = Image.open(ex).convert("RGB")
before = measure(pil)
a = np.array(pil.convert("L")) > 18
ys, xs = np.where(a)
after = measure(pil.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
out["recrop"] = {"image": os.path.basename(ex), "before": before, "after": after}
for tag, v in (("as shipped", before), ("cropped to the disc", after)):
    print("   %-20s disc %.0f%%  outside %.0f%%  P(intolerant) %.4f"
          % (tag, 100 * v["coverage"], 100 * v["outside"], v["prob"]))

# ---- 3. the eyes in Figure 4a --------------------------------------------------------
print("\n3. the eyes shown in Figure 4a")
eyes = [l.strip() for l in open(FIG4_EYES, encoding="utf-8") if l.strip()]
vals = [measure(os.path.join(DEV, e)) for e in eyes if os.path.exists(os.path.join(DEV, e))]
out["figure4_eyes"] = dict(n=len(vals),
                           mean_coverage=round(float(np.mean([v["coverage"] for v in vals])), 3),
                           mean_outside=round(float(np.mean([v["outside"] for v in vals])), 3),
                           max_outside=round(float(np.max([v["outside"] for v in vals])), 3))
f = out["figure4_eyes"]
print("   n=%d  disc %.0f%%  outside %.0f%% on average (worst %.0f%%)"
      % (f["n"], 100 * f["mean_coverage"], 100 * f["mean_outside"], 100 * f["max_outside"]))

dst = os.path.join(ROOT, "results", "gradcam_framing.json")
json.dump(out, open(dst, "w"), indent=2)
print("\nsaved %s" % dst)
