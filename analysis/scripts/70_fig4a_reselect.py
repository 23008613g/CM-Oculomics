# -*- coding: utf-8 -*-
"""
70_fig4a_reselect.py - choose the Figure 4a example eyes from ONE camera, by a stated rule.

WHY: the previous Figure 4a showed seven intolerant eyes of which six were VISUCAM images and
seven tolerant eyes that were all Topcon images (scripts/62). The VISUCAM writes a laterality
label and fixation marker and frames the fundus differently, so the two rows differed by camera
as well as by outcome, and blind reviewers read the marks as a class difference. The examples
had also been hand-picked. This script re-selects them by a rule that anyone can re-run:

  * held-out test split only (no model shown here saw these images in training);
  * VISUCAM 200 only (the camera with most intolerant eyes; marks present on every image);
  * one image per patient (the first by file name);
  * classified correctly at 0.5 by BOTH models whose attention is drawn: the released DINOv2
    model (dino_deploy.pth) and the RETFound single model (retfound_antivegf/best.pth);
  * the same number per class, the smaller eligible count capped at 7, drawn at random
    (seed 0) from the eligible images.

It then writes the original, DINOv2 and RETFound Grad-CAM tiles exactly as scripts/30 and
scripts/05 draw them, and measures fundus coverage and the share of the displayed map that
falls outside the fundus (the quantity used in the Grad-CAM framing limitation, scripts/59).

USAGE   python scripts/70_fig4a_reselect.py        (GPU recommended, ~2 min)
OUTPUT  results/fig4a_eyes.json; tiles in results/figures/gradcam_dino/ and gradcam/
"""
import os, sys, json
import numpy as np, pandas as pd
import torch, torch.nn as nn, timm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from tcm_retina.models.backbone import FundusClassifier
from tcm_retina.data.dataset import build_transforms, GRADE_CLASSES

DATA = os.path.join(ROOT, "data_anon")
GD = os.path.join(ROOT, "results", "figures", "gradcam_dino")
GR = os.path.join(ROOT, "results", "figures", "gradcam")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES, N_PER, SEED = 224, 7, 0
VISUCAM = (2400, 2040)


def dino():
    b = timm.create_model("vit_large_patch14_dinov2", pretrained=False, num_classes=0, img_size=RES, drop_rate=0.2)
    m = nn.Sequential(b, nn.Sequential(nn.LayerNorm(b.num_features), nn.Dropout(0.2), nn.Linear(b.num_features, 2)))
    m.load_state_dict(torch.load(os.path.join(ROOT, "results", "dino_experiment", "dino_deploy.pth"), map_location="cpu"))
    return m.to(DEVICE).eval()


def retfound():
    m = FundusClassifier("vit_large_patch16_224", num_classes=2, pretrained=False, retfound=False)
    m.load_state_dict(torch.load(os.path.join(ROOT, "results", "retfound_antivegf", "best.pth"), map_location="cpu"))
    return m.to(DEVICE).eval()


def rs_dino(t, s=RES // 14):     # scripts/30: last s*s tokens -> 16x16 grid
    x = t[:, -s * s:, :]
    return x.reshape(t.size(0), s, s, t.size(2)).permute(0, 3, 1, 2)


def rs_ret(t, h=14, w=14):       # scripts/05: drop CLS -> 14x14 grid
    return t[:, 1:, :].reshape(t.size(0), h, w, t.size(2)).permute(0, 3, 1, 2)


tf = build_transforms(RES, train=False)
sp = pd.read_csv(os.path.join(DATA, "splits.csv"))
te = sp[sp.split == "test"].copy()
te["vis"] = te.anon_image.map(lambda f: Image.open(os.path.join(DATA, "images_with_overlays", f)).size == VISUCAM)
cand = te[te.vis].sort_values("anon_image").groupby("anon_id").head(1).reset_index(drop=True)

mD, mR = dino(), retfound()
pD, pR = [], []
with torch.no_grad():
    for fn in cand.anon_image:
        x = tf(Image.open(os.path.join(DATA, "images", fn)).convert("RGB").resize((RES, RES))).unsqueeze(0).to(DEVICE)
        pD.append(float(torch.softmax(mD(x), 1)[0, 1])); pR.append(float(torch.softmax(mR(x), 1)[0, 1]))
cand["p_dino"], cand["p_ret"] = pD, pR
cand["y"] = (cand.grade == "PDR").astype(int)
cand["ok"] = ((cand.p_dino >= 0.5) == (cand.y == 1)) & ((cand.p_ret >= 0.5) == (cand.y == 1))
n_per = min(N_PER, int(((cand.grade == "PDR") & cand.ok).sum()), int(((cand.grade == "NPDR") & cand.ok).sum()))
out = {"rule": "held-out test, VISUCAM 200, one image per patient, correct at 0.5 by both models, "
               "%d per class drawn at random (seed %d)" % (n_per, SEED),
       "candidates": {g: int((cand.grade == g).sum()) for g in ("PDR", "NPDR")},
       "eligible": {g: int(((cand.grade == g) & cand.ok).sum()) for g in ("PDR", "NPDR")}, "eyes": {}}
print("VISUCAM test patients:", out["candidates"], " eligible:", out["eligible"])

camD = GradCAM(model=mD, target_layers=[mD[0].blocks[-1].norm1], reshape_transform=rs_dino)
camR = GradCAM(model=mR, target_layers=[mR.backbone.blocks[-1].norm1], reshape_transform=rs_ret)
frame = []
for g in ("PDR", "NPDR"):
    el = cand[(cand.grade == g) & cand.ok]
    pick = el.sample(n_per, random_state=SEED).sort_values("anon_image")
    cls = GRADE_CLASSES.index(g)
    out["eyes"][g] = []
    for _, r in pick.iterrows():
        fn = r.anon_image
        pil = Image.open(os.path.join(DATA, "images", fn)).convert("RGB").resize((RES, RES))
        rgb = np.array(pil).astype(np.float32) / 255.0
        x = tf(pil).unsqueeze(0).to(DEVICE)
        gD = camD(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
        gR = camR(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
        Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(GD, f"orig_{g}_{fn}.png"))
        Image.fromarray(show_cam_on_image(rgb, gD, use_rgb=True)).save(os.path.join(GD, f"cam_{g}_{fn}.png"))
        Image.fromarray(show_cam_on_image(rgb, gR, use_rgb=True)).save(os.path.join(GR, f"cam_{g}_{fn}.png"))
        # framing, as in scripts/59: disc on the 224 grid, min-max stretched map
        m = np.array(pil.convert("L")).astype(float) > 18
        s = gD - gD.min(); s = s / (s.max() + 1e-8)
        frame.append((float(m.mean()), float(s[~m].sum() / s.sum())))
        out["eyes"][g].append({"image": fn, "p_dino": round(r.p_dino, 3), "p_retfound": round(r.p_ret, 3),
                               "coverage": round(frame[-1][0], 3), "outside": round(frame[-1][1], 3)})
        print("  %-5s %-18s DINOv2 %.3f  RETFound %.3f  coverage %.2f  outside %.2f"
              % (g, fn, r.p_dino, r.p_ret, *frame[-1]))
out["framing"] = {"n": len(frame), "mean_coverage": round(float(np.mean([f[0] for f in frame])), 3),
                  "mean_outside": round(float(np.mean([f[1] for f in frame])), 3),
                  "max_outside": round(float(np.max([f[1] for f in frame])), 3)}
json.dump(out, open(os.path.join(ROOT, "results", "fig4a_eyes.json"), "w", encoding="utf-8"), indent=2)
print("framing:", out["framing"]); print("saved results/fig4a_eyes.json")
