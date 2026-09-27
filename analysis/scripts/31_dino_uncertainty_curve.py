# -*- coding: utf-8 -*-
"""Load saved DINOv2 deploy model, run MC-dropout on the held-out test set, and save the
selective-prediction curve + pooled-OOF calibration arrays for Fig 7. No retraining."""
import os, sys, json
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from tcm_retina.data.dataset import build_transforms, GRADE_CLASSES
IMG = os.path.join(ROOT, "data_anon", "images"); DATA = os.path.join(ROOT, "data_anon")
OUT = os.path.join(ROOT, "results", "dino_experiment")
DEV = "cuda" if torch.cuda.is_available() else "cpu"; RES = 224
g2i = {c: i for i, c in enumerate(GRADE_CLASSES)}

def make_model():
    b = timm.create_model("vit_large_patch14_dinov2", pretrained=False, num_classes=0, img_size=RES, drop_rate=0.2)
    h = nn.Sequential(nn.LayerNorm(b.num_features), nn.Dropout(0.2), nn.Linear(b.num_features, 2))
    return nn.Sequential(b, h)

class DS(Dataset):
    def __init__(self, rows):
        self.rows = rows.reset_index(drop=True); self.tf = build_transforms(RES, False, None)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        return self.tf(Image.open(os.path.join(IMG, r["anon_image"])).convert("RGB")), g2i[r["grade"]], i

def enable_dropout(m):
    for mod in m.modules():
        if mod.__class__.__name__.startswith("Dropout"): mod.train()

@torch.no_grad()
def main():
    m = make_model(); m.load_state_dict(torch.load(os.path.join(OUT, "dino_deploy.pth"), map_location="cpu"))
    m.to(DEV).eval()
    sp = pd.read_csv(os.path.join(DATA, "splits.csv"))
    test = sp[sp.split == "test"].dropna(subset=["grade"]).reset_index(drop=True)
    ds = DS(test); ld = DataLoader(ds, 16, shuffle=False, num_workers=0); ys = test["grade"].map(g2i).values
    enable_dropout(m); runs = []
    for _ in range(30):
        ps = np.zeros(len(ds))
        for x, y, idx in ld:
            ps[idx.numpy()] = torch.softmax(m(x.to(DEV)), 1)[:, 1].float().cpu().numpy()
        runs.append(ps)
    runs = np.array(runs); mean_p = runs.mean(0); std_p = runs.std(0)
    order = np.argsort(std_p); cov, bacc = [], []
    for kf in np.linspace(0.1, 1.0, 10):
        sel = order[:max(1, int(len(order) * kf))]
        cov.append(kf); bacc.append(balanced_accuracy_score(ys[sel], (mean_p[sel] >= 0.5).astype(int)))
    np.savez(os.path.join(OUT, "dino_uncertainty_curve.npz"),
             coverage=np.array(cov), bacc=np.array(bacc), mean_p=mean_p, ys=ys, std_p=std_p)
    print("AUC(test, MC mean)=%.3f" % roc_auc_score(ys, mean_p))
    print("saved dino_uncertainty_curve.npz  cov/bacc:", list(zip(np.round(cov, 2), np.round(bacc, 3))))

if __name__ == "__main__":
    main()
