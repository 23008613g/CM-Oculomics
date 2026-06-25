# -*- coding: utf-8 -*-
"""
Fine-tune a vision foundation model for anti-VEGF intolerance (5-fold, patient-level CV).

Primary backbone: DINOv2 (ViT-L/14, Apache-2.0); set `backbone: retfound` in the config
to fine-tune the RETFound comparator instead. Class-weighted cross-entropy handles the
~9:1 imbalance; AMP + cosine schedule; early stopping on validation balanced accuracy.
Supply your own de-identified manifest (see src/dataset.py). NO patient data is included.
"""
import argparse, yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from .model import FundusClassifier, build_dinov2
from .dataset import FundusDataset, build_transforms
from PIL import Image


class _DS(torch.utils.data.Dataset):
    def __init__(self, df, train, size=224):
        self.df = df.reset_index(drop=True); self.tf = build_transforms(size, train)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r["image_path"]).convert("RGB")), int(r["label"])


def class_weights(labels, device):
    c = np.bincount(labels, minlength=2).astype(float)
    w = c.sum() / (2 * np.maximum(c, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def train_one_fold(tr, va, cfg, device):
    tr_ds, va_ds = _DS(tr, True), _DS(va, False)
    tr_ld = DataLoader(tr_ds, cfg["batch_size"], shuffle=True, num_workers=4, drop_last=True)
    va_ld = DataLoader(va_ds, cfg["batch_size"], shuffle=False, num_workers=4)
    if cfg.get("backbone", "dinov2") == "retfound":
        model = FundusClassifier(2, cfg.get("retfound_weights")).to(device)
    else:
        model = build_dinov2(2, img_size=cfg.get("image_size", 224)).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights(tr["label"].values, device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, bad = -1, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        ys, ps = _eval(model, va_ld, device)
        auc = roc_auc_score(ys, ps); ba = balanced_accuracy_score(ys, (ps >= 0.5).astype(int))
        if ba > best: best, bad = ba, 0; torch.save(model.state_dict(), cfg["out_weights"])
        else:
            bad += 1
            if bad >= cfg.get("patience", 10): break
    return best


@torch.no_grad()
def _eval(model, ld, device):
    model.eval(); ys, ps = [], []
    for x, y in ld:
        p = torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy()
        ps.extend(p); ys.extend(y.numpy())
    return np.array(ys), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(cfg["manifest"])
    y = df["label"].values
    if "patient_id" in df.columns:
        splitter = StratifiedGroupKFold(cfg["folds"], shuffle=True, random_state=cfg["seed"])
        folds = splitter.split(df, y, df["patient_id"].values)
    else:
        splitter = StratifiedKFold(cfg["folds"], shuffle=True, random_state=cfg["seed"])
        folds = splitter.split(df, y)
    aucs = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        cfg["out_weights"] = f"weights/fold{k}.pth"
        best = train_one_fold(df.iloc[tr_idx], df.iloc[va_idx], cfg, device)
        print(f"fold{k}: best balanced_acc={best:.3f}")
        aucs.append(best)
    print(f"mean balanced_acc: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")


if __name__ == "__main__":
    main()
