# -*- coding: utf-8 -*-
"""多任务训练:共享影像编码器,主任务=anti-VEGF耐受性,辅助任务=中医证型。
对比 baseline(单任务,aux_weight=0)与 multitask(aux_weight>0),看证型监督是否帮主任务。"""
import os, json, time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             recall_score, f1_score, accuracy_score)

from tcm_retina.data.dataset import build_transforms, GRADE_CLASSES, SYNDROME_CLASSES
from tcm_retina.models.multitask import MultiTaskNet


class MTDataset(Dataset):
    def __init__(self, rows, image_dir, image_size, train, aug=None):
        self.rows = rows.reset_index(drop=True); self.image_dir = image_dir
        self.tf = build_transforms(image_size, train, aug)
        self.g2i = {c: i for i, c in enumerate(GRADE_CLASSES)}
        self.s2i = {c: i for i, c in enumerate(SYNDROME_CLASSES)}

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = self.tf(Image.open(os.path.join(self.image_dir, r["anon_image"])).convert("RGB"))
        y_tol = self.g2i[r["grade"]]
        y_syn = self.s2i.get(r["tcm_syndrome"], -1)  # -1 = 无证型,损失里忽略
        return x, y_tol, y_syn

    def tol_counts(self):
        vc = self.rows["grade"].value_counts(); return [int(vc.get(c, 0)) for c in GRADE_CLASSES]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, ps, probs = [], [], []
    for x, yt, _ in loader:
        x = x.to(device)
        logit, _ = model(x)
        prob = torch.softmax(logit, 1).cpu().numpy()
        ps.extend(prob.argmax(1)); probs.extend(prob[:, 1]); ys.extend(yt.numpy())
    ys, ps, probs = np.array(ys), np.array(ps), np.array(probs)
    return {"auc": roc_auc_score(ys, probs), "pr_auc": average_precision_score(ys, probs),
            "balanced_acc": balanced_accuracy_score(ys, ps),
            "sensitivity": recall_score(ys, ps, pos_label=1, zero_division=0),
            "specificity": recall_score(ys, ps, pos_label=0, zero_division=0),
            "_ys": ys, "_pos": probs}


def run_fold(cfg, fold, aux_weight, log):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(cfg["data"]["splits_csv"]).dropna(subset=["grade"])
    tr = df[(df.split == "train_val") & (df.fold != fold)]
    va = df[(df.split == "train_val") & (df.fold == fold)]
    img_dir, sz, aug = cfg["data"]["image_dir"], cfg["data"]["image_size"], cfg.get("aug", {})
    tr_ds = MTDataset(tr, img_dir, sz, True, aug); va_ds = MTDataset(va, img_dir, sz, False)
    bs, nw = cfg["train"]["batch_size"], cfg["data"].get("num_workers", 4)
    tr_ld = DataLoader(tr_ds, bs, shuffle=True, num_workers=nw, drop_last=True)
    va_ld = DataLoader(va_ds, bs, shuffle=False, num_workers=nw)

    model = MultiTaskNet(retfound=cfg["model"].get("retfound", True),
                         drop_rate=cfg["model"].get("drop_rate", 0.2)).to(device)
    counts = np.array(tr_ds.tol_counts(), float)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    crit_tol = nn.CrossEntropyLoss(weight=w)
    crit_syn = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略无证型样本
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_val, best = -1, None; monitor = cfg["train"].get("monitor", "balanced_acc")
    patience, bad = cfg["train"].get("early_stop_patience", 12), 0
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time()
        for x, yt, ys in tr_ld:
            x, yt, ys = x.to(device), yt.to(device), ys.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                lt, lsn = model(x)
                loss = crit_tol(lt, yt) + aux_weight * crit_syn(lsn, ys)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        m = evaluate(model, va_ld, device)
        if m[monitor] > best_val: best_val, best, bad = m[monitor], m, 0
        else:
            bad += 1
            if bad >= patience: break
    log(f"  [aux={aux_weight} f{fold}] best auc={best['auc']:.3f} bacc={best['balanced_acc']:.3f} "
        f"sens={best['sensitivity']:.3f} ({time.time()-t0:.0f}s/ep)")
    return best


def run(cfg, aux_weight, tag):
    out_dir = cfg["output"]["dir"]; os.makedirs(out_dir, exist_ok=True)
    def log(msg):
        print(msg, flush=True)
        with open(os.path.join(out_dir, f"train_{tag}.log"), "a", encoding="utf-8") as fh: fh.write(msg+"\n")
    log(f"==== 多任务: tag={tag} aux_weight={aux_weight} ====")
    res = []
    for fold in range(5):
        b = run_fold(cfg, fold, aux_weight, log)
        np.savez(os.path.join(out_dir, f"valpred_{tag}_fold{fold}.npz"), ys=b["_ys"], pos=b["_pos"])
        res.append(b)
    keys = ["auc", "pr_auc", "balanced_acc", "sensitivity", "specificity"]
    summary = {k: f"{np.mean([r[k] for r in res]):.4f} ± {np.std([r[k] for r in res]):.4f}" for k in keys}
    with open(os.path.join(out_dir, f"summary_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n==== {tag} 5折汇总 ===="); [log(f"  {k:14s}: {v}") for k, v in summary.items()]
    return summary
