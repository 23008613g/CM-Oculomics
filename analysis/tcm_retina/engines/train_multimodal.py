# -*- coding: utf-8 -*-
"""多模态训练:眼底图(+临床指标)(+中医证型)→ anti-VEGF 耐受性。
支持消融:--mode image / image_clinical / image_clinical_syndrome"""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tcm_retina.data.multimodal_dataset import MultimodalDataset
from tcm_retina.data.dataset import GRADE_CLASSES
from tcm_retina.models.multimodal import MultimodalNet
from tcm_retina.engines.train_classify import _threshold_report  # 复用阈值分析


def _metrics(ys, ps, probs):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 roc_auc_score, average_precision_score, recall_score,
                                 confusion_matrix)
    pos = probs[:, 1]
    out = {
        "accuracy": accuracy_score(ys, ps),
        "balanced_acc": balanced_accuracy_score(ys, ps),
        "macro_f1": f1_score(ys, ps, average="macro"),
        "auc": roc_auc_score(ys, pos),
        "pr_auc": average_precision_score(ys, pos),
        "sensitivity": recall_score(ys, ps, pos_label=1, zero_division=0),
        "specificity": recall_score(ys, ps, pos_label=0, zero_division=0),
    }
    out["_cm"] = confusion_matrix(ys, ps).tolist()
    out["_ys"] = ys; out["_pos"] = pos
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, ps, probs = [], [], []
    for img, tab, y in loader:
        img, tab = img.to(device), tab.to(device)
        logit = model(img, tab)
        prob = torch.softmax(logit, 1).cpu().numpy()
        ps.extend(prob.argmax(1)); probs.extend(prob); ys.extend(y.numpy())
    return _metrics(np.array(ys), np.array(ps), np.array(probs))


def run_fold(cfg, fold, mode, log):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_clinical = mode in ("image_clinical", "image_clinical_syndrome")
    use_syndrome = mode == "image_clinical_syndrome"

    df = pd.read_csv(cfg["data"]["splits_csv"]).dropna(subset=["grade"])
    pts = pd.read_csv(cfg["data"]["patients_csv"])
    tr = df[(df.split == "train_val") & (df.fold != fold)]
    va = df[(df.split == "train_val") & (df.fold == fold)]

    norm = MultimodalDataset.compute_norm(pts) if use_clinical else None
    img_dir, sz, aug = cfg["data"]["image_dir"], cfg["data"]["image_size"], cfg.get("aug", {})
    tr_ds = MultimodalDataset(tr, img_dir, pts, sz, True, use_clinical, use_syndrome, aug, norm)
    va_ds = MultimodalDataset(va, img_dir, pts, sz, False, use_clinical, use_syndrome, None, norm)
    bs, nw = cfg["train"]["batch_size"], cfg["data"].get("num_workers", 4)
    tr_ld = DataLoader(tr_ds, bs, shuffle=True, num_workers=nw, drop_last=True)
    va_ld = DataLoader(va_ds, bs, shuffle=False, num_workers=nw)

    fusion_type = cfg["model"].get("fusion", "concat")
    if fusion_type == "gated":
        from tcm_retina.models.multimodal import GatedFusionNet
        model = GatedFusionNet(num_classes=2, tab_dim=tr_ds.tab_dim(),
                               retfound=cfg["model"].get("retfound", True),
                               drop_rate=cfg["model"].get("drop_rate", 0.2)).to(device)
    else:
        model = MultimodalNet(num_classes=2, tab_dim=tr_ds.tab_dim(),
                              retfound=cfg["model"].get("retfound", True),
                              drop_rate=cfg["model"].get("drop_rate", 0.2)).to(device)

    counts = np.array(tr_ds.class_counts(), float)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_val, best = -1, None
    monitor, patience, bad = cfg["train"].get("monitor", "balanced_acc"), cfg["train"].get("early_stop_patience", 12), 0
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); run = 0
        for img, tab, y in tr_ld:
            img, tab, y = img.to(device), tab.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = crit(model(img, tab), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run += loss.item()
        sched.step()
        m = evaluate(model, va_ld, device)
        log(f"  [{mode} f{fold}] ep{ep:02d} loss={run/len(tr_ld):.3f} auc={m['auc']:.3f} "
            f"pr={m['pr_auc']:.3f} sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
            f"bacc={m['balanced_acc']:.3f} ({time.time()-t0:.0f}s)")
        if m[monitor] > best_val:
            best_val, best, bad = m[monitor], m, 0
        else:
            bad += 1
            if bad >= patience: break
    return best


def run(cfg, mode):
    out_dir = cfg["output"]["dir"]; os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"train_{mode}.log")
    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh: fh.write(msg + "\n")

    log(f"==== 多模态消融: mode={mode} ====")
    fold_res = []
    for fold in range(5):
        b = run_fold(cfg, fold, mode, log)
        fold_res.append(b)
        # 保存逐样本验证集概率(供 DeLong 配对检验)
        np.savez(os.path.join(out_dir, f"valpred_{mode}_fold{fold}.npz"),
                 ys=b["_ys"], pos=b["_pos"])
        log(f"  >> fold{fold} best: auc={b['auc']:.3f} bacc={b['balanced_acc']:.3f} "
            f"sens={b['sensitivity']:.3f} spec={b['specificity']:.3f}")

    keys = ["auc", "pr_auc", "balanced_acc", "sensitivity", "specificity", "macro_f1"]
    summary = {k: f"{np.mean([r[k] for r in fold_res]):.4f} ± {np.std([r[k] for r in fold_res]):.4f}"
               for k in keys}
    with open(os.path.join(out_dir, f"summary_{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n==== {mode} 5折汇总 ====")
    for k, v in summary.items(): log(f"  {k:14s}: {v}")
    return summary
