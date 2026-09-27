# -*- coding: utf-8 -*-
"""引擎①训练循环:眼底图 → 单标签分类(中医证型 或 DR分级)。"""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, cohen_kappa_score, roc_auc_score,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize

from tcm_retina.data.dataset import FundusDataset, get_classes
from tcm_retina.models.backbone import FundusClassifier


def make_loaders(cfg):
    df = pd.read_csv(cfg["data"]["splits_csv"])
    task = cfg["task"]
    label_col = "tcm_syndrome" if task == "tcm_syndrome" else "grade"
    df = df.dropna(subset=[label_col])
    fold = cfg["train"]["fold"]
    tr = df[(df.split == "train_val") & (df.fold != fold)]
    va = df[(df.split == "train_val") & (df.fold == fold)]
    img_dir = cfg["data"]["image_dir"]
    sz = cfg["data"]["image_size"]
    aug = cfg.get("aug", {})
    tr_ds = FundusDataset(tr, img_dir, task, sz, train=True, aug=aug)
    va_ds = FundusDataset(va, img_dir, task, sz, train=False)
    nw = cfg["data"].get("num_workers", 4)
    bs = cfg["train"]["batch_size"]
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return tr_ld, va_ld, tr_ds


def class_weights(counts, device):
    counts = np.array(counts, dtype=np.float32)
    w = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, classes, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logit = model(x)
        prob = torch.softmax(logit, 1).cpu().numpy()
        ps.extend(prob.argmax(1)); probs.extend(prob); ys.extend(y.numpy())
    ys, ps, probs = np.array(ys), np.array(ps), np.array(probs)
    out = {
        "accuracy": accuracy_score(ys, ps),
        "balanced_acc": balanced_accuracy_score(ys, ps),
        "macro_f1": f1_score(ys, ps, average="macro"),
        "kappa": cohen_kappa_score(ys, ps),
    }
    if len(classes) == 2:
        # 二分类(不平衡):正类=索引1(如 PDR=anti-VEGF不耐受)
        from sklearn.metrics import average_precision_score, recall_score
        pos = probs[:, 1]
        try:
            out["auc"] = roc_auc_score(ys, pos)             # ROC-AUC
            out["pr_auc"] = average_precision_score(ys, pos)  # PR-AUC(不平衡更重要)
        except Exception:
            out["auc"] = out["pr_auc"] = float("nan")
        out["macro_auc"] = out["auc"]   # 兼容早停/汇总键
        out["sensitivity"] = recall_score(ys, ps, pos_label=1, zero_division=0)  # 不耐受召回
        out["specificity"] = recall_score(ys, ps, pos_label=0, zero_division=0)  # 耐受召回
    else:
        try:
            Yb = label_binarize(ys, classes=list(range(len(classes))))
            out["macro_auc"] = roc_auc_score(Yb, probs, average="macro", multi_class="ovr")
        except Exception:
            out["macro_auc"] = float("nan")
    out["_cm"] = confusion_matrix(ys, ps).tolist()
    out["_report"] = classification_report(ys, ps, target_names=classes, zero_division=0)
    out["_ys"] = ys          # 真实标签(供阈值分析)
    out["_probs"] = probs    # 预测概率
    return out


def run(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    classes = get_classes(cfg["task"])
    tr_ld, va_ld, tr_ds = make_loaders(cfg)
    print(f"[data] task={cfg['task']} fold={cfg['train']['fold']} "
          f"train={len(tr_ld.dataset)} val={len(va_ld.dataset)} "
          f"classes={classes} train_counts={tr_ds.class_counts()}")

    model = FundusClassifier(cfg["model"]["backbone"], len(classes),
                             pretrained=cfg["model"].get("pretrained", True),
                             drop_rate=cfg["model"].get("drop_rate", 0.1),
                             retfound=cfg["model"].get("retfound", False),
                             retfound_repo=cfg["model"].get("retfound_repo", "sabarimj/retfound"),
                             retfound_file=cfg["model"].get("retfound_file", "RETFound_cfp_weights.pth")
                             ).to(device)

    crit = nn.CrossEntropyLoss(
        weight=class_weights(tr_ds.class_counts(), device)
        if cfg["train"].get("class_weight") == "balanced" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = cfg["train"].get("amp", True) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = cfg["output"]["dir"]; os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")
    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    log(f"[start] task={cfg['task']} fold={cfg['train']['fold']} "
        f"backbone={cfg['model']['backbone']} epochs={epochs} device={device}")
    best_key = cfg["train"].get("monitor", "macro_f1")  # 不平衡二分类建议设 pr_auc / auc
    best_val, best_metrics = -1, None
    patience = cfg["train"].get("early_stop_patience", 8); bad = 0

    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); running = 0.0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            running += loss.item() * x.size(0)
        sched.step()
        m = evaluate(model, va_ld, classes, device)
        if len(classes) == 2:
            line = (f"ep{ep:02d} loss={running/len(tr_ld.dataset):.3f} "
                    f"auc={m['auc']:.3f} pr_auc={m['pr_auc']:.3f} "
                    f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
                    f"bacc={m['balanced_acc']:.3f} ({time.time()-t0:.0f}s)")
        else:
            line = (f"ep{ep:02d} loss={running/len(tr_ld.dataset):.3f} "
                    f"acc={m['accuracy']:.3f} bacc={m['balanced_acc']:.3f} "
                    f"f1={m['macro_f1']:.3f} auc={m['macro_auc']:.3f} "
                    f"kappa={m['kappa']:.3f} ({time.time()-t0:.0f}s)")
        log(line)
        if m[best_key] > best_val:
            best_val = m[best_key]; best_metrics = m; bad = 0
            if cfg["output"].get("save_best", True):
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
        else:
            bad += 1
            if bad >= patience:
                print(f"[early stop] no improve in {patience} epochs"); break

    # 保存结果
    fold = cfg["train"]["fold"]
    res = {k: v for k, v in best_metrics.items() if not k.startswith("_")}
    res["best_metric"] = best_key; res["fold"] = fold
    res["confusion_matrix"] = best_metrics["_cm"]; res["classes"] = classes

    # 二分类:保存验证集概率 + 阈值优化(高敏感度 / Youden)
    if len(classes) == 2:
        ys = best_metrics["_ys"]; pos = best_metrics["_probs"][:, 1]
        np.savez(os.path.join(out_dir, f"valpred_fold{fold}.npz"), ys=ys, pos=pos)
        res["threshold_analysis"] = _threshold_report(ys, pos, log)

    with open(os.path.join(out_dir, f"metrics_fold{fold}.json"),
              "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    log("\n[BEST] " + str({k: round(v, 4) for k, v in res.items()
                           if isinstance(v, float)}))
    log(best_metrics["_report"])
    return res


def _threshold_report(ys, pos, log):
    """二分类阈值分析:默认0.5 / Youden最优 / 满足敏感度>=0.8 的阈值。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(ys, pos)
    youden = thr[np.argmax(tpr - fpr)]
    out = {}
    # 找满足敏感度>=0.8的最大阈值(尽量保特异度)
    sens_target = 0.8
    ok = [t for t, s in zip(thr, tpr) if s >= sens_target]
    high_sens_thr = max(ok) if ok else 0.0
    for name, t in [("default_0.5", 0.5), ("youden", float(youden)),
                    ("sens>=0.8", float(high_sens_thr))]:
        pred = (pos >= t).astype(int)
        tp = int(((pred == 1) & (ys == 1)).sum()); fn = int(((pred == 0) & (ys == 1)).sum())
        tn = int(((pred == 0) & (ys == 0)).sum()); fp = int(((pred == 1) & (ys == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        out[name] = {"threshold": round(t, 3), "sensitivity": round(sens, 3),
                     "specificity": round(spec, 3), "TP": tp, "FN": fn, "TN": tn, "FP": fp}
    log("\n[阈值分析] (正类=不耐受/PDR)")
    for k, v in out.items():
        log(f"  {k:12s} thr={v['threshold']:.3f} sens={v['sensitivity']:.3f} "
            f"spec={v['specificity']:.3f}  TP={v['TP']} FN={v['FN']} TN={v['TN']} FP={v['FP']}")
    return out
