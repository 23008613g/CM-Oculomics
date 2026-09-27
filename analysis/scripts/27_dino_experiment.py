# -*- coding: utf-8 -*-
"""
DINOv2 (and RETFound re-check) backbone comparison — mirrors train_perpatient.py
EXACTLY (same folds from splits.csv, class-weighted CE, AdamW wd=0.05, cosine,
AMP, balanced-acc early stop patience 10, 30 epochs, effective batch size 16).
The ONLY things that change per run: the backbone encoder, its input resolution
(patch-valid), and the independently-tuned learning rate. Effective batch size is
held at 16 across all backbones via gradient accumulation when VRAM is tight.

Usage:
  python 27_dino_experiment.py dryrun --backbone dinov2_l --res 224
  python 27_dino_experiment.py dryrun --backbone dinov2_l --res 518 --bs 2
  python 27_dino_experiment.py sweep  --backbone dinov2_l --res 224          # LR sweep on fold0
  python 27_dino_experiment.py cv     --backbone dinov2_l --res 224 --lr 5e-5
  python 27_dino_experiment.py labeleff --backbone dinov2_l --res 224 --lr 5e-5
  python 27_dino_experiment.py external --backbone dinov2_l --res 224 --lr 5e-5
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tcm_retina.data.dataset import build_transforms, GRADE_CLASSES

IMG = os.path.join(ROOT, "data_anon", "images")
DDR = os.path.join(ROOT, "external_data", "ddr")
OUT = os.path.join(ROOT, "results", "dino_experiment"); os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EFF_BS = 16          # effective batch size, held constant across ALL backbones
EPOCHS = 30
PATIENCE = 10

BACKBONES = {
    "retfound":  dict(timm="vit_large_patch16_224", retfound=True,  res=224),
    "dinov2_l":  dict(timm="vit_large_patch14_dinov2", retfound=False, res=224),
    "dinov2_b":  dict(timm="vit_base_patch14_dinov2",  retfound=False, res=224),
    # Comparison baselines, restored here so they can be retrained alongside the others.
    # Configurations taken from the original runs recorded in results/backbone_*/train.log.
    # img_size=False marks models that do not accept an img_size override.
    "swinv2":    dict(timm="swinv2_tiny_window8_256", retfound=False, res=256, img_size=False),
    "vit_b":     dict(timm="vit_base_patch16_224",    retfound=False, res=224),
    "resnet50":  dict(timm="resnet50",                retfound=False, res=224, img_size=False),
}


def make_model(kind, res):
    spec = BACKBONES[kind]
    if spec["retfound"]:
        from tcm_retina.models.backbone import FundusClassifier
        return FundusClassifier(spec["timm"], 2, pretrained=False, retfound=True, drop_rate=0.2)
    # DINOv2 (or other timm) — pretrained generalist weights, img_size override
    kw = dict(pretrained=True, num_classes=0, drop_rate=0.2)
    if spec.get("img_size", True):
        kw["img_size"] = res          # convnets and fixed-window transformers reject this
    backbone = timm.create_model(spec["timm"], **kw)
    feat = backbone.num_features
    head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(0.2), nn.Linear(feat, 2))
    return nn.Sequential(backbone, head)


class DS(Dataset):
    def __init__(self, rows, train, res):
        self.rows = rows.reset_index(drop=True)
        self.tf = build_transforms(res, train,
                                   dict(hflip=True, rotate=15, color_jitter=0.1,
                                        random_resized_crop=True) if train else None)
        self.g2i = {c: i for i, c in enumerate(GRADE_CLASSES)}
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = self.tf(Image.open(os.path.join(IMG, r["anon_image"])).convert("RGB"))
        return x, self.g2i[r["grade"]], i
    def counts(self):
        vc = self.rows["grade"].value_counts(); return [int(vc.get(c, 0)) for c in GRADE_CLASSES]


@torch.no_grad()
def infer(model, ds, res, bs=16):
    model.eval(); ld = DataLoader(ds, bs, shuffle=False, num_workers=4)
    probs = np.zeros(len(ds))
    for x, y, idx in ld:
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            p = torch.softmax(model(x.to(DEVICE)), 1)[:, 1].float().cpu().numpy()
        probs[idx.numpy()] = p
    return probs


def train_fold(tr, va, kind, res, lr, per_step_bs, return_model=False):
    from sklearn.metrics import balanced_accuracy_score
    accum = max(1, EFF_BS // per_step_bs)
    tr_ds = DS(tr, True, res); va_ds = DS(va, False, res)
    model = make_model(kind, res).to(DEVICE)
    cnt = np.array(tr_ds.counts(), float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    tr_ld = DataLoader(tr_ds, per_step_bs, shuffle=True, num_workers=4, drop_last=True)
    yv = (va["grade"] == "PDR").astype(int).values
    best, best_prob, bad = -1, None, 0
    peak = 0
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        for bi, (x, y, _) in enumerate(tr_ld):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = crit(model(x), y) / accum
            scaler.scale(loss).backward()
            if (bi + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
        sch.step()
        if DEVICE == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        prob = infer(model, va_ds, res)
        ba = balanced_accuracy_score(yv, (prob >= 0.5).astype(int))
        if ba > best:
            best, best_prob, bad = ba, prob.copy(), 0
            if return_model: best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    if return_model:
        model.load_state_dict(best_state)
        return best_prob, peak, model
    del model; torch.cuda.empty_cache()
    return best_prob, peak


def load_tv():
    sp = pd.read_csv(os.path.join(ROOT, "data_anon", "splits.csv")).dropna(subset=["grade"])
    return sp[sp.split == "train_val"].copy()


class DDRDS(Dataset):
    def __init__(self, lab, res):
        self.lab = lab.reset_index(drop=True); self.tf = build_transforms(res, False, None)
    def __len__(self): return len(self.lab)
    def __getitem__(self, i):
        r = self.lab.iloc[i]
        x = self.tf(Image.open(os.path.join(DDR, "images", r["image"])).convert("RGB"))
        return x, int(r["ddr_grade"] == 4), i


@torch.no_grad()
def eval_ddr(model, res, bs=16):
    lab = pd.read_csv(os.path.join(DDR, "subset_labels.csv"))
    ds = DDRDS(lab, res); ld = DataLoader(ds, bs, shuffle=False, num_workers=4)
    ys = np.zeros(len(ds)); ps = np.zeros(len(ds))
    model.eval()
    for x, y, idx in ld:
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            p = torch.softmax(model(x.to(DEVICE)), 1)[:, 1].float().cpu().numpy()
        ps[idx.numpy()] = p; ys[idx.numpy()] = y.numpy()
    return ys, ps


def run_cv(kind, res, lr, per_step_bs, frac=1.0, tag=""):
    tv = load_tv(); oof = []; peaks = []
    rng = np.random.default_rng(42)
    for k in range(5):
        tr = tv[tv.fold != k].copy(); va = tv[tv.fold == k].copy()
        if frac < 1.0:  # label-efficiency: subsample training rows (stratified by grade), SAME seed
            keep = []
            for g, grp in tr.groupby("grade"):
                n = max(1, int(round(len(grp) * frac)))
                keep.append(grp.sample(n, random_state=42))
            tr = pd.concat(keep)
        prob, peak = train_fold(tr, va, kind, res, lr, per_step_bs)
        peaks.append(peak)
        va = va.copy(); va["prob"] = prob; va["y"] = (va["grade"] == "PDR").astype(int)
        oof.append(va[["anon_id", "anon_image", "grade", "y", "prob", "fold"]])
        print(f"  [{tag}] fold{k} done  peakVRAM={peak:.1f}GB", flush=True)
    oof = pd.concat(oof, ignore_index=True)
    fn = os.path.join(OUT, f"oof_{kind}_res{res}_frac{frac}{tag}.csv")
    oof.to_csv(fn, index=False)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(oof["y"], oof["prob"])
    print(f"  [{tag}] pooled OOF AUC={auc:.4f}  saved {os.path.basename(fn)}  peakVRAM~{max(peaks):.1f}GB", flush=True)
    return fn, auc, max(peaks)


def dryrun(kind, res, per_step_bs):
    """Build model + 2 training steps; report peak VRAM and accum needed."""
    tv = load_tv(); tr = tv[tv.fold != 0].head(per_step_bs * 3).copy()
    accum = max(1, EFF_BS // per_step_bs)
    t0 = time.time()
    ds = DS(tr, True, res); model = make_model(kind, res).to(DEVICE)
    crit = nn.CrossEntropyLoss(); opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    ld = DataLoader(ds, per_step_bs, shuffle=True, num_workers=2)
    model.train()
    for bi, (x, y, _) in enumerate(ld):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        if bi >= 1: break
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"DRYRUN {kind} @res{res} per_step_bs={per_step_bs} accum={accum} "
          f"-> peakVRAM={peak:.1f}GB  params={nparam:.0f}M  {time.time()-t0:.1f}s")
    del model; torch.cuda.empty_cache()
    return peak


def run_standard(kind, res, per_step_bs):
    """Full Standard scope: LR sweep (fold0) -> 5-fold CV -> label-efficiency -> external DDR."""
    from sklearn.metrics import roc_auc_score
    t0 = time.time(); summary = {"backbone": kind, "res": res}
    tv = load_tv()
    # 1) LR sweep on fold0
    tr0, va0 = tv[tv.fold != 0], tv[tv.fold == 0]
    y0 = (va0["grade"] == "PDR").astype(int).values
    sweep = {}
    for lr in [1e-5, 5e-5, 1e-4]:
        prob, _ = train_fold(tr0, va0, kind, res, lr, per_step_bs)
        sweep[lr] = float(roc_auc_score(y0, prob))
        print(f"[sweep] {kind}@{res} lr={lr:.0e} fold0 AUC={sweep[lr]:.4f}", flush=True)
    best_lr = max(sweep, key=sweep.get)
    summary["sweep"] = {f"{k:.0e}": v for k, v in sweep.items()}; summary["best_lr"] = f"{best_lr:.0e}"
    print(f"[sweep] BEST lr={best_lr:.0e}", flush=True)
    # 2) full 5-fold CV
    fn, auc, peak = run_cv(kind, res, best_lr, per_step_bs, tag="_main")
    summary["cv_oof_csv"] = os.path.basename(fn); summary["cv_pooled_auc"] = auc; summary["peak_vram_gb"] = peak
    # 3) label-efficiency
    le = {}
    for frac in [0.10, 0.25, 0.50, 1.0]:
        f2, a2, _ = run_cv(kind, res, best_lr, per_step_bs, frac=frac, tag="_labeleff")
        le[str(frac)] = {"csv": os.path.basename(f2), "pooled_auc": a2}
    summary["label_efficiency"] = le
    # 4) external DDR (train one model on fold!=0, early-stop on fold0, score DDR)
    _bp, _pk, model = train_fold(tv[tv.fold != 0], tv[tv.fold == 0], kind, res, best_lr, per_step_bs, return_model=True)
    ys, ps = eval_ddr(model, res)
    np.savez(os.path.join(OUT, f"ddr_{kind}_res{res}.npz"), ys=ys, pos=ps)
    summary["ddr_auc"] = float(roc_auc_score(ys, ps))
    del model; torch.cuda.empty_cache()
    summary["runtime_min"] = round((time.time() - t0) / 60, 1)
    json.dump(summary, open(os.path.join(OUT, f"summary_{kind}_res{res}.json"), "w"), indent=2)
    print(f"[STANDARD DONE] {kind}@{res} best_lr={best_lr:.0e} cvAUC={auc:.4f} ddrAUC={summary['ddr_auc']:.4f} "
          f"runtime={summary['runtime_min']}min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dryrun", "sweep", "cv", "labeleff", "external", "standard"])
    ap.add_argument("--backbone", default="dinov2_l")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=16)   # per-step batch size
    args = ap.parse_args()

    if args.mode == "standard":
        run_standard(args.backbone, args.res, args.bs)
    elif args.mode == "dryrun":
        dryrun(args.backbone, args.res, args.bs)
    elif args.mode == "sweep":
        tv = load_tv(); tr = tv[tv.fold != 0]; va = tv[tv.fold == 0]
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score
        res = {}
        for lr in [1e-5, 5e-5, 1e-4]:
            prob, peak = train_fold(tr, va, args.backbone, args.res, lr, args.bs)
            y = (va["grade"] == "PDR").astype(int).values
            res[lr] = roc_auc_score(y, prob)
            print(f"SWEEP {args.backbone}@{args.res} lr={lr:.0e} fold0 AUC={res[lr]:.4f}", flush=True)
        best = max(res, key=res.get)
        print(f"SWEEP BEST lr={best:.0e} (AUC={res[best]:.4f})")
        json.dump({str(k): v for k, v in res.items()} | {"best_lr": best},
                  open(os.path.join(OUT, f"sweep_{args.backbone}_res{args.res}.json"), "w"), indent=2)
    elif args.mode == "cv":
        run_cv(args.backbone, args.res, args.lr, args.bs)
    elif args.mode == "labeleff":
        for frac in [0.10, 0.25, 0.50, 1.0]:
            run_cv(args.backbone, args.res, args.lr, args.bs, frac=frac, tag="_labeleff")
    elif args.mode == "external":
        # train one model on full train_val, eval zero-fine-tuning on DDR
        from sklearn.metrics import roc_auc_score
        tv = load_tv()
        # use fold!=0 as train (mirror that RETFound external used a single trained model)
        prob, peak = None, None
        model = make_model(args.backbone, args.res).to(DEVICE)
        # quick full-train (reuse train_fold by treating all tv as train, fold0 as val for early stop)
        tr = tv[tv.fold != 0]; va = tv[tv.fold == 0]
        _bp, _pk = train_fold(tr, va, args.backbone, args.res, args.lr, args.bs)
        print("external mode: retrain done; (DDR scoring to be added)", flush=True)
