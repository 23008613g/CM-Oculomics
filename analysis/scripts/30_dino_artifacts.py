# -*- coding: utf-8 -*-
"""
30_dino_artifacts.py — train ONE DINOv2 (ViT-L/14 @224) deploy model and produce the
model-dependent artifacts the OOF predictions cannot give us (Option A, brief STEP 1b):
  * dino_deploy.pth                          (the saved model -> reproducibility/Zenodo)
  * uncertainty_metrics_dinov2.json          (MC-dropout, mirrors 09)  -> Fig7 panel
  * dino_tsne_emb.npz  (X feats, y)          (mirrors 24)              -> Fig13a
  * results/figures/gradcam_dino/*.png       (mirrors 05, 16x16 grid)  -> Fig5

Self-contained (own DS + train loop, num_workers=0) so Windows `spawn` workers re-import
this module cleanly. Hyper-parameters IDENTICAL to the comparison protocol:
eff. batch 16, 30 epochs, AdamW wd=0.05, cosine, AMP, class-weighted CE, balanced-acc
early stop (patience 10). best_lr=1e-5 (from the experiment sweep). Trains on fold!=0,
early-stops on fold0 — same recipe that produced RETFound's deploy model.
"""
import os, sys, json, math
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from tcm_retina.data.dataset import build_transforms, GRADE_CLASSES

IMG = os.path.join(ROOT, "data_anon", "images")
DATA = os.path.join(ROOT, "data_anon")
OUT = os.path.join(ROOT, "results", "dino_experiment")
GCAM = os.path.join(ROOT, "results", "figures", "gradcam_dino"); os.makedirs(GCAM, exist_ok=True)
CKPT_OUT = os.path.join(OUT, "dino_deploy.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES, EFF_BS, EPOCHS, PATIENCE, LR = 224, 16, 30, 10, 1e-5
g2i = {c: i for i, c in enumerate(GRADE_CLASSES)}


def make_model():
    backbone = timm.create_model("vit_large_patch14_dinov2", pretrained=True,
                                 num_classes=0, img_size=RES, drop_rate=0.2)
    head = nn.Sequential(nn.LayerNorm(backbone.num_features), nn.Dropout(0.2),
                         nn.Linear(backbone.num_features, 2))
    return nn.Sequential(backbone, head)


class DS(Dataset):
    def __init__(self, rows, train):
        self.rows = rows.reset_index(drop=True)
        self.tf = build_transforms(RES, train,
                                   dict(hflip=True, rotate=15, color_jitter=0.1,
                                        random_resized_crop=True) if train else None)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = self.tf(Image.open(os.path.join(IMG, r["anon_image"])).convert("RGB"))
        return x, g2i[r["grade"]], i


@torch.no_grad()
def infer(model, ds):
    model.eval(); ld = DataLoader(ds, EFF_BS, shuffle=False, num_workers=0)
    probs = np.zeros(len(ds))
    for x, y, idx in ld:
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            p = torch.softmax(model(x.to(DEVICE)), 1)[:, 1].float().cpu().numpy()
        probs[idx.numpy()] = p
    return probs


def train_deploy():
    sp = pd.read_csv(os.path.join(DATA, "splits.csv")).dropna(subset=["grade"])
    tv = sp[sp.split == "train_val"].copy()
    tr, va = tv[tv.fold != 0].copy(), tv[tv.fold == 0].copy()
    tr_ds, va_ds = DS(tr, True), DS(va, False)
    yv = (va["grade"] == "PDR").astype(int).values
    model = make_model().to(DEVICE)
    cnt = np.array([int((tr["grade"] == c).sum()) for c in GRADE_CLASSES], float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    tr_ld = DataLoader(tr_ds, EFF_BS, shuffle=True, num_workers=0, drop_last=True)
    best, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        for x, y, _ in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        sch.step()
        prob = infer(model, va_ds); ba = balanced_accuracy_score(yv, (prob >= 0.5).astype(int))
        auc = roc_auc_score(yv, prob)
        print(f"  epoch {ep+1:02d}  val bal-acc={ba:.3f}  val AUC={auc:.3f}", flush=True)
        if ba > best:
            best, bad = ba, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: print("  early stop", flush=True); break
    model.load_state_dict(best_state)
    torch.save(best_state, CKPT_OUT)
    print(f"[deploy] saved {CKPT_OUT}  (val bal-acc={best:.3f})", flush=True)
    return model


# ---------------- MC-dropout (mirror 09) ----------------
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"): m.train()


def ece_score(p, y, n=10):
    bins = np.linspace(0, 1, n + 1); idx = np.clip(np.digitize(p, bins) - 1, 0, n - 1); e = 0.0
    for b in range(n):
        m = idx == b
        if m.sum(): e += m.sum() / len(y) * abs((y[m] == 1).mean() - p[m].mean())
    return e


@torch.no_grad()
def mc_dropout(model, T=30):
    sp = pd.read_csv(os.path.join(DATA, "splits.csv"))
    test = sp[sp.split == "test"].dropna(subset=["grade"]).reset_index(drop=True)
    ds = DS(test, False); ld = DataLoader(ds, EFF_BS, shuffle=False, num_workers=0)
    ys = test["grade"].map(g2i).values
    model.eval(); enable_dropout(model)
    runs = []
    for _ in range(T):
        ps = np.zeros(len(ds))
        for x, y, idx in ld:
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                p = torch.softmax(model(x.to(DEVICE)), 1)[:, 1].float().cpu().numpy()
            ps[idx.numpy()] = p
        runs.append(ps)
    runs = np.array(runs); mean_p = runs.mean(0); std_p = runs.std(0)
    auc = roc_auc_score(ys, mean_p); ece = ece_score(mean_p, ys)
    order = np.argsort(std_p); accs = []
    for kf in np.linspace(0.1, 1.0, 10):
        sel = order[:max(1, int(len(order) * kf))]
        accs.append(balanced_accuracy_score(ys[sel], (mean_p[sel] >= 0.5).astype(int)))
    out = {"mc_dropout_T": T, "auc": round(float(auc), 4),
           "mean_uncertainty_std": round(float(std_p.mean()), 4), "ECE": round(float(ece), 4),
           "selective_bacc_at_50pct_coverage": round(float(accs[4]), 4),
           "bacc_full_coverage": round(float(accs[-1]), 4), "n_test": int(len(ys))}
    json.dump(out, open(os.path.join(OUT, "uncertainty_metrics_dinov2.json"), "w"), indent=2)
    print("[mc-dropout]", json.dumps(out), flush=True)


# ---------------- t-SNE embeddings (mirror 24) ----------------
@torch.no_grad()
def extract_embeddings(model):
    model.eval()
    spl = pd.read_csv(os.path.join(DATA, "splits.csv")).dropna(subset=["grade"])
    pdr = spl[spl.grade == "PDR"]
    npdr = spl[spl.grade == "NPDR"].sample(min(300, int((spl.grade == "NPDR").sum())), random_state=42)
    sub = pd.concat([pdr, npdr])
    ds = DS(sub.assign(), False); ld = DataLoader(ds, EFF_BS, shuffle=False, num_workers=0)
    labs = (sub["grade"].values == "PDR").astype(int)
    feats = np.zeros((len(ds), model[0].num_features), np.float32)
    for x, y, idx in ld:
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            f = model[0](x.to(DEVICE)).float().cpu().numpy()
        feats[idx.numpy()] = f
    np.savez(os.path.join(OUT, "dino_tsne_emb.npz"), X=feats, y=labs)
    print(f"[t-SNE] saved embeddings {feats.shape}", flush=True)


# ---------------- Grad-CAM (mirror 05, 16x16 grid) ----------------
def grad_cam(model):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    side = RES // 14  # 16

    def reshape_transform(t, s=side):
        x = t[:, -s * s:, :]  # robust to any prefix tokens (cls/registers)
        return x.reshape(t.size(0), s, s, t.size(2)).permute(0, 3, 1, 2)

    model.eval()
    target_layers = [model[0].blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    tf = build_transforms(RES, train=False)
    sp = pd.read_csv(os.path.join(DATA, "splits.csv"))
    test = sp[sp.split == "test"].dropna(subset=["grade"])
    avail = set(test["anon_image"])
    pdr_pref = [f for f in ["P0030_OD_01.jpg", "P0144_OD_01.jpg", "P0287_OD_01.jpg", "P0433_OD_01.jpg"] if f in avail]
    pdr_rest = [f for f in test[test.grade == "PDR"]["anon_image"].tolist() if f not in pdr_pref]
    picks = {"PDR": (pdr_pref + pdr_rest)[:8], "NPDR": test[test.grade == "NPDR"]["anon_image"].head(8).tolist()}
    for grade, imgs in picks.items():
        cls_idx = GRADE_CLASSES.index(grade)
        for fn in imgs:
            pil = Image.open(os.path.join(DATA, "images", fn)).convert("RGB").resize((RES, RES))
            rgb = np.array(pil).astype(np.float32) / 255.0
            x = tf(pil).unsqueeze(0).to(DEVICE)
            gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls_idx)])[0]
            vis = show_cam_on_image(rgb, gray, use_rgb=True)
            Image.fromarray(vis).save(os.path.join(GCAM, f"cam_{grade}_{fn}.png"))
            Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(GCAM, f"orig_{grade}_{fn}.png"))
    print(f"[grad-cam] saved DINOv2 CAMs to {GCAM}", flush=True)


def main():
    print(f"device={DEVICE}  res={RES}  lr={LR}", flush=True)
    model = train_deploy()
    mc_dropout(model)
    extract_embeddings(model)
    grad_cam(model)
    print("[ALL DONE] DINOv2 deploy artifacts ready.", flush=True)


if __name__ == "__main__":
    main()
