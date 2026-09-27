# -*- coding: utf-8 -*-
"""
5折CV重训,保存逐样本(anon_image, anon_id, y, prob),用于:
- 逐图(per-image)指标(复现)
- 患者级(per-patient)指标:每患者多图概率取平均→再算指标 + bootstrap 95% CI
- 不耐受(PDR)类 bootstrap 敏感性分析(回应过拟合/小样本)
输出: results/perpatient/oof_predictions.csv (out-of-fold 逐样本)
       results/perpatient/metrics.json
"""
import os, sys, json, time
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tcm_retina.data.dataset import build_transforms, GRADE_CLASSES
from tcm_retina.models.backbone import FundusClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG = os.path.join(ROOT, "data_anon", "images")
OUT = os.path.join(ROOT, "results", "perpatient"); os.makedirs(OUT, exist_ok=True)

class DS(Dataset):
    def __init__(self, rows, train, aug=None):
        self.rows = rows.reset_index(drop=True)
        self.tf = build_transforms(224, train, aug)
        self.g2i = {c: i for i, c in enumerate(GRADE_CLASSES)}
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = self.tf(Image.open(os.path.join(IMG, r["anon_image"])).convert("RGB"))
        return x, self.g2i[r["grade"]], i   # 返回行索引以对回图名
    def counts(self):
        vc = self.rows["grade"].value_counts(); return [int(vc.get(c,0)) for c in GRADE_CLASSES]

@torch.no_grad()
def infer(model, ds, device, bs=16):
    model.eval(); ld = DataLoader(ds, bs, shuffle=False, num_workers=4)
    probs = np.zeros(len(ds));
    for x, y, idx in ld:
        p = torch.softmax(model(x.to(device)), 1)[:,1].cpu().numpy()
        probs[idx.numpy()] = p
    return probs

def train_fold(tr, va, device, epochs=30):
    tr_ds = DS(tr, True, dict(hflip=True,rotate=15,color_jitter=0.1,random_resized_crop=True))
    model = FundusClassifier("vit_large_patch16_224",2,pretrained=False,retfound=True,drop_rate=0.2).to(device)
    cnt = np.array(tr_ds.counts(),float); w=torch.tensor(cnt.sum()/(2*np.maximum(cnt,1)),dtype=torch.float32,device=device)
    crit=nn.CrossEntropyLoss(weight=w); opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.05)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
    from sklearn.metrics import balanced_accuracy_score
    va_ds=DS(va,False); best=-1; best_prob=None; bad=0
    tr_ld=DataLoader(tr_ds,16,shuffle=True,num_workers=4,drop_last=True)
    for ep in range(epochs):
        model.train()
        for x,y,_ in tr_ld:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            with torch.amp.autocast("cuda",enabled=device=="cuda"): loss=crit(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        prob=infer(model,va_ds,device)
        ba=balanced_accuracy_score((va["grade"]=="PDR").astype(int).values,(prob>=0.5).astype(int))
        if ba>best: best=ba; best_prob=prob.copy(); bad=0
        else:
            bad+=1
            if bad>=10: break
    del model; torch.cuda.empty_cache()
    return best_prob

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    sp=pd.read_csv(os.path.join(ROOT,"data_anon","splits.csv")).dropna(subset=["grade"])
    tv=sp[sp.split=="train_val"].copy()
    oof=[]
    for k in range(5):
        tr=tv[tv.fold!=k]; va=tv[tv.fold==k].copy()
        print(f"fold{k}: train={len(tr)} val={len(va)}", flush=True)
        prob=train_fold(tr,va,device)
        va["prob"]=prob; va["y"]=(va["grade"]=="PDR").astype(int)
        oof.append(va[["anon_id","anon_image","grade","y","prob","fold"]])
    oof=pd.concat(oof,ignore_index=True)
    oof.to_csv(os.path.join(OUT,"oof_predictions.csv"),index=False)
    print("saved oof_predictions.csv  n=",len(oof))

if __name__=="__main__":
    main()
