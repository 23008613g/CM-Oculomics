# -*- coding: utf-8 -*-
"""
硬创新A:中医证型引导的表征学习(Syndrome-Guided Representation Learning, SGRL)
============================================================
不同于普通多任务(加个证型预测头),SGRL 用【证型监督对比损失】结构化约束特征空间:
让同一中医证型的眼底特征在嵌入空间聚拢、不同证型推远,作为先验正则化主任务(耐受性预测)。
对比 baseline(无SGRL) vs SGRL,看是否提升主任务 + 是否让特征更有中医语义。
诚实:小数据,不保证有效,如实报。
"""
import os, json, time
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, average_precision_score

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tcm_retina.data.dataset import build_transforms, GRADE_CLASSES, SYNDROME_CLASSES
import timm

class DS(Dataset):
    def __init__(self, rows, image_dir, sz, train, aug=None):
        self.rows=rows.reset_index(drop=True); self.image_dir=image_dir
        self.tf=build_transforms(sz,train,aug)
        self.g2i={c:i for i,c in enumerate(GRADE_CLASSES)}; self.s2i={c:i for i,c in enumerate(SYNDROME_CLASSES)}
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i]
        x=self.tf(Image.open(os.path.join(self.image_dir,r["anon_image"])).convert("RGB"))
        return x, self.g2i[r["grade"]], self.s2i.get(r["tcm_syndrome"],-1)
    def counts(self):
        vc=self.rows["grade"].value_counts(); return [int(vc.get(c,0)) for c in GRADE_CLASSES]

class SGRLNet(nn.Module):
    def __init__(self, retfound=True, drop=0.2, proj=128):
        super().__init__()
        self.backbone=timm.create_model("vit_large_patch16_224",pretrained=False,num_classes=0,drop_rate=drop)
        if retfound:
            from huggingface_hub import hf_hub_download
            sd=torch.load(hf_hub_download("sabarimj/retfound","RETFound_cfp_weights.pth"),map_location="cpu",weights_only=False)
            sd=sd.get("model",sd); own=self.backbone.state_dict()
            mt={k:v for k,v in sd.items() if not k.startswith("decoder") and k!="mask_token" and k in own and own[k].shape==v.shape}
            own.update(mt); self.backbone.load_state_dict(own,strict=False); print(f"[RETFound] {len(mt)}层")
        d=self.backbone.num_features
        self.norm=nn.LayerNorm(d); self.drop=nn.Dropout(drop)
        self.head=nn.Linear(d,2)                    # 主任务:耐受性
        self.proj=nn.Sequential(nn.Linear(d,proj),nn.ReLU(),nn.Linear(proj,proj))  # 对比投影头
    def forward(self,x):
        f=self.drop(self.norm(self.backbone(x)))
        return self.head(f), F.normalize(self.proj(f),dim=1)

def supcon_loss(emb, labels, temp=0.1):
    """监督对比损失(证型作标签);忽略label=-1。"""
    valid = labels>=0
    if valid.sum()<2: return torch.tensor(0.0, device=emb.device)
    emb=emb[valid]; labels=labels[valid]
    sim=torch.mm(emb,emb.t())/temp
    sim=sim - sim.max(1,keepdim=True)[0].detach()
    exp=torch.exp(sim)
    mask=(labels.unsqueeze(0)==labels.unsqueeze(1)).float()
    mask.fill_diagonal_(0)
    denom=exp.sum(1,keepdim=True)-torch.exp(torch.zeros_like(sim.diag())).unsqueeze(1)
    log_prob=sim - torch.log(exp.sum(1,keepdim=True)+1e-9)
    mp=(mask*log_prob).sum(1)/(mask.sum(1)+1e-9)
    return -mp.mean()

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); ys,ps,pos=[],[],[]
    for x,yt,_ in loader:
        x=x.to(device); logit,_=model(x); p=torch.softmax(logit,1).cpu().numpy()
        ps.extend(p.argmax(1)); pos.extend(p[:,1]); ys.extend(yt.numpy())
    ys,ps,pos=np.array(ys),np.array(ps),np.array(pos)
    return {"auc":roc_auc_score(ys,pos),"pr_auc":average_precision_score(ys,pos),
            "balanced_acc":balanced_accuracy_score(ys,ps),
            "sensitivity":recall_score(ys,ps,pos_label=1,zero_division=0),
            "specificity":recall_score(ys,ps,pos_label=0,zero_division=0),"_ys":ys,"_pos":pos}

def run_fold(cfg,fold,scl,log):
    device="cuda" if torch.cuda.is_available() else "cpu"
    df=pd.read_csv(cfg["splits"]).dropna(subset=["grade"])
    tr=df[(df.split=="train_val")&(df.fold!=fold)]; va=df[(df.split=="train_val")&(df.fold==fold)]
    tr_ds=DS(tr,cfg["img"],224,True,cfg.get("aug",{})); va_ds=DS(va,cfg["img"],224,False)
    tr_ld=DataLoader(tr_ds,cfg["bs"],shuffle=True,num_workers=4,drop_last=True)
    va_ld=DataLoader(va_ds,cfg["bs"],shuffle=False,num_workers=4)
    model=SGRLNet(retfound=True,drop=0.2).to(device)
    cnt=np.array(tr_ds.counts(),float); w=torch.tensor(cnt.sum()/(2*np.maximum(cnt,1)),dtype=torch.float32,device=device)
    crit=nn.CrossEntropyLoss(weight=w)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=0.05)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["epochs"])
    scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
    best=-1; bestm=None; bad=0
    for ep in range(1,cfg["epochs"]+1):
        model.train(); t0=time.time()
        for x,yt,ys in tr_ld:
            x,yt,ys=x.to(device),yt.to(device),ys.to(device); opt.zero_grad()
            with torch.amp.autocast("cuda",enabled=device=="cuda"):
                logit,emb=model(x); loss=crit(logit,yt)+scl*supcon_loss(emb,ys)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step(); m=evaluate(model,va_ld,device)
        if m["balanced_acc"]>best: best,bestm,bad=m["balanced_acc"],m,0
        else:
            bad+=1
            if bad>=cfg.get("patience",10): break
    log(f"  [scl={scl} f{fold}] auc={bestm['auc']:.3f} bacc={bestm['balanced_acc']:.3f} sens={bestm['sensitivity']:.3f} ({time.time()-t0:.0f}s/ep)")
    return bestm

def run(cfg,scl,tag):
    out=cfg["out"]; os.makedirs(out,exist_ok=True)
    def log(m):
        print(m,flush=True); open(os.path.join(out,f"train_{tag}.log"),"a",encoding="utf-8").write(m+"\n")
    log(f"==== SGRL tag={tag} supcon_scale={scl} ====")
    res=[]
    for f in range(5):
        b=run_fold(cfg,f,scl,log); res.append(b)
        np.savez(os.path.join(out,f"valpred_{tag}_fold{f}.npz"),ys=b["_ys"],pos=b["_pos"])
    keys=["auc","pr_auc","balanced_acc","sensitivity","specificity"]
    summ={k:f"{np.mean([r[k] for r in res]):.4f} ± {np.std([r[k] for r in res]):.4f}" for k in keys}
    json.dump(summ,open(os.path.join(out,f"summary_{tag}.json"),"w"),ensure_ascii=False,indent=2)
    log("\n=== "+tag+" 5折 ==="); [log(f"  {k}: {v}") for k,v in summ.items()]
    return summ

if __name__=="__main__":
    cfg=dict(splits="data_anon/splits.csv",img="data_anon/images",
             bs=16,lr=5e-5,epochs=35,patience=10,
             aug=dict(hflip=True,rotate=15,color_jitter=0.1,random_resized_crop=True),
             out="results/sgrl_antivegf")
    import sys
    smoke="--smoke" in sys.argv
    if smoke: cfg["epochs"]=1
    # baseline(无对比) vs SGRL(有证型对比)
    if "--sgrl-only" in sys.argv:
        run(cfg,0.5,"sgrl")
    else:
        run(cfg,0.0,"baseline"); run(cfg,0.5,"sgrl")
