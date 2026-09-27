# -*- coding: utf-8 -*-
"""
数据增强消融:用【公开DDR数据】扩充训练集,缓解小样本(尤其PDR只有95张)。
对比:
  - 'inhouse'  : 仅用本院数据训练(5折,与主结果同口径)
  - 'augmented': 本院训练折 + DDR外部数据 一起训练
两者都【只在本院验证折】上评估(保证公平,评估集不含DDR)。
看加入公开数据是否提升AUC稳健性 / 外部泛化。
"""
import os, json, time
import numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tcm_retina.data.dataset import build_transforms, GRADE_CLASSES
from tcm_retina.models.backbone import FundusClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INHOUSE_IMG = os.path.join(ROOT, "data_anon", "images")
DDR_IMG = os.path.join(ROOT, "external_data", "ddr", "train_images")

class MixDS(Dataset):
    """rows: DataFrame with columns [path, label('NPDR'/'PDR')]"""
    def __init__(self, rows, sz, train, aug=None):
        self.rows=rows.reset_index(drop=True); self.tf=build_transforms(sz,train,aug)
        self.g2i={c:i for i,c in enumerate(GRADE_CLASSES)}
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i]
        x=self.tf(Image.open(r["path"]).convert("RGB"))
        return x, self.g2i[r["label"]]
    def counts(self):
        vc=self.rows["label"].value_counts(); return [int(vc.get(c,0)) for c in GRADE_CLASSES]

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); ys,ps,pos=[],[],[]
    for x,y in loader:
        x=x.to(device); p=torch.softmax(model(x),1).cpu().numpy()
        ps.extend(p.argmax(1)); pos.extend(p[:,1]); ys.extend(y.numpy())
    ys,ps,pos=np.array(ys),np.array(ps),np.array(pos)
    return {"auc":roc_auc_score(ys,pos),"pr_auc":average_precision_score(ys,pos),
            "balanced_acc":balanced_accuracy_score(ys,ps),
            "sensitivity":recall_score(ys,ps,pos_label=1,zero_division=0),
            "specificity":recall_score(ys,ps,pos_label=0,zero_division=0),"_ys":ys,"_pos":pos}

def inhouse_rows(df_sub):
    return pd.DataFrame({"path":[os.path.join(INHOUSE_IMG,n) for n in df_sub["anon_image"]],
                         "label":df_sub["grade"].values})

def ddr_rows():
    lab=pd.read_csv(os.path.join(ROOT,"external_data","ddr","train_subset_labels.csv"))
    lab=lab[[os.path.exists(os.path.join(DDR_IMG,n)) for n in lab["image"]]]
    return pd.DataFrame({"path":[os.path.join(DDR_IMG,n) for n in lab["image"]],"label":lab["label"].values})

def run_fold(fold, mode, cfg, log):
    device="cuda" if torch.cuda.is_available() else "cpu"
    sp=pd.read_csv(os.path.join(ROOT,"data_anon","splits.csv")).dropna(subset=["grade"])
    tr=sp[(sp.split=="train_val")&(sp.fold!=fold)]; va=sp[(sp.split=="train_val")&(sp.fold==fold)]
    tr_rows=inhouse_rows(tr)
    if mode=="augmented":
        tr_rows=pd.concat([tr_rows, ddr_rows()], ignore_index=True)
    va_rows=inhouse_rows(va)   # 评估只用本院
    tr_ds=MixDS(tr_rows,224,True,cfg["aug"]); va_ds=MixDS(va_rows,224,False)
    tr_ld=DataLoader(tr_ds,cfg["bs"],shuffle=True,num_workers=4,drop_last=True)
    va_ld=DataLoader(va_ds,cfg["bs"],shuffle=False,num_workers=4)
    model=FundusClassifier("vit_large_patch16_224",2,pretrained=False,retfound=True,drop_rate=0.2).to(device)
    cnt=np.array(tr_ds.counts(),float); w=torch.tensor(cnt.sum()/(2*np.maximum(cnt,1)),dtype=torch.float32,device=device)
    crit=nn.CrossEntropyLoss(weight=w)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=0.05)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["epochs"])
    scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
    best=-1; bestm=None; bad=0
    for ep in range(1,cfg["epochs"]+1):
        model.train(); t0=time.time()
        for x,y in tr_ld:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            with torch.amp.autocast("cuda",enabled=device=="cuda"): loss=crit(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step(); m=evaluate(model,va_ld,device)
        if m["balanced_acc"]>best: best,bestm,bad=m["balanced_acc"],m,0
        else:
            bad+=1
            if bad>=cfg["patience"]: break
    log(f"  [{mode} f{fold}] auc={bestm['auc']:.3f} bacc={bestm['balanced_acc']:.3f} sens={bestm['sensitivity']:.3f} (train_n={len(tr_rows)}, {time.time()-t0:.0f}s/ep)")
    return bestm

def run(mode, cfg):
    out=os.path.join(ROOT,"results","augment_antivegf"); os.makedirs(out,exist_ok=True)
    def log(m): print(m,flush=True); open(os.path.join(out,f"train_{mode}.log"),"a",encoding="utf-8").write(m+"\n")
    log(f"==== augment mode={mode} ====")
    res=[]
    for f in range(5):
        b=run_fold(f,mode,cfg,log); res.append(b)
        np.savez(os.path.join(out,f"valpred_{mode}_fold{f}.npz"),ys=b["_ys"],pos=b["_pos"])
    keys=["auc","pr_auc","balanced_acc","sensitivity","specificity"]
    summ={k:f"{np.mean([r[k] for r in res]):.4f} ± {np.std([r[k] for r in res]):.4f}" for k in keys}
    json.dump(summ,open(os.path.join(out,f"summary_{mode}.json"),"w"),ensure_ascii=False,indent=2)
    log("\n=== "+mode+" 5折 ==="); [log(f"  {k}: {v}") for k,v in summ.items()]
    return summ

if __name__=="__main__":
    cfg=dict(bs=16,lr=5e-5,epochs=30,patience=10,
             aug=dict(hflip=True,rotate=15,color_jitter=0.1,random_resized_crop=True))
    if "--smoke" in sys.argv: cfg["epochs"]=1
    only=[a.split("=")[1] for a in sys.argv if a.startswith("--mode=")]
    modes=only if only else ["inhouse","augmented"]
    R={}
    for m in modes: R[m]=run(m,cfg)
    print("\n=== 增强消融对比 ===")
    for m,s in R.items(): print(f"  {m:10s} AUC={s['auc']} bacc={s['balanced_acc']} sens={s['sensitivity']}")
