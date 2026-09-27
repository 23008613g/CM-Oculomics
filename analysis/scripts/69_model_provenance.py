# -*- coding: utf-8 -*-
"""
69_model_provenance.py - one row per family of trained models: what it was trained on, how its
stopping point and learning rate were chosen, and which results it produced.

WHY: all three blind reviewers asked which data the external seed models and the released
weights were trained on, whether the in-distribution anchor of the performance estimators was
ever seen in training, and which model drew which figure. The answers were only in code
(scripts/27, 30, 38 and the RETFound code base). This table states them, with the counts read
from the split file and the released checkpoint identified by its hash.

Sources checked: scripts/27_dino_experiment.py (run_cv, run_standard: early stopping on the
evaluated fold; learning-rate sweep scored by AUC on fold 0), scripts/38_external2_benchmark.py
(seed models: folds != 0, early stop on fold 0, lr 1e-5 / 5e-5, seeds 0-4),
scripts/30_dino_artifacts.py (deployed model, Grad-CAM, t-SNE, MC-dropout), scripts/76 (same-task
control on DDR),
results/retfound_antivegf/metrics_fold0.json (RETFound single model validated on fold 0).

USAGE   python scripts/69_model_provenance.py
OUTPUT  results/model_provenance.csv, results/model_provenance.json
"""
import os, sys, json, hashlib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sp = pd.read_csv(os.path.join(ROOT, "data_anon", "splits.csv"))
tv, te = sp[sp.split == "train_val"], sp[sp.split == "test"]
f14, f0 = tv[tv.fold != 0], tv[tv.fold == 0]
n = dict(cv=len(tv), cv_pdr=int((tv.grade == "PDR").sum()), f14=len(f14), f14_pdr=int((f14.grade == "PDR").sum()),
         f0=len(f0), f0_pdr=int((f0.grade == "PDR").sum()), te=len(te), te_pdr=int((te.grade == "PDR").sum()))
ss = pd.read_csv(os.path.join(ROOT, "results", "same_task", "split.csv"))      # scripts/76
st = {k: int((ss.part == p).sum()) for k, p in (("train", "train"), ("es", "early_stop"), ("an", "anchor"))}
st.update({k + "_pdr": int(ss[ss.part == p].y.sum()) for k, p in (("train", "train"), ("es", "early_stop"), ("an", "anchor"))})
sha = hashlib.sha256(open(os.path.join(ROOT, "results", "dino_experiment", "dino_deploy.pth"), "rb").read()).hexdigest()

CV_STOP = ("each fold model stops early on the fold it then predicts (balanced accuracy at 0.5, "
           "patience 10, at most 30 epochs); learning rate chosen on fold 0")
F14 = "cross-validation folds 1-4 (%d images, %d intolerant)" % (n["f14"], n["f14_pdr"])
STOP0 = "early stopping on fold 0 (%d images, %d intolerant)" % (n["f0"], n["f0_pdr"])
rows = [
    dict(models="Cross-validation run 1 (primary)", backbone="DINOv2 ViT-L/14, 224 px",
         training="5 fold models, each trained on the other 4 folds of the cross-validation set "
                  "(%d images, %d intolerant)" % (n["cv"], n["cv_pdr"]),
         selection=CV_STOP + " (1e-5)", runs="1",
         used_for="Sections 2.1, 2.7 and Supplementary Note S4; Figs 2, 3a-b; Figures S8a, S8c-d, S9b, S10b, S11, S12, S13b-d",
         saved="out-of-fold predictions only"),
    dict(models="Cross-validation runs 2-3", backbone="DINOv2 ViT-L/14, 224 px",
         training="as run 1", selection="as run 1", runs="2 (different random seeds)",
         used_for="Section 2.3 (mean and SD over runs 1-3)", saved="out-of-fold predictions only"),
    dict(models="RETFound cross-validation runs 1-3", backbone="RETFound ViT-L/16, 224 px",
         training="as DINOv2 run 1", selection=CV_STOP + " (own sweep)", runs="3",
         used_for="Section 2.3; Fig 3a; calibration comparison in Supplementary Note S4",
         saved="out-of-fold predictions only"),
    dict(models="ImageNet baselines", backbone="Swin-V2 (256 px), ViT-B/16 (224 px), ResNet-50 (224 px)",
         training="as DINOv2 run 1", selection=CV_STOP + " (own sweep)", runs="1 each",
         used_for="Section 2.3; Fig 3a", saved="out-of-fold predictions only"),
    dict(models="Label-efficiency runs", backbone="DINOv2 and RETFound, 224 px",
         training="as run 1, with training rows subsampled within each fold to 10, 25, 50 and 100% "
                  "(stratified by outcome); the validation fold is never subsampled",
         selection=CV_STOP, runs="1 per fraction", used_for="Supplementary Note S4; Figure S13a",
         saved="out-of-fold predictions only"),
    dict(models="External seed models", backbone="DINOv2 (lr 1e-5) and RETFound (lr 5e-5), 224 px",
         training=F14, selection=STOP0, runs="5 per backbone (seeds 0-4)",
         used_for="held-out test (Section 2.2); seven external cohorts (Sections 2.4-2.6; Figs 3c, "
                  "4, 5, 6, S4-S6; Data S1); in-distribution anchor of the estimators",
         saved="checkpoints kept, not released"),
    dict(models="Deployed model (dino_deploy.pth)", backbone="DINOv2 ViT-L/14, 224 px (lr 1e-5)",
         training=F14, selection=STOP0, runs="1",
         used_for="single-model DDR result (Section 2.4, Figure S8b); selective prediction on the held-out "
                  "split (Figure S10a); DINOv2 Grad-CAM (Fig 7a); t-SNE (Figure S9a); web application (Figure S7)",
         saved="released (Hugging Face, Apache-2.0); sha256 " + sha),
    dict(models="RETFound single model", backbone="RETFound ViT-L/16",
         training=F14, selection=STOP0, runs="1", used_for="RETFound Grad-CAM (Fig 7a)",
         saved="kept, not released (CC BY-NC weights)"),
    dict(models="Same-task control models", backbone="DINOv2 (lr 1e-5) and RETFound (lr 5e-5), 224 px",
         training="public DDR training images of grades 1-4, proliferative versus non-proliferative: "
                  "70%% of them (%d images, %d proliferative); no development-cohort image" % (st["train"], st["train_pdr"]),
         selection="early stopping on 15%% of DDR (%d images, %d proliferative); anchor the remaining 15%% "
                   "(%d, %d); split stratified at image level (DDR has no patient identifiers)"
                   % (st["es"], st["es_pdr"], st["an"], st["an_pdr"]),
         runs="5 per backbone (seeds 0-4)",
         used_for="same-task control of the estimators (Section 2.5; Supplementary Note S2, Table S4, Figure S14)",
         saved="checkpoints kept, not released; public-cohort predictions released"),
    dict(models="Syndrome and fusion models", backbone="see Supplementary Note S1",
         training="cross-validation set, same patient-level folds", selection="as run 1", runs="1 each",
         used_for="Supplementary Note S1 only", saved="predictions only"),
]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "results", "model_provenance.csv"), index=False)
json.dump({"counts": n, "deploy_sha256": sha, "rows": rows},
          open(os.path.join(ROOT, "results", "model_provenance.json"), "w", encoding="utf-8"), indent=2)
print(n); print(df[["models", "training", "runs"]].to_string(index=False))
print("no model's training or early-stopping data include the held-out split:",
      set(te.anon_id).isdisjoint(set(tv.anon_id)))
