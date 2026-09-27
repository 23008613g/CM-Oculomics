# -*- coding: utf-8 -*-
"""
76_same_task_control.py - do the published performance estimators fail when the task does NOT
change?

WHY: in the paper the estimators were anchored on an in-house split labelled for progression under
anti-VEGF and applied to public cohorts labelled for current PDR, class-balanced to 50%. All three
blind reviewers pointed out that the estimators assume a shared labelling function, so their failure
there may reflect the task change (and the designed prior shift) rather than cohort dependence. This
control removes the task change: models are trained for PDR versus NPDR on one public cohort (DDR),
anchored on a held-out part of the same cohort at its natural prevalence, and the estimators are then
applied to the other public cohorts, both in the class-balanced subsets used in the paper and in the
full cohorts at their natural prevalence.

Protocol, identical to the paper except for the training data (scripts/27, scripts/38):
  * DDR training images of grades 1-4 (grade 4 = PDR), split stratified 70/15/15 (seed 2026) into
    training, early stopping and anchor parts; DDR has no patient identifiers, so images of one
    patient may fall in different parts (stated as a limitation);
  * DINOv2 ViT-L/14 (lr 1e-5) and RETFound ViT-L/16 (lr 5e-5), 224 px, class-weighted cross-entropy,
    AdamW (weight decay 0.05), cosine schedule, AMP, effective batch 16, at most 30 epochs, early
    stopping on balanced accuracy at 0.5 (patience 10); seeds 0-4 for each backbone.
  * Each model scores, in the same process: the anchor; the class-balanced public subsets of the six
    other cohorts (the image lists of the paper); and the full APTOS, EyePACS, Messidor-2 and IDRiD
    cohorts, grades 1-4, at natural prevalence (DeepDRiD and JSIEC have no full grade lists locally).
  * Checkpoints go to D:/CM-Oculomics_same_task (the project drive is nearly full and synced to the
    cloud); predictions are saved per model and set, and a model whose predictions all exist is
    skipped, so the run can be resumed.

USAGE   python scripts/76_same_task_control.py            (full run; 3 h 46 min on one RTX 3090, 2026-09-27)
        python scripts/76_same_task_control.py --smoke    (one model, one epoch, 64 images per set)
OUTPUT  results/same_task/pred_<backbone>_seed<k>_<set>.npz, split.csv, log.txt
"""
import os, sys, time, json, random, argparse, importlib.util
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tcm_retina.data.dataset import build_transforms          # noqa: E402

EXT = os.path.join(ROOT, "external_data")
OUT = os.path.join(ROOT, "results", "same_task")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES, EFF_BS, EPOCHS, PATIENCE, BS = 224, 16, 30, 10, 16
LR = {"dinov2_l": 1e-5, "retfound": 5e-5}
KINDS, SEEDS = ["dinov2_l", "retfound"], [0, 1, 2, 3, 4]
BALANCED = ["jsiec", "deepdrid", "aptos", "idrid", "messidor2", "eyepacs"]
NATURAL = ["aptos", "eyepacs", "messidor2", "idrid"]
WORKERS = 8
CKPT_DIR = r"D:/CM-Oculomics_same_task"


class DS(Dataset):
    """rows: columns path (absolute), y (0/1)."""
    def __init__(self, rows, train):
        self.rows = rows.reset_index(drop=True)
        self.tf = build_transforms(RES, train, dict(hflip=True, rotate=15, color_jitter=0.1,
                                                    random_resized_crop=True) if train else None)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), int(r["y"]), i


def load_exp():
    spec = importlib.util.spec_from_file_location("exp27", os.path.join(ROOT, "scripts", "27_dino_experiment.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def log(msg):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(OUT, "log.txt"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def infer(model, rows):
    model.eval()
    ld = DataLoader(DS(rows, False), BS, shuffle=False, num_workers=WORKERS, persistent_workers=False)
    probs = np.zeros(len(rows))
    for x, _, idx in ld:
        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            probs[idx.numpy()] = torch.softmax(model(x.to(DEVICE)), 1)[:, 1].float().cpu().numpy()
    return probs


def train(exp, kind, tr, va, max_epochs, max_batches=None):
    """The training loop of scripts/27 train_fold, with a dataset that reads any image path."""
    model = exp.make_model(kind, RES).to(DEVICE)
    cnt = np.array([(tr.y == 0).sum(), (tr.y == 1).sum()], float)
    w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[kind], weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    ld = DataLoader(DS(tr, True), BS, shuffle=True, num_workers=WORKERS, drop_last=True)
    accum = max(1, EFF_BS // BS)
    best, best_state, bad, best_ep = -1, None, 0, -1
    for ep in range(max_epochs):
        t0 = time.time(); model.train(); opt.zero_grad()
        for bi, (x, y, _) in enumerate(ld):
            if max_batches and bi >= max_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = crit(model(x), y) / accum
            scaler.scale(loss).backward()
            if (bi + 1) % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
        sch.step()
        prob = infer(model, va)
        ba = balanced_accuracy_score(va.y, (prob >= 0.5).astype(int))
        log("   %s epoch %2d  val bal-acc %.3f  (%.0f s)" % (kind, ep, ba, time.time() - t0))
        if ba > best:
            best, bad, best_ep = ba, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, best_ep


def ddr_split():
    fn = os.path.join(OUT, "split.csv")
    if os.path.exists(fn):
        lab = pd.read_csv(fn)
        if "path" not in lab:                          # released split.csv carries no local paths
            lab["path"] = lab.image.map(lambda f: os.path.join(EXT, "ddr", "train_images", f))
        return lab
    lab = pd.read_csv(os.path.join(EXT, "ddr", "train.txt"), sep=" ", header=None, names=["image", "grade"])
    lab = lab[lab.grade.between(1, 4)].copy()
    lab["path"] = lab.image.map(lambda f: os.path.join(EXT, "ddr", "train_images", f))
    lab = lab[lab.path.map(os.path.exists)].reset_index(drop=True)
    lab["y"] = (lab.grade == 4).astype(int)
    tr, rest = train_test_split(lab, test_size=0.30, stratify=lab.y, random_state=2026)
    es, an = train_test_split(rest, test_size=0.50, stratify=rest.y, random_state=2026)
    lab["part"] = "train"
    lab.loc[es.index, "part"] = "early_stop"; lab.loc[an.index, "part"] = "anchor"
    lab.to_csv(fn, index=False)
    return lab


def target_sets():
    sets = {}
    for ds in BALANCED:
        sub = pd.read_csv(os.path.join(ROOT, "results", "external2", "subset_labels_%s.csv" % ds))
        sets["bal_" + ds] = pd.DataFrame({"path": sub.image.map(lambda f: os.path.join(EXT, ds, "images", str(f))),
                                          "y": (sub.grade == 4).astype(int)})
    for ds in NATURAL:
        lab = pd.read_csv(os.path.join(EXT, ds, "labels.csv"))
        lab.columns = [c.strip().lower() for c in lab.columns]
        lab = lab[lab.grade.between(1, 4)]
        sets["nat_" + ds] = pd.DataFrame({"path": lab.image.map(lambda f: os.path.join(EXT, ds, "images", str(f))),
                                          "y": (lab.grade == 4).astype(int)})
    for k, v in sets.items():
        missing = int((~v.path.map(os.path.exists)).sum())
        if missing:
            raise SystemExit("%s: %d images missing" % (k, missing))
    return sets


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global OUT
    if a.smoke:
        OUT = os.path.join(ROOT, "results", "same_task_smoke")
    os.makedirs(OUT, exist_ok=True)
    exp = load_exp()
    lab = ddr_split()
    tr, es, an = (lab[lab.part == p].reset_index(drop=True) for p in ("train", "early_stop", "anchor"))
    sets = {"anchor": an, **target_sets()}
    log("DDR grades 1-4: train %d (%d PDR), early stop %d (%d), anchor %d (%d)"
        % (len(tr), tr.y.sum(), len(es), es.y.sum(), len(an), an.y.sum()))
    log("targets: " + ", ".join("%s %d (%d PDR)" % (k, len(v), v.y.sum()) for k, v in sets.items()))
    runs = [(k, s) for k in KINDS for s in SEEDS]
    if a.smoke:
        runs = [("dinov2_l", 0)]
        sets = {k: v.sample(min(64, len(v)), random_state=0).reset_index(drop=True) for k, v in sets.items()}
    for kind, seed in runs:
        todo = [s for s in sets if not os.path.exists(os.path.join(OUT, "pred_%s_seed%d_%s.npz" % (kind, seed, s)))]
        if not todo:
            log("skip %s seed %d (all predictions exist)" % (kind, seed)); continue
        t0 = time.time(); set_seed(seed)
        log("train %s seed %d" % (kind, seed))
        model, best, best_ep = train(exp, kind, tr, es, 1 if a.smoke else EPOCHS, 20 if a.smoke else None)
        log("   best early-stop bal-acc %.3f at epoch %d; training %.1f min" % (best, best_ep, (time.time() - t0) / 60))
        if not a.smoke:
            os.makedirs(CKPT_DIR, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       os.path.join(CKPT_DIR, "ckpt_%s_seed%d.pth" % (kind, seed)))
        for s in todo:
            t1 = time.time(); p = infer(model, sets[s]); y = sets[s].y.values
            np.savez(os.path.join(OUT, "pred_%s_seed%d_%s.npz" % (kind, seed, s)), ys=y, pos=p)
            log("   scored %-15s n=%5d  AUC %.3f  bal-acc %.3f  (%.0f s)"
                % (s, len(y), roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan"),
                   balanced_accuracy_score(y, (p > 0.5).astype(int)), time.time() - t1))
        del model; torch.cuda.empty_cache()
        log("done %s seed %d in %.1f min" % (kind, seed, (time.time() - t0) / 60))
    log("ALL DONE")


if __name__ == "__main__":
    main()
