# -*- coding: utf-8 -*-
"""
50_mask_effect.py — does removing the burned-in overlays change what the model predicts?

WHY: script 49 blanks patient name, hospital ID and timestamps that were burned into the
black surround of some images. Strictly, changing the input means the models should be
retrained. Before spending a day on that, measure whether it matters at all: run the SAME
trained checkpoints over the original and the masked copies and compare, image by image.

If the predictions are effectively identical, masking is cosmetic, the published numbers
stand, and only the released image set needs replacing. If they move, retraining is required.

USAGE  python scripts/50_mask_effect.py
OUTPUT results/mask_effect.json
"""
import os, sys, json, importlib.util
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
OUT = os.path.join(ROOT, "results", "external2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES = 224


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


EXP = _load(os.path.join(ROOT, "scripts", "27_dino_experiment.py"), "exp27")
_ORIG_DL = EXP.DataLoader
EXP.DataLoader = lambda *a, **k: _ORIG_DL(*a, **{**k, "num_workers": 0})

ORIG_DIR = os.path.join(ROOT, "data_anon", "images")
MASK_DIR = os.path.join(ROOT, "data_anon", "images_masked")


def score_all(kind, seed, img_dir, rows):
    """Predicted P(intolerant) for every row, reading images from img_dir."""
    EXP.IMG = img_dir                      # DS builds paths from EXP.IMG
    ds = EXP.DS(rows, False, RES)
    ck = os.path.join(OUT, f"ckpt_{kind}_seed{seed}.pth")
    model = EXP.make_model(kind, RES).to(DEVICE)
    model.load_state_dict(torch.load(ck, map_location=DEVICE))
    p = EXP.infer(model, ds, RES)
    del model
    torch.cuda.empty_cache()
    return p


def main():
    sp = pd.read_csv(os.path.join(ROOT, "data_anon", "splits.csv")).dropna(subset=["grade"])
    from PIL import Image
    sp["res"] = sp["anon_image"].map(
        lambda f: "%dx%d" % Image.open(os.path.join(ORIG_DIR, f)).size)
    y = np.array([EXP.GRADE_CLASSES.index(g) for g in sp["grade"]])
    touched = sp["res"].isin(["720x576", "2592x1728"]).values

    res = {}
    for kind in ("dinov2_l", "retfound"):
        for seed in (0,):
            tag = f"{kind}_seed{seed}"
            print(f"[{tag}] scoring {len(sp)} images, original ...", flush=True)
            p0 = score_all(kind, seed, ORIG_DIR, sp)
            print(f"[{tag}] scoring {len(sp)} images, masked ...", flush=True)
            p1 = score_all(kind, seed, MASK_DIR, sp)
            d = np.abs(p1 - p0)
            flips = int(((p0 > 0.5) != (p1 > 0.5)).sum())
            r = dict(
                auc_original=float(roc_auc_score(y, p0)),
                auc_masked=float(roc_auc_score(y, p1)),
                auc_delta=float(roc_auc_score(y, p1) - roc_auc_score(y, p0)),
                max_abs_prob_change=float(d.max()),
                mean_abs_prob_change=float(d.mean()),
                mean_abs_change_masked_images=float(d[touched].mean()),
                mean_abs_change_untouched_images=float(d[~touched].mean()),
                pearson_r=float(np.corrcoef(p0, p1)[0, 1]),
                class_flips=flips, n=int(len(sp)))
            res[tag] = r
            print(f"  AUC {r['auc_original']:.4f} -> {r['auc_masked']:.4f} "
                  f"(delta {r['auc_delta']:+.4f})")
            print(f"  |prob change|: mean {r['mean_abs_prob_change']:.5f}, "
                  f"max {r['max_abs_prob_change']:.5f}, r = {r['pearson_r']:.6f}")
            print(f"  on masked images only: mean {r['mean_abs_change_masked_images']:.5f}; "
                  f"on untouched images: {r['mean_abs_change_untouched_images']:.5f}")
            print(f"  predicted-class flips: {flips} of {len(sp)}")
            print()

    json.dump(res, open(os.path.join(ROOT, "results", "mask_effect.json"), "w"), indent=2)
    print("saved results/mask_effect.json")
    worst = max(r["auc_delta"] for r in res.values())
    print()
    print("VERDICT: %s" % ("masking is inert; published numbers stand"
                           if max(abs(r["auc_delta"]) for r in res.values()) < 0.005
                           and max(r["class_flips"] for r in res.values()) <= 2
                           else "masking moves predictions; retraining needed"))


if __name__ == "__main__":
    main()
