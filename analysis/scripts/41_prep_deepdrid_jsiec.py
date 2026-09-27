# -*- coding: utf-8 -*-
"""
41_prep_deepdrid_jsiec.py — build balanced PDR-vs-NPDR subsets for two cohorts whose images
live in nested folders (so the generic flat-directory prep in 38 does not apply).

Same rule as every other external set: PDR = highest grade, NPDR = intermediate grades,
"no DR" excluded, class-balanced with seed 42.

DeepDRiD  China, multi-centre. Per-image DR level 0-4 in left_eye/right_eye columns.
JSIEC     Joint Shantou International Eye Centre, China. Class folders DR1/DR2/DR3,
          where DR3 is proliferative. "Blur fundus with suspected PDR" is EXCLUDED
          (suspected, and ungradable quality).
"""
import os, sys, glob
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "external_data")
OUT = os.path.join(ROOT, "results", "external2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEEPDRID = "D:/kagglehub/datasets/nancyhisham/deepdrid/versions/1/regular_fundus_images"
JSIEC = "D:/kagglehub/datasets/linchundan/fundusimage1000/versions/4/1000images"


def junction(dst, src):
    if os.path.exists(dst):
        print(f"[link] {dst} exists")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rc = os.system(f'cmd /c mklink /J "{dst}" "{os.path.normpath(src)}" >nul 2>&1')
    print(f"[link] {'ok' if rc == 0 else 'FAILED'}  {dst} -> {src}")


def balance(df, name):
    pos = df[df.grade == 4]
    neg = df[df.grade.isin([1, 2, 3])]
    n = min(len(pos), len(neg))
    rng = np.random.RandomState(42)
    sub = pd.concat([pos.sample(n, random_state=rng) if len(pos) > n else pos,
                     neg.sample(n, random_state=rng) if len(neg) > n else neg])
    sub = sub.sample(frac=1, random_state=42).reset_index(drop=True)
    sub["label"] = np.where(sub.grade == 4, "PDR", "NPDR")
    fn = os.path.join(OUT, f"subset_labels_{name}.csv")
    sub[["image", "grade", "label"]].to_csv(fn, index=False)
    print(f"[{name}] balanced subset: {len(sub)} images ({n} PDR / {n} NPDR) -> {fn}")
    print(sub.groupby(["grade", "label"]).size().to_string())
    return sub


def do_deepdrid():
    rows = []
    for split in ["regular-fundus-training", "regular-fundus-validation"]:
        csv = os.path.join(DEEPDRID, split, f"{split}.csv")
        if not os.path.exists(csv):
            print(f"[deepdrid] missing {csv}")
            continue
        d = pd.read_csv(csv)
        # each row is one image; exactly one of the two eye columns is populated
        lvl = d.left_eye_DR_Level.fillna(d.right_eye_DR_Level)
        d = d.assign(grade=lvl).dropna(subset=["grade"])
        # the CSV path omits the "Images" level that actually exists on disk:
        # csv "\regular-fundus-training\1\1_l1.jpg"  ->  "regular-fundus-training/Images/1/1_l1.jpg"
        d["image"] = (split + "/Images/" + d.patient_id.astype(str)
                      + "/" + d.image_id.astype(str) + ".jpg")
        rows.append(d[["image", "grade"]])
    d = pd.concat(rows, ignore_index=True)
    d["grade"] = d.grade.astype(int)
    root = os.path.join(EXT, "deepdrid", "images")
    junction(root, DEEPDRID)
    d = d[[os.path.exists(os.path.join(root, p)) for p in d.image]]
    print(f"[deepdrid] {len(d)} images found on disk")
    print(d.groupby("grade").size().to_string())
    balance(d, "deepdrid")


def do_jsiec():
    root = os.path.join(EXT, "jsiec", "images")
    junction(root, JSIEC)
    grade_of = {"0.3.DR1": 1, "1.0.DR2": 2, "1.1.DR3": 4}   # DR3 = proliferative
    rows = []
    for folder, g in grade_of.items():
        p = os.path.join(root, folder)
        if not os.path.isdir(p):
            print(f"[jsiec] missing {folder}")
            continue
        for f in os.listdir(p):
            if os.path.isfile(os.path.join(p, f)):
                rows.append({"image": f"{folder}/{f}", "grade": g})
    d = pd.DataFrame(rows)
    print(f"[jsiec] {len(d)} images (DR1->1, DR2->2, DR3->4; 'suspected PDR' excluded)")
    print(d.groupby("grade").size().to_string())
    balance(d, "jsiec")


if __name__ == "__main__":
    do_deepdrid()
    print()
    do_jsiec()
