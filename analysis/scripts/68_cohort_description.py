# -*- coding: utf-8 -*-
"""
68_cohort_description.py - the cohort as it was actually analysed.

WHY: Section 5.2 described the cohort from the clinical file (486 records) and said that the
remaining three of the 489 imaged patients had images only. The files say otherwise: 465
patients have both images and a clinical record, 24 imaged patients (43 images) have no
clinical record, and 21 clinical records have no images. The demographics printed in the paper
were therefore computed over a set that includes 21 patients who contributed no image. This
script recomputes them over the 465 analysed patients, counts eyes and images per eye, and
tabulates camera by partition and outcome.

USAGE   python scripts/68_cohort_description.py
OUTPUT  results/cohort_description.json
"""
import os, sys, json
import numpy as np, pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
DATA = os.path.join(ROOT, "data_anon")
DEVICE = {(2400, 2040): "VISUCAM 200", (720, 576): "Topcon + frame grabber",
          (2592, 1728): "Canon digital back"}
SEX, AGE, SYN, DUR = "性别", "年龄", "中医证型", "糖尿病年限（年）"
SYN_EN = {"痰瘀阻滞证": "phlegm-stasis", "阴虚夹瘀证": "yin-deficiency with stasis",
          "脾肾两虚证": "spleen-kidney deficiency", "脾肾亏虚证": "spleen-kidney deficiency", "气阴两虚证": "qi-yin deficiency"}

man = pd.read_csv(os.path.join(DATA, "manifest.csv"))
sp = pd.read_csv(os.path.join(DATA, "splits.csv"))
pts = pd.read_csv(os.path.join(DATA, "patients_anon.csv"), encoding="utf-8-sig")
img_ids, clin_ids = set(man.anon_id), set(pts.anon_id)
an_ids = set(sp.anon_id)
out = {"imaged_patients": len(img_ids), "images": len(man), "clinical_records": len(clin_ids),
       "both": len(img_ids & clin_ids), "imaged_without_record": len(img_ids - clin_ids),
       "records_without_images": len(clin_ids - img_ids),
       "images_without_record": int((~man.anon_id.isin(clin_ids)).sum()),
       "images_without_record_pdr": int(((~man.anon_id.isin(clin_ids)) & (man.grade == "PDR")).sum())}
assert an_ids == img_ids & clin_ids, "analysed patients are not exactly the linked set"

# eyes and images per eye, analysed set
eyes = sp.groupby(["anon_id", "eye"]).size()
out["analysed"] = {"images": int(len(sp)), "patients": int(sp.anon_id.nunique()), "eyes": int(len(eyes)),
                   "images_per_eye": {int(k): int(v) for k, v in eyes.value_counts().sort_index().items()},
                   "intolerant_images": int((sp.grade == "PDR").sum()),
                   "intolerant_eyes": int(sp[sp.grade == "PDR"].groupby(["anon_id", "eye"]).ngroups),
                   "intolerant_patients": int(sp[sp.grade == "PDR"].anon_id.nunique()),
                   "patients_with_both_eyes": int((sp.groupby("anon_id").eye.nunique() == 2).sum())}

# demographics over the 465 analysed patients
a = pts[pts.anon_id.isin(an_ids)].copy()
def ms(c, nd=1):
    v = pd.to_numeric(a[c], errors="coerce")
    return [round(float(v.mean()), nd), round(float(v.std(ddof=1)), nd)]
sex = a[SEX].value_counts().to_dict()
out["demographics_analysed"] = {
    "n": int(len(a)), "male": int(sex.get("男", 0)), "female": int(sex.get("女", 0)),
    "age_mean_sd": ms(AGE), "age_range": [int(a[AGE].min()), int(a[AGE].max())],
    "duration_mean_sd": ms(DUR), "hba1c_mean_sd": ms("HbA1c"), "bmi_mean_sd": ms("BMI"),
    "hypertension_n": int(a.hypertension.sum()), "smoking_n": int(a.smoking.sum()),
    "syndrome": {SYN_EN.get(k, k): int(v) for k, v in a[SYN].value_counts().items()}}

# camera by partition and outcome
sp["device"] = sp.anon_image.map(
    lambda f: DEVICE.get(Image.open(os.path.join(DATA, "images_with_overlays", f)).size, "other"))
tab = sp.groupby(["device", "split"]).agg(images=("grade", "size"),
                                         pdr=("grade", lambda g: int((g == "PDR").sum())))
out["device_by_split"] = {"%s | %s" % k: {"images": int(v.images), "pdr": int(v.pdr)} for k, v in tab.iterrows()}
out["device_totals"] = {k: {"images": int(len(g)), "pdr": int((g.grade == "PDR").sum()),
                            "pdr_rate": round(float((g.grade == "PDR").mean()), 3)}
                        for k, g in sp.groupby("device")}

json.dump(out, open(os.path.join(ROOT, "results", "cohort_description.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
