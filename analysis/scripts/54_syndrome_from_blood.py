# -*- coding: utf-8 -*-
"""
54_syndrome_from_blood.py — can the TCM syndrome be predicted from blood and metabolic
indicators? Re-derived, because the value quoted in Supplementary Note S1 and Figure S3
(AUC 0.587) exists only as a literal inside old plotting scripts: no file under results/
and no console log produces it, so it is not traceable to a run.

This analysis uses tabular clinical data only and is unaffected by the image masking.
Protocol: patient-level stratified five-fold CV, multinomial logistic regression on
standardized indicators, one-vs-rest macro AUC pooled over the out-of-fold predictions.

USAGE   python scripts/54_syndrome_from_blood.py
OUTPUT  results/syndrome_from_blood.json
"""
import os, sys, json, io
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

SYN = "中医证型"
BLOOD = ["HbA1c", "LDL", "HDL", "Triglyceride"]
METAB = BLOOD + ["BMI", "年龄", "糖尿病年限（年）"]


def run(df, feats, name, seed=42):
    d = df.dropna(subset=[SYN]).copy()
    X = d[feats].apply(pd.to_numeric, errors="coerce")
    y = d[SYN].astype("category")
    classes = list(y.cat.categories)
    yi = y.cat.codes.values
    oof = np.zeros((len(d), len(classes)))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, va in skf.split(X, yi):
        # multinomial is the default for multiclass in this scikit-learn
        mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=2000))
        mdl.fit(X.iloc[tr], yi[tr])
        oof[va] = mdl.predict_proba(X.iloc[va])
    macro = roc_auc_score(yi, oof, multi_class="ovr", average="macro")
    bacc = balanced_accuracy_score(yi, oof.argmax(1))
    print("  %-34s n=%d  macro AUC (OvR) = %.4f   balanced acc = %.3f"
          % (name, len(d), macro, bacc))
    return {"features": feats, "n": int(len(d)), "classes": classes,
            "macro_auc_ovr": round(float(macro), 4), "balanced_acc": round(float(bacc), 3)}


def main():
    p = pd.read_csv(os.path.join(ROOT, "data_anon", "patients_anon.csv"), encoding="utf-8-sig")
    print("patients with a recorded syndrome: %d" % p[SYN].notna().sum())
    print("class counts: %s" % dict(p[SYN].value_counts()))
    print()
    print("Predicting syndrome from tabular indicators (patient-level 5-fold CV,")
    print("multinomial logistic regression, macro one-vs-rest AUC on pooled out-of-fold):")
    # "blood and metabolic indicators" in Supplementary Note S1 means the laboratory
    # panel plus BMI. Age and diabetes duration are reported separately because the
    # same note shows syndrome varies strongly with them, so folding them in here
    # would re-import that association rather than test the laboratory data.
    out = {"blood_only": run(p, BLOOD, "blood indicators only (4)"),
           "blood_metabolic_bmi": run(p, BLOOD + ["BMI"],
                                      "blood + BMI (5)"),
           "blood_metabolic": run(p, METAB, "blood + BMI + age/duration (7)")}
    out["note"] = ("Re-derived because the previously quoted 0.587 could not be traced to any "
                   "file under results/ or any console log. Chance level for four balanced "
                   "classes is 0.5 under the one-vs-rest macro AUC.")
    json.dump(out, io.open(os.path.join(ROOT, "results", "syndrome_from_blood.json"),
                           "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print()
    print("saved results/syndrome_from_blood.json")


if __name__ == "__main__":
    main()
