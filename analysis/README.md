# Analysis code and public outputs

This folder contains the scripts that produced every number and figure in the paper, and the
model outputs on the **public** cohorts. No development-cohort data are included: no images, no
per-image or per-patient predictions, no identity information. Scripts that need the development
cohort will not run without it; de-identified data may be requested from the lead contact (see
the paper, *Resource availability*).

## Layout

```
analysis/
├── scripts/                 numbered analysis scripts (each has a WHY/USAGE/OUTPUT docstring)
├── tcm_retina/              training / dataset / model code used by the scripts and run*.py
├── configs/                 training configurations (RETFound comparator, syndrome, fusion)
├── run.py, run_multimodal.py, run_multitask.py
└── public_outputs/
    ├── external_predictions/   pred_<backbone>_seed<k>_<cohort>.npz  (ys = label, pos = score)
    │                           subset_labels_<cohort>.csv  (the class-balanced public image lists,
    │                           in the same order as the .npz arrays)
    ├── aggregate/              summary JSON/CSV files behind the text and figures
    └── same_task/              same-task control (public cohorts only): DDR split, per-model
                                predictions, estimators.json, same_task_rows.csv, training log
```

## What reproduces what (figure and section numbers of the submitted paper)

| Paper element | Script(s) | Needs development data? |
|---|---|---|
| Training protocol, cross-validation, label efficiency | `27_dino_experiment.py` | yes |
| Case study and held-out split (Sections 2.1-2.2) | `56`, `61`, `60`, `62` | yes |
| External seed models and external scoring (Sections 2.4-2.6) | `38_external2_benchmark.py` | yes (training); scoring outputs are in `public_outputs/` |
| Fig. 4, fixed- and random-effects pooling, variance components | `40_external_meta.py`, `72_random_effects.py` | **no**: copy `public_outputs/external_predictions/*.npz` to `results/external2/` and `aggregate/heldout_test.json` to `results/` |
| Published estimators and the no-shift baseline (Fig. 5) | `46_aline_atc_doc.py`, `75_estimator_baselines.py` | yes (held-out anchor); `75` runs from `aggregate/aline_atc_doc.json` |
| Label-free signals (Figure S4, Table S2), sub-cohort analysis (Figure S5) | `42`, `44`, `45`, `47` | yes (development embeddings) |
| Same-task control of the estimators (Section 2.5, Table S4, Figure S14) | `76_same_task_control.py`, `77_same_task_estimators.py` | **no**: public cohorts only. `77`: set `SAME_TASK_DIR=public_outputs/same_task` and copy `aggregate/estimator_baselines.json` to `results/external2/`; `76` retrains the ten models (under 4 h on one RTX 3090) |
| Cost of local validation (Fig. 6, Figure S6) | `74_local_validation_cost.py` | **no**: same predictions, in `results/external2/` |
| Per-model table and model provenance (Data S1, Table S3) | `64_per_seed_table.py`, `69_model_provenance.py` | anchor rows and counts only |
| Recalibration and decision curves (Supplementary Note S4, Figures S10-S11) | `73_recalibration.py`, `32_figures_dinov2.py` | yes |
| Headline internal metrics, supporting numbers | `57`, `66` | yes |
| Vascular measures by camera; cohort description | `67_biomarker_by_device.py`, `68_cohort_description.py` | yes |
| Overlay masking | `49_mask_overlays.py` | yes |
| Figures 1-3, 7 and S8-S13 | `63` (Fig. 1), `32_figures_dinov2.py`, `70` (Fig. 7a eye selection) | yes |
| Figure S1, Figure S7 | `71`, `58` (+ `figS7_capture/make_app_figS7.py`) | S7: example images |

## Recomputing the external AUCs from the released predictions

```python
import numpy as np
from sklearn.metrics import roc_auc_score
cohorts = ["ddr", "jsiec", "deepdrid", "aptos", "idrid", "messidor2", "eyepacs"]
for c in cohorts:
    for k in ("dinov2_l", "retfound"):
        z = [np.load(f"public_outputs/external_predictions/pred_{k}_seed{s}_{c}.npz") for s in range(5)]
        ens = np.mean([x["pos"] for x in z], 0)
        print(c, k, round(roc_auc_score(z[0]["ys"], ens), 3))   # five-seed ensemble, as in the paper
```

The public cohorts must be obtained from their distributors (see the paper's references); the
image lists above identify exactly which images were used.
