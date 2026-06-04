# Anti-VEGF Intolerance Prediction in Diabetic Retinopathy from Color Fundus Photographs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20537894.svg)](https://doi.org/10.5281/zenodo.20537894)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Code and reproducibility resources for the paper:

> **Linking Retinal Imaging, Traditional Chinese Medicine Syndromes, and Anti-VEGF Treatment Outcome in Diabetic Retinopathy: An Interpretable AI Framework for Objective Syndrome Characterization and Intolerance Prediction**
> *(Authors, affiliations, journal/DOI — TBD)*

This repository provides the training, inference, explainability, and imaging-biomarker code, plus an interactive demo, to reproduce the main results of the paper. **No patient data are included** (see *Data & Privacy* below).

---

## 1. Overview

We predict **anti-VEGF intolerance** (operationalized as progression to proliferative DR despite anti-VEGF therapy) from a single **low-cost color fundus photograph (CFP)**, using the retinal foundation model **RETFound** fine-tuned for binary classification. The framework adds:

- **Interpretable imaging biomarkers** of intolerance — retinal vascular density, vascular-skeleton length, and vascular fractal dimension (box-counting).
- **Explainability** via Grad-CAM adapted to the vision transformer.
- **Trustworthy-AI** evaluation (uncertainty, calibration, selective prediction) and **patient-level** metrics with bootstrap confidence intervals.

Main results (5-fold cross-validation): image-level AUC ≈ 0.95; patient-level AUC 0.919 (95% CI 0.877–0.952); external validation on DDR (zero fine-tuning) AUC 0.858.

---

## 2. Repository structure

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE                      # CC BY-NC 4.0 (inherits RETFound restriction; see §6)
├── .gitignore                  # blocks all patient data / identity maps
├── app.py                      # Gradio demo (Hugging Face Space ready)
├── configs/
│   └── default.yaml            # training / inference config
├── src/
│   ├── model.py                # RETFound (ViT-L/16) encoder + classification head
│   ├── dataset.py              # fundus image dataset + transforms
│   ├── train.py                # 5-fold fine-tuning (class-weighted, AMP, early stopping)
│   ├── inference.py            # single-image / batch inference -> risk score
│   ├── gradcam.py              # Grad-CAM for ViT (token->grid reshape)
│   └── biomarkers.py           # vascular density / skeleton length / fractal dimension
└── assets/
    └── example_fundus.jpg      # (optional) a NON-patient example image for the demo
```

---

## 3. Installation

```bash
# option A: pip
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# option B: conda
conda env create -f environment.yml
conda activate antivegf-cfp
```

Python ≥ 3.10 and PyTorch ≥ 2.1 (CUDA build recommended). A single 24 GB GPU (e.g. RTX 3090) is sufficient.

---

## 4. Model weights

This work **fine-tunes RETFound**. Two steps:

1. **Obtain RETFound pretrained weights** from the official source (Hugging Face / the RETFound repository). RETFound weights are released under **CC BY-NC 4.0** — see §6.
2. **Our fine-tuned weights** (anti-VEGF intolerance head) are archived as an attachment on the Zenodo record for this repository: **https://doi.org/10.5281/zenodo.20537894** (download `best.pth` from the *Files* section). Released under **CC BY-NC 4.0** in compliance with the RETFound license (§6). Place the checkpoint at `weights/best.pth` (path configurable in `configs/default.yaml`).

> If fine-tuned weights are not yet released, `src/train.py` reproduces them from RETFound + your own ethically-approved data.

---

## 5. Reproducing the results

```bash
# 1. Fine-tune (5-fold CV) on your own de-identified dataset
python src/train.py --config configs/default.yaml

# 2. Single-image inference: risk score
python src/inference.py --image path/to/fundus.jpg --weights weights/best.pth

# 3. Grad-CAM explanation
python src/gradcam.py --image path/to/fundus.jpg --weights weights/best.pth --out cam.png

# 4. Imaging biomarkers (vascular density / skeleton / fractal dimension)
python src/biomarkers.py --image path/to/fundus.jpg

# 5. Interactive demo (local)
python app.py     # opens a Gradio UI
```

A data manifest CSV (columns: `image_path,label[,patient_id]`) is expected for training; a template is documented in `src/dataset.py`. **You must supply your own ethically-approved data — none is distributed here.**

---

## 6. License and compliance (please read)

- **This repository's code** is released under **CC BY-NC 4.0** (see `LICENSE`).
- **RETFound dependency:** RETFound and its pretrained weights are licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. Verified from the official RETFound `LICENSE` file. Key terms:
  - **NonCommercial only** — "reproduce and Share the Licensed Material … for NonCommercial purposes only"; NonCommercial means "not primarily intended for or directed towards commercial advantage or monetary compensation."
  - **Derivatives** (including fine-tuned weights) — "produce, reproduce, and Share Adapted Material for NonCommercial purposes only," and the applied license "must not prevent recipients from complying with" the original license.
  - **Attribution** — retain creator identification, copyright notice, license reference, and indicate any modifications.
- **Consequently, any fine-tuned weights derived here are also CC BY-NC 4.0**, may be used for **non-commercial research only**, must **cite/attribute RETFound**, and must **state that modifications were made**. Commercial use requires separate permission from the RETFound authors.
- Please also cite RETFound: Zhou Y, et al. *A foundation model for generalizable disease detection from retinal images.* Nature 2023. https://doi.org/10.1038/s41586-023-06555-x

---

## 7. Data & privacy

- **No patient data, images, or identity-mapping files are included in this repository**, by design (enforced via `.gitignore`).
- The in-house dataset used in the paper cannot be shared publicly due to patient-privacy and ethics constraints; de-identified data may be available from the corresponding author on reasonable request, subject to institutional and ethical approval.
- **External dataset (DDR):** used for zero-fine-tuning external validation. DDR is publicly available:
  - Li T, et al. *Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening.* Inf Sci 2019. Repository: https://github.com/nkicsl/DDR-dataset

---

## 8. Citation

```bibtex
@article{TBD_anti_vegf_cfp,
  title   = {Linking Retinal Imaging, Traditional Chinese Medicine Syndromes, and Anti-VEGF Treatment Outcome in Diabetic Retinopathy: An Interpretable AI Framework for Objective Syndrome Characterization and Intolerance Prediction},
  author  = {TBD},
  journal = {Engineering},
  year    = {TBD},
  doi     = {TBD}
}
```

This repository is archived on Zenodo (cite the code/data archive separately):

```bibtex
@software{CM_Oculomics_zenodo,
  title     = {CM-Oculomics: reproducibility code for anti-VEGF intolerance prediction from color fundus photographs},
  author    = {TBD},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20537894},
  url       = {https://doi.org/10.5281/zenodo.20537894}
}
```

## 9. Acknowledgments

This work builds on **RETFound** (Zhou et al., Nature 2023) and uses the public **DDR** dataset for external validation.
