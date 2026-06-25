# CM-Oculomics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20537894.svg)](https://doi.org/10.5281/zenodo.20537894)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-CM--Oculomics-yellow)](https://huggingface.co/spaces/fc28/CM-Oculomics)

**🔗 Interactive demo (no install): https://huggingface.co/spaces/fc28/CM-Oculomics**

Code, an interactive demo, and reproducibility resources for the paper:

> **CM-Oculomics: Extending the Oculomics Paradigm to Chinese Medicine for Interpretable Anti-VEGF Intolerance Prediction in Diabetic Retinopathy**
> *(Authors, affiliations, journal/DOI — TBD)*

This repository provides the inference, explainability, and imaging-biomarker code, plus an interactive web demo, for the main results of the paper. **No patient data are included** (see *Data & Privacy* below).

---

## 1. Overview

We predict **anti-VEGF intolerance** — operationalized as progression to proliferative DR despite an adequate anti-VEGF course (ranibizumab, six consecutive monthly injections + 12-month follow-up) — from a single **low-cost color fundus photograph (CFP)**, using a **generalist vision foundation model, DINOv2 (ViT-L/14)**, fine-tuned for binary classification. The framework adds:

- **Interpretable imaging biomarkers** of intolerance — retinal vascular density, vascular-skeleton length, and vascular fractal dimension (box-counting).
- **Explainability** via Grad-CAM adapted to the vision transformer.
- **Trustworthy-AI** evaluation (uncertainty, calibration, selective prediction) and **patient-level** metrics with bootstrap confidence intervals.
- An **objective digital characterization of Traditional Chinese Medicine (TCM) syndromes** referenced against fundus imaging — extending the oculomics paradigm to Chinese medicine.

**Main results (five-fold cross-validation):** patient-level AUC **0.901** (95% CI 0.850–0.939); image-level pooled out-of-fold AUC **0.938** (95% CI 0.903–0.967); external validation on the public DDR dataset (zero fine-tuning) AUC **0.921**.

**Backbone choice.** In a like-for-like, resolution-matched comparison, DINOv2 was statistically indistinguishable from the retinal-specialist foundation model **RETFound** on internal data (DeLong p = 0.68), generalized **significantly better** on external data (0.921 vs 0.858, p < 0.001), matched it under label scarcity, and — being **Apache-2.0** licensed — permits unrestricted release of weights and this public demo. DINOv2 is therefore the primary encoder; RETFound is retained as a comparator in the paper.

---

## 2. Repository structure

```
.
├── README.md
├── README_HFSpace.md            # README for a Hugging Face Space (rename to README.md inside the Space)
├── requirements.txt
├── environment.yml
├── LICENSE                      # Apache-2.0
├── .gitignore                  # blocks all patient data / identity maps / weights
├── app.py                      # Gradio demo (Hugging Face Space ready) — DINOv2
├── configs/
│   └── default.yaml            # inference config
├── src/
│   ├── model.py                # DINOv2 (ViT-L/14) primary + RETFound comparator
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

Python ≥ 3.10 and PyTorch ≥ 2.1. A GPU is recommended for training; the demo runs on CPU.

---

## 4. Model weights

The primary model is **DINOv2 (ViT-L/14)** fine-tuned for anti-VEGF intolerance. The fine-tuned weights (`dino_deploy.pth`) are released under **Apache-2.0** and hosted on the Hugging Face Hub:

- **Hugging Face model repo:** `https://huggingface.co/<your-org>/CM-Oculomics` *(set after upload)* — file `dino_deploy.pth`.
- The app resolves weights automatically: `WEIGHTS_PATH` (local) → `weights/dino_deploy.pth` → auto-download from `WEIGHTS_URL` (set to the HF resolve URL) → a clearly-labeled placeholder head if none are found.

Place the checkpoint at `weights/dino_deploy.pth`, or set `WEIGHTS_URL` to the direct download link, e.g.:

```bash
export WEIGHTS_URL="https://huggingface.co/<your-org>/CM-Oculomics/resolve/main/dino_deploy.pth"
```

DINOv2's permissive Apache-2.0 license means these weights can be shared and used (including commercially) with attribution — no NonCommercial restriction.

---

## 5. Usage

```bash
# single-image inference: risk score
python src/inference.py --image path/to/fundus.jpg --weights weights/dino_deploy.pth

# Grad-CAM explanation
python src/gradcam.py --image path/to/fundus.jpg --weights weights/dino_deploy.pth --out cam.png

# imaging biomarkers (vascular density / skeleton / fractal dimension)
python src/biomarkers.py --image path/to/fundus.jpg

# interactive demo (local)
python app.py     # opens a Gradio UI at http://127.0.0.1:7860
```

Training on your own ethically-approved data uses `src/train.py` with a manifest CSV (`image_path,label[,patient_id]`); a template is documented in `src/dataset.py`. **You must supply your own data — none is distributed here.**

### Interactive demo (`app.py`)

`app.py` is a [Gradio](https://gradio.app) app that takes one color fundus photograph and returns (1) an **anti-VEGF intolerance risk score**, (2) a **Grad-CAM** heatmap, and (3) three **vascular biomarkers**. A live public instance is hosted on Hugging Face Spaces — **https://huggingface.co/spaces/fc28/CM-Oculomics** — and it also runs locally with `python app.py` (CPU is sufficient; to self-host see `README_HFSpace.md`).

**Research prototype — not a medical device. Not for clinical use.** No patient data are bundled.

---

## 6. License

- **This repository's code and the released DINOv2 fine-tuned weights** are licensed under the **Apache License 2.0** (see `LICENSE`), consistent with the permissive license of the DINOv2 backbone (Meta AI, Apache-2.0). Free for research and commercial use with attribution.
- **RETFound** (Zhou et al., Nature 2023) is used in the paper only as a *comparator* under its CC BY-NC 4.0 license; its weights are **not redistributed** in this repository.
- Please cite both this work and the underlying foundation models (DINOv2; and RETFound where the comparison is used) — see *Citation*.

---

## 7. Data & privacy

- **No patient data, images, or identity-mapping files are included in this repository**, by design (enforced via `.gitignore`).
- The in-house cohort cannot be shared publicly due to patient-privacy and ethics constraints; de-identified data may be available from the corresponding author on reasonable request, subject to institutional and ethical approval.
- **External dataset (DDR):** used for zero-fine-tuning external validation; publicly available at https://github.com/nkicsl/DDR-dataset (Li T, et al. *Inf Sci* 2019).

---

## 8. Citation

```bibtex
@article{CM_Oculomics,
  title   = {CM-Oculomics: Extending the Oculomics Paradigm to Chinese Medicine for Interpretable Anti-VEGF Intolerance Prediction in Diabetic Retinopathy},
  author  = {TBD},
  journal = {Engineering},
  year    = {TBD},
  doi     = {TBD}
}

@software{CM_Oculomics_zenodo,
  title     = {CM-Oculomics: code and demo for anti-VEGF intolerance prediction from color fundus photographs},
  author    = {TBD},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20537894},
  url       = {https://doi.org/10.5281/zenodo.20537894}
}
```

## 9. Acknowledgments

This work builds on **DINOv2** (Oquab et al., 2024; Apache-2.0) as the primary encoder, uses **RETFound** (Zhou et al., Nature 2023) as a comparator, and the public **DDR** dataset for external validation.
