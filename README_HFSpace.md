---
title: Anti-VEGF Intolerance Prediction (CFP)
emoji: 👁️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

<!--
============================================================================
HOW TO USE THIS FILE  (do not commit it as-is to the GitHub repo)
============================================================================
This file is the README for a *Hugging Face Space*. On a HF Space the file MUST
be named exactly `README.md`, and the YAML block above (the part between the
two `---` lines) is what tells HF to launch a Gradio app from `app.py`.

Because the GitHub repo already has a different `README.md` (the developer
README), keep this one named `README_HFSpace.md` in the GitHub repo, and when
you push to the HF Space, rename it to `README.md` there. See section
"Deploy to a Hugging Face Space" below.
============================================================================
-->

# Anti-VEGF Intolerance Prediction from Color Fundus Photographs

Upload a **color fundus photograph** to obtain:

1. an **anti-VEGF intolerance risk score** (0–1),
2. a **Grad-CAM** explanation heatmap, and
3. three interpretable **vascular biomarkers** (density, skeleton length, fractal dimension).

Built on the retinal foundation model **RETFound** (CC BY-NC 4.0, NonCommercial).
**Research prototype — not a medical device. Not for clinical use.**
No patient data are bundled with this Space.

## Model weights

The fine-tuned weights are downloaded automatically at startup from the project's
Zenodo record. To override, set a Space **secret/variable**:

- `WEIGHTS_URL` — a direct download URL for `best.pth`
  (default points at the Zenodo record), **or**
- `WEIGHTS_PATH` — a local path if you upload `best.pth` into the Space.

If no weights are found, the demo still runs but clearly labels its output as a
placeholder (random head).

## Links

- Code & full reproducibility: https://github.com/23008613g/CM-Oculomics
- Archive (DOI): https://doi.org/10.5281/zenodo.20537894
- RETFound: https://github.com/rmaphoh/RETFound_MAE

---

## Deploy to a Hugging Face Space (step by step)

1. Create a Space at https://huggingface.co/new-space
   - **SDK: Gradio**, **Hardware: CPU basic** (free) is enough for a demo;
     pick a small GPU if you want faster inference.
2. Clone the empty Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   ```
3. Copy these files from this project into the Space repo:
   ```
   app.py
   requirements.txt
   src/            (the whole folder)
   README_HFSpace.md   ->  rename to README.md  in the Space
   assets/example_fundus.jpg   (optional, non-patient example image)
   ```
   Do **not** copy the GitHub `README.md` (it lacks the YAML header HF needs).
4. (Optional) In the Space **Settings → Variables and secrets**, set
   `WEIGHTS_URL` to your `best.pth` download link if the default Zenodo URL is
   not yet live.
5. Commit & push:
   ```bash
   git add app.py requirements.txt src README.md
   git push
   ```
   The Space builds automatically and serves the Gradio app.

> Note: `pytorch-grad-cam`, `opencv-python`, `scikit-image`, `timm` are all in
> `requirements.txt`, so the Space installs everything it needs. The ViT-L model
> on CPU takes a few seconds per image on first run.
