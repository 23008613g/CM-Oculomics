---
title: CM-Oculomics
emoji: 👁️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

<!--
============================================================================
HOW TO USE THIS FILE
============================================================================
This is the README for a *Hugging Face Space*. On a Space the file MUST be named
exactly `README.md`, and the YAML block above tells HF to launch the Gradio app
from `app.py`. In the GitHub repo it is kept as `README_HFSpace.md` (so it does
not overwrite the developer README); when you push to the Space, rename it to
`README.md` there. Deploy steps are at the bottom.
============================================================================
-->

# CM-Oculomics

Upload a **color fundus photograph** to obtain:

1. an **anti-VEGF intolerance risk score** (0–1),
2. a **Grad-CAM** explanation heatmap, and
3. three interpretable **vascular biomarkers** (density, skeleton length, fractal dimension).

Built on the generalist vision foundation model **DINOv2 (ViT-L/14)**, fine-tuned for
anti-VEGF intolerance prediction. Weights are released under **Apache-2.0**.
**Research prototype — not a medical device. Not for clinical use.**
No patient data are bundled with this Space.

## Model weights

The fine-tuned weights (`dino_deploy.pth`, ~1.2 GB) are downloaded at startup from a
Hugging Face model repository. Set a Space **variable**:

- `WEIGHTS_URL` — direct download URL, e.g.
  `https://huggingface.co/<your-user>/CM-Oculomics/resolve/main/dino_deploy.pth`

(or `WEIGHTS_PATH` if you upload the file directly into the Space). If no weights are
found the demo still runs but clearly labels its output as a placeholder.

## Links

- Code & full reproducibility: https://github.com/23008613g/CM-Oculomics
- Archive (DOI): https://doi.org/10.5281/zenodo.20537894

---

## Deploy to a Hugging Face Space (step by step)

> Prerequisites: a free Hugging Face account and the CLI
> (`pip install -U huggingface_hub`); log in once with `huggingface-cli login`.

**1. Host the weights in a HF model repo** (one-time):

```bash
huggingface-cli repo create CM-Oculomics --type model           # -> <your-user>/CM-Oculomics
huggingface-cli upload <your-user>/CM-Oculomics \
    "path/to/dino_deploy.pth" dino_deploy.pth                    # uploads the 1.2 GB checkpoint (LFS)
```

**2. Create the Space**: huggingface.co → **New → Space** → SDK **Gradio**,
hardware **CPU basic** (free) is enough. This creates
`https://huggingface.co/spaces/<your-user>/CM-Oculomics`.

**3. Push the app to the Space**:

```bash
git clone https://huggingface.co/spaces/<your-user>/CM-Oculomics space && cd space
cp ../app.py ../requirements.txt .
cp -r ../src .
cp ../README_HFSpace.md README.md          # the Space README MUST be named README.md
# (optional) cp -r ../assets .
git add . && git commit -m "CM-Oculomics demo (DINOv2)" && git push
```

**4. Point the Space at the weights**: Space → **Settings → Variables and secrets**
→ add variable `WEIGHTS_URL` =
`https://huggingface.co/<your-user>/CM-Oculomics/resolve/main/dino_deploy.pth`.

The Space builds, downloads the weights on first boot (a few minutes for 1.2 GB;
cached afterwards), and serves a public URL. On free CPU, inference is a few
seconds per image — fine for a demo; upgrade to a small GPU for snappier response.

> `timm`, `pytorch-grad-cam` (`grad-cam`), `opencv-python`, `scikit-image` are all in
> `requirements.txt`, so the Space installs everything it needs.
