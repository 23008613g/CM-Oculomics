# -*- coding: utf-8 -*-
"""
Gradio demo — Hugging Face Space ready.

Input : one color fundus photograph
Output: (1) anti-VEGF intolerance risk score
        (2) Grad-CAM heatmap
        (3) three vascular imaging biomarkers
            (vascular density / skeleton length / fractal dimension)

LICENSE / COMPLIANCE: this demo uses RETFound-derived weights (CC BY-NC 4.0).
NonCommercial research use only; attribute RETFound; indicate modifications.
This demo is a research prototype and NOT a medical device — not for clinical use.
NO patient data is bundled with this Space.

Weights resolution order:
  1. env var WEIGHTS_PATH (local path), if it exists;
  2. a local weights/best.pth, if it exists;
  3. auto-download from the Zenodo record (env WEIGHTS_URL, default below);
  4. fall back to a randomly-initialized head (demo runs but predictions are
     meaningless) and the UI clearly says so.
"""
import os
import tempfile
import urllib.request

import numpy as np
import torch
import gradio as gr
from PIL import Image

from src.model import FundusClassifier
from src.dataset import build_transforms
from src.biomarkers import compute_biomarkers
from src.gradcam import reshape_transform
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = float(os.environ.get("THRESHOLD", "0.5"))

# Direct-download URL for the fine-tuned weights. By default we point at the
# Zenodo record for this project; override with the WEIGHTS_URL env var.
# (The file name on Zenodo must be best.pth, or adjust the URL accordingly.)
WEIGHTS_URL = os.environ.get(
    "WEIGHTS_URL",
    "https://zenodo.org/records/20537895/files/best.pth?download=1",
)
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "weights/best.pth")


def _resolve_weights() -> str | None:
    """Return a local path to the weights, downloading from Zenodo if needed."""
    if os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH
    # try to download
    if WEIGHTS_URL:
        try:
            os.makedirs(os.path.dirname(WEIGHTS_PATH) or ".", exist_ok=True)
            print(f"[weights] downloading from {WEIGHTS_URL} ...")
            urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
            if os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 1_000_000:
                print("[weights] download OK")
                return WEIGHTS_PATH
            print("[weights] downloaded file looks too small; ignoring")
        except Exception as e:  # noqa: BLE001 - demo must not crash on network errors
            print(f"[weights] download failed: {e}")
    return None


# ---- load model once ----
_model = FundusClassifier(num_classes=2, retfound_weights=None)
_wpath = _resolve_weights()
if _wpath:
    _model.load_state_dict(torch.load(_wpath, map_location="cpu"))
    _WEIGHTS_OK = True
else:
    _WEIGHTS_OK = False
_model = _model.to(DEVICE).eval()
_tf = build_transforms(224, train=False)
_cam = GradCAM(model=_model, target_layers=[_model.backbone.blocks[-1].norm1],
               reshape_transform=reshape_transform)

_DEVICE_NOTE = ("Running on GPU." if DEVICE == "cuda"
                else "Running on CPU — first inference may take a few seconds.")


def analyze(image: Image.Image):
    if image is None:
        return "Please upload a fundus image.", None, None
    pil = image.convert("RGB")

    # --- risk score (no_grad is fine here) ---
    x = _tf(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(_model(x), 1)[0, 1].item()
    label = "INTOLERANT (high risk)" if prob >= THRESH else "Tolerant (low risk)"
    warn = "" if _WEIGHTS_OK else (
        "\n[!] No fine-tuned weights available — this output is a PLACEHOLDER "
        "(random head). Provide WEIGHTS_PATH / WEIGHTS_URL for real predictions."
    )
    risk_text = (f"Anti-VEGF intolerance risk: {prob:.3f}\n"
                 f"Prediction (threshold {THRESH}): {label}{warn}\n\n"
                 f"{_DEVICE_NOTE}\nResearch prototype — NOT for clinical use.")

    # --- Grad-CAM (needs gradients; must NOT be inside torch.no_grad) ---
    rgb = np.array(pil.resize((224, 224))).astype(np.float32) / 255.0
    x_cam = _tf(pil).unsqueeze(0).to(DEVICE)
    grayscale = _cam(input_tensor=x_cam,
                     targets=[ClassifierOutputTarget(1)])[0]
    cam_img = Image.fromarray(show_cam_on_image(rgb, grayscale, use_rgb=True))

    # --- biomarkers (write to a temp file for the cv2/PIL pipeline) ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil.save(tmp.name)
        bm = compute_biomarkers(tmp.name)
    bm_text = "\n".join(f"{k.replace('_', ' ')}: {v}" for k, v in bm.items())

    return risk_text, cam_img, bm_text


title = "Anti-VEGF Intolerance Prediction from Color Fundus Photographs"
desc = ("Upload a color fundus photograph to obtain an anti-VEGF intolerance risk "
        "score, a Grad-CAM explanation, and three interpretable vascular biomarkers "
        "(density, skeleton length, fractal dimension). Built on RETFound "
        "(CC BY-NC 4.0, NonCommercial). Research prototype — not a medical device.")

# wire up an example image if a non-patient sample is shipped in assets/
_examples = None
_example_path = "assets/example_fundus.jpg"
if os.path.exists(_example_path):
    _examples = [[_example_path]]

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil", label="Color fundus photograph"),
    outputs=[
        gr.Textbox(label="Intolerance risk"),
        gr.Image(label="Grad-CAM explanation"),
        gr.Textbox(label="Vascular imaging biomarkers"),
    ],
    title=title,
    description=desc,
    examples=_examples,
    article=("Weights are CC BY-NC 4.0 (inherited from RETFound). NonCommercial "
             "research use only. No patient data are bundled with this Space."),
)

if __name__ == "__main__":
    demo.launch()
