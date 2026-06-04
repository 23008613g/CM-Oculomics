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

Weights: set env var WEIGHTS_PATH or place the checkpoint at weights/best.pth.
The model auto-falls back to a randomly-initialized head if no weights are found
(demo will run but predictions are meaningless until real weights are provided).
"""
import os
import tempfile
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
WEIGHTS = os.environ.get("WEIGHTS_PATH", "weights/best.pth")
THRESH = float(os.environ.get("THRESHOLD", "0.5"))

# ---- load model once ----
_model = FundusClassifier(num_classes=2, retfound_weights=None)
if os.path.exists(WEIGHTS):
    _model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    _WEIGHTS_OK = True
else:
    _WEIGHTS_OK = False
_model = _model.to(DEVICE).eval()
_tf = build_transforms(224, train=False)
_cam = GradCAM(model=_model, target_layers=[_model.backbone.blocks[-1].norm1],
               reshape_transform=reshape_transform)


def analyze(image: Image.Image):
    if image is None:
        return "Please upload a fundus image.", None, None
    pil = image.convert("RGB")

    # risk score
    x = _tf(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(_model(x), 1)[0, 1].item()
    label = "INTOLERANT (high risk)" if prob >= THRESH else "Tolerant (low risk)"
    warn = "" if _WEIGHTS_OK else \
        "\n⚠ No fine-tuned weights found — output is a placeholder. Set WEIGHTS_PATH."
    risk_text = (f"Anti-VEGF intolerance risk: {prob:.3f}\n"
                 f"Prediction (threshold {THRESH}): {label}{warn}\n\n"
                 f"Research prototype — NOT for clinical use.")

    # Grad-CAM
    rgb = np.array(pil.resize((224, 224))).astype(np.float32) / 255.0
    gray = _cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
    cam_img = Image.fromarray(show_cam_on_image(rgb, gray, use_rgb=True))

    # biomarkers (write to temp file for the cv2/PIL pipeline)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil.save(tmp.name)
        bm = compute_biomarkers(tmp.name)
    bm_text = "\n".join(f"{k.replace('_',' ')}: {v}" for k, v in bm.items())

    return risk_text, cam_img, bm_text


title = "Anti-VEGF Intolerance Prediction from Color Fundus Photographs"
desc = ("Upload a color fundus photograph to obtain an anti-VEGF intolerance risk score, "
        "a Grad-CAM explanation, and three interpretable vascular biomarkers. "
        "Built on RETFound (CC BY-NC 4.0, NonCommercial). Research prototype — not a medical device.")

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
    article="Weights are CC BY-NC 4.0 (inherited from RETFound). NonCommercial research use only.",
)

if __name__ == "__main__":
    demo.launch()
