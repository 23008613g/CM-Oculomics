# -*- coding: utf-8 -*-
"""
Gradio demo — Hugging Face Space ready.

Input : one color fundus photograph
Output: (1) anti-VEGF intolerance risk score (color-coded clinical card)
        (2) Grad-CAM heatmap (explains the PREDICTED class)
        (3) three vascular imaging biomarkers (as a table)

LICENSE / COMPLIANCE: this demo uses a DINOv2 (ViT-L/14) backbone fine-tuned in-house;
the released weights are distributed under Apache-2.0 (DINOv2's permissive license).
This demo is a research prototype and NOT a medical device — not for clinical use.
NO patient data is bundled with this Space.

Weights resolution order:
  1. env var WEIGHTS_PATH (local path), if it exists;
  2. a local weights/dino_deploy.pth, if it exists;
  3. auto-download from WEIGHTS_URL (e.g. the Hugging Face model repo), if set;
  4. fall back to a randomly-initialized head (demo runs but predictions are
     meaningless) and the UI clearly says so.
"""
import os
import tempfile
import urllib.request

import numpy as np
import cv2
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

WEIGHTS_URL = os.environ.get("WEIGHTS_URL", "")  # set to the Zenodo DINOv2 weights URL once uploaded
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "weights/dino_deploy.pth")

# colormap for the heatmap: TURBO (20) is perceptually clearer than JET
_CMAP = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)


def _resolve_weights() -> str | None:
    """Return a local path to the weights, downloading from Zenodo if needed."""
    if os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH
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


# ---- build DINOv2 (ViT-L/14) — primary backbone, matches the paper ----
import timm
import torch.nn as nn
_GRID = 224 // 14  # 16x16 patch-token grid

def _build_model():
    backbone = timm.create_model("vit_large_patch14_dinov2", pretrained=False,
                                 num_classes=0, img_size=224, drop_rate=0.2)
    head = nn.Sequential(nn.LayerNorm(backbone.num_features), nn.Dropout(0.2),
                         nn.Linear(backbone.num_features, 2))
    return nn.Sequential(backbone, head)

def _dino_reshape(t, s=_GRID):
    x = t[:, -s * s:, :]                      # keep patch tokens (drop cls/register tokens)
    return x.reshape(t.size(0), s, s, t.size(2)).permute(0, 3, 1, 2)

# ---- load model once ----
_model = _build_model()
_wpath = _resolve_weights()
if _wpath:
    _model.load_state_dict(torch.load(_wpath, map_location="cpu"))
    _WEIGHTS_OK = True
else:
    _WEIGHTS_OK = False
_model = _model.to(DEVICE).eval()
_tf = build_transforms(224, train=False)
_cam = GradCAM(model=_model, target_layers=[_model[0].blocks[-1].norm1],
               reshape_transform=_dino_reshape)

_DEVICE_NOTE = ("GPU" if DEVICE == "cuda"
                else "CPU — first inference may take a few seconds")

_BM_META = {
    "vascular_density": ("Vascular density", "fraction of FOV"),
    "vascular_skeleton_length": ("Vascular-skeleton length", "normalized"),
    "vascular_fractal_dimension": ("Vascular fractal dimension", "box-counting"),
}

# Group means from the study (tolerant=NPDR vs intolerant=PDR). All three
# biomarkers are HIGHER in intolerant eyes (neovascularization burden); we use
# the midpoint of the two group means as an interpretive reference. This is a
# population-level interpretation, NOT a diagnosis.
_BM_REF = {
    # key: (tolerant_mean, intolerant_mean)
    "vascular_density": (0.497, 0.633),
    "vascular_skeleton_length": (0.275, 0.379),
    "vascular_fractal_dimension": (1.734, 1.801),
}


def _bm_interpret(key, val):
    """Return (level_label, css_class, text) for a biomarker value."""
    if key not in _BM_REF:
        return ("", "mid", "")
    lo, hi = _BM_REF[key]
    mid = (lo + hi) / 2.0
    if val >= hi:
        return ("High", "hi",
                "above the intolerant-group average — pattern associated with "
                "higher neovascularization burden")
    if val >= mid:
        return ("Elevated", "hi",
                "above the midpoint between groups — leans toward the "
                "intolerant pattern")
    if val >= lo:
        return ("Borderline", "mid",
                "between group averages — intermediate")
    return ("Low", "lo",
            "below the tolerant-group average — pattern associated with lower "
            "neovascularization burden")


def analyze(image):
    if image is None:
        return (
            "<div class='risk-card neutral'><b>Please upload a fundus image to begin.</b></div>",
            None,
            "<div class='bm-empty'>Biomarkers will appear here after analysis.</div>",
        )
    pil = image.convert("RGB")

    # --- risk score ---
    x = _tf(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(_model(x), 1)[0]
    prob = probs[1].item()                       # P(intolerant)
    pred_class = int(probs.argmax().item())      # the class the model predicts
    high = prob >= THRESH
    pct = prob * 100.0
    label = "INTOLERANT — high risk" if high else "TOLERANT — low risk"
    css_cls = "high" if high else "low"
    warn = "" if _WEIGHTS_OK else (
        "<div class='warn'>⚠ No fine-tuned weights available — this output is a "
        "PLACEHOLDER (random head). Set WEIGHTS_PATH / WEIGHTS_URL for real "
        "predictions.</div>"
    )
    interp = ("Higher probability suggests the eye may be less responsive to "
              "anti-VEGF therapy; such cases may warrant closer follow-up or "
              "earlier consideration of escalation. "
              if high else
              "Lower probability suggests the eye is more likely to respond to "
              "anti-VEGF therapy. ")
    risk_html = f"""
    <div class='risk-card {css_cls}'>
      <div class='risk-row'>
        <span class='risk-dot'></span>
        <span class='risk-label'>{label}</span>
      </div>
      <div class='risk-value'>{prob:.3f}</div>
      <div class='risk-sub'>probability of anti-VEGF intolerance</div>
      <div class='bar'><div class='bar-fill' style='width:{pct:.1f}%'></div>
        <div class='bar-thresh' style='left:{THRESH*100:.0f}%'></div></div>
      <div class='risk-meta'>decision threshold {THRESH:.2f} &nbsp;·&nbsp; running on {_DEVICE_NOTE}</div>
      <div class='risk-interp'>{interp}</div>
      {warn}
    </div>
    """

    # --- Grad-CAM: explain the PREDICTED class (fixes the "flat" heatmap) ---
    rgb = np.array(pil.resize((224, 224))).astype(np.float32) / 255.0
    grayscale = _cam(input_tensor=_tf(pil).unsqueeze(0).to(DEVICE),
                     targets=[ClassifierOutputTarget(pred_class)])[0]
    # robust per-image normalization for crisp contrast
    g = grayscale - grayscale.min()
    g = g / (g.max() + 1e-8)
    cam_img = Image.fromarray(
        show_cam_on_image(rgb, g, use_rgb=True, colormap=_CMAP, image_weight=0.5)
    )

    # --- biomarkers -> HTML cards (fully styled, dark-theme safe) ---
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil.save(tmp.name)
        bm = compute_biomarkers(tmp.name)
    items = ""
    for key, val in bm.items():
        name, unit = _BM_META.get(key, (key.replace("_", " "), ""))
        level, lvl_cls, text = _bm_interpret(key, float(val))
        badge = (f"<span class='bm-badge {lvl_cls}'>{level}</span>"
                 if level else "")
        note = f"<div class='bm-note'>{text}</div>" if text else ""
        items += (
            f"<div class='bm-item'>"
            f"<div class='bm-top'>"
            f"<div class='bm-name'>{name}</div>"
            f"<div class='bm-val'>{val}</div>"
            f"<div class='bm-unit'>{unit}</div></div>"
            f"<div class='bm-bottom'>{badge}{note}</div>"
            f"</div>"
        )
    bm_html = (f"<div class='bm-grid'>{items}</div>"
               "<div class='bm-foot'>Interpretation is relative to study "
               "group averages (tolerant vs intolerant); higher vascular "
               "density, skeleton length, and fractal dimension reflect greater "
               "neovascularization burden. Population-level context, not a "
               "diagnosis.</div>")

    return risk_html, cam_img, bm_html


# ---------------------------------------------------------------------------
# UI  — clinical / medical styling
# ---------------------------------------------------------------------------
CSS = """
:root {
  --bg:#f4f8fa; --panel:#ffffff; --ink:#10242b; --muted:#5a6b73;
  --line:#dbe7ea; --teal:#0b6e6e; --teal-d:#08504f; --cyan:#10b3c4;
  --blue:#1f6feb; --green:#15936a; --green-d:#0c6647;
  --red:#c23a36; --red-d:#8f2622;
  --mono:'JetBrains Mono','SFMono-Regular',ui-monospace,monospace;
}
/* ===== light base with subtle tech grid ===== */
.gradio-container {max-width:1160px !important; margin:auto; color:var(--ink) !important;
  background:
    radial-gradient(900px 360px at 50% -120px, rgba(16,179,196,.10), transparent 70%),
    linear-gradient(rgba(11,110,110,.04) 1px, transparent 1px) 0 0/28px 28px,
    linear-gradient(90deg, rgba(11,110,110,.04) 1px, transparent 1px) 0 0/28px 28px,
    var(--bg) !important;}
.gradio-container .gr-check-radio label, .gradio-container label span {
  cursor:pointer !important;}
.gradio-container input[type=radio], .gradio-container input[type=checkbox] {
  width:18px !important; height:18px !important; cursor:pointer !important;
  accent-color:var(--teal) !important; appearance:auto !important;
  -webkit-appearance:auto !important; opacity:1 !important;}
/* ===== header: dark tech bar (kept for a touch of sci-fi) ===== */
#hdr {text-align:center; padding:24px 18px 18px; border-radius:18px;
  background:linear-gradient(135deg,#06363b 0%,#0b4f55 55%,#0a2c45 100%);
  border:1px solid rgba(16,179,196,.30);
  box-shadow:0 10px 30px rgba(6,40,42,.20), inset 0 0 60px rgba(16,179,196,.08);
  position:relative; overflow:hidden;}
#hdr:before {content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,transparent,#3fe0ec,transparent);
  animation:scan 5s linear infinite;}
@keyframes scan {0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
#hdr h1 {font-size:1.6rem; margin:0; font-weight:800; letter-spacing:.3px;
  background:linear-gradient(90deg,#d7fbf8,#3fe0ec 55%,#9fd2ff);
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; position:relative;}
#hdr .tag {display:inline-block; margin-top:11px; padding:4px 15px;
  border:1px solid rgba(63,224,236,.5); border-radius:999px; color:#bff4f1;
  background:rgba(63,224,236,.10); font-size:.76rem; font-weight:700;
  letter-spacing:.9px; text-transform:uppercase; position:relative;}
#sub {text-align:center; color:var(--muted); font-size:.92rem; margin:14px auto 4px;
  max-width:800px; line-height:1.6;}
.section-title {font-size:.74rem; font-weight:800; color:var(--teal);
  text-transform:uppercase; letter-spacing:1.2px; margin:2px 0 8px;
  display:flex; align-items:center; gap:8px;}
.section-title:before {content:''; width:8px; height:8px; border-radius:2px;
  background:var(--cyan); box-shadow:0 0 8px var(--cyan);}
/* ===== risk card (kept colorful: clinical semantics) ===== */
.risk-card {border-radius:16px; padding:18px 20px; position:relative;
  overflow:hidden; color:#fff;}
.risk-card * {color:#fff !important;}
.risk-card.high {background:linear-gradient(135deg,#c23a36,#7a1f1c);
  box-shadow:0 8px 24px rgba(194,58,54,.30);}
.risk-card.low {background:linear-gradient(135deg,#15936a,#0a5238);
  box-shadow:0 8px 24px rgba(21,147,106,.28);}
.risk-card.neutral {background:linear-gradient(135deg,#5c6f76,#3f4f55);}
.risk-row {display:flex; align-items:center; gap:8px;}
.risk-dot {width:10px; height:10px; border-radius:50%; background:#fff;
  box-shadow:0 0 0 4px rgba(255,255,255,.22), 0 0 12px rgba(255,255,255,.8);}
.risk-label {font-size:1.02rem; font-weight:800; letter-spacing:.4px;}
.risk-value {font-family:var(--mono); font-size:3rem; font-weight:800;
  line-height:1.05; margin:6px 0 0; text-shadow:0 0 22px rgba(255,255,255,.35);}
.risk-sub {opacity:.92; font-size:.84rem;}
.bar {position:relative; height:10px; background:rgba(255,255,255,.28);
  border-radius:8px; margin:14px 0 6px;}
.bar-fill {position:absolute; height:100%; border-radius:8px; background:#fff;
  box-shadow:0 0 12px rgba(255,255,255,.7);}
.bar-thresh {position:absolute; top:-4px; width:2px; height:18px; background:#ffe08a;}
.risk-meta {font-family:var(--mono); font-size:.74rem; opacity:.9; margin-top:6px;}
.risk-interp {font-size:.82rem; opacity:.96; margin-top:10px;
  border-top:1px solid rgba(255,255,255,.26); padding-top:8px;}
.warn {background:#fff3cd !important; padding:8px 10px;
  border:1px solid rgba(122,91,0,.3); border-radius:8px; margin-top:10px;
  font-size:.82rem;}
.warn * {color:#7a5b00 !important;}
#foot {text-align:center; color:var(--muted); font-size:.78rem; margin-top:18px;
  line-height:1.6;}
#foot a {color:var(--teal); text-decoration:none; font-weight:600;}
/* Grad-CAM image framed like a viewport */
.cam-frame {border-radius:14px !important; padding:6px !important;
  background:linear-gradient(135deg,#06363b,#0a2c3a) !important;
  border:1px solid rgba(16,179,196,.4) !important;
  box-shadow:0 6px 18px rgba(6,40,42,.22), inset 0 0 30px rgba(16,179,196,.08) !important;}
.cam-frame img {border-radius:9px !important;
  box-shadow:0 0 0 1px rgba(63,224,236,.35) !important;}
.cam-frame, .cam-frame * {border-color:rgba(16,179,196,.4) !important;}
.cam-note {color:var(--muted); font-size:.78rem; margin-top:6px; text-align:center;}
/* ===== model card ===== */
.mcard {background:var(--panel); border:1px solid var(--line); border-radius:14px;
  padding:14px 16px; box-shadow:0 2px 10px rgba(8,80,79,.06);}
.mcard, .mcard * {color:var(--ink) !important;}
.mcard .row {display:flex; justify-content:space-between; padding:6px 0;
  border-bottom:1px dashed var(--line); font-size:.85rem;}
.mcard .row:last-child {border-bottom:none;}
.mcard .k {color:var(--muted) !important;}
.mcard .v {color:var(--teal) !important; font-weight:700; font-family:var(--mono);}
.mcard-note {margin-top:10px; font-size:.78rem; color:var(--muted) !important;
  line-height:1.5;}
.mcard-note a {color:var(--teal) !important; text-decoration:none;}
/* ===== biomarker cards (dark tech readout, echoes the risk card) ===== */
.bm-grid {display:flex; flex-direction:column; gap:8px;}
.bm-item {padding:12px 15px; border-radius:13px; position:relative; overflow:hidden;
  background:linear-gradient(135deg,#06363b,#0a2c3a);
  border:1px solid rgba(16,179,196,.32);
  box-shadow:0 6px 18px rgba(6,40,42,.22), inset 0 0 34px rgba(16,179,196,.07);}
.bm-item:after {content:''; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(transparent 50%, rgba(255,255,255,.025) 50%);
  background-size:100% 4px;}
.bm-item, .bm-item * {color:#dcf6f4 !important;}
.bm-top {display:flex; align-items:baseline; gap:10px; position:relative;}
.bm-name {flex:1; color:#bfe9e6 !important; font-size:.84rem; font-weight:600;}
.bm-val {font-family:var(--mono); font-size:1.45rem; font-weight:800;
  color:#3fe0ec !important; text-shadow:0 0 16px rgba(63,224,236,.55);
  min-width:80px; text-align:right; letter-spacing:.5px;}
.bm-unit {color:#8fb6b3 !important; font-size:.7rem; min-width:96px;
  text-align:right;}
.bm-bottom {display:flex; align-items:center; gap:8px; margin-top:8px;
  position:relative;}
.bm-badge {font-size:.66rem; font-weight:800; letter-spacing:.5px;
  text-transform:uppercase; padding:2px 8px; border-radius:999px;
  border:1px solid; white-space:nowrap;}
.bm-badge.hi {color:#ffb3b1 !important; border-color:rgba(255,120,116,.55);
  background:rgba(255,90,87,.16);}
.bm-badge.mid {color:#ffe08a !important; border-color:rgba(255,224,138,.5);
  background:rgba(255,224,138,.14);}
.bm-badge.lo {color:#8ff0c6 !important; border-color:rgba(25,201,138,.5);
  background:rgba(25,201,138,.16);}
.bm-note {color:#a9d4d1 !important; font-size:.74rem; line-height:1.4;}
.bm-foot {color:var(--muted); font-size:.73rem; line-height:1.5; margin-top:10px;
  padding-top:8px; border-top:1px solid var(--line);}
.bm-empty {color:var(--muted); font-size:.82rem; padding:14px;
  border:1px dashed var(--line); border-radius:12px; text-align:center;}
/* ===== steps ===== */
.steps {display:flex; gap:12px; margin:4px 0 2px;}
.step {flex:1; background:var(--panel); border:1px solid var(--line);
  border-radius:14px; padding:13px 15px; transition:.2s;
  box-shadow:0 2px 10px rgba(8,80,79,.06);}
.step:hover {border-color:var(--cyan); transform:translateY(-2px);
  box-shadow:0 6px 18px rgba(16,179,196,.18);}
.step .num {display:inline-flex; width:28px; height:28px; border-radius:8px;
  background:linear-gradient(135deg,var(--teal),var(--cyan)); color:#fff;
  align-items:center; justify-content:center; font-weight:800; font-size:.85rem;
  font-family:var(--mono); box-shadow:0 0 14px rgba(16,179,196,.4);}
.step .t {font-weight:800; color:var(--ink) !important; margin:8px 0 2px;
  font-size:.92rem;}
.step .d {color:var(--muted) !important; font-size:.8rem; line-height:1.45;}
/* ===== buttons ===== */
.gradio-container button.primary, .gradio-container .primary {
  background:linear-gradient(135deg,var(--teal),var(--cyan)) !important;
  color:#fff !important; border:none !important; font-weight:800 !important;
  box-shadow:0 4px 16px rgba(16,179,196,.32) !important;}
/* hide Gradio's auto-localized footer (Use via API / Settings / Built with Gradio)
   so the page stays consistently English regardless of the viewer's browser locale */
footer {display:none !important;}
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)

_example_path = "assets/example_fundus.jpg"
_has_example = os.path.exists(_example_path)

with gr.Blocks(theme=THEME, css=CSS, title="CM-Oculomics — Anti-VEGF Intolerance Prediction") as demo:
    gr.HTML(
        "<div id='hdr'><h1>CM-Oculomics</h1>"
        "<div style='font-size:1.05rem;font-weight:500;color:rgba(255,255,255,0.58);margin:2px 0 5px'>"
        "Anti-VEGF Intolerance Prediction from Color Fundus Photographs</div>"
        "<span class='tag'>Interpretable AI · Diabetic Retinopathy · Integrative Medicine</span></div>"
    )
    gr.HTML(
        "<div id='sub'>Predict anti-VEGF intolerance, visualize the supporting "
        "evidence with Grad-CAM, and quantify retinal vascular biomarkers — "
        "from a single low-cost fundus image.</div>"
    )

    # --- How it works ---
    gr.HTML("<div class='section-title' style='margin-top:6px'>How it works</div>")
    gr.HTML(
        "<div class='steps'>"
        "<div class='step'><span class='num'>1</span>"
        "<div class='t'>Upload</div><div class='d'>Provide one color fundus "
        "photograph (either eye).</div></div>"
        "<div class='step'><span class='num'>2</span>"
        "<div class='t'>Encode &amp; classify</div><div class='d'>A fine-tuned "
        "DINOv2 vision foundation model estimates intolerance risk.</div></div>"
        "<div class='step'><span class='num'>3</span>"
        "<div class='t'>Explain</div><div class='d'>Grad-CAM and vascular "
        "biomarkers show the evidence behind the score.</div></div>"
        "</div>"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            with gr.Group():
                gr.HTML("<div class='section-title'>Input</div>")
                inp = gr.Image(type="pil", label="Color fundus photograph",
                               height=340)
                with gr.Row():
                    btn = gr.Button("Analyze", variant="primary", scale=3)
                    clr = gr.ClearButton(value="Reset", scale=1)
                if _has_example:
                    gr.Examples(examples=[[_example_path]], inputs=inp,
                                label="Example (public DDR sample)")

            # --- Model card (descriptive only; no performance numbers) ---
            gr.HTML("<div class='section-title' style='margin-top:14px'>About the model</div>")
            gr.HTML(
                "<div class='mcard'>"
                "<div class='row'><span class='k'>Backbone</span>"
                "<span class='v'>DINOv2 ViT-L/14 (vision foundation model)</span></div>"
                "<div class='row'><span class='k'>Task</span>"
                "<span class='v'>anti-VEGF intolerance (binary)</span></div>"
                "<div class='row'><span class='k'>Training data</span>"
                "<span class='v'>de-identified DR fundus images</span></div>"
                "<div class='row'><span class='k'>Explainability</span>"
                "<span class='v'>Grad-CAM + vascular biomarkers</span></div>"
                "<div class='mcard-note'>Full methodology and evaluation are "
                "reported in the accompanying paper and "
                "<a href='https://github.com/23008613g/CM-Oculomics'>code "
                "repository</a>.</div>"
                "</div>"
            )

        with gr.Column(scale=6):
            gr.HTML("<div class='section-title'>Risk assessment</div>")
            risk_out = gr.HTML()
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>Model attention (Grad-CAM)</div>")
                    cam_out = gr.Image(label=None, height=240, show_label=False,
                                       elem_classes="cam-frame")
                    gr.HTML("<div class='cam-note'>Warm colors = regions driving "
                            "the prediction</div>")
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-title'>Vascular biomarkers</div>")
                    bm_out = gr.HTML(
                        "<div class='bm-empty'>Biomarkers will appear here "
                        "after analysis.</div>"
                    )

    gr.HTML(
        "<div id='foot'><b>For research use only.</b><br>"
        "Model: DINOv2 ViT-L/14, fine-tuned in-house · weights under Apache-2.0. "
        "No patient data are bundled.<br>"
        "<a href='https://github.com/23008613g/CM-Oculomics'>Code</a></div>"
    )

    btn.click(analyze, inputs=inp, outputs=[risk_out, cam_out, bm_out])
    clr.add([inp, risk_out, cam_out, bm_out])

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
                show_api=False)
