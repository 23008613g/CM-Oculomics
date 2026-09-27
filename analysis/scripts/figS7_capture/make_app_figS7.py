# -*- coding: utf-8 -*-
"""
make_app_figS7.py - rebuild the Figure S7 capture copy of the application from release_repo/app.py.

WHY: Figure S7 must show exactly what the released application shows. The capture copy differs
from app.py only in that it can start in the post-Analyze state (Gradio serialises component
values at construction, so the result has to be passed as `value=`). Rebuilding it from app.py
every time, instead of editing a separate copy, keeps the wording in the figure identical to the
wording users see.

USAGE   python scripts/figS7_capture/make_app_figS7.py
OUTPUT  scripts/figS7_capture/app_figS7.py
"""
import os, io

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "..", "release_repo", "app.py")
s = io.open(APP, encoding="utf-8").read()

PREFILL = '''
# ---------------------------------------------------------------- Figure S7 prefill
# Set FIGS7_IMAGE to render the page in its post-Analyze state, for figure capture only.
_PF_IN = _PF_RISK = _PF_CAM = None
_PF_BM = "<div class='bm-empty'>Biomarkers will appear here after analysis.</div>"
_pf_path = os.environ.get("FIGS7_IMAGE", "")
if _pf_path:
    _PF_IN = Image.open(_pf_path).convert("RGB")
    _PF_RISK, _PF_CAM, _PF_BM = analyze(_PF_IN)
    print("[figS7] prefilled from %s" % os.path.basename(_pf_path), flush=True)

'''
anchor = "with gr.Blocks("
assert s.count(anchor) == 1
s = s.replace(anchor, PREFILL + anchor)
subs = [
    ('inp = gr.Image(type="pil", label="Color fundus photograph",',
     'inp = gr.Image(type="pil", label="Color fundus photograph", value=_PF_IN,'),
    ("risk_out = gr.HTML()", "risk_out = gr.HTML(_PF_RISK)"),
    ("cam_out = gr.Image(label=None, height=240, show_label=False,",
     "cam_out = gr.Image(label=None, height=240, show_label=False, value=_PF_CAM,"),
    ('''bm_out = gr.HTML(
                        "<div class='bm-empty'>Biomarkers will appear here "
                        "after analysis.</div>"
                    )''', "bm_out = gr.HTML(_PF_BM)"),
]
for a, b in subs:
    assert s.count(a) == 1, a[:50]
    s = s.replace(a, b)
io.open(os.path.join(HERE, "app_figS7.py"), "w", encoding="utf-8").write(s)
print("wrote app_figS7.py from app.py")
