# -*- coding: utf-8 -*-
"""
58_build_figS7.py - compose Supplementary Figure S7 from screenshots of the released
application running the retrained weights.

WHY: the original figure's source files were lost, and its three panels were screenshots of
the application driven by the pre-masking model, so every probability and biomarker in them
is superseded. The two example eyes were recovered by matching the biomarker values recorded
in Supplementary Note S3 against the whole cohort (exact matches: P0001_OS_01 low risk,
P0189_OS_01 high risk). The application was then run locally from release_repo/ with the
retrained weights, at the manuscript's Youden operating point (0.618) rather than the app
default of 0.50, and captured with headless Edge at 2x device scale.

Panel (a) shows the full landing page; panels (b) and (c) are cropped to the result region,
which is what the reader needs to see and what was illegible in the previous version. The
figure is built wider than the other supplementary figures because screenshot text needs the
pixels.

USAGE   python scripts/58_build_figS7.py
INPUT   results/figS7/panel_a_interface.png, panel_low.png, panel_high.png
OUTPUT  results/figS7/FigureS7_app_demo.png
"""
import os, sys
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "figS7")
DST = os.path.join(SRC, "FigureS7_app_demo.png")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PAGE_COLS = (76, 2124)          # the page content, without the browser gutters
FULL_ROWS = (0, 2950)           # whole landing page, trimmed below the footer
RESULT_ROWS = (1030, 3470)      # input + risk + Grad-CAM + biomarkers + footnote (longer since 2026-09-27)

BG = (255, 255, 255)
GAP = 54
MARGIN = 40
LABEL_H = 74
BORDER = (214, 219, 226)


def _font(size, bold=False):
    for name in (("arialbd.ttf", "seguisb.ttf") if bold else ("arial.ttf", "segoeui.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def looks_like_app(path):
    """Refuse a panel that is not a rendered app page.

    A failed capture (server not yet up, or already shut down) produces a browser error page:
    a ~100 kB, almost uniformly light image. A real capture is 0.8-2.3 MB and contains the app's
    dark header, attention panel and biomarker cards. This guard exists because a stray
    capture once overwrote a good panel with an ERR_CONNECTION_REFUSED page and the figure
    was rebuilt from it without anyone looking.
    """
    import numpy as np
    size = os.path.getsize(path)
    a = np.asarray(Image.open(path).convert("L"), dtype=float)
    dark = float((a < 80).mean())
    ok = size > 400_000 and dark > 0.05
    return ok, size, dark


def crop(name, rows):
    path = os.path.join(SRC, name)
    ok, size, dark = looks_like_app(path)
    if not ok:
        raise SystemExit("REFUSING to build: %s does not look like an app capture "
                         "(%d bytes, %.1f%% dark pixels). Recapture it and look at it."
                         % (name, size, 100 * dark))
    im = Image.open(path).convert("RGB")
    return im.crop((PAGE_COLS[0], rows[0], PAGE_COLS[1], min(rows[1], im.size[1])))


def scaled(im, w):
    return im.resize((w, int(round(im.size[1] * w / im.size[0]))), Image.LANCZOS)


def main():
    a = crop("panel_a_interface.png", FULL_ROWS)
    b = crop("panel_low.png", RESULT_ROWS)
    c = crop("panel_high.png", RESULT_ROWS)
    print("crops: a %s | b %s | c %s" % (a.size, b.size, c.size))

    # right column holds b over c; the left column (a) matches their combined height
    wr = 980
    b_s, c_s = scaled(b, wr), scaled(c, wr)
    right_h = b_s.size[1] + LABEL_H + c_s.size[1] + LABEL_H + GAP
    a_s = scaled(a, int(round(a.size[0] * (right_h - LABEL_H) / a.size[1])))

    W = MARGIN * 2 + a_s.size[0] + GAP + wr
    H = MARGIN * 2 + right_h
    fig = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(fig)
    f_lab = _font(46, bold=True)
    f_cap = _font(34)

    def place(im, x, y, letter, caption):
        d.text((x, y), letter, font=f_lab, fill=(0, 0, 0))
        tw = d.textlength(letter, font=f_lab)
        d.text((x + tw + 18, y + 10), caption, font=f_cap, fill=(45, 50, 58))
        fig.paste(im, (x, y + LABEL_H))
        d.rectangle([x, y + LABEL_H, x + im.size[0] - 1, y + LABEL_H + im.size[1] - 1],
                    outline=BORDER, width=2)

    x0 = MARGIN
    place(a_s, x0, MARGIN, "a", "Application interface")
    x1 = x0 + a_s.size[0] + GAP
    place(b_s, x1, MARGIN, "b", "Lower-risk score  (0.397; eye remained NPDR)")
    place(c_s, x1, MARGIN + LABEL_H + b_s.size[1] + GAP, "c",
          "Higher-risk score  (0.971; eye progressed to PDR)")

    fig.save(DST, format="PNG", optimize=True)
    print("wrote %s  %dx%d  %.2f MB"
          % (DST, fig.size[0], fig.size[1], os.path.getsize(DST) / 1e6))


if __name__ == "__main__":
    main()
