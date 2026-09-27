# -*- coding: utf-8 -*-
"""
62_overlay_audit.py - which burned-in overlays remain in the training images, and do they
track the outcome?

WHY: 49_mask_overlays.py removed identifying text from the Canon and Topcon images but copied
the VISUCAM 200 images unchanged, because their overlays (an 'OD/OS' laterality label, top
left, and a fixation-target indicator, bottom right) do not identify a patient. VISUCAM is also
the device with by far the highest PDR rate, so those marks are a device signature and hence a
candidate shortcut. Blind reviewers noticed that every intolerant example in Fig. 4a carries
the marks and no tolerant example does. This script measures, on the images the models were
actually trained on (data_anon/images, i.e. after masking):
  * overlay presence by device and by outcome class;
  * whether, WITHIN VISUCAM, overlay presence varies at all (if every VISUCAM image carries it,
    it cannot drive discrimination inside that stratum, where the model's AUC is 0.962);
  * the device of each eye shown in Fig. 4a.

Overlay pixels are bright pixels OUTSIDE the fundus disc inside the two corner boxes; the disc
is found with the same routine as the masking script, so retina is never counted.

USAGE   python scripts/62_overlay_audit.py
OUTPUT  results/overlay_audit.json
"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
IMG = os.path.join(ROOT, "data_anon", "images")          # the masked images used for training
SPLITS = os.path.join(ROOT, "data_anon", "splits.csv")
DEVICE = {(2400, 2040): "VISUCAM 200", (720, 576): "Topcon + frame grabber",
          (2592, 1728): "Canon digital back"}
BOXES = {"top-left": (0.00, 0.00, 0.22, 0.12), "bottom-right": (0.80, 0.80, 1.00, 1.00)}
MIN_PX = 200            # bright out-of-disc pixels needed to call a mark present

spec = importlib.util.spec_from_file_location("m49", os.path.join(ROOT, "scripts", "49_mask_overlays.py"))
m49 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m49)

FIG4A = ["P0030_OD_01.jpg", "P0144_OD_01.jpg", "P0274_OS_01.jpg", "P0287_OD_01.jpg",
         "P0312_OD_02.jpg", "P0433_OD_01.jpg", "P0436_OD_01.jpg",          # intolerant
         "P0011_OD_01.jpg", "P0083_OD_01.jpg", "P0122_OD_01.jpg", "P0194_OD_01.jpg",
         "P0284_OD_01.jpg", "P0338_OD_01.jpg", "P0353_OD_01.jpg"]           # tolerant


def marks(path):
    a = np.array(Image.open(path).convert("RGB"))
    h, w = a.shape[:2]
    disc = m49.fundus_mask(a)
    out = {}
    for name, (x0, y0, x1, y1) in BOXES.items():
        ys, ye, xs, xe = int(y0 * h), int(y1 * h), int(x0 * w), int(x1 * w)
        box, d = a[ys:ye, xs:xe], disc[ys:ye, xs:xe]
        out[name] = int(((box.max(2) > 60) & ~d).sum())
    return (w, h), out


sp = pd.read_csv(SPLITS)
rows = []
for f in sorted(os.listdir(IMG)):
    size, px = marks(os.path.join(IMG, f))
    rows.append({"anon_image": f, "device": DEVICE.get(size, "other"),
                 "tl_px": px["top-left"], "br_px": px["bottom-right"]})
d = pd.DataFrame(rows).merge(sp, on="anon_image", how="left")
d["overlay"] = (d.tl_px >= MIN_PX) | (d.br_px >= MIN_PX)
d["y"] = d.grade.eq("PDR").astype(int)
an = d[d.split.isin(["train_val", "test"])]          # the 945 analysed images

out = {"n_images_all": int(len(d)), "n_analysed": int(len(an)), "min_px": MIN_PX, "by_device": {}}
print("OVERLAY PRESENCE IN TRAINING IMAGES (after masking), analysed set n = %d" % len(an))
for dev, g in an.groupby("device"):
    rec = {"images": int(len(g)), "pdr": int(g.y.sum()),
           "overlay_rate": round(float(g.overlay.mean()), 4),
           "overlay_rate_pdr": round(float(g[g.y == 1].overlay.mean()), 4) if g.y.sum() else None,
           "overlay_rate_npdr": round(float(g[g.y == 0].overlay.mean()), 4)}
    out["by_device"][dev] = rec
    print("  %-24s n=%4d PDR=%3d  overlay %5.1f%%  (PDR %s, NPDR %5.1f%%)"
          % (dev, rec["images"], rec["pdr"], 100 * rec["overlay_rate"],
             "%5.1f%%" % (100 * rec["overlay_rate_pdr"]) if rec["overlay_rate_pdr"] is not None else "  n/a",
             100 * rec["overlay_rate_npdr"]))

ct = pd.crosstab(an.overlay, an.y)
out["overlay_x_class"] = {"overlay_pdr": int(ct.loc[True, 1]) if True in ct.index else 0,
                          "overlay_npdr": int(ct.loc[True, 0]) if True in ct.index else 0,
                          "none_pdr": int(ct.loc[False, 1]) if False in ct.index else 0,
                          "none_npdr": int(ct.loc[False, 0]) if False in ct.index else 0}
o = out["overlay_x_class"]
print("\nOVERLAY x CLASS (all devices)")
print("  overlay present: %d PDR / %d NPDR   (PDR rate %.1f%%)"
      % (o["overlay_pdr"], o["overlay_npdr"], 100 * o["overlay_pdr"] / max(1, o["overlay_pdr"] + o["overlay_npdr"])))
print("  overlay absent : %d PDR / %d NPDR   (PDR rate %.1f%%)"
      % (o["none_pdr"], o["none_npdr"], 100 * o["none_pdr"] / max(1, o["none_pdr"] + o["none_npdr"])))

vis = an[an.device == "VISUCAM 200"]
out["visucam_overlay_values"] = {"tl_px_min": int(vis.tl_px.min()), "br_px_min": int(vis.br_px.min()),
                                 "n_without_overlay": int((~vis.overlay).sum())}
print("\nWITHIN VISUCAM: images without a detectable mark = %d of %d  (min px TL %d, BR %d)"
      % (out["visucam_overlay_values"]["n_without_overlay"], len(vis),
         out["visucam_overlay_values"]["tl_px_min"], out["visucam_overlay_values"]["br_px_min"]))
nonvis = an[an.device != "VISUCAM 200"]
print("OUTSIDE VISUCAM: images with a detectable mark = %d of %d" % (int(nonvis.overlay.sum()), len(nonvis)))

f4 = d.set_index("anon_image").loc[FIG4A, ["grade", "device", "overlay", "split"]]
out["fig4a"] = {k: {"grade": v.grade, "device": v.device, "overlay": bool(v.overlay), "split": v.split}
                for k, v in f4.iterrows()}
print("\nFIG. 4a EXAMPLES")
print(f4.to_string())

json.dump(out, open(os.path.join(ROOT, "results", "overlay_audit.json"), "w"), indent=2)
print("\nsaved results/overlay_audit.json")
