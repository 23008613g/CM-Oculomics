# -*- coding: utf-8 -*-
"""
52_biomarkers_postmask.py — recompute the handcrafted imaging biomarkers on the
overlay-masked images, and measure what the burned-in text had been contributing.

WHY: 12_imaging_biomarkers.py takes its field-of-view mask as `gray > 12`, so the
burned-in overlay text fell inside the FOV. Two features are directly exposed to it:
  bright_lesion_area = ((g > 180) & (r > 180) & mask)  -- white text satisfies this exactly
  contrast, red_energy, green_red_ratio                -- computed over the same mask
and the black-hat vessel filter responds to text edges. On the 358 images that carried an
overlay, part of the "exudate area" was therefore text, not retina.

The feature definitions are copied verbatim from script 12, so the only thing that differs
between the two runs is the pixels.

USAGE   python scripts/52_biomarkers_postmask.py           # report only
        python scripts/52_biomarkers_postmask.py --write   # replace the two result CSVs
OUTPUT  results/imaging_biomarkers.csv, results/biomarker_stats.csv
        (backups: *_preMask.csv)   diff: results/biomarkers_postmask_diff.json
"""
import os, sys, json, argparse, shutil, warnings
import numpy as np, pandas as pd, cv2
from PIL import Image
from scipy import stats
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_anon"); OUT = os.path.join(ROOT, "results")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
MASKED = os.path.join(DATA, "images")
ORIG   = os.path.join(DATA, "images_with_overlays")
FEATS = ["vessel_density", "vessel_skeleton", "fractal_dim", "dark_lesion_area",
         "bright_lesion_area", "red_energy", "green_red_ratio", "contrast"]


def fov_mask(gray):
    _, m = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    return (m > 0)


def fractal_dimension(Z):
    Z = Z > 0
    if Z.sum() == 0: return 0.0
    def boxcount(Z, k):
        S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])
    sizes = 2 ** np.arange(1, 7)
    counts = [max(boxcount(Z, s), 1) for s in sizes]
    return -np.polyfit(np.log(sizes), np.log(counts), 1)[0]


def extract(path):
    try: pil = Image.open(path).convert("RGB")
    except Exception: return None
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (512, 512))
    b, g, r = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = fov_mask(gray); area = mask.sum() + 1
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ge = clahe.apply(g)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bh = cv2.morphologyEx(ge, cv2.MORPH_BLACKHAT, kernel)
    _, vessel = cv2.threshold(bh, 15, 255, cv2.THRESH_BINARY)
    vessel = (vessel > 0) & mask
    skel = skeletonize(vessel)
    dark = ((g < 60) & (r < 90) & mask)
    bright = ((g > 180) & (r > 180) & mask)
    return dict(vessel_density=vessel.sum() / area, vessel_skeleton=skel.sum() / area,
                fractal_dim=fractal_dimension(vessel),
                dark_lesion_area=dark.sum() / area, bright_lesion_area=bright.sum() / area,
                red_energy=r[mask].mean() / 255.0,
                green_red_ratio=(g[mask].mean() + 1) / (r[mask].mean() + 1),
                contrast=gray[mask].std() / 255.0)


def score_dir(img_dir, man, tag):
    rows = []
    for _i, r in man.iterrows():
        f = extract(os.path.join(img_dir, r["anon_image"]))
        if f is None: continue
        f["anon_image"] = r["anon_image"]; f["label"] = r["grade"]; rows.append(f)
        if len(rows) % 250 == 0: print("  [%s] %d" % (tag, len(rows)), flush=True)
    return pd.DataFrame(rows)


def stats_table(df):
    npdr, pdr = df[df.label == "NPDR"], df[df.label == "PDR"]
    y = (df.label == "PDR").astype(int)
    rows = []
    for c in FEATS:
        _u, p = stats.mannwhitneyu(npdr[c].dropna(), pdr[c].dropna())
        auc = roc_auc_score(y, df[c]); auc = max(auc, 1 - auc)
        rows.append({"biomarker": c, "NPDR_mean": npdr[c].mean(), "PDR_mean": pdr[c].mean(),
                     "p_value": p, "single_AUC": auc, "sig": "*" if p < 0.05 else ""})
    return pd.DataFrame(rows).sort_values("single_AUC", ascending=False)


def main(args):
    man = pd.read_csv(os.path.join(DATA, "manifest.csv")).dropna(subset=["grade"])
    print("extracting from masked images ...", flush=True)
    new = score_dir(MASKED, man, "masked")
    print("extracting from originals ...", flush=True)
    old = score_dir(ORIG, man, "orig")

    sizes = {f: Image.open(os.path.join(ORIG, f)).size for f in new["anon_image"]}
    j = old.merge(new, on="anon_image", suffixes=("_o", "_m"))
    j["touched"] = j["anon_image"].map(lambda f: sizes[f] in [(720, 576), (2592, 1728)])
    t = j[j.touched]

    print()
    print("=" * 78)
    print("WHAT THE BURNED-IN TEXT WAS CONTRIBUTING  (%d overlay-bearing images)" % len(t))
    print("=" * 78)
    print("  %-20s %12s %12s %10s" % ("feature", "originals", "masked", "change"))
    contrib = {}
    for c in FEATS:
        a, b = t[c + "_o"].mean(), t[c + "_m"].mean()
        pct = 100 * (b - a) / max(abs(a), 1e-12)
        contrib[c] = {"orig_mean": float(a), "masked_mean": float(b), "pct_change": float(pct)}
        print("  %-20s %12.5f %12.5f %9.1f%%" % (c, a, b, pct))

    so, sn = stats_table(old), stats_table(new)
    print()
    print("  single-feature AUC for PDR (whole cohort, n=%d)" % len(new))
    print("  %-20s %10s %10s   %12s %12s" % ("biomarker", "AUC orig", "AUC mask", "p orig", "p mask"))
    for c in FEATS:
        ro = so[so.biomarker == c].iloc[0]; rn = sn[sn.biomarker == c].iloc[0]
        flag = ""
        if (ro.p_value < 0.05) != (rn.p_value < 0.05):
            flag = "   <- significance FLIPS"
        print("  %-20s %10.4f %10.4f   %12.2e %12.2e%s"
              % (c, ro.single_AUC, rn.single_AUC, ro.p_value, rn.p_value, flag))

    json.dump({"n_overlay_images": int(len(t)), "per_feature_on_overlay_images": contrib,
               "stats_orig": so.to_dict("records"), "stats_masked": sn.to_dict("records")},
              open(os.path.join(OUT, "biomarkers_postmask_diff.json"), "w"), indent=2)
    print("\ndiff -> results/biomarkers_postmask_diff.json")

    if args.write:
        for df, name in ((new, "imaging_biomarkers.csv"), (sn, "biomarker_stats.csv")):
            src = os.path.join(OUT, name)
            bak = os.path.join(OUT, name.replace(".csv", "_preMask.csv"))
            if os.path.exists(src) and not os.path.exists(bak):
                shutil.copy2(src, bak); print("backed up -> %s" % os.path.basename(bak))
            df.to_csv(src, index=False)
            print("wrote %s (%d rows)" % (name, len(df)))
    else:
        print("nothing written; re-run with --write to replace the result CSVs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args())
