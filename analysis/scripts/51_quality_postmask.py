# -*- coding: utf-8 -*-
"""
51_quality_postmask.py — recompute the no-reference image-quality score on the
overlay-masked images, and measure how much the masking moved it.

WHY: script 17 builds the quality score from the WHOLE image. Its foreground mask is a
plain `gray > 12` threshold, so the burned-in overlay text counted as retina, and the
sharpness term is the Laplacian variance of the whole frame, which text edges inflate.
Masking therefore changes the score of the 358 images that carried an overlay, and
because the composite score is a percentile rank across the cohort, the Low/Medium/High
tier boundaries move for every image. Figure 10b-d currently mixes pre-mask quality
scores with post-mask model predictions.

The metric definition is copied from 17_quality_aware.py unchanged, so the only thing
that differs between the two runs is the pixels.

USAGE   python scripts/51_quality_postmask.py            # report only
        python scripts/51_quality_postmask.py --write    # also replace results/image_quality.csv
OUTPUT  results/image_quality.csv          (backup: results/image_quality_preMask.csv)
        results/quality_postmask_diff.json
"""
import os, sys, json, argparse, shutil
import numpy as np, pandas as pd, cv2
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_anon"); R = os.path.join(ROOT, "results")
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

MASKED = os.path.join(DATA, "images")                  # active set, overlays removed
ORIG   = os.path.join(DATA, "images_with_overlays")    # originals, kept read-only


def quality(path):
    """Verbatim from 17_quality_aware.py — do not 'improve' it, the point is comparability."""
    try: pil = Image.open(path).convert("RGB")
    except Exception: return None
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR); img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY); mask = m > 0
    if mask.sum() < 1000:
        return dict(sharpness=0, contrast=0, brightness=0, fov_ratio=0, illum_unif=0)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray[mask].std()
    bright = gray[mask].mean()
    fov = mask.sum() / mask.size
    blur = cv2.GaussianBlur(gray, (0, 0), 30).astype(float)
    illum = 1.0 / (1.0 + blur[mask].std() / 50.0)
    return dict(sharpness=sharp, contrast=contrast, brightness=bright,
                fov_ratio=fov, illum_unif=illum)


def score_dir(img_dir, manifest):
    rows = []
    for _i, r in manifest.iterrows():
        q = quality(os.path.join(img_dir, r["anon_image"]))
        if q is None: continue
        q["anon_image"] = r["anon_image"]; q["grade"] = r["grade"]; rows.append(q)
    df = pd.DataFrame(rows)
    feats = ["sharpness", "contrast", "fov_ratio", "illum_unif"]
    norm = pd.DataFrame({f: df[f].rank(pct=True) for f in feats})
    norm["brightness"] = 1 - (df["brightness"] - df["brightness"].median()).abs().rank(pct=True)
    df["quality_score"] = norm.mean(axis=1)
    df["quality_tier"] = pd.qcut(df["quality_score"], [0, .25, .75, 1.0],
                                 labels=["Low", "Medium", "High"])
    return df


def main(args):
    man = pd.read_csv(os.path.join(DATA, "manifest.csv")).dropna(subset=["grade"])
    print("manifest rows with a grade: %d" % len(man))

    print("scoring masked images ...", flush=True)
    new = score_dir(MASKED, man)
    print("scoring originals ...", flush=True)
    old = score_dir(ORIG, man)

    j = old.merge(new, on="anon_image", suffixes=("_orig", "_mask"))
    # which images actually carried an overlay
    sizes = {f: Image.open(os.path.join(ORIG, f)).size for f in j["anon_image"]}
    j["touched"] = j["anon_image"].map(lambda f: sizes[f] in [(720, 576), (2592, 1728)])
    t, u = j[j.touched], j[~j.touched]

    print()
    print("=" * 74)
    print("EFFECT OF MASKING ON THE QUALITY SCORE")
    print("=" * 74)
    print("  overlay-bearing images : %d" % len(t))
    print("  untouched images       : %d" % len(u))
    for nm, sub in (("overlay-bearing", t), ("untouched", u)):
        d = (sub.quality_score_mask - sub.quality_score_orig)
        print("  %-16s composite score change: mean %+.4f, |max| %.4f"
              % (nm, d.mean(), d.abs().max()))
    for f in ["sharpness", "contrast", "brightness", "fov_ratio", "illum_unif"]:
        a, b = t[f + "_orig"], t[f + "_mask"]
        print("    %-12s overlay images: %.3f -> %.3f  (%+.1f%%)"
              % (f, a.mean(), b.mean(), 100 * (b.mean() - a.mean()) / max(abs(a.mean()), 1e-9)))
    moved = int((j.quality_tier_orig.astype(str) != j.quality_tier_mask.astype(str)).sum())
    print()
    print("  images that changed quality tier: %d of %d (%.1f%%)"
          % (moved, len(j), 100.0 * moved / len(j)))
    print("  tier counts  before: %s" % dict(old.quality_tier.value_counts()))
    print("  tier counts  after : %s" % dict(new.quality_tier.value_counts()))

    # --- what this does to the reported per-tier AUCs, using the canonical OOF ---
    oof = pd.read_csv(os.path.join(R, "dino_experiment",
                                   "oof_dinov2_l_res224_frac1.0_main.csv"))
    out = {"n_overlay_images": len(t), "n_untouched": len(u), "tier_changes": moved,
           "per_tier_auc": {}, "exclusion_curve": {}}
    print()
    print("  per-tier AUC (canonical DINOv2 OOF, oof_dinov2_l_res224_frac1.0_main.csv)")
    for lbl, q in (("pre-mask quality", old), ("post-mask quality", new)):
        d = oof.merge(q[["anon_image", "quality_tier", "quality_score"]], on="anon_image")
        aucs = {}
        for tier in ["Low", "Medium", "High"]:
            s = d[d.quality_tier == tier]
            aucs[tier] = round(float(roc_auc_score(s.y, s.prob)), 4) if s.y.nunique() > 1 else None
        out["per_tier_auc"][lbl] = aucs
        curve = {}
        for e in [0, 10, 20, 30]:
            thr = d.quality_score.quantile(e / 100); s = d[d.quality_score >= thr]
            curve[str(e)] = round(float(roc_auc_score(s.y, s.prob)), 4)
        out["exclusion_curve"][lbl] = curve
        print("    %-18s low %s  medium %s  high %s | exclude 0/10/20/30%%: %s"
              % (lbl, aucs["Low"], aucs["Medium"], aucs["High"],
                 " ".join(str(v) for v in curve.values())))

    json.dump(out, open(os.path.join(R, "quality_postmask_diff.json"), "w"), indent=2)
    print()
    print("diff -> results/quality_postmask_diff.json")

    if args.write:
        src = os.path.join(R, "image_quality.csv")
        bak = os.path.join(R, "image_quality_preMask.csv")
        if os.path.exists(src) and not os.path.exists(bak):
            shutil.copy2(src, bak); print("backed up existing CSV -> %s" % bak)
        new.to_csv(src, index=False)
        print("wrote %s from the masked images (%d rows)" % (src, len(new)))
    else:
        print("nothing written; re-run with --write to replace results/image_quality.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args())
