# -*- coding: utf-8 -*-
"""
Interpretable retinal imaging biomarkers of anti-VEGF intolerance.

Computes, from a single color fundus photograph:
  - vascular density        : fraction of FOV occupied by the vessel mask
  - vascular skeleton length: normalized length of the vessel skeleton
  - vascular fractal dimension : box-counting fractal dimension of the vessels

Vessels are enhanced with CLAHE + morphological black-hat on the green channel.
PIL is used to read images (supports non-ASCII paths).
"""
import argparse
import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize


def _fov_mask(gray):
    _, m = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    return m > 0


def _vessel_mask(img_bgr, mask):
    g = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ge = clahe.apply(g)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bh = cv2.morphologyEx(ge, cv2.MORPH_BLACKHAT, kernel)
    _, vessel = cv2.threshold(bh, 15, 255, cv2.THRESH_BINARY)
    return (vessel > 0) & mask


def fractal_dimension(binary):
    """Box-counting fractal dimension of a binary structure.

    D = -slope of log N(eps) vs log eps, where N(eps) is the number of boxes of
    side eps that intersect the structure.
    """
    Z = binary > 0
    if Z.sum() == 0:
        return 0.0

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k * k))[0])

    sizes = 2 ** np.arange(1, 7)
    counts = [max(boxcount(Z, int(s)), 1) for s in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return float(-coeffs[0])


def compute_biomarkers(image_path: str, size: int = 512) -> dict:
    pil = Image.open(image_path).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _fov_mask(gray)
    area = max(int(mask.sum()), 1)

    vessel = _vessel_mask(img, mask)
    density = float(vessel.sum() / area)
    skeleton_len = float(skeletonize(vessel).sum() / area)
    fd = fractal_dimension(vessel)
    return {
        "vascular_density": round(density, 4),
        "vascular_skeleton_length": round(skeleton_len, 4),
        "vascular_fractal_dimension": round(fd, 4),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute retinal vascular biomarkers.")
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    for k, v in compute_biomarkers(args.image).items():
        print(f"{k}: {v}")
