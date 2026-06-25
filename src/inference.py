# -*- coding: utf-8 -*-
"""Single-image (or batch) inference: anti-VEGF intolerance risk score."""
import argparse
import torch
from PIL import Image
from .model import load_dinov2
from .dataset import build_transforms


def predict(image_path: str, weights: str, device: str = None, threshold: float = 0.5):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(weights, device)   # primary backbone: DINOv2 (ViT-L/14)
    tf = build_transforms(224, train=False)
    x = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)[0, 1].item()
    return {
        "intolerance_risk": round(prob, 4),
        "prediction": "intolerant" if prob >= threshold else "tolerant",
        "threshold": threshold,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="anti-VEGF intolerance risk from a fundus image.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    out = predict(args.image, args.weights, threshold=args.threshold)
    for k, v in out.items():
        print(f"{k}: {v}")
