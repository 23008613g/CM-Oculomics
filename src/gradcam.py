# -*- coding: utf-8 -*-
"""Grad-CAM explanation for the DINOv2 (ViT-L/14) intolerance classifier.

ViT tokens are reshaped to a 16x16 grid (224/14) for spatial attribution.
"""
import argparse
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .model import load_dinov2, dinov2_gradcam_target, dinov2_reshape
from .dataset import build_transforms


def reshape_transform(tensor, grid=16):
    return dinov2_reshape(tensor, grid)


def gradcam(image_path: str, weights: str, out_path: str = "cam.png",
            device: str = None, target_class: int = 1):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(weights, device)
    target_layers = dinov2_gradcam_target(model)
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)

    pil = Image.open(image_path).convert("RGB").resize((224, 224))
    rgb = np.array(pil).astype(np.float32) / 255.0
    x = build_transforms(224, train=False)(pil).unsqueeze(0).to(device)
    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])[0]
    vis = show_cam_on_image(rgb, grayscale, use_rgb=True)
    Image.fromarray(vis).save(out_path)
    print("saved:", out_path)
    return vis


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grad-CAM for the intolerance classifier.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="cam.png")
    args = ap.parse_args()
    gradcam(args.image, args.weights, args.out)
