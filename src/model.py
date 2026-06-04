# -*- coding: utf-8 -*-
"""
RETFound (ViT-L/16) encoder + binary classification head for anti-VEGF intolerance.

NOTE ON LICENSE: this model fine-tunes RETFound (Zhou et al., Nature 2023), whose
weights are CC BY-NC 4.0 (NonCommercial). Any released fine-tuned weights inherit
this restriction — NonCommercial research use only, with attribution to RETFound.
"""
import torch
import torch.nn as nn
import timm


class FundusClassifier(nn.Module):
    """ViT-L/16 backbone (RETFound-initialized) + linear head.

    Args:
        num_classes: 2 (tolerant / intolerant).
        retfound_weights: path to RETFound .pth, or None to start from timm init.
        drop_rate: dropout for the head.
    """

    def __init__(self, num_classes: int = 2, retfound_weights: str | None = None,
                 backbone: str = "vit_large_patch16_224", drop_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False,
                                          num_classes=0, drop_rate=drop_rate)
        if retfound_weights:
            self._load_retfound(retfound_weights)
        d = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(drop_rate), nn.Linear(d, num_classes)
        )

    def _load_retfound(self, path: str):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        own = self.backbone.state_dict()
        matched = {k: v for k, v in sd.items()
                   if not k.startswith("decoder") and k != "mask_token"
                   and k in own and own[k].shape == v.shape}
        own.update(matched)
        self.backbone.load_state_dict(own, strict=False)
        print(f"[RETFound] loaded {len(matched)}/{len(own)} encoder tensors")

    def forward(self, x, return_feat: bool = False):
        f = self.backbone(x)
        logits = self.head(f)
        return (logits, f) if return_feat else logits


def load_finetuned(weights_path: str, device: str = "cuda"):
    """Load a fully fine-tuned classifier checkpoint for inference."""
    model = FundusClassifier(num_classes=2, retfound_weights=None)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()
