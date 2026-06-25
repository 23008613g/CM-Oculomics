# -*- coding: utf-8 -*-
"""
Models for anti-VEGF intolerance prediction.

PRIMARY: DINOv2 (ViT-L/14) — a generalist vision foundation model (Meta AI, Apache-2.0),
fine-tuned for binary classification. Released fine-tuned weights (dino_deploy.pth) are
Apache-2.0 (free for research and commercial use with attribution). Use `build_dinov2`
/ `load_dinov2`.

COMPARATOR: RETFound (ViT-L/16) — a retinal-domain foundation model (Zhou et al.,
Nature 2023, CC BY-NC 4.0). Provided via `FundusClassifier` for the head-to-head
comparison reported in the paper; RETFound weights are NOT redistributed here.
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
    """Load a fully fine-tuned RETFound (comparator) classifier checkpoint."""
    model = FundusClassifier(num_classes=2, retfound_weights=None)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# PRIMARY model: DINOv2 (ViT-L/14) generalist vision foundation model
# --------------------------------------------------------------------------- #
def build_dinov2(num_classes: int = 2, img_size: int = 224, drop_rate: float = 0.2):
    """DINOv2 (ViT-L/14) backbone + LayerNorm/Dropout/Linear head, as an nn.Sequential.

    The 224-px input yields a 16x16 = 256 patch-token grid. The returned module's
    state_dict matches the released `dino_deploy.pth` (keys '0.*' backbone, '1.*' head).
    """
    backbone = timm.create_model("vit_large_patch14_dinov2", pretrained=False,
                                 num_classes=0, img_size=img_size, drop_rate=drop_rate)
    head = nn.Sequential(nn.LayerNorm(backbone.num_features),
                         nn.Dropout(drop_rate),
                         nn.Linear(backbone.num_features, num_classes))
    return nn.Sequential(backbone, head)


def load_dinov2(weights_path: str, device: str = "cuda", img_size: int = 224):
    """Load the fine-tuned DINOv2 classifier (dino_deploy.pth) for inference."""
    model = build_dinov2(num_classes=2, img_size=img_size)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model.to(device).eval()


def dinov2_gradcam_target(model):
    """Grad-CAM target layer for the DINOv2 Sequential model."""
    return [model[0].blocks[-1].norm1]


def dinov2_reshape(tensor, grid: int = 16):
    """reshape_transform for Grad-CAM on DINOv2 (keep the last grid*grid patch tokens)."""
    x = tensor[:, -grid * grid:, :]
    return x.reshape(tensor.size(0), grid, grid, tensor.size(2)).permute(0, 3, 1, 2)
