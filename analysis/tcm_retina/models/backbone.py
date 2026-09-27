# -*- coding: utf-8 -*-
"""
影像编码器 + 分类头。
- 基线:timm 的 Swin Transformer V2(ImageNet 预训练)。
- RETFound:timm 的 vit_large_patch16_224,载入 RETFound MAE 自监督权重(160万眼底图)。
  权重来自 HF 镜像 sabarimj/retfound(非 gated),已验证为官方 ViT-L/16 MAE 架构。
"""
import torch
import torch.nn as nn
import timm


class FundusClassifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int,
                 pretrained: bool = True, drop_rate: float = 0.1,
                 retfound: bool = False, retfound_repo: str = "sabarimj/retfound",
                 retfound_file: str = "RETFound_cfp_weights.pth"):
        super().__init__()
        # RETFound 用 vit_large_patch16_224,从外部载权重,故 timm pretrained=False
        timm_pretrained = pretrained and not retfound
        self.backbone = timm.create_model(
            backbone, pretrained=timm_pretrained, num_classes=0, drop_rate=drop_rate,
        )
        if retfound:
            self._load_retfound(retfound_repo, retfound_file)

        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )
        self.feat_dim = feat_dim

    def _load_retfound(self, repo: str, fname: str):
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo, fname)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        own = self.backbone.state_dict()
        # 只取 encoder 权重(忽略 decoder_*, mask_token);名称对齐 timm ViT
        matched, skipped = {}, []
        for k, v in sd.items():
            if k.startswith("decoder") or k == "mask_token":
                continue
            if k in own and own[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)
        own.update(matched)
        missing = self.backbone.load_state_dict(own, strict=False)
        print(f"[RETFound] 载入 {len(matched)}/{len(own)} 层权重;"
              f"跳过(形状不符/非encoder) {len(skipped)} 个")
        if skipped[:5]:
            print(f"[RETFound] 跳过示例: {skipped[:5]}")

    def forward(self, x, return_feat: bool = False):
        f = self.backbone(x)
        logits = self.head(f)
        if return_feat:
            return logits, f
        return logits
