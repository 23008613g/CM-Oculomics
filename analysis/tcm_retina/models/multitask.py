# -*- coding: utf-8 -*-
"""多任务模型:共享RETFound影像编码器 → 双头(anti-VEGF耐受性 + 中医证型)。
证型作为辅助监督信号,期望共享表征正则化能帮主任务(耐受性)。"""
import torch
import torch.nn as nn
import timm


class MultiTaskNet(nn.Module):
    def __init__(self, backbone="vit_large_patch16_224", n_tol=2, n_syn=4,
                 retfound=True, drop_rate=0.2,
                 retfound_repo="sabarimj/retfound",
                 retfound_file="RETFound_cfp_weights.pth"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False,
                                          num_classes=0, drop_rate=drop_rate)
        if retfound:
            self._load_retfound(retfound_repo, retfound_file)
        d = self.backbone.num_features
        self.shared = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop_rate))
        self.head_tol = nn.Linear(d, n_tol)   # 主任务:耐受性
        self.head_syn = nn.Linear(d, n_syn)   # 辅助任务:中医证型

    def _load_retfound(self, repo, fname):
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo, fname)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        own = self.backbone.state_dict()
        matched = {k: v for k, v in sd.items()
                   if not k.startswith("decoder") and k != "mask_token"
                   and k in own and own[k].shape == v.shape}
        own.update(matched)
        self.backbone.load_state_dict(own, strict=False)
        print(f"[RETFound] 载入 {len(matched)}/{len(own)} 层")

    def forward(self, img):
        f = self.shared(self.backbone(img))
        return self.head_tol(f), self.head_syn(f)
