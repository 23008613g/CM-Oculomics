# -*- coding: utf-8 -*-
"""多模态模型:RETFound影像编码器 + 表格MLP + 融合分类头。"""
import torch
import torch.nn as nn
import timm


class MultimodalNet(nn.Module):
    def __init__(self, backbone="vit_large_patch16_224", num_classes=2,
                 tab_dim=0, retfound=True, drop_rate=0.2,
                 retfound_repo="sabarimj/retfound",
                 retfound_file="RETFound_cfp_weights.pth", fusion_dim=256):
        super().__init__()
        # 影像分支
        self.backbone = timm.create_model(backbone, pretrained=False,
                                          num_classes=0, drop_rate=drop_rate)
        if retfound:
            self._load_retfound(retfound_repo, retfound_file)
        img_dim = self.backbone.num_features

        # 表格分支(若有)
        self.tab_dim = tab_dim
        if tab_dim > 0:
            self.tab_mlp = nn.Sequential(
                nn.Linear(tab_dim, 64), nn.ReLU(), nn.Dropout(drop_rate),
                nn.Linear(64, 64), nn.ReLU(),
            )
            fused_in = img_dim + 64
        else:
            self.tab_mlp = None
            fused_in = img_dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_in), nn.Dropout(drop_rate),
            nn.Linear(fused_in, fusion_dim), nn.ReLU(), nn.Dropout(drop_rate),
            nn.Linear(fusion_dim, num_classes),
        )

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

    def forward(self, img, tab=None):
        f = self.backbone(img)
        if self.tab_mlp is not None and tab is not None and tab.shape[1] > 0:
            t = self.tab_mlp(tab)
            f = torch.cat([f, t], dim=1)
        return self.fusion(f)


class GatedFusionNet(nn.Module):
    """改进版多模态:影像特征投影 + 表格(临床/证型)gated cross-attention 融合。
    设计目的:让低维的表格/证型特征不被1024维影像特征淹没,获得公平的影响力。"""
    def __init__(self, backbone="vit_large_patch16_224", num_classes=2,
                 tab_dim=0, retfound=True, drop_rate=0.2,
                 retfound_repo="sabarimj/retfound",
                 retfound_file="RETFound_cfp_weights.pth", proj_dim=256):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False,
                                          num_classes=0, drop_rate=drop_rate)
        if retfound:
            MultimodalNet._load_retfound(self, retfound_repo, retfound_file)
        img_dim = self.backbone.num_features
        self.img_proj = nn.Sequential(nn.Linear(img_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.tab_dim = tab_dim
        if tab_dim > 0:
            self.tab_enc = nn.Sequential(
                nn.Linear(tab_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(),
                nn.Linear(proj_dim, proj_dim), nn.GELU(),
            )
            # gate:由表格特征决定对影像特征的调制(让证型能"门控"影像信息)
            self.gate = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.Sigmoid())
            self.cross_q = nn.Linear(proj_dim, proj_dim)
            self.cross_kv = nn.Linear(proj_dim, proj_dim)
            fused_in = proj_dim * 2
        else:
            self.tab_enc = None
            fused_in = proj_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_in), nn.Dropout(drop_rate),
            nn.Linear(fused_in, proj_dim), nn.GELU(), nn.Dropout(drop_rate),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, img, tab=None):
        fi = self.img_proj(self.backbone(img))             # [B, P]
        if self.tab_enc is not None and tab is not None and tab.shape[1] > 0:
            ft = self.tab_enc(tab)                          # [B, P]
            # cross-attention(标量注意力):表格query影像
            q = self.cross_q(ft); k = self.cross_kv(fi)
            attn = torch.sigmoid((q * k).sum(-1, keepdim=True) / (k.shape[-1] ** 0.5))
            fi_attended = fi * attn
            g = self.gate(torch.cat([fi_attended, ft], dim=1))
            fused = torch.cat([fi_attended * g, ft], dim=1)
            return self.head(fused)
        return self.head(fi)
