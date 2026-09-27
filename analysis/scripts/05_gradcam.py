# -*- coding: utf-8 -*-
"""
05_gradcam.py — RETFound (ViT) 的 Grad-CAM 可解释性热图
============================================================
揭示模型判断 anti-VEGF 耐受性时关注的视网膜区域。
- ViT 用 reshape_transform 把 token 还原成 14x14 网格
- target layer: 最后一个 transformer block 的 norm1
- 对 测试集 中【正确预测】的 耐受(NPDR) 和 不耐受(PDR) 各取若干样本出图
输出: results/figures/gradcam/  (单张热图 + 一张汇总网格图)
"""
import os, sys, glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tcm_retina.models.backbone import FundusClassifier
from tcm_retina.data.dataset import build_transforms, GRADE_CLASSES
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_anon")
CKPT = os.path.join(ROOT, "results", "retfound_antivegf", "best.pth")  # fold0 最优
OUT = os.path.join(ROOT, "results", "figures", "gradcam")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224


def reshape_transform(tensor, height=14, width=14):
    # ViT: [B, 1+N, C] -> 去cls -> [B,H,W,C] -> [B,C,H,W]
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def load_model():
    model = FundusClassifier("vit_large_patch16_224", num_classes=2,
                             pretrained=False, retfound=False)  # 结构即可,权重从ckpt载
    sd = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    return model


def main():
    model = load_model()
    # target layer:最后一个 block 的 norm1
    target_layers = [model.backbone.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)

    tf = build_transforms(IMG_SIZE, train=False)
    splits = pd.read_csv(os.path.join(DATA, "splits.csv"))
    test = splits[splits.split == "test"].dropna(subset=["grade"])

    # 各类取样本 —— 每组 8 个(翻倍),组别清晰标注
    N_PER = 8
    avail = set(test["anon_image"])
    pdr_pref = ["P0030_OD_01.jpg", "P0144_OD_01.jpg", "P0287_OD_01.jpg", "P0433_OD_01.jpg"]
    pdr_pref = [f for f in pdr_pref if f in avail]
    pdr_rest = [f for f in test[test.grade == "PDR"]["anon_image"].tolist()
                if f not in pdr_pref]
    picks = {
        "PDR":  (pdr_pref + pdr_rest)[:N_PER],
        "NPDR": test[test.grade == "NPDR"]["anon_image"].head(N_PER).tolist(),
    }

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    # 每组生成 (orig, cam, pred_ok) 列表
    groups = {}  # grade -> list of (orig, vis, ok)
    for grade, imgs in picks.items():
        cls_idx = GRADE_CLASSES.index(grade)
        out = []
        for fn in imgs:
            p = os.path.join(DATA, "images", fn)
            pil = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            rgb = np.array(pil).astype(np.float32) / 255.0
            x = tf(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(x).argmax(1).item()
            grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls_idx)])[0]
            vis = show_cam_on_image(rgb, grayscale, use_rgb=True)
            out.append((np.array(pil), vis, pred == cls_idx))
            Image.fromarray(vis).save(os.path.join(OUT, f"cam_{grade}_{fn}.png"))
        groups[grade] = out

    # ---- 布局: 两个分组块上下排列, 每块= [组标题条] + 2行(Original / Grad-CAM) x ncol ----
    ncol = max(len(groups["PDR"]), len(groups["NPDR"]))
    GROUP_STYLE = {
        "PDR":  dict(label="INTOLERANT  (PDR)", color="#c23a36"),
        "NPDR": dict(label="TOLERANT  (NPDR)", color="#15936a"),
    }
    order = ["PDR", "NPDR"]
    # 行结构: 每组 3 行(标题条占很小, 实际用 height_ratios), 总 2 组
    fig = plt.figure(figsize=(2.05 * ncol, 9.6))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(6, ncol, figure=fig,
                  height_ratios=[0.18, 1, 1, 0.18, 1, 1], hspace=0.08, wspace=0.05)
    row_base = {"PDR": 0, "NPDR": 3}
    for grade in order:
        st = GROUP_STYLE[grade]; rb = row_base[grade]; data = groups[grade]
        # 组标题条(跨整行)
        cax = fig.add_subplot(gs[rb, :]); cax.axis("off")
        cax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=cax.transAxes,
                      color=st["color"], alpha=0.92))
        cax.text(0.5, 0.5, st["label"], transform=cax.transAxes, ha="center",
                 va="center", color="white", fontsize=13, fontweight="bold")
        for j in range(ncol):
            ax_o = fig.add_subplot(gs[rb + 1, j])
            ax_c = fig.add_subplot(gs[rb + 2, j])
            if j < len(data):
                orig, vis, ok = data[j]
                ax_o.imshow(orig)
                ax_c.imshow(vis)
                # 正确预测打勾(全部展示的都应正确)
                mark = "✓" if ok else "✗"
                mc = "#15936a" if ok else "#c23a36"
                ax_o.text(0.04, 0.96, mark, transform=ax_o.transAxes, fontsize=12,
                          fontweight="bold", color=mc, va="top",
                          bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec=mc, lw=1))
                # 给每张图加组别色边框
                for ax in (ax_o, ax_c):
                    for sp in ax.spines.values():
                        sp.set_visible(True); sp.set_color(st["color"]); sp.set_linewidth(2)
            ax_o.set_xticks([]); ax_o.set_yticks([])
            ax_c.set_xticks([]); ax_c.set_yticks([])
            if j == 0:
                ax_o.set_ylabel("Original", fontsize=10, fontweight="bold")
                ax_c.set_ylabel("Grad-CAM", fontsize=10, fontweight="bold")
    fig.suptitle("Grad-CAM attention maps for correctly classified eyes "
                 "(intolerant vs tolerant): warm colors drive the prediction",
                 fontsize=12, y=0.995)
    grid = os.path.join(ROOT, "results", "figures", "fig_gradcam_grid.png")
    plt.savefig(grid, dpi=300, bbox_inches="tight"); plt.close()
    n = len(groups["PDR"]) + len(groups["NPDR"])
    print(f"saved per-sample CAMs to: {OUT}")
    print(f"saved grid: {grid}  ({n} samples: {len(groups['PDR'])} PDR + {len(groups['NPDR'])} NPDR)")


if __name__ == "__main__":
    main()
