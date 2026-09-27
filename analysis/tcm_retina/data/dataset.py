# -*- coding: utf-8 -*-
"""眼底图数据集(支持 中医证型 / DR分级 两种标签)。"""
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 标签映射(固定顺序,保证可复现)
SYNDROME_CLASSES = ["气阴两虚证", "痰瘀阻滞证", "脾肾两虚证", "阴虚夹瘀证"]
GRADE_CLASSES = ["NPDR", "PDR"]


def get_classes(task: str):
    return SYNDROME_CLASSES if task == "tcm_syndrome" else GRADE_CLASSES


def build_transforms(image_size: int, train: bool, aug: dict | None = None):
    aug = aug or {}
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        ops = []
        if aug.get("random_resized_crop", True):
            ops.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
        else:
            ops.append(transforms.Resize((image_size, image_size)))
        if aug.get("hflip", True):
            ops.append(transforms.RandomHorizontalFlip())
        if aug.get("rotate", 0):
            ops.append(transforms.RandomRotation(aug["rotate"]))
        if aug.get("color_jitter", 0):
            cj = aug["color_jitter"]
            ops.append(transforms.ColorJitter(cj, cj, cj))
        ops += [transforms.ToTensor(), norm]
        return transforms.Compose(ops)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), norm,
    ])


class FundusDataset(Dataset):
    """
    splits_df: 含列 anon_image, anon_id, grade, tcm_syndrome, fold, split
    rows: 已过滤好的子集(由 train.py 按 fold/split 切好后传入)
    """
    def __init__(self, rows: pd.DataFrame, image_dir: str, task: str,
                 image_size: int, train: bool, aug: dict | None = None):
        self.rows = rows.reset_index(drop=True)
        self.image_dir = image_dir
        self.task = task
        self.label_col = "tcm_syndrome" if task == "tcm_syndrome" else "grade"
        self.classes = get_classes(task)
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        self.tf = build_transforms(image_size, train, aug)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(os.path.join(self.image_dir, r["anon_image"])).convert("RGB")
        x = self.tf(img)
        y = self.cls2idx[r[self.label_col]]
        return x, y

    def class_counts(self):
        vc = self.rows[self.label_col].value_counts()
        return [int(vc.get(c, 0)) for c in self.classes]
