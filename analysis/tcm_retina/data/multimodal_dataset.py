# -*- coding: utf-8 -*-
"""多模态数据集:眼底图 + 临床指标(+中医证型)→ anti-VEGF 耐受性。"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from tcm_retina.data.dataset import build_transforms, GRADE_CLASSES

# 临床数值特征(标准化用)
NUM_FEATS = ["年龄", "糖尿病年限（年）", "BMI", "HbA1c", "LDL", "HDL", "Triglyceride"]
BIN_FEATS = ["hypertension", "smoking"]
SYNDROME_CLASSES = ["气阴两虚证", "痰瘀阻滞证", "脾肾两虚证", "阴虚夹瘀证"]


class MultimodalDataset(Dataset):
    """
    返回 (image_tensor, tab_vector, label)
    tab_vector 由配置决定包含:数值指标 / 二值 / 证型one-hot
    """
    def __init__(self, rows, image_dir, patients_df, image_size, train,
                 use_clinical=True, use_syndrome=True, aug=None,
                 norm_stats=None):
        self.rows = rows.reset_index(drop=True)
        self.image_dir = image_dir
        self.tf = build_transforms(image_size, train, aug)
        self.use_clinical = use_clinical
        self.use_syndrome = use_syndrome
        self.cls2idx = {c: i for i, c in enumerate(GRADE_CLASSES)}  # NPDR/PDR
        # 病人表索引(anon_id -> 行)
        self.pt = patients_df.set_index("anon_id")
        self.norm_stats = norm_stats  # (mean,std) for NUM_FEATS,训练集算好传入

    @staticmethod
    def compute_norm(patients_df):
        sub = patients_df[NUM_FEATS].apply(pd.to_numeric, errors="coerce")
        return sub.mean().values, sub.std().replace(0, 1).values

    def tab_dim(self):
        d = 0
        if self.use_clinical:
            d += len(NUM_FEATS) + len(BIN_FEATS)
        if self.use_syndrome:
            d += len(SYNDROME_CLASSES)
        return d

    def _tab_vector(self, anon_id):
        parts = []
        row = self.pt.loc[anon_id] if anon_id in self.pt.index else None
        if self.use_clinical:
            if row is not None:
                vals = pd.to_numeric(row[NUM_FEATS], errors="coerce").values.astype(float)
            else:
                vals = np.full(len(NUM_FEATS), np.nan)
            if self.norm_stats is not None:
                mean, std = self.norm_stats
                vals = (vals - mean) / std
            vals = np.nan_to_num(vals, nan=0.0)
            parts.append(vals)
            # 二值特征
            bin_vals = []
            for b in BIN_FEATS:
                v = row[b] if (row is not None and b in row) else 0
                bin_vals.append(float(v) if pd.notna(v) else 0.0)
            parts.append(np.array(bin_vals))
        if self.use_syndrome:
            oh = np.zeros(len(SYNDROME_CLASSES))
            if row is not None and "中医证型" in row and pd.notna(row["中医证型"]):
                syn = row["中医证型"]
                if syn in SYNDROME_CLASSES:
                    oh[SYNDROME_CLASSES.index(syn)] = 1.0
            parts.append(oh)
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(os.path.join(self.image_dir, r["anon_image"])).convert("RGB")
        x = self.tf(img)
        tab = torch.tensor(self._tab_vector(r["anon_id"]))
        y = self.cls2idx[r["grade"]]
        return x, tab, y

    def class_counts(self):
        vc = self.rows["grade"].value_counts()
        return [int(vc.get(c, 0)) for c in GRADE_CLASSES]
