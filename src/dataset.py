# -*- coding: utf-8 -*-
"""
Fundus image dataset and transforms.

Expects a manifest CSV with columns:
    image_path,label[,patient_id]
where label in {0,1}  (0 = tolerant / NPDR, 1 = intolerant / PDR).
patient_id is optional but recommended for patient-level cross-validation splits.

NO patient data is distributed with this repository; supply your own
ethically-approved, de-identified manifest and images.
"""
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = False):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), norm,
    ])


class FundusDataset(Dataset):
    def __init__(self, manifest_csv: str, image_size: int = 224, train: bool = False):
        self.df = pd.read_csv(manifest_csv)
        assert {"image_path", "label"}.issubset(self.df.columns), \
            "manifest must contain columns: image_path,label[,patient_id]"
        self.tf = build_transforms(image_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tf(Image.open(r["image_path"]).convert("RGB"))
        return x, int(r["label"])
