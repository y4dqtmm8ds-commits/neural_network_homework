# Shared training engine for wafer defect classification.
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.vit_wafer import CNNStemViTWafer, ViTTinyWafer
from models.unet_dpfee_hybrid import UNetDPFEEHybrid
from models.dpfee_geometry_hybrid import DPFEEGeometryHybrid
from models.capsule_head import CapsuleHardClassHead, capsule_margin_loss
from models.base import BaseModel
from data.dataset import BaseDataset, add_geometric_priors
from data.geometric_feature_utils import FEATURE_NAMES, extract_geometric_features
from data.pseudo_mask_utils import generate_pseudo_masks


def load_config_file(path):
    text = Path(path).read_text(encoding="utf-8")
    if path.lower().endswith(".json"):
        return json.loads(text)
    config = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        if value.lower() in ("true", "false"):
            config[key] = value.lower() == "true"
        elif value.lower() in ("none", "null"):
            config[key] = None
        elif value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            config[key] = items
        else:
            try:
                config[key] = int(value)
            except ValueError:
                try:
                    config[key] = float(value)
                except ValueError:
                    config[key] = value.strip("'\"")
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is requested, but this PyTorch installation cannot use CUDA. "
            "Install a CUDA-enabled PyTorch build, then rerun with --device cuda."
        )

    return device


def amp_enabled(device, requested_amp):
    return bool(requested_amp and device.type == "cuda")


def describe_device(device):
    info = {
        "device": str(device),
        "use_cuda": device.type == "cuda",
        "gpu_name": None,
        "cuda_version": torch.version.cuda,
        "amp": False,
    }

    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)

    return info


class WaferDataset(BaseDataset):
    def __init__(
        self,
        npz_path,
        augment=False,
        shift_prob=0.15,
        max_shift=2,
        morph_aug=False,
        morph_prob=0.1,
        morph_kernel=3,
        use_geo_prior=False,
        use_geometric_features=False,
        geo_mean=None,
        geo_std=None,
        class_names=None,
        use_hardclass_aug=False,
        hardclass_aug_prob=0.15,
        use_scratch_aug=False,
        scratch_aug_prob=0.15,
        class_specific_aug=False,
        use_pseudo_mask=False,
    ):
        data = np.load(npz_path, allow_pickle=True)
        raw_x = data["x"].astype(np.float32)
        self.x = raw_x.copy()
        self.y = data["y"].astype(np.int64)
        self.augment = augment
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self.morph_aug = morph_aug
        self.morph_prob = morph_prob
        self.morph_kernel = morph_kernel
        self.use_geo_prior = bool(use_geo_prior)
        self.use_geometric_features = bool(use_geometric_features)
        self.class_names = list(class_names or [])
        self.use_hardclass_aug = bool(use_hardclass_aug)
        self.hardclass_aug_prob = float(hardclass_aug_prob)
        self.use_scratch_aug = bool(use_scratch_aug)
        self.scratch_aug_prob = float(scratch_aug_prob)
        self.class_specific_aug = bool(class_specific_aug)
        self.use_pseudo_mask = bool(use_pseudo_mask)
        self.name_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.hard_aug_ids = {
            self.name_to_idx[name]
            for name in ["Loc", "Scratch", "Edge-Loc", "Edge-Ring"]
            if name in self.name_to_idx
        }
        self.geo_mean = None if geo_mean is None else np.asarray(geo_mean, dtype=np.float32)
        self.geo_std = None if geo_std is None else np.asarray(geo_std, dtype=np.float32)
        self.geo_features = None
        if self.use_geometric_features:
            self.geo_features = np.stack(
                [extract_geometric_features(item) for item in raw_x],
                axis=0,
            ).astype(np.float32)
            if self.geo_mean is not None and self.geo_std is not None:
                self.set_geo_standardization(self.geo_mean, self.geo_std)
        self.sample_ids = np.arange(len(self.y), dtype=np.int64)
        self.sample_weights = None
        self.sample_quality = None

        # waferMap 原始值通常是 0/1/2，这里统一缩放到 [0,1]
        # 如果最大值不是 2，也做保护处理
        max_val = self.x.max()
        if max_val > 0 and max_val != 0:
            self.x = self.x / max_val

        if self.x.ndim == 3:
            # [N, H, W] -> [N, 1, H, W]
            self.x = self.x[:, None, :, :]
        elif self.x.ndim != 4:
            raise ValueError(f"Invalid x shape in {npz_path}: {self.x.shape}")

    def set_geo_standardization(self, mean, std):
        self.geo_mean = np.asarray(mean, dtype=np.float32)
        self.geo_std = np.asarray(std, dtype=np.float32)
        self.geo_std = np.where(self.geo_std < 1e-6, 1.0, self.geo_std).astype(np.float32)
        if self.geo_features is not None:
            self.geo_features = ((self.geo_features - self.geo_mean) / self.geo_std).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def apply_cleanlab_mask(
        self,
        keep_mask,
        mode="remove",
        issue_weight=0.3,
        min_keep_per_class=1,
        issues_path=None,
        hard_class_ids=None,
        weight_normal=0.3,
        weight_confusing=0.5,
        weight_strong=0.1,
        quality_source="cleanlab",
    ):
        keep_mask = np.asarray(keep_mask).astype(bool)
        if keep_mask.shape[0] != len(self.y):
            raise ValueError(f"Cleanlab mask length {keep_mask.shape[0]} != dataset length {len(self.y)}")

        if mode == "remove":
            final_keep = keep_mask.copy()
            for cls in np.unique(self.y):
                cls_idx = np.where(self.y == cls)[0]
                kept = cls_idx[final_keep[cls_idx]]
                if len(kept) < int(min_keep_per_class):
                    issue_idx = cls_idx[~keep_mask[cls_idx]]
                    restore = issue_idx[: max(0, int(min_keep_per_class) - len(kept))]
                    final_keep[restore] = True
            removed = int((~final_keep).sum())
            self.x = self.x[final_keep]
            self.y = self.y[final_keep]
            self.sample_ids = self.sample_ids[final_keep]
            if self.geo_features is not None:
                self.geo_features = self.geo_features[final_keep]
            self.sample_weights = None
            self.sample_quality = None
            return removed

        if mode == "downweight":
            self.sample_weights = np.ones(len(self.y), dtype=np.float32)
            self.sample_weights[~keep_mask] = float(issue_weight)
            return int((~keep_mask).sum())

        issue_rows = {}
        if issues_path:
            with open(issues_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    idx = int(row.get("dataset_index", row.get("sample_id", -1)))
                    if idx >= 0:
                        issue_rows[idx] = row

        if mode == "relabel":
            changed = 0
            for idx, row in issue_rows.items():
                if idx >= len(self.y):
                    continue
                original = int(row.get("original_label", self.y[idx]))
                predicted = int(row.get("predicted_label", original))
                quality = float(row.get("label_quality_score", 1.0))
                if quality_source == "self_confidence":
                    quality = float(row.get("self_confidence", quality))
                elif quality_source == "constant":
                    quality = 1.0
                max_prob = float(row.get("max_pred_prob", 0.0))
                if quality < 0.2 and max_prob > 0.95 and original != predicted:
                    self.y[idx] = predicted
                    changed += 1
            self.sample_weights = None
            return changed

        if mode == "class_aware_downweight":
            hard_class_ids = set(hard_class_ids or [])
            self.sample_weights = np.ones(len(self.y), dtype=np.float32)
            self.sample_quality = np.ones(len(self.y), dtype=np.float32)
            affected = 0
            for idx, row in issue_rows.items():
                if idx >= len(self.y):
                    continue
                quality = float(row.get("label_quality_score", 1.0))
                max_prob = float(row.get("max_pred_prob", 0.0))
                original = int(row.get("original_label", self.y[idx]))
                predicted = int(row.get("predicted_label", original))
                is_confusing = str(row.get("is_confusing_pair", "False")).lower() == "true"
                if quality < 0.1 and max_prob > 0.98:
                    weight = float(weight_strong)
                elif is_confusing or original in hard_class_ids or predicted in hard_class_ids:
                    weight = float(weight_confusing)
                else:
                    weight = float(weight_normal)
                self.sample_weights[idx] = weight
                self.sample_quality[idx] = quality
                affected += 1
            return affected

        raise ValueError(f"Unknown cleanlab mode: {mode}")

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)

        class_name = self.class_names[int(self.y[idx])] if self.class_names else ""

        if self.augment and self.class_specific_aug:
            x = class_specific_augment(
                x,
                class_name,
                shift_prob=self.shift_prob,
                max_shift=self.max_shift,
                hardclass_aug_prob=self.hardclass_aug_prob if self.use_hardclass_aug else 0.0,
                morph_prob=self.morph_prob,
                scratch_aug_prob=self.scratch_aug_prob if self.use_scratch_aug else 0.0,
            )
        elif self.augment:
            x = random_wafer_augment(x, shift_prob=self.shift_prob, max_shift=self.max_shift)
        if self.morph_aug:
            x = random_morph_augment(x, prob=self.morph_prob, kernel_size=self.morph_kernel)
        if self.use_hardclass_aug and int(self.y[idx]) in self.hard_aug_ids and not self.class_specific_aug:
            x = random_morph_augment(x, prob=self.hardclass_aug_prob, kernel_size=self.morph_kernel)
        if self.use_scratch_aug and class_name in {"Scratch", "Loc"} and not self.class_specific_aug:
            x = random_scratch_augment(x, prob=self.scratch_aug_prob)
        if self.use_geo_prior:
            x = add_geometric_priors(x)
        geo = None
        if self.use_geometric_features:
            geo = torch.from_numpy(self.geo_features[idx]).float()
        pseudo_mask = None
        if self.use_pseudo_mask:
            pseudo_mask = generate_pseudo_masks_torch(x).float()

        if self.sample_weights is not None or self.sample_quality is not None:
            weight = 1.0 if self.sample_weights is None else float(self.sample_weights[idx])
            quality = 1.0 if self.sample_quality is None else float(self.sample_quality[idx])
            if pseudo_mask is not None and geo is not None:
                return x, y, geo, pseudo_mask, torch.tensor(weight, dtype=torch.float32), torch.tensor(quality, dtype=torch.float32)
            if pseudo_mask is not None:
                return x, y, pseudo_mask, torch.tensor(weight, dtype=torch.float32), torch.tensor(quality, dtype=torch.float32)
            if geo is not None:
                return x, y, geo, torch.tensor(weight, dtype=torch.float32), torch.tensor(quality, dtype=torch.float32)
            return x, y, torch.tensor(weight, dtype=torch.float32), torch.tensor(quality, dtype=torch.float32)
        if pseudo_mask is not None and geo is not None:
            return x, y, geo, pseudo_mask
        if pseudo_mask is not None:
            return x, y, pseudo_mask
        if geo is not None:
            return x, y, geo
        return x, y


def shift_with_zero_fill(x, dy, dx):
    if dy == 0 and dx == 0:
        return x

    _, h, w = x.shape
    out = torch.zeros_like(x)

    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)

    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, src_y0:src_y1, src_x0:src_x1]
    return out


def random_wafer_augment(x, shift_prob=0.15, max_shift=2):
    """Discrete-preserving D4 transforms plus small zero-filled translations."""
    k = torch.randint(0, 4, (1,)).item()
    if k:
        x = torch.rot90(x, k=k, dims=(-2, -1))

    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(-1,))
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=(-2,))

    if max_shift > 0 and torch.rand(()) < shift_prob:
        dy = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        dx = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        x = shift_with_zero_fill(x, dy=dy, dx=dx)

    return x.contiguous()


def random_rotate_flip(x, allow_shift=False, shift_prob=0.15, max_shift=2):
    k = torch.randint(0, 4, (1,), device=x.device).item()
    if k:
        x = torch.rot90(x, k=k, dims=(-2, -1))
    if torch.rand((), device=x.device) < 0.5:
        x = torch.flip(x, dims=(-1,))
    if torch.rand((), device=x.device) < 0.5:
        x = torch.flip(x, dims=(-2,))
    if allow_shift and max_shift > 0 and torch.rand((), device=x.device) < shift_prob:
        dy = torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item()
        dx = torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item()
        x = shift_with_zero_fill(x, dy=dy, dx=dx)
    return x.contiguous()


def discrete_to_onehot(labels, channels, dtype, device):
    if channels == 3:
        return F.one_hot(labels.long().clamp(0, 2), num_classes=3).permute(2, 0, 1).to(dtype=dtype, device=device)
    return (labels.float() / 2.0).unsqueeze(0).to(dtype=dtype, device=device)


def random_morph_augment(x, prob=0.1, kernel_size=3):
    if prob <= 0 or torch.rand(()) >= prob:
        return x
    channels, height, width = x.shape
    if channels == 3:
        labels = torch.argmax(x, dim=0)
    else:
        labels = torch.round(x.squeeze(0) * 2.0).long().clamp(0, 2)

    defect = (labels > 0).float().unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    op = torch.randint(0, 3, (1,)).item()
    if op == 0:
        morphed = F.max_pool2d(defect, kernel_size=kernel_size, stride=1, padding=pad)
        labels = torch.where(morphed.squeeze(0).squeeze(0) > 0, torch.maximum(labels, torch.ones_like(labels)), labels)
    elif op == 1:
        eroded = 1.0 - F.max_pool2d(1.0 - defect, kernel_size=kernel_size, stride=1, padding=pad)
        labels = torch.where((defect.squeeze(0).squeeze(0) > 0) & (eroded.squeeze(0).squeeze(0) <= 0), torch.zeros_like(labels), labels)
    else:
        drop_mask = torch.rand((height, width), device=x.device) < 0.03
        labels = torch.where((labels > 0) & drop_mask, torch.zeros_like(labels), labels)
    return discrete_to_onehot(labels, channels, x.dtype, x.device).contiguous()


def random_scratch_augment(x, prob=0.15):
    if prob <= 0 or torch.rand((), device=x.device) >= prob:
        return x
    channels, height, width = x.shape
    labels = torch.argmax(x, dim=0) if channels == 3 else torch.round(x.squeeze(0) * 2.0).long().clamp(0, 2)
    defect = labels > 0
    if not bool(defect.any()):
        return x
    op = torch.randint(0, 4, (1,), device=x.device).item()
    defect_f = defect.float().unsqueeze(0).unsqueeze(0)
    if op == 0:
        eroded = 1.0 - F.max_pool2d(1.0 - defect_f, kernel_size=3, stride=1, padding=1)
        labels = torch.where(defect & (eroded.squeeze(0).squeeze(0) <= 0), torch.zeros_like(labels), labels)
    elif op == 1:
        thick = F.max_pool2d(defect_f, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0) > 0
        labels = torch.where(thick, torch.maximum(labels, torch.ones_like(labels)), labels)
    elif op == 2:
        drop = (torch.rand((height, width), device=x.device) < 0.04) & defect
        labels = torch.where(drop, torch.zeros_like(labels), labels)
    else:
        ys, xs = torch.where(defect)
        if ys.numel() > 1:
            y0 = ys[torch.randint(0, ys.numel(), (1,), device=x.device)]
            x0 = xs[torch.randint(0, xs.numel(), (1,), device=x.device)]
            length = int(torch.randint(1, 4, (1,), device=x.device).item())
            direction = int(torch.randint(0, 4, (1,), device=x.device).item())
            dy, dx = [(1, 0), (0, 1), (1, 1), (1, -1)][direction]
            for step in range(-length, length + 1):
                yy = int(y0.item()) + dy * step
                xx = int(x0.item()) + dx * step
                if 0 <= yy < height and 0 <= xx < width:
                    labels[yy, xx] = torch.maximum(labels[yy, xx], torch.tensor(1, device=x.device, dtype=labels.dtype))
    return discrete_to_onehot(labels, channels, x.dtype, x.device).contiguous()


def generate_pseudo_masks_torch(x):
    channels, height, width = x.shape
    labels = torch.argmax(x[:3], dim=0) if channels >= 3 else torch.round(x.squeeze(0) * 2.0).long().clamp(0, 2)
    defect = (labels > 0).float()
    yy = torch.arange(height, device=x.device, dtype=x.dtype)[:, None]
    xx = torch.arange(width, device=x.device, dtype=x.dtype)[None, :]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(cx, cy)
    edge_band = (radius >= 0.78).float()
    center_ring = (((radius <= 0.34) | ((radius >= 0.32) & (radius <= 0.72))).float())
    defect4 = defect[None, None]
    neigh = F.conv2d(
        defect4,
        torch.ones((1, 1, 3, 3), dtype=x.dtype, device=x.device),
        padding=1,
    ).squeeze(0).squeeze(0) - defect
    gray = labels.float()[None, None]
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype,
        device=x.device,
    )[None, None]
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1).squeeze(0).squeeze(0)
    gy = F.conv2d(gray, sobel_y, padding=1).squeeze(0).squeeze(0)
    edge_response = torch.sqrt(gx * gx + gy * gy)
    scratch_like = ((defect > 0) & ((neigh <= 3) | (edge_response > 0))).float()
    return torch.stack(
        [
            defect,
            defect * edge_band,
            scratch_like,
            defect * center_ring,
        ],
        dim=0,
    ).to(dtype=x.dtype)


def class_specific_augment(
    x,
    class_name,
    shift_prob=0.15,
    max_shift=2,
    hardclass_aug_prob=0.15,
    morph_prob=0.1,
    scratch_aug_prob=0.1,
):
    if class_name == "Donut":
        return random_rotate_flip(x, allow_shift=False)
    if class_name == "Scratch":
        x = random_rotate_flip(x, allow_shift=False)
        return random_scratch_augment(x, prob=scratch_aug_prob)
    if class_name == "Loc":
        x = random_rotate_flip(x, allow_shift=True, shift_prob=shift_prob, max_shift=max_shift)
        return random_morph_augment(x, prob=morph_prob, kernel_size=3)
    if class_name == "Edge-Loc":
        x = random_rotate_flip(x, allow_shift=True, shift_prob=min(shift_prob, 0.1), max_shift=min(max_shift, 1))
        return random_morph_augment(x, prob=hardclass_aug_prob, kernel_size=3)
    if class_name == "Edge-Ring":
        return random_rotate_flip(x, allow_shift=True, shift_prob=min(shift_prob, 0.1), max_shift=min(max_shift, 1))
    return random_wafer_augment(x, shift_prob=shift_prob, max_shift=max_shift)


def tta_variants(x):
    variants = []
    for k in range(4):
        xr = torch.rot90(x, k=k, dims=(-2, -1))
        variants.append(xr)
        variants.append(torch.flip(xr, dims=(-1,)))
    return variants


class SimpleWaferCNN(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.silu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = F.silu(out + identity, inplace=True)

        return out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ECABlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = self.conv(y).transpose(1, 2).unsqueeze(-1)
        return x * self.sigmoid(y)


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        maxv = F.adaptive_max_pool2d(x, 1)
        x = x * self.sigmoid(self.mlp(avg) + self.mlp(maxv))
        avg_spatial = x.mean(dim=1, keepdim=True)
        max_spatial = x.max(dim=1, keepdim=True).values
        return x * self.sigmoid(self.spatial(torch.cat([avg_spatial, max_spatial], dim=1)))


class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x):
        _, _, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).transpose(2, 3)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


def build_attention(attention, channels):
    if attention == "none":
        return nn.Identity()
    if attention == "se":
        return SEBlock(channels, reduction=8)
    if attention == "eca":
        return ECABlock(channels)
    if attention == "cbam":
        return CBAMBlock(channels, reduction=8)
    if attention == "ca":
        return CoordinateAttention(channels, reduction=16)
    raise ValueError(f"Unknown attention: {attention}")


class EdgeBranch(nn.Module):
    def __init__(self, in_channels, out_channels, branch_type="fixed"):
        super().__init__()
        self.branch_type = branch_type
        if branch_type == "fixed":
            sobel_x = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)
            self.extractor = None
            edge_channels = 2
        elif branch_type == "learnable":
            self.extractor = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            edge_channels = out_channels
        else:
            raise ValueError(f"Unknown edge branch type: {branch_type}")

        self.project = nn.Sequential(
            nn.Conv2d(edge_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, out_size):
        if self.branch_type == "fixed":
            if x.shape[1] == 3:
                gray = x[:, 1:2] * 0.5 + x[:, 2:3]
            else:
                gray = x.mean(dim=1, keepdim=True)
            gx = F.conv2d(gray, self.sobel_x, padding=1)
            gy = F.conv2d(gray, self.sobel_y, padding=1)
            x = torch.cat([gx.abs(), gy.abs()], dim=1)
        else:
            x = self.extractor(x)
        x = F.adaptive_avg_pool2d(x, out_size)
        return self.project(x)


class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0, se_reduction=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=se_reduction)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + identity, inplace=True)
        return out


class ResWaferCNN(nn.Module):
    def __init__(self, num_classes, width=48, dropout=0.25, in_channels=1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
        )

        self.features = nn.Sequential(
            ResidualBlock(width, width, stride=1, dropout=dropout * 0.5),
            ResidualBlock(width, width * 2, stride=2, dropout=dropout * 0.5),
            ResidualBlock(width * 2, width * 2, stride=1, dropout=dropout * 0.5),
            ResidualBlock(width * 2, width * 4, stride=2, dropout=dropout),
            ResidualBlock(width * 4, width * 4, stride=1, dropout=dropout),
            ResidualBlock(width * 4, width * 4, stride=1, dropout=dropout),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


class SEWaferCNN(nn.Module):
    def __init__(self, num_classes, width=48, dropout=0.25, in_channels=1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            SEResidualBlock(width, width, stride=1, dropout=dropout * 0.5),
            SEResidualBlock(width, width * 2, stride=2, dropout=dropout * 0.5),
            SEResidualBlock(width * 2, width * 2, stride=1, dropout=dropout * 0.5),
            SEResidualBlock(width * 2, width * 4, stride=2, dropout=dropout),
            SEResidualBlock(width * 4, width * 4, stride=1, dropout=dropout),
            SEResidualBlock(width * 4, width * 4, stride=1, dropout=dropout),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x


class DualHeadSEWaferCNN(nn.Module):
    def __init__(self, num_classes, width=48, dropout=0.25, in_channels=1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            SEResidualBlock(width, width, stride=1, dropout=dropout * 0.5),
            SEResidualBlock(width, width * 2, stride=2, dropout=dropout * 0.5),
            SEResidualBlock(width * 2, width * 2, stride=1, dropout=dropout * 0.5),
            SEResidualBlock(width * 2, width * 4, stride=2, dropout=dropout),
            SEResidualBlock(width * 4, width * 4, stride=1, dropout=dropout),
            SEResidualBlock(width * 4, width * 4, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.main_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(width * 4, num_classes),
        )
        self.aux_head = nn.Sequential(
            nn.Dropout(p=min(dropout + 0.1, 0.6)),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.flatten(self.pool(x))
        return self.main_head(x), self.aux_head(x)


class DualHeadHybridViT(nn.Module):
    def __init__(
        self,
        num_classes,
        width=48,
        dropout=0.25,
        vit_dim=128,
        vit_depth=2,
        vit_heads=4,
        in_channels=1,
    ):
        super().__init__()
        if vit_dim % vit_heads != 0:
            raise ValueError("vit_dim must be divisible by vit_heads")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            SEResidualBlock(width, width, stride=1, dropout=dropout * 0.5),
            SEResidualBlock(width, width * 2, stride=2, dropout=dropout * 0.5),
            SEResidualBlock(width * 2, width * 2, stride=2, dropout=dropout * 0.5),
        )
        self.token_projection = nn.Conv2d(width * 2, vit_dim, kernel_size=1, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1, 64, vit_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_dim,
            nhead=vit_heads,
            dim_feedforward=vit_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=vit_depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(vit_dim)
        self.main_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(vit_dim, num_classes))
        self.aux_head = nn.Sequential(
            nn.Dropout(min(dropout + 0.1, 0.6)),
            nn.Linear(vit_dim, num_classes),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def position_tokens(self, height, width):
        if height == 8 and width == 8:
            return self.position_embedding
        pos = self.position_embedding.reshape(1, 8, 8, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(height, width), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, height * width, -1)

    def forward(self, x):
        x = self.token_projection(self.stem(x))
        height, width = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_tokens(height, width)
        tokens = self.norm(self.transformer(tokens)).mean(dim=1)
        return self.main_head(tokens), self.aux_head(tokens)


class DenseShallowBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=24, dropout=0.0):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = torch.cat([self.conv3(x), self.conv5(x)], dim=1)
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class DPFEELiteWaferCNN(BaseModel):
    def __init__(
        self,
        num_classes,
        width=48,
        dropout=0.25,
        in_channels=1,
        attention="none",
        use_edge_branch=False,
        edge_branch_type="fixed",
        feature_map_size=0,
    ):
        super().__init__()
        self.input_projection = (
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
            if int(in_channels) > 3 else nn.Identity()
        )
        backbone_in_channels = 3 if int(in_channels) > 3 else in_channels
        shallow_width = max(width // 2, 24)
        growth = max(width // 3, 16)
        self.feature_dim = width * 4
        self.use_edge_branch = bool(use_edge_branch)
        self.feature_map_size = int(feature_map_size or 0)

        self.branch_a_stem = nn.Sequential(
            nn.Conv2d(backbone_in_channels, shallow_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(shallow_width),
            nn.ReLU(inplace=True),
        )
        a_channels = shallow_width
        self.branch_a_blocks = nn.ModuleList()
        for _ in range(3):
            block = DenseShallowBlock(a_channels, growth_rate=growth, dropout=dropout * 0.35)
            self.branch_a_blocks.append(block)
            a_channels += growth * 2
        self.branch_a_transition = nn.Sequential(
            nn.Conv2d(a_channels, width * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )

        self.branch_b = nn.Sequential(
            nn.Conv2d(backbone_in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            SEResidualBlock(width, width, stride=1, dropout=dropout * 0.35),
            SEResidualBlock(width, width * 2, stride=2, dropout=dropout * 0.5),
            SEResidualBlock(width * 2, width * 2, stride=1, dropout=dropout * 0.5),
            SEResidualBlock(width * 2, width * 4, stride=2, dropout=dropout),
            SEResidualBlock(width * 4, width * 4, stride=1, dropout=dropout),
        )

        if self.use_edge_branch:
            self.edge_branch = EdgeBranch(
                in_channels=backbone_in_channels,
                out_channels=width,
                branch_type=edge_branch_type,
            )
        else:
            self.edge_branch = None

        fused_channels = width * 6 + (width if self.use_edge_branch else 0)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, width * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            SEBlock(width * 4, reduction=8),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.post_attention = build_attention(attention, width * 4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward_feature_map(self, x):
        x = self.input_projection(x)
        a = self.branch_a_stem(x)
        for block in self.branch_a_blocks:
            a = block(a)
        a = self.branch_a_transition(a)

        b = self.branch_b(x)
        if a.shape[-2:] != b.shape[-2:]:
            a = F.adaptive_avg_pool2d(a, b.shape[-2:])
        parts = [a, b]
        if self.edge_branch is not None:
            parts.append(self.edge_branch(x, b.shape[-2:]))
        x = torch.cat(parts, dim=1)
        x = self.fusion(x)
        if self.feature_map_size > 0:
            x = F.adaptive_avg_pool2d(x, (self.feature_map_size, self.feature_map_size))
        x = self.post_attention(x)
        return x

    def forward_features(self, x):
        x = self.forward_feature_map(x)
        return self.flatten(self.pool(x))

    def forward(self, x):
        x = self.forward_feature_map(x)
        return self.classifier(x)


class DualHeadDPFEELiteWaferCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        width=48,
        dropout=0.25,
        in_channels=1,
        attention="none",
        use_edge_branch=False,
        edge_branch_type="fixed",
        use_component_head=False,
        component_dim=None,
        use_capsule_head=False,
        capsule_hard_class_count=4,
        capsule_dim=8,
        capsule_routing_iters=3,
        use_scratchness_head=False,
        feature_map_size=0,
    ):
        super().__init__()
        self.use_component_head = bool(use_component_head)
        self.use_capsule_head = bool(use_capsule_head)
        self.use_scratchness_head = bool(use_scratchness_head)
        self.component_dim = int(component_dim or num_classes)
        self.backbone = DPFEELiteWaferCNN(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            in_channels=in_channels,
            attention=attention,
            use_edge_branch=use_edge_branch,
            edge_branch_type=edge_branch_type,
            feature_map_size=feature_map_size,
        )
        feature_dim = self.backbone.feature_dim
        self.main_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.aux_head = nn.Sequential(
            nn.Dropout(min(dropout + 0.1, 0.6)),
            nn.Linear(feature_dim, num_classes),
        )
        if self.use_component_head:
            self.component_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, self.component_dim),
            )
        else:
            self.component_head = None
        if self.use_capsule_head:
            self.capsule_head = CapsuleHardClassHead(
                in_channels=feature_dim,
                num_hard_classes=int(capsule_hard_class_count),
                capsule_dim=int(capsule_dim),
                routing_iters=int(capsule_routing_iters),
            )
        else:
            self.capsule_head = None
        if self.use_scratchness_head:
            self.scratchness_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, 1),
            )
        else:
            self.scratchness_head = None

    def forward(self, x):
        feature_map = self.backbone.forward_feature_map(x)
        features = self.backbone.flatten(self.backbone.pool(feature_map))
        main_logits = self.main_head(features)
        aux_logits = self.aux_head(features)
        payload = {}
        if self.component_head is not None:
            payload["component_logits"] = self.component_head(features)
        if self.capsule_head is not None:
            payload["capsule_logits"] = self.capsule_head(feature_map)
        if self.scratchness_head is not None:
            payload["scratchness_logits"] = self.scratchness_head(features).squeeze(1)
        if not payload:
            return main_logits, aux_logits
        return main_logits, aux_logits, payload


class TwoStageNormalDefectDPFEE(nn.Module):
    def __init__(
        self,
        num_classes,
        width=48,
        dropout=0.25,
        in_channels=1,
        attention="none",
        use_edge_branch=False,
        edge_branch_type="fixed",
        normal_class_idx=None,
        defect_loss_weight=1.0,
        feature_map_size=0,
    ):
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("Two-stage model requires at least two classes")
        self.num_classes = int(num_classes)
        self.normal_class_idx = int(self.num_classes - 1 if normal_class_idx is None else normal_class_idx)
        self.defect_class_ids = [idx for idx in range(self.num_classes) if idx != self.normal_class_idx]
        self.defect_loss_weight = float(defect_loss_weight)
        self.backbone = DPFEELiteWaferCNN(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            in_channels=in_channels,
            attention=attention,
            use_edge_branch=use_edge_branch,
            edge_branch_type=edge_branch_type,
            feature_map_size=feature_map_size,
        )
        feature_dim = self.backbone.feature_dim
        self.binary_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 2))
        self.defect_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, self.num_classes - 1))

    def forward_parts(self, x):
        feature_map = self.backbone.forward_feature_map(x)
        features = self.backbone.flatten(self.backbone.pool(feature_map))
        return self.binary_head(features), self.defect_head(features)

    def combine_logits(self, binary_logits, defect_logits):
        combined = binary_logits.new_full((binary_logits.shape[0], self.num_classes), -1e4)
        combined[:, self.normal_class_idx] = binary_logits[:, 0]
        combined[:, self.defect_class_ids] = binary_logits[:, 1:2] + defect_logits
        return combined

    def forward(self, x):
        binary_logits, defect_logits = self.forward_parts(x)
        return self.combine_logits(binary_logits, defect_logits)

    def two_stage_loss(self, x, y, label_smoothing=0.0, sample_weight=None):
        binary_logits, defect_logits = self.forward_parts(x)
        binary_target = (y != self.normal_class_idx).long()
        binary_loss = F.cross_entropy(binary_logits, binary_target, reduction="none")
        if sample_weight is not None:
            binary_loss = (binary_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
        else:
            binary_loss = binary_loss.mean()

        defect_mask = y != self.normal_class_idx
        if defect_mask.any():
            global_to_local = {
                int(global_idx): int(local_idx)
                for local_idx, global_idx in enumerate(self.defect_class_ids)
            }
            local_target = torch.tensor(
                [global_to_local[int(item)] for item in y[defect_mask].detach().cpu().tolist()],
                dtype=torch.long,
                device=y.device,
            )
            defect_loss = F.cross_entropy(
                defect_logits[defect_mask],
                local_target,
                reduction="none",
                label_smoothing=float(label_smoothing),
            )
            if sample_weight is not None:
                defect_weight = sample_weight[defect_mask]
                defect_loss = (defect_loss * defect_weight).sum() / defect_weight.sum().clamp_min(1e-6)
            else:
                defect_loss = defect_loss.mean()
        else:
            defect_loss = defect_logits.sum() * 0.0
        return binary_loss + self.defect_loss_weight * defect_loss


class DPFEETransformerTail(nn.Module):
    def __init__(
        self,
        num_classes,
        width=48,
        dropout=0.25,
        vit_dim=192,
        vit_depth=2,
        vit_heads=4,
        in_channels=1,
        attention="none",
        use_edge_branch=False,
        edge_branch_type="fixed",
    ):
        super().__init__()
        self.backbone = DPFEELiteWaferCNN(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            in_channels=in_channels,
            attention=attention,
            use_edge_branch=use_edge_branch,
            edge_branch_type=edge_branch_type,
        )
        if vit_dim % vit_heads != 0:
            raise ValueError("vit_dim must be divisible by vit_heads")
        self.token_projection = nn.Conv2d(self.backbone.feature_dim, vit_dim, kernel_size=1, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1, 16 * 16, vit_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=vit_dim,
            nhead=vit_heads,
            dim_feedforward=vit_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=vit_depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(vit_dim)
        self.main_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(vit_dim, num_classes))
        self.aux_head = nn.Sequential(
            nn.Dropout(min(dropout + 0.1, 0.6)),
            nn.Linear(vit_dim, num_classes),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        feature_map = self.backbone.forward_feature_map(x)
        tokens = self.token_projection(feature_map).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        tokens = self.transformer(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        return self.main_head(pooled), self.aux_head(pooled)


def load_label_info(data_dir):
    path = Path(data_dir) / "label_map.json"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find label_map.json in {data_dir}")

    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)

    class_names = info["class_names"]
    num_classes = info["num_classes"]
    image_size = info["image_size"]
    return class_names, num_classes, image_size


def compute_class_weights(dataset, num_classes):
    y = dataset.y
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)

    # 避免除零
    counts = np.maximum(counts, 1.0)

    # 常用 class weight: total / (num_classes * count)
    weights = len(y) / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32), counts


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        log_pt = -F.cross_entropy(logits, target, reduction="none")
        pt = log_pt.exp().clamp(min=1e-6, max=1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def build_criterion(loss_name, class_weights=None, label_smoothing=0.0, focal_gamma=2.0):
    if loss_name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_name in ("focal", "qfl"):
        return FocalLoss(
            gamma=focal_gamma,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    raise ValueError(f"Unknown loss: {loss_name}")


def primary_logits(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def output_aux_payload(output):
    if isinstance(output, (tuple, list)) and len(output) >= 3 and isinstance(output[2], dict):
        return output[2]
    return {}


def prediction_logits(output, aux_eval_weight=0.0):
    logits = primary_logits(output)
    if is_dual_head_output(output) and aux_eval_weight > 0:
        logits = logits + float(aux_eval_weight) * output[1]
    return logits


def model_forward(model, x, geo=None):
    if geo is not None:
        try:
            return model(x, geo)
        except TypeError:
            return model(x)
    return model(x)


def scratchness_labels_from_geo(
    geo,
    geo_mean,
    geo_std,
    anisotropy_threshold=8.0,
    skeleton_threshold=3.0,
    aspect_threshold=4.0,
):
    if geo_mean is not None and geo_std is not None:
        mean = torch.as_tensor(geo_mean, dtype=geo.dtype, device=geo.device)
        std = torch.as_tensor(geo_std, dtype=geo.dtype, device=geo.device).clamp_min(1e-6)
        raw = geo * std + mean
    else:
        raw = geo
    name_to_idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    anisotropy = raw[:, name_to_idx["anisotropy"]]
    skeleton_norm = raw[:, name_to_idx["skeleton_norm"]]
    aspect = raw[:, name_to_idx["bbox_aspect_ratio"]]
    label = (
        (anisotropy > float(anisotropy_threshold))
        | (skeleton_norm > float(skeleton_threshold))
        | (aspect > float(aspect_threshold))
    )
    return label.float()


def apply_capsule_fusion(logits, capsule_logits, hard_class_ids, mode):
    mode = mode or "none"
    if capsule_logits is None or mode == "none" or not hard_class_ids:
        return logits
    fused = logits.clone()
    hard_ids = torch.tensor(list(hard_class_ids), device=logits.device, dtype=torch.long)
    if mode == "logit_add":
        fused[:, hard_ids] = fused[:, hard_ids] + capsule_logits
        return fused
    if mode == "rerank_hard":
        top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).indices
        hard_mask = torch.isin(top2, hard_ids).any(dim=1)
        if hard_mask.any():
            local = capsule_logits[hard_mask].argmax(dim=1)
            chosen = hard_ids[local]
            hard_values = fused[hard_mask][:, hard_ids]
            replacement = hard_values.max(dim=1).values + 1e-3
            rows = torch.where(hard_mask)[0]
            fused[rows, chosen] = replacement
        return fused
    return fused


def is_dual_head_output(output):
    return isinstance(output, (tuple, list)) and len(output) >= 2


def per_sample_supervised_loss(logits, target, loss_name, class_weights=None, focal_gamma=2.0, label_smoothing=0.0):
    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if loss_name == "ce":
        return ce
    if loss_name in ("focal", "qfl"):
        log_pt = -F.cross_entropy(logits, target, reduction="none")
        pt = log_pt.exp().clamp(min=1e-6, max=1.0)
        return ((1.0 - pt) ** focal_gamma) * ce
    raise ValueError(f"Unknown loss: {loss_name}")


def build_hard_class_weights(class_names, hard_classes, hard_weight, device):
    weights = torch.ones(len(class_names), dtype=torch.float32, device=device)
    for name in hard_classes:
        if name in class_names:
            weights[class_names.index(name)] = float(hard_weight)
    return weights


def find_normal_class_idx(class_names):
    aliases = {"normal", "none", "no defect", "no-defect", "nonedefect"}
    for idx, name in enumerate(class_names):
        if str(name).strip().lower() in aliases:
            return idx
    return None


def defect_class_indices(class_names):
    normal_idx = find_normal_class_idx(class_names)
    return [idx for idx in range(len(class_names)) if idx != normal_idx]


def defect_macro_f1_score(y_true, y_pred, class_names):
    labels = defect_class_indices(class_names)
    if not labels:
        return None
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def class_counts_dict(y, class_names):
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=len(class_names)).astype(int)
    return {class_names[i]: int(counts[i]) for i in range(len(class_names))}


def build_confusion_pairs(class_names):
    names = [
        ("Loc", "Scratch"),
        ("Scratch", "Loc"),
        ("Loc", "Edge-Loc"),
        ("Edge-Loc", "Loc"),
        ("Scratch", "Edge-Loc"),
        ("Edge-Loc", "Scratch"),
        ("Edge-Loc", "Edge-Ring"),
        ("Edge-Ring", "Edge-Loc"),
        ("Donut", "Center"),
        ("Center", "Donut"),
    ]
    pairs = []
    for true_name, confusing_name in names:
        if true_name in class_names and confusing_name in class_names:
            pairs.append((class_names.index(true_name), class_names.index(confusing_name)))
    return pairs


def confusion_margin_loss(logits, target, pairs, margin=0.2):
    if not pairs:
        return logits.new_tensor(0.0)
    losses = []
    for true_idx, confusing_idx in pairs:
        mask = target == int(true_idx)
        if mask.any():
            true_logit = logits[mask, true_idx]
            confusing_logit = logits[mask, confusing_idx]
            losses.append(F.relu(confusing_logit + float(margin) - true_logit))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.cat(losses).mean()


def supervised_training_loss(
    output,
    target,
    loss_name,
    class_weights=None,
    focal_gamma=2.0,
    label_smoothing=0.0,
    aux_loss_weight=0.0,
    aux_focal_gamma=3.0,
    hard_sample_weights=None,
    loc_class_idx=None,
    loc_loss_boost=0.0,
    confusion_pairs=None,
    confusion_margin=0.2,
    confusion_lambda=0.0,
    sample_weight=None,
    sample_quality=None,
    qfl_quality_min=0.5,
    qfl_quality_max=1.0,
    unet_loss_weight=0.0,
    pseudo_mask_loss_weight=0.0,
    component_loss_weight=0.0,
    component_matrix=None,
    scratchness_loss_weight=0.0,
    scratchness_thresholds=None,
    geo_mean=None,
    geo_std=None,
    capsule_loss_weight=0.0,
    capsule_hard_ids=None,
):
    logits = primary_logits(output)
    sample_loss = per_sample_supervised_loss(
        logits,
        target,
        loss_name=loss_name,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    if loss_name == "qfl" and sample_quality is not None:
        quality = sample_quality.to(sample_loss.device).float()
        quality = quality.clamp(min=float(qfl_quality_min), max=float(qfl_quality_max))
        sample_loss = sample_loss * quality
    if loc_class_idx is not None and loc_loss_boost > 0:
        sample_loss = sample_loss * torch.where(
            target == int(loc_class_idx),
            torch.full_like(sample_loss, 1.0 + float(loc_loss_boost)),
            torch.ones_like(sample_loss),
        )
    if sample_weight is not None:
        sample_weight = sample_weight.to(sample_loss.device).float()
        loss = (sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
    else:
        loss = sample_loss.mean()
    if confusion_lambda > 0 and confusion_pairs:
        loss = loss + float(confusion_lambda) * confusion_margin_loss(
            logits,
            target,
            confusion_pairs,
            margin=confusion_margin,
        )

    if is_dual_head_output(output) and aux_loss_weight > 0:
        aux_logits = output[1]
        aux_loss = per_sample_supervised_loss(
            aux_logits,
            target,
            loss_name="focal",
            class_weights=class_weights,
            focal_gamma=aux_focal_gamma,
            label_smoothing=label_smoothing,
        )
        if loss_name == "qfl" and sample_quality is not None:
            quality = sample_quality.to(aux_loss.device).float()
            quality = quality.clamp(min=float(qfl_quality_min), max=float(qfl_quality_max))
            aux_loss = aux_loss * quality
        if hard_sample_weights is not None:
            aux_loss = aux_loss * hard_sample_weights[target]
        if loc_class_idx is not None and loc_loss_boost > 0:
            aux_loss = aux_loss * torch.where(
                target == int(loc_class_idx),
                torch.full_like(aux_loss, 1.0 + float(loc_loss_boost)),
                torch.ones_like(aux_loss),
            )
        if sample_weight is not None:
            aux_total = (aux_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
        else:
            aux_total = aux_loss.mean()
        if confusion_lambda > 0 and confusion_pairs:
            aux_total = aux_total + float(confusion_lambda) * confusion_margin_loss(
                aux_logits,
                target,
                confusion_pairs,
                margin=confusion_margin,
            )
        loss = loss + float(aux_loss_weight) * aux_total
    aux_payload = output_aux_payload(output)
    component_logits = aux_payload.get("component_logits")
    if component_logits is not None and component_loss_weight > 0:
        if component_matrix is None:
            component_target = F.one_hot(target, num_classes=component_logits.shape[1]).float()
        else:
            component_target = component_matrix.to(component_logits.device).float()[target]
        component_loss = F.binary_cross_entropy_with_logits(
            component_logits,
            component_target,
            reduction="none",
        ).mean(dim=1)
        if sample_weight is not None:
            component_total = (component_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
        else:
            component_total = component_loss.mean()
        loss = loss + float(component_loss_weight) * component_total
    if unet_loss_weight > 0 and "mask_loss" in aux_payload:
        loss = loss + float(unet_loss_weight) * aux_payload["mask_loss"]
    return loss


def build_balanced_sampler(dataset, num_classes):
    counts = np.bincount(dataset.y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    sample_weights = 1.0 / counts[dataset.y]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_model(
    model_name,
    num_classes,
    width=48,
    dropout=0.25,
    vit_dim=128,
    vit_depth=2,
    vit_heads=4,
    vit_patch_size=8,
    in_channels=1,
    attention="none",
    use_edge_branch=False,
    edge_branch_type="fixed",
    image_size=64,
    use_unet_structure_branch=False,
    freeze_dpfee_backbone=False,
    unet_entropy_weight=1.0,
    use_component_head=False,
    component_dim=None,
    use_geometric_features=False,
    geo_feature_dim=18,
    geo_mlp_hidden=64,
    geo_dropout=0.1,
    use_scratchness_head=False,
    use_capsule_head=False,
    capsule_hard_class_count=4,
    capsule_dim=8,
    capsule_routing_iters=3,
    dpfee_feature_map_size=0,
    normal_class_idx=None,
    defect_loss_weight=1.0,
):
    if use_geometric_features:
        dpfee_backbone = DPFEELiteWaferCNN(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            in_channels=in_channels,
            attention=attention,
            use_edge_branch=use_edge_branch,
            edge_branch_type=edge_branch_type,
            feature_map_size=dpfee_feature_map_size,
        )
        return DPFEEGeometryHybrid(
            dpfee_backbone=dpfee_backbone,
            num_classes=num_classes,
            geo_feature_dim=geo_feature_dim,
            geo_mlp_hidden=geo_mlp_hidden,
            geo_dropout=geo_dropout,
            dropout=dropout,
            use_scratchness_head=use_scratchness_head,
            use_capsule_head=use_capsule_head,
            capsule_hard_class_count=capsule_hard_class_count,
            capsule_dim=capsule_dim,
            capsule_routing_iters=capsule_routing_iters,
        )
    if use_unet_structure_branch or model_name == "unet_dpfee_hybrid":
        dpfee_backbone = DPFEELiteWaferCNN(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            in_channels=in_channels,
            attention=attention,
            use_edge_branch=use_edge_branch,
            edge_branch_type=edge_branch_type,
            feature_map_size=dpfee_feature_map_size,
        )
        model = UNetDPFEEHybrid(
            dpfee_backbone=dpfee_backbone,
            num_classes=num_classes,
            dropout=dropout,
            in_channels=in_channels,
            unet_entropy_weight=unet_entropy_weight,
        )
        if freeze_dpfee_backbone:
            model.freeze_dpfee_backbone()
        return model
    if model_name == "simple":
        return SimpleWaferCNN(num_classes=num_classes, in_channels=in_channels)
    if model_name == "resnet":
        return ResWaferCNN(num_classes=num_classes, width=width, dropout=dropout, in_channels=in_channels)
    if model_name == "seresnet":
        return SEWaferCNN(num_classes=num_classes, width=width, dropout=dropout, in_channels=in_channels)
    if model_name == "dpfee":
        return DPFEELiteWaferCNN(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            in_channels=in_channels,
            attention=attention,
            use_edge_branch=use_edge_branch,
            edge_branch_type=edge_branch_type,
            feature_map_size=dpfee_feature_map_size,
        )
    if model_name in ("dpfee_dual", "dual_dpfee"):
        return DualHeadDPFEELiteWaferCNN(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            in_channels=in_channels,
            attention=attention,
            use_edge_branch=use_edge_branch,
            edge_branch_type=edge_branch_type,
            use_component_head=use_component_head,
            component_dim=component_dim,
            use_capsule_head=use_capsule_head,
            capsule_hard_class_count=capsule_hard_class_count,
            capsule_dim=capsule_dim,
            capsule_routing_iters=capsule_routing_iters,
            use_scratchness_head=use_scratchness_head,
            feature_map_size=dpfee_feature_map_size,
        )
    if model_name in ("dual_seresnet", "dualhead_seresnet"):
        return DualHeadSEWaferCNN(num_classes=num_classes, width=width, dropout=dropout, in_channels=in_channels)
    if model_name == "two_stage_dpfee":
        return TwoStageNormalDefectDPFEE(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            in_channels=in_channels,
            attention=attention,
            use_edge_branch=use_edge_branch,
            edge_branch_type=edge_branch_type,
            normal_class_idx=normal_class_idx,
            defect_loss_weight=defect_loss_weight,
            feature_map_size=dpfee_feature_map_size,
        )
    if model_name == "dual_hybridvit":
        return DualHeadHybridViT(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            vit_dim=vit_dim,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            in_channels=in_channels,
        )
    if model_name == "vit_tiny_wafer":
        return ViTTinyWafer(
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=image_size,
            patch_size=vit_patch_size,
            embed_dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            dropout=dropout,
            dual_head=True,
        )
    if model_name == "cnn_stem_vit_wafer":
        return CNNStemViTWafer(
            num_classes=num_classes,
            in_channels=in_channels,
            embed_dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            dropout=dropout,
            dual_head=True,
        )
    if model_name == "dpfee_transformer_tail":
        return DPFEETransformerTail(
            num_classes=num_classes,
            width=width,
            dropout=dropout,
            vit_dim=vit_dim,
            vit_depth=vit_depth,
            vit_heads=vit_heads,
            in_channels=in_channels,
            attention=attention,
            use_edge_branch=use_edge_branch,
            edge_branch_type=edge_branch_type,
        )

    raise ValueError(f"Unknown model: {model_name}")


class ModelEma:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.num_updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        ema_state = self.module.state_dict()
        model_state = model.state_dict()

        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


class ExpertBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class ExpertResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, width=32):
        super().__init__()
        self.in_channels = int(in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(width, width, 2, stride=1)
        self.layer2 = self._make_layer(width, width * 2, 2, stride=2)
        self.layer3 = self._make_layer(width * 2, width * 4, 2, stride=2)
        self.layer4 = self._make_layer(width * 4, width * 4, 2, stride=1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 4, num_classes))

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride):
        layers = [ExpertBasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ExpertBasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


def load_expert_model(expert_model_path, in_channels, device):
    ckpt = torch.load(expert_model_path, map_location=device, weights_only=False)
    hard_ids = ckpt.get("hard_ids", [2, 4, 7])
    expert_in_channels = int(ckpt.get("in_channels", in_channels))
    model = ExpertResNet18(in_channels=expert_in_channels, num_classes=len(hard_ids)).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, hard_ids


def apply_expert_gating(logits, x, expert_model, hard_ids, threshold):
    probs = F.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    max_prob = probs.gather(1, pred[:, None]).squeeze(1)
    gate_mask = torch.zeros_like(pred, dtype=torch.bool)
    for cls_id in hard_ids:
        gate_mask |= pred == int(cls_id)
    gate_mask &= max_prob < float(threshold)
    final = pred.clone()
    replaced = int(gate_mask.sum().item())
    if replaced > 0:
        local_to_global = {i: int(hard_ids[i]) for i in range(len(hard_ids))}
        expert_x = x[gate_mask]
        if expert_x.shape[1] != getattr(expert_model, "in_channels", expert_x.shape[1]):
            expert_x = expert_x[:, : expert_model.in_channels]
        expert_local = expert_model(expert_x).argmax(dim=1)
        expert_global = torch.tensor(
            [local_to_global[int(i)] for i in expert_local.detach().cpu().tolist()],
            dtype=final.dtype,
            device=final.device,
        )
        final[gate_mask] = expert_global
    return final, replaced


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes,
    class_names=None,
    criterion=None,
    use_tta=False,
    use_amp=False,
    aux_eval_weight=0.0,
    expert_model=None,
    expert_hard_ids=None,
    expert_threshold=0.85,
    capsule_hard_ids=None,
    capsule_fusion_mode="none",
):
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_num = 0
    gated_replaced = 0

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    for batch in loader:
        geo = None
        if len(batch) == 5:
            x, y, geo, sample_weight, _sample_quality = batch
            geo = geo.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
        elif len(batch) == 4:
            x, y, sample_weight, _sample_quality = batch
            sample_weight = sample_weight.to(device, non_blocking=True)
        elif len(batch) == 3:
            x, y, sample_weight = batch
            if sample_weight.ndim > 1:
                geo = sample_weight.to(device, non_blocking=True)
                sample_weight = None
            else:
                sample_weight = sample_weight.to(device, non_blocking=True)
        else:
            x, y = batch
            sample_weight = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_tta:
                logits = torch.stack(
                    [prediction_logits(model_forward(model, x_aug, geo), aux_eval_weight) for x_aug in tta_variants(x)],
                    dim=0,
                ).mean(dim=0)
                if capsule_fusion_mode != "none":
                    output = model_forward(model, x, geo)
                    logits = apply_capsule_fusion(
                        logits,
                        output_aux_payload(output).get("capsule_logits"),
                        capsule_hard_ids or [],
                        capsule_fusion_mode,
                    )
            else:
                output = model_forward(model, x, geo)
                logits = prediction_logits(output, aux_eval_weight)
                logits = apply_capsule_fusion(
                    logits,
                    output_aux_payload(output).get("capsule_logits"),
                    capsule_hard_ids or [],
                    capsule_fusion_mode,
                )
            loss = criterion(logits, y)

        if expert_model is not None:
            preds, replaced = apply_expert_gating(
                logits,
                x,
                expert_model,
                expert_hard_ids or [],
                expert_threshold,
            )
            gated_replaced += replaced
        else:
            preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

        total_loss += loss.item() * y.size(0)
        total_num += y.size(0)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    metrics = {
        "loss": total_loss / max(total_num, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "gated_replaced": gated_replaced,
    }
    if class_names is not None and find_normal_class_idx(class_names) is not None:
        metrics["defect_macro_f1"] = defect_macro_f1_score(y_true, y_pred, class_names)

    return metrics, y_true, y_pred


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    ema=None,
    use_amp=False,
    loss_name=None,
    class_weights=None,
    focal_gamma=2.0,
    label_smoothing=0.0,
    aux_loss_weight=0.0,
    aux_focal_gamma=3.0,
    hard_sample_weights=None,
    loc_class_idx=None,
    loc_loss_boost=0.0,
    confusion_pairs=None,
    confusion_margin=0.2,
    confusion_lambda=0.0,
    qfl_quality_min=0.5,
    qfl_quality_max=1.0,
    unet_loss_weight=0.0,
    pseudo_mask_loss_weight=0.0,
    component_loss_weight=0.0,
    component_matrix=None,
    scratchness_loss_weight=0.0,
    scratchness_thresholds=None,
    scratchness_start_epoch=1,
    scratchness_hard_ids=None,
    current_epoch=1,
    geo_mean=None,
    geo_std=None,
    capsule_loss_weight=0.0,
    capsule_hard_ids=None,
    teacher_model=None,
    distill_loss_weight=0.0,
    distill_temperature=2.0,
    aux_eval_weight=0.0,
):
    model.train()

    total_loss = 0.0
    total_num = 0
    gated_replaced = 0
    mask_loss_total = 0.0
    mask_activation_total = 0.0

    for batch in loader:
        sample_quality = None
        geo = None
        pseudo_mask = None
        if len(batch) == 6:
            x, y, geo, pseudo_mask, sample_weight, sample_quality = batch
            geo = geo.to(device, non_blocking=True)
            pseudo_mask = pseudo_mask.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            sample_quality = sample_quality.to(device, non_blocking=True)
        elif len(batch) == 5:
            x, y, third, sample_weight, sample_quality = batch
            if third.ndim > 2:
                pseudo_mask = third.to(device, non_blocking=True)
                geo = None
            else:
                geo = third.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            sample_quality = sample_quality.to(device, non_blocking=True)
        elif len(batch) == 4:
            x, y, sample_weight, sample_quality = batch
            sample_weight = sample_weight.to(device, non_blocking=True)
            sample_quality = sample_quality.to(device, non_blocking=True)
        elif len(batch) == 3:
            x, y, sample_weight = batch
            if sample_weight.ndim > 1:
                if sample_weight.ndim > 3:
                    pseudo_mask = sample_weight.to(device, non_blocking=True)
                else:
                    geo = sample_weight.to(device, non_blocking=True)
                sample_weight = None
            else:
                sample_weight = sample_weight.to(device, non_blocking=True)
        else:
            x, y = batch
            sample_weight = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            if hasattr(model, "two_stage_loss"):
                loss = model.two_stage_loss(
                    x,
                    y,
                    label_smoothing=label_smoothing,
                    sample_weight=sample_weight,
                )
                output = model_forward(model, x, geo)
            else:
                output = model_forward(model, x, geo)
            if (not hasattr(model, "two_stage_loss")) and is_dual_head_output(output):
                loss = supervised_training_loss(
                    output,
                    y,
                    loss_name=loss_name,
                    class_weights=class_weights,
                    focal_gamma=focal_gamma,
                    label_smoothing=label_smoothing,
                    aux_loss_weight=aux_loss_weight,
                    aux_focal_gamma=aux_focal_gamma,
                    hard_sample_weights=hard_sample_weights,
                    loc_class_idx=loc_class_idx,
                    loc_loss_boost=loc_loss_boost,
                    confusion_pairs=confusion_pairs,
                    confusion_margin=confusion_margin,
                    confusion_lambda=confusion_lambda,
                    sample_weight=sample_weight,
                    sample_quality=sample_quality,
                    qfl_quality_min=qfl_quality_min,
                    qfl_quality_max=qfl_quality_max,
                    unet_loss_weight=unet_loss_weight,
                    component_loss_weight=component_loss_weight,
                    component_matrix=component_matrix,
                )
                payload = output_aux_payload(output)
                if (
                    scratchness_loss_weight > 0
                    and int(current_epoch) >= int(scratchness_start_epoch)
                    and geo is not None
                    and "scratchness_logits" in payload
                ):
                    thresholds = scratchness_thresholds or {}
                    scratch_target = scratchness_labels_from_geo(
                        geo,
                        geo_mean,
                        geo_std,
                        anisotropy_threshold=thresholds.get("anisotropy", 8.0),
                        skeleton_threshold=thresholds.get("skeleton", 3.0),
                        aspect_threshold=thresholds.get("aspect", 4.0),
                    )
                    scratch_loss = F.binary_cross_entropy_with_logits(
                        payload["scratchness_logits"],
                        scratch_target,
                        reduction="none",
                    )
                    if scratchness_hard_ids:
                        hard_ids_tensor = torch.as_tensor(
                            scratchness_hard_ids,
                            dtype=y.dtype,
                            device=y.device,
                        )
                        hard_mask = torch.isin(y, hard_ids_tensor)
                        if hard_mask.any():
                            scratch_loss = scratch_loss[hard_mask]
                            scratch_weight = sample_weight[hard_mask] if sample_weight is not None else None
                        else:
                            scratch_loss = scratch_loss[:0]
                            scratch_weight = None
                    else:
                        scratch_weight = sample_weight
                    if scratch_loss.numel() == 0:
                        scratch_loss = payload["scratchness_logits"].sum() * 0.0
                    elif scratch_weight is not None:
                        scratch_loss = (scratch_loss * scratch_weight).sum() / scratch_weight.sum().clamp_min(1e-6)
                    else:
                        scratch_loss = scratch_loss.mean()
                    loss = loss + float(scratchness_loss_weight) * scratch_loss
                if capsule_loss_weight > 0 and "capsule_logits" in payload:
                    loss = loss + float(capsule_loss_weight) * capsule_margin_loss(
                        payload["capsule_logits"],
                        y,
                        capsule_hard_ids or [],
                    )
                if pseudo_mask_loss_weight > 0 and pseudo_mask is not None and "mask_logits" in payload:
                    mask_loss = F.binary_cross_entropy_with_logits(
                        payload["mask_logits"],
                        pseudo_mask,
                        reduction="none",
                    ).mean(dim=(1, 2, 3))
                    if sample_weight is not None:
                        mask_loss = (mask_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
                    else:
                        mask_loss = mask_loss.mean()
                    loss = loss + float(pseudo_mask_loss_weight) * mask_loss
            elif not hasattr(model, "two_stage_loss"):
                logits = primary_logits(output)
                if sample_weight is not None or (loss_name == "qfl" and sample_quality is not None):
                    sample_loss = per_sample_supervised_loss(
                        logits,
                        y,
                        loss_name=loss_name,
                        class_weights=class_weights,
                        focal_gamma=focal_gamma,
                        label_smoothing=label_smoothing,
                    )
                    if loss_name == "qfl" and sample_quality is not None:
                        quality = sample_quality.float().clamp(
                            min=float(qfl_quality_min),
                            max=float(qfl_quality_max),
                        )
                        sample_loss = sample_loss * quality
                    if sample_weight is None:
                        sample_weight = torch.ones_like(sample_loss)
                    loss = (sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
                else:
                    loss = criterion(logits, y)
            if teacher_model is not None and distill_loss_weight > 0:
                with torch.no_grad():
                    teacher_output = model_forward(teacher_model, x, None)
                    teacher_logits = prediction_logits(teacher_output, aux_eval_weight)
                student_logits = prediction_logits(output, aux_eval_weight)
                temp = float(distill_temperature)
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits / temp, dim=1),
                    F.softmax(teacher_logits / temp, dim=1),
                    reduction="batchmean",
                ) * (temp * temp)
                loss = loss + float(distill_loss_weight) * distill_loss

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * y.size(0)
        total_num += y.size(0)
        aux_payload = output_aux_payload(output)
        if "mask_loss" in aux_payload:
            mask_loss_total += float(aux_payload["mask_loss"].detach().item()) * y.size(0)
        if "mask_activation_mean" in aux_payload:
            mask_activation_total += float(aux_payload["mask_activation_mean"].detach().item()) * y.size(0)

    avg_loss = total_loss / max(total_num, 1)
    model._last_mask_loss = mask_loss_total / max(total_num, 1) if mask_loss_total > 0 else None
    model._last_mask_activation_mean = mask_activation_total / max(total_num, 1) if mask_activation_total > 0 else None
    return avg_loss


def plot_training_curves(history, out_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.plot(epochs, history["val_accuracy"], label="val_accuracy")
    plt.plot(epochs, history["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # 在格子里写数字
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_per_class_recall(y_true, y_pred, class_names, out_path):
    recalls = recall_score(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, recalls)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Recall")
    plt.title("Per-Class Recall")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return recalls


def save_geo_prior_visualization(dataset, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = torch.from_numpy(dataset.x[0])
    geo = add_geometric_priors(x)
    image = x[:3].max(dim=0).values if x.shape[0] >= 3 else x[0]
    radial = geo[-2]
    edge = geo[-1]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, arr, title, cmap in [
        (axes[0], image.numpy(), "wafer", "gray"),
        (axes[1], radial.numpy(), "radial_distance", "viridis"),
        (axes[2], edge.numpy(), "local_edge_response", "magma"),
    ]:
        ax.imshow(arr, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    path = out_dir / "geo_prior_sample.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def load_human_corrections(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find corrections file: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [{"sample_id": int(k), "corrected_label": int(v)} for k, v in data.items()]
        return [
            {
                "sample_id": int(item["sample_id"]),
                "corrected_label": int(item.get("corrected_label", item.get("label"))),
            }
            for item in data
        ]
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"sample_id": int(row["sample_id"]), "corrected_label": int(row["corrected_label"])})
    return rows


def apply_human_corrections_to_dataset(dataset, corrections_path):
    corrections = load_human_corrections(corrections_path)
    applied = []
    for item in corrections:
        idx = int(item["sample_id"])
        corrected = int(item["corrected_label"])
        if 0 <= idx < len(dataset.y):
            old = int(dataset.y[idx])
            dataset.y[idx] = corrected
            applied.append({"sample_id": idx, "old_label": old, "corrected_label": corrected})
    Path("data").mkdir(exist_ok=True)
    Path("data/corrections_v1.json").write_text(json.dumps(applied, indent=2), encoding="utf-8")
    return applied


def apply_human_review_to_dataset(
    dataset,
    review_csv,
    class_names,
    split_name="train",
    ambiguous_weight=0.6,
    relabel_weight=0.8,
    human_review_filter=None,
    ignore_unlabeled=True,
):
    if str(split_name).lower() != "train":
        print(f"[WARN] Human review rows for split={split_name} are ignored for training.")
        return {"applied": 0, "by_class": {}, "by_decision": {}}
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    rows_by_id = {}
    with open(review_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("split") or "train").strip().lower() != "train":
                continue
            decision = (row.get("human_decision") or "").strip().lower()
            if decision == "" and ignore_unlabeled:
                continue
            if decision not in {"keep", "ambiguous", "relabel", "remove"}:
                raise ValueError(
                    f"Invalid human_decision={decision!r} for sample_id={row.get('sample_id')}"
                )
            if human_review_filter:
                pair = (row.get("confusion_pair") or "").strip()
                original = (row.get("original_label") or "").strip()
                pred = (row.get("w3_pred") or "").strip()
                if pair != human_review_filter and f"{original}_to_{pred}" != human_review_filter:
                    continue
            sample_id = int(row.get("sample_id"))
            rows_by_id[sample_id] = row

    if not rows_by_id:
        return {"applied": 0, "by_class": {}, "by_decision": {}}

    if dataset.sample_weights is None:
        dataset.sample_weights = np.ones(len(dataset.y), dtype=np.float32)
    if dataset.sample_quality is None:
        dataset.sample_quality = np.ones(len(dataset.y), dtype=np.float32)

    keep_mask = np.ones(len(dataset.y), dtype=bool)
    by_class = {name: {"keep": 0, "ambiguous": 0, "relabel": 0, "remove": 0} for name in class_names}
    by_decision = {"keep": 0, "ambiguous": 0, "relabel": 0, "remove": 0}
    relabel_label_distribution = {}
    affected_confusion_pairs = {}
    applied = 0

    for idx, sample_id in enumerate(dataset.sample_ids.tolist()):
        row = rows_by_id.get(int(sample_id))
        if row is None:
            continue
        decision = (row.get("human_decision") or "").strip().lower()
        old_label = int(dataset.y[idx])
        old_name = class_names[old_label]
        pair_name = (row.get("confusion_pair") or "").strip()
        if not pair_name:
            pair_name = f"{row.get('original_label', old_name)}_to_{row.get('w3_pred', '')}"
        affected_confusion_pairs[pair_name] = int(affected_confusion_pairs.get(pair_name, 0)) + 1
        if decision == "keep":
            dataset.sample_weights[idx] = 1.0
        elif decision == "ambiguous":
            dataset.sample_weights[idx] = float(ambiguous_weight)
        elif decision == "relabel":
            human_label = (row.get("human_label") or "").strip()
            if human_label == "":
                raise ValueError(f"sample_id={sample_id} has relabel decision but empty human_label")
            if human_label in label_to_idx:
                new_label = label_to_idx[human_label]
            else:
                try:
                    new_label = int(human_label)
                except ValueError as exc:
                    raise ValueError(f"Invalid human_label={human_label!r} for sample_id={sample_id}") from exc
                if new_label < 0 or new_label >= len(class_names):
                    raise ValueError(f"Invalid human_label index={new_label} for sample_id={sample_id}")
            dataset.y[idx] = new_label
            dataset.sample_weights[idx] = float(relabel_weight)
            relabel_name = class_names[new_label]
            relabel_label_distribution[relabel_name] = int(relabel_label_distribution.get(relabel_name, 0)) + 1
        elif decision == "remove":
            keep_mask[idx] = False
        by_class[old_name][decision] += 1
        by_decision[decision] += 1
        applied += 1

    if not keep_mask.all():
        dataset.x = dataset.x[keep_mask]
        dataset.y = dataset.y[keep_mask]
        dataset.sample_ids = dataset.sample_ids[keep_mask]
        if dataset.geo_features is not None:
            dataset.geo_features = dataset.geo_features[keep_mask]
        dataset.sample_weights = dataset.sample_weights[keep_mask]
        dataset.sample_quality = dataset.sample_quality[keep_mask]

    return {
        "applied": int(applied),
        "by_class": by_class,
        "by_decision": by_decision,
        "relabel_label_distribution": relabel_label_distribution,
        "affected_confusion_pairs": affected_confusion_pairs,
    }


def flatten_human_review_stats(stats, ambiguous_weight, relabel_weight, human_review_filter):
    by_decision = stats.get("by_decision", {})
    return {
        "applied_sample_count": int(stats.get("applied", 0) or 0),
        "keep_count": int(by_decision.get("keep", 0) or 0),
        "ambiguous_count": int(by_decision.get("ambiguous", 0) or 0),
        "relabel_count": int(by_decision.get("relabel", 0) or 0),
        "remove_count": int(by_decision.get("remove", 0) or 0),
        "relabel_label_distribution": stats.get("relabel_label_distribution", {}),
        "affected_confusion_pairs": stats.get("affected_confusion_pairs", {}),
        "ambiguous_weight": float(ambiguous_weight),
        "relabel_weight": float(relabel_weight),
        "filter_used": human_review_filter,
    }


def load_component_matrix(class_names, mapping_path=None):
    if mapping_path is None:
        return np.eye(len(class_names), dtype=np.float32), list(class_names)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    component_names = mapping.get("component_names") or mapping.get("components")
    class_components = mapping.get("class_components") or mapping.get("mapping") or mapping
    if component_names is None:
        seen = []
        for name in class_names:
            values = class_components.get(name, [name])
            if isinstance(values, str):
                values = [values]
            for item in values:
                if item not in seen:
                    seen.append(item)
        component_names = seen
    comp_to_idx = {name: i for i, name in enumerate(component_names)}
    matrix = np.zeros((len(class_names), len(component_names)), dtype=np.float32)
    for class_idx, class_name in enumerate(class_names):
        values = class_components.get(class_name, [class_name])
        if isinstance(values, str):
            values = [values]
        for item in values:
            if item not in comp_to_idx:
                raise ValueError(f"Unknown component {item!r} for class {class_name!r}")
            matrix[class_idx, comp_to_idx[item]] = 1.0
    return matrix, list(component_names)


@torch.no_grad()
def save_unet_mask_visualizations(model, dataset, class_names, out_dir, device, max_per_class=10):
    model.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in class_names}
    mask_sums = []
    entropy_sums = []
    for idx in range(len(dataset)):
        label = int(dataset.y[idx])
        class_name = class_names[label]
        if counts[class_name] >= max_per_class:
            continue
        x = torch.from_numpy(dataset.x[idx]).unsqueeze(0).to(device)
        output = model(x)
        payload = output_aux_payload(output)
        if "masks" not in payload:
            break
        masks = payload["masks"][0].detach().cpu().numpy()
        image = dataset.x[idx].max(axis=0)
        mask_sums.append(float(masks.mean()))
        flat = masks.reshape(3, -1) + 1e-6
        prob = flat / flat.sum(axis=1, keepdims=True)
        entropy = float((-(prob * np.log(prob)).sum(axis=1) / np.log(flat.shape[1])).mean())
        entropy_sums.append(entropy)
        fig, axes = plt.subplots(1, 5, figsize=(12, 3))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("wafer")
        names = ["defect", "edge", "radial"]
        for mi, name in enumerate(names):
            axes[mi + 1].imshow(masks[mi], cmap="magma", vmin=0, vmax=1)
            axes[mi + 1].set_title(name)
        axes[4].imshow(image, cmap="gray")
        axes[4].imshow(masks.max(axis=0), cmap="magma", alpha=0.45, vmin=0, vmax=1)
        axes[4].set_title("overlay")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{class_name} idx={idx}")
        class_dir = out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(class_dir / f"{idx:06d}.png", dpi=160)
        plt.close(fig)
        counts[class_name] += 1
        if all(v >= max_per_class for v in counts.values()):
            break
    return {
        "mask_activation_mean": float(np.mean(mask_sums)) if mask_sums else None,
        "mask_entropy": float(np.mean(entropy_sums)) if entropy_sums else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Optional JSON config. CLI arguments override config values.")
    parser.add_argument("--exp_name", "--exp-name", dest="exp_name", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="outputs/baseline_cnn")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Optional. If None, read from label_map.json.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="Use mixed precision on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--class-weight", action="store_true",
                        help="Use class-weighted CrossEntropyLoss")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal", "qfl"])
    parser.add_argument("--loss-type", type=str, default=None,
                        choices=["ce", "focal", "ce_ls", "focal_ls", "qfl", "qfl_ls"],
                        help="Convenience alias for loss plus label smoothing experiments.")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--qfl-gamma", type=float, default=1.5)
    parser.add_argument("--qfl-quality-source", type=str, default="cleanlab",
                        choices=["cleanlab", "self_confidence", "constant"])
    parser.add_argument("--qfl-quality-min", type=float, default=0.5)
    parser.add_argument("--qfl-quality-max", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="simple",
                        choices=[
                            "simple", "resnet", "seresnet", "dual_seresnet",
                            "dualhead_seresnet", "dual_hybridvit", "dpfee",
                            "dpfee_dual", "dual_dpfee", "vit_tiny_wafer",
                            "cnn_stem_vit_wafer", "dpfee_transformer_tail",
                            "unet_dpfee_hybrid", "two_stage_dpfee",
                        ])
    parser.add_argument("--use-unet-structure-branch", action="store_true", default=False)
    parser.add_argument("--unet-loss-weight", type=float, default=0.05)
    parser.add_argument("--unet-entropy-weight", type=float, default=1.0)
    parser.add_argument("--freeze-dpfee-backbone", action="store_true", default=False)
    parser.add_argument("--dual-head", action="store_true",
                        help="Record dual-head intent; dual-head model names enable it automatically.")
    parser.add_argument("--two-stage-normal-defect", action="store_true", default=False,
                        help="Use shared DPFEE backbone with Normal-vs-Defect binary head and defect-only head.")
    parser.add_argument("--defect-loss-weight", type=float, default=1.0,
                        help="Loss weight for defect_head in --two-stage-normal-defect mode.")
    parser.add_argument("--width", type=int, default=48,
                        help="Base channel width for --model resnet")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--vit-dim", type=int, default=128)
    parser.add_argument("--vit-depth", type=int, default=2)
    parser.add_argument("--vit-heads", type=int, default=4)
    parser.add_argument("--vit-patch-size", type=int, default=8)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--attention", type=str, default="none", choices=["none", "se", "eca", "cbam", "ca"],
                        help="Optional post-fusion attention for DPFEE models.")
    parser.add_argument("--use-edge-branch", action="store_true", default=False,
                        help="Add an edge-aware branch to DPFEE models.")
    parser.add_argument("--edge-branch-type", type=str, default="fixed", choices=["fixed", "learnable"])
    parser.add_argument("--dpfee-feature-map-size", type=int, default=0,
                        help="Optionally adapt DPFEE fused feature map to a fixed H=W before pooling.")
    parser.add_argument("--use_geo_prior", "--use-geo-prior", dest="use_geo_prior",
                        action="store_true", default=False)
    parser.add_argument("--disable_geo_prior", "--disable-geo-prior", dest="use_geo_prior",
                        action="store_false")
    parser.add_argument("--dry_run_geo", "--dry-run-geo", dest="dry_run_geo",
                        action="store_true", default=False)
    parser.add_argument("--train-aug", "--train-augment", dest="train_augment", action="store_true", default=False)
    parser.add_argument("--no-train-aug", "--no-train-augment", dest="train_augment", action="store_false")
    parser.add_argument("--shift-prob", type=float, default=0.15)
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--use-morph-aug", action="store_true", default=False)
    parser.add_argument("--morph-prob", type=float, default=0.1)
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--use-hardclass-aug", action="store_true", default=False)
    parser.add_argument("--hardclass-aug-prob", type=float, default=0.15)
    parser.add_argument("--use-scratch-aug", action="store_true", default=False)
    parser.add_argument("--scratch-aug-prob", type=float, default=0.15)
    parser.add_argument("--class-specific-aug", action="store_true", default=False)
    parser.add_argument("--balanced-sampler", dest="balanced_sampler", action="store_true", default=False)
    parser.add_argument("--no-balanced-sampler", dest="balanced_sampler", action="store_false")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "plateau"])
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--patience", "--early-stop-patience", dest="patience", type=int, default=15,
                        help="Early stopping patience on val_macro_f1. 0 disables early stopping.")
    parser.add_argument("--tta", dest="tta", nargs="?", const="rotate_flip", default=False,
                        choices=[False, "rotate_flip", "class_aware"],
                        help="Use rotate/flip TTA for final test evaluation. Optional modes: rotate_flip/class_aware.")
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--tta-val", action="store_true",
                        help="Also use TTA during every validation epoch (substantially slower).")
    parser.add_argument("--ema", dest="ema", action="store_true", default=False)
    parser.add_argument("--no-ema", dest="ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-warmup-epochs", type=int, default=5,
                        help="Evaluate raw model before this epoch when EMA is enabled.")
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Optional checkpoint to initialize model weights before training.")
    parser.add_argument("--aux-loss-weight", type=float, default=0.4,
                        help="Auxiliary head loss weight for dual-head models.")
    parser.add_argument("--aux-focal-gamma", type=float, default=3.0,
                        help="Focal gamma used by the auxiliary hard-class head.")
    parser.add_argument("--aux-eval-weight", type=float, default=0.0,
                        help="Auxiliary logits weight during validation/test/TTA.")
    parser.add_argument("--hard-class-weight", "--hard-class-boost", dest="hard_class_weight", type=float, default=2.0,
                        help="Auxiliary head class weight for Loc/Scratch/Edge-Loc.")
    parser.add_argument("--hard-classes", nargs="+",
                        default=["Loc", "Scratch", "Edge-Loc", "Donut"])
    parser.add_argument("--loc-loss-boost", type=float, default=0.0,
                        help="Extra multiplicative boost for Loc samples, e.g. 0.5 means 1.5x.")
    parser.add_argument("--use-confusion-margin", action="store_true", default=False)
    parser.add_argument("--confusion-margin", type=float, default=0.2)
    parser.add_argument("--confusion-lambda", type=float, default=0.1)
    parser.add_argument("--use-clean-labels", action="store_true", default=False)
    parser.add_argument("--cleanlab-mask-path", type=str, default=None)
    parser.add_argument(
        "--cleanlab-mode",
        type=str,
        default="remove",
        choices=["remove", "downweight", "relabel", "class_aware_downweight"],
    )
    parser.add_argument("--cleanlab-remove-frac", type=float, default=0.02)
    parser.add_argument("--cleanlab-downweight", type=float, default=0.3)
    parser.add_argument("--cleanlab-weight-normal", type=float, default=0.3)
    parser.add_argument("--cleanlab-weight-confusing", type=float, default=0.5)
    parser.add_argument("--cleanlab-weight-strong", type=float, default=0.1)
    parser.add_argument("--cleanlab_weights", "--cleanlab-weights", nargs=3, type=float, default=None)
    parser.add_argument("--cleanlab-min-keep-per-class", type=int, default=1)
    parser.add_argument("--cleanlab-issues-path", type=str, default=None)
    parser.add_argument("--use-human-review", action="store_true", default=False)
    parser.add_argument("--human-review-csv", type=str, default=None)
    parser.add_argument("--human-review-filter", type=str, default=None)
    parser.add_argument("--ignore-unlabeled-human-review", action="store_true", default=False)
    parser.add_argument("--ambiguous-weight", type=float, default=0.6)
    parser.add_argument("--relabel-weight", type=float, default=0.8)
    parser.add_argument("--use-component-head", action="store_true", default=False)
    parser.add_argument("--component-loss-weight", type=float, default=0.0)
    parser.add_argument("--component-mapping-path", type=str, default=None)
    parser.add_argument("--use-pseudo-mask", action="store_true", default=False)
    parser.add_argument("--pseudo-mask-loss-weight", type=float, default=0.05)
    parser.add_argument("--freeze-dpfee", action="store_true", default=False)
    parser.add_argument("--freeze-unet", action="store_true", default=False)
    parser.add_argument("--use-geometric-features", action="store_true", default=False)
    parser.add_argument("--geo-feature-dim", type=int, default=len(FEATURE_NAMES))
    parser.add_argument("--geo-mlp-hidden", type=int, default=64)
    parser.add_argument("--geo-dropout", type=float, default=0.1)
    parser.add_argument("--use-scratchness-head", action="store_true", default=False)
    parser.add_argument("--scratchness-loss-weight", type=float, default=0.1)
    parser.add_argument("--scratchness-start-epoch", type=int, default=1)
    parser.add_argument("--scratchness-hard-classes", nargs="+", default=["Loc", "Scratch", "Edge-Loc"])
    parser.add_argument("--scratchness-inference", choices=["none"], default="none")
    parser.add_argument("--scratchness-anisotropy-threshold", type=float, default=8.0)
    parser.add_argument("--scratchness-skeleton-threshold", type=float, default=3.0)
    parser.add_argument("--scratchness-aspect-threshold", type=float, default=4.0)
    parser.add_argument("--use-teacher-distillation", action="store_true", default=False)
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--distill-loss-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--use-capsule-head", action="store_true", default=False)
    parser.add_argument("--capsule-hard-classes", nargs="+", default=["Loc", "Scratch", "Edge-Loc", "Edge-Ring"])
    parser.add_argument("--capsule-loss-weight", type=float, default=0.1)
    parser.add_argument("--capsule-routing-iters", type=int, default=3)
    parser.add_argument("--capsule-dim", type=int, default=8)
    parser.add_argument("--capsule-fusion-mode", choices=["none", "logit_add", "rerank_hard"], default="rerank_hard")
    parser.add_argument("--enable_expert_gating", "--enable-expert-gating", action="store_true", default=False)
    parser.add_argument("--disable_expert_gating", "--disable-expert-gating", dest="enable_expert_gating", action="store_false")
    parser.add_argument("--expert_model_path", "--expert-model-path", type=str, default=None)
    parser.add_argument("--expert_threshold", "--expert-threshold", type=float, default=0.85)
    parser.add_argument("--apply_human_corrections", "--apply-human-corrections", type=str, default=None)
    config_args, remaining_argv = parser.parse_known_args()
    if config_args.config:
        config = load_config_file(config_args.config)
        valid_dests = {action.dest for action in parser._actions}
        parser.set_defaults(**{k: v for k, v in config.items() if k in valid_dests})
    args = parser.parse_args()
    if args.exp_name:
        args.out = str(Path("outputs64") / args.exp_name)
    if args.cleanlab_weights is not None:
        args.cleanlab_weight_normal = float(args.cleanlab_weights[0])
        args.cleanlab_weight_confusing = float(args.cleanlab_weights[1])
        args.cleanlab_weight_strong = float(args.cleanlab_weights[2])
    if args.loss_type is not None:
        if args.loss_type.startswith("focal"):
            args.loss = "focal"
        elif args.loss_type.startswith("qfl"):
            args.loss = "qfl"
        else:
            args.loss = "ce"
        if args.loss_type.endswith("_ls") and args.label_smoothing <= 0:
            args.label_smoothing = 0.05
    else:
        args.loss_type = (
            f"{args.loss}_ls" if args.loss in ("ce", "focal", "qfl") and args.label_smoothing > 0
            else args.loss
        )
    if args.dual_head and args.model == "seresnet":
        args.model = "dual_seresnet"
    if args.dual_head and args.model == "dpfee":
        args.model = "dpfee_dual"
    if args.two_stage_normal_defect:
        args.model = "two_stage_dpfee"
    if args.dry_run_geo:
        args.use_geo_prior = True
        args.epochs = 1
        args.patience = 0

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names, num_classes, saved_image_size = load_label_info(data_dir)
    image_size = args.image_size if args.image_size is not None else saved_image_size

    train_path = data_dir / f"wafer_{image_size}_train.npz"
    val_path = data_dir / f"wafer_{image_size}_val.npz"
    test_path = data_dir / f"wafer_{image_size}_test.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Cannot find {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Cannot find {test_path}")

    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Num classes: {num_classes}")
    print(f"[INFO] Image size: {image_size}")
    normal_class_idx = find_normal_class_idx(class_names)
    if normal_class_idx is not None:
        print(f"[INFO] Normal class detected: {class_names[normal_class_idx]} (idx={normal_class_idx})")

    device = resolve_device(args.device)
    use_amp = amp_enabled(device, args.amp)
    device_info = describe_device(device)
    device_info["amp"] = use_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] CUDA version: {torch.version.cuda}")
    if device.type == "cuda":
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        print("[INFO] CUDA device: None")
    print(f"[INFO] AMP: {use_amp}")

    need_geo_features = bool(args.use_geometric_features or args.use_scratchness_head)
    train_set = WaferDataset(
        train_path,
        augment=args.train_augment,
        shift_prob=args.shift_prob,
        max_shift=args.max_shift,
        morph_aug=args.use_morph_aug,
        morph_prob=args.morph_prob,
        morph_kernel=args.morph_kernel,
        use_geo_prior=args.use_geo_prior,
        use_geometric_features=need_geo_features,
        class_names=class_names,
        use_hardclass_aug=args.use_hardclass_aug,
        hardclass_aug_prob=args.hardclass_aug_prob,
        use_scratch_aug=args.use_scratch_aug,
        scratch_aug_prob=args.scratch_aug_prob,
        class_specific_aug=args.class_specific_aug,
        use_pseudo_mask=args.use_pseudo_mask,
    )
    geo_stats = {}
    if need_geo_features:
        geo_mean = train_set.geo_features.mean(axis=0).astype(np.float32)
        geo_std = train_set.geo_features.std(axis=0).astype(np.float32)
        geo_std = np.where(geo_std < 1e-6, 1.0, geo_std).astype(np.float32)
        train_set.set_geo_standardization(geo_mean, geo_std)
        geo_stats = {
            "feature_names": FEATURE_NAMES,
            "mean": geo_mean.astype(float).tolist(),
            "std": geo_std.astype(float).tolist(),
        }
    else:
        geo_mean = None
        geo_std = None
    val_set = WaferDataset(
        val_path,
        use_geo_prior=args.use_geo_prior,
        use_geometric_features=need_geo_features,
        geo_mean=geo_mean,
        geo_std=geo_std,
        class_names=class_names,
        use_pseudo_mask=args.use_pseudo_mask,
    )
    test_set = WaferDataset(
        test_path,
        use_geo_prior=args.use_geo_prior,
        use_geometric_features=need_geo_features,
        geo_mean=geo_mean,
        geo_std=geo_std,
        class_names=class_names,
        use_pseudo_mask=args.use_pseudo_mask,
    )
    args.input_channels = int(train_set.x.shape[1]) + (2 if args.use_geo_prior else 0)
    if args.dry_run_geo:
        geo_path = save_geo_prior_visualization(train_set, "logs/geo_prior_vis")
        print(f"[INFO] Saved geo prior dry-run visualization: {geo_path}")
    cleanlab_removed_or_weighted = 0
    human_corrections_applied = []
    if args.apply_human_corrections:
        human_corrections_applied = apply_human_corrections_to_dataset(train_set, args.apply_human_corrections)
        print(f"[INFO] Applied human corrections in memory: {len(human_corrections_applied)}")
    cleanlab_class_counts_before = np.bincount(train_set.y, minlength=num_classes).astype(int).tolist()
    cleanlab_class_counts_after = cleanlab_class_counts_before
    if args.use_clean_labels:
        if not args.cleanlab_mask_path:
            raise ValueError("--use-clean-labels requires --cleanlab-mask-path")
        keep_mask = np.load(args.cleanlab_mask_path)
        issues_path = args.cleanlab_issues_path
        if issues_path is None:
            candidate = Path(args.cleanlab_mask_path).parent / "cleanlab_label_issues.csv"
            issues_path = str(candidate) if candidate.exists() else None
        hard_class_ids = [
            class_names.index(name)
            for name in args.hard_classes
            if name in class_names
        ]
        cleanlab_removed_or_weighted = train_set.apply_cleanlab_mask(
            keep_mask,
            mode=args.cleanlab_mode,
            issue_weight=args.cleanlab_downweight,
            min_keep_per_class=args.cleanlab_min_keep_per_class,
            issues_path=issues_path,
            hard_class_ids=hard_class_ids,
            weight_normal=args.cleanlab_weight_normal,
            weight_confusing=args.cleanlab_weight_confusing,
            weight_strong=args.cleanlab_weight_strong,
            quality_source=args.qfl_quality_source,
        )
        print(
            f"[INFO] Cleanlab mode={args.cleanlab_mode}, affected={cleanlab_removed_or_weighted}, "
            f"train samples now={len(train_set)}"
        )
        cleanlab_class_counts_after = np.bincount(train_set.y, minlength=num_classes).astype(int).tolist()
    human_review_stats = {"applied": 0, "by_class": {}, "by_decision": {}, "relabel_label_distribution": {}}
    if args.use_human_review:
        if not args.human_review_csv:
            raise ValueError("--use-human-review requires --human-review-csv")
        human_review_stats = apply_human_review_to_dataset(
            train_set,
            args.human_review_csv,
            class_names,
            split_name="train",
            ambiguous_weight=args.ambiguous_weight,
            relabel_weight=args.relabel_weight,
            human_review_filter=args.human_review_filter,
            ignore_unlabeled=args.ignore_unlabeled_human_review,
        )
        print(f"[INFO] Applied human review decisions: {human_review_stats['applied']}")
        print(f"[INFO] Human review by decision: {human_review_stats['by_decision']}")
        cleanlab_class_counts_after = np.bincount(train_set.y, minlength=num_classes).astype(int).tolist()
    human_review_applied_summary = flatten_human_review_stats(
        human_review_stats,
        args.ambiguous_weight,
        args.relabel_weight,
        args.human_review_filter,
    )

    print(f"[INFO] Train samples: {len(train_set)}")
    print(f"[INFO] Val samples:   {len(val_set)}")
    print(f"[INFO] Test samples:  {len(test_set)}")
    print(f"[INFO] Input channels: {args.input_channels}")
    print(f"[INFO] Train class counts: {class_counts_dict(train_set.y, class_names)}")
    print(f"[INFO] Val class counts:   {class_counts_dict(val_set.y, class_names)}")
    print(f"[INFO] Test class counts:  {class_counts_dict(test_set.y, class_names)}")
    capsule_hard_ids = [class_names.index(name) for name in args.capsule_hard_classes if name in class_names]
    scratchness_hard_ids = [class_names.index(name) for name in args.scratchness_hard_classes if name in class_names]
    component_matrix_np, component_names = load_component_matrix(
        class_names,
        args.component_mapping_path if args.use_component_head else None,
    )
    component_matrix = torch.tensor(component_matrix_np, dtype=torch.float32, device=device)

    train_sampler = build_balanced_sampler(train_set, num_classes) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        width=args.width,
        dropout=args.dropout,
        vit_dim=args.vit_dim,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        vit_patch_size=args.vit_patch_size,
        in_channels=args.input_channels,
        attention=args.attention,
        use_edge_branch=args.use_edge_branch,
        edge_branch_type=args.edge_branch_type,
        image_size=image_size,
        use_unet_structure_branch=args.use_unet_structure_branch,
        freeze_dpfee_backbone=args.freeze_dpfee_backbone,
        unet_entropy_weight=args.unet_entropy_weight,
        use_component_head=args.use_component_head,
        component_dim=len(component_names),
        use_geometric_features=args.use_geometric_features,
        geo_feature_dim=args.geo_feature_dim,
        geo_mlp_hidden=args.geo_mlp_hidden,
        geo_dropout=args.geo_dropout,
        use_scratchness_head=args.use_scratchness_head,
        use_capsule_head=args.use_capsule_head,
        capsule_hard_class_count=len(capsule_hard_ids),
        capsule_dim=args.capsule_dim,
        capsule_routing_iters=args.capsule_routing_iters,
        dpfee_feature_map_size=args.dpfee_feature_map_size,
        normal_class_idx=normal_class_idx,
        defect_loss_weight=args.defect_loss_weight,
    ).to(device)
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        init_state = init_ckpt.get("selected_model_state_dict")
        if init_state is None:
            init_state = init_ckpt.get("ema_model_state_dict")
        if init_state is None:
            init_state = init_ckpt["model_state_dict"]
        try:
            model.load_state_dict(init_state)
        except RuntimeError:
            current = model.state_dict()
            remapped = {}
            for key, value in init_state.items():
                candidates = [key]
                if key.startswith("backbone."):
                    candidates.append("dpfee_backbone." + key[len("backbone."):])
                for candidate in candidates:
                    if candidate in current and current[candidate].shape == value.shape:
                        remapped[candidate] = value
                        break
            current.update(remapped)
            model.load_state_dict(current)
            print(f"[INFO] Partially initialized {len(remapped)} tensors from checkpoint.")
        print(f"[INFO] Initialized model from: {args.init_checkpoint}")
    if args.freeze_dpfee and hasattr(model, "freeze_dpfee_backbone"):
        model.freeze_dpfee_backbone()
        print("[INFO] Frozen DPFEE backbone.")
    if args.freeze_unet and hasattr(model, "freeze_unet_branch"):
        model.freeze_unet_branch()
        print("[INFO] Frozen U-Net branch.")
    print(f"[INFO] Model: {args.model}")
    dual_head_enabled = args.model in (
        "dual_seresnet",
        "dualhead_seresnet",
        "dual_hybridvit",
        "dpfee_dual",
        "dual_dpfee",
        "vit_tiny_wafer",
        "cnn_stem_vit_wafer",
        "dpfee_transformer_tail",
        "unet_dpfee_hybrid",
    )
    if args.use_unet_structure_branch:
        dual_head_enabled = True
    print(f"[INFO] Dual head: {dual_head_enabled}")
    print(f"[INFO] Component head: {args.use_component_head}, weight={args.component_loss_weight}")
    print(f"[INFO] Geometric features: {args.use_geometric_features}, dim={args.geo_feature_dim}")
    print(
        f"[INFO] Scratchness head: {args.use_scratchness_head}, "
        f"weight={args.scratchness_loss_weight}, start_epoch={args.scratchness_start_epoch}, "
        f"hard_ids={scratchness_hard_ids}, inference={args.scratchness_inference}"
    )
    print(f"[INFO] Capsule head: {args.use_capsule_head}, hard_ids={capsule_hard_ids}, fusion={args.capsule_fusion_mode}")
    print(f"[INFO] U-Net structure branch: {args.use_unet_structure_branch}, weight={args.unet_loss_weight}")
    print(f"[INFO] Train augment: {args.train_augment}")
    print(
        f"[INFO] Hardclass aug: {args.use_hardclass_aug}, p={args.hardclass_aug_prob}; "
        f"scratch aug: {args.use_scratch_aug}, p={args.scratch_aug_prob}; "
        f"class-specific: {args.class_specific_aug}; morph={args.use_morph_aug}, p={args.morph_prob}"
    )
    print(f"[INFO] Pseudo mask: {args.use_pseudo_mask}, weight={args.pseudo_mask_loss_weight}")
    print(f"[INFO] Balanced sampler: {args.balanced_sampler}")
    print(f"[INFO] Test-time augmentation: {args.tta}")
    print(f"[INFO] EMA: {args.ema}")
    print(f"[INFO] DPFEE attention: {args.attention}")
    print(f"[INFO] Edge branch: {args.use_edge_branch} ({args.edge_branch_type})")
    print(f"[INFO] DPFEE feature map size: {args.dpfee_feature_map_size or 'native'}")

    class_weights_for_loss = None
    class_counts = np.bincount(train_set.y, minlength=num_classes).astype(np.float32)
    if args.class_weight:
        class_weights, class_counts = compute_class_weights(train_set, num_classes)
        print("[INFO] Class counts:")
        for name, count, weight in zip(class_names, class_counts, class_weights):
            print(f"  {name:12s} count={int(count):6d}, weight={weight.item():.4f}")
        class_weights_for_loss = class_weights.to(device)

    criterion = build_criterion(
        loss_name=args.loss,
        class_weights=class_weights_for_loss,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.qfl_gamma if args.loss == "qfl" else args.focal_gamma,
    )
    print(f"[INFO] Loss: {args.loss}")
    if args.loss == "focal":
        print(f"[INFO] Focal gamma: {args.focal_gamma}")
    if args.loss == "qfl":
        print(
            f"[INFO] QFL gamma: {args.qfl_gamma}, "
            f"quality=[{args.qfl_quality_min}, {args.qfl_quality_max}], "
            f"source={args.qfl_quality_source}"
        )
    hard_sample_weights = build_hard_class_weights(
        class_names,
        hard_classes=args.hard_classes,
        hard_weight=args.hard_class_weight,
        device=device,
    )
    loc_class_idx = class_names.index("Loc") if "Loc" in class_names else None
    confusion_pairs = build_confusion_pairs(class_names) if args.use_confusion_margin else []
    if args.use_confusion_margin:
        print(f"[INFO] Confusion margin pairs: {confusion_pairs}")

    expert_model = None
    expert_hard_ids = None
    if args.enable_expert_gating:
        if not args.expert_model_path:
            raise ValueError("--enable_expert_gating requires --expert_model_path")
        expert_model, expert_hard_ids = load_expert_model(
            args.expert_model_path,
            in_channels=args.input_channels,
            device=device,
        )
        print(
            f"[INFO] Expert gating enabled: model={args.expert_model_path}, "
            f"threshold={args.expert_threshold}, hard_ids={expert_hard_ids}"
        )

    teacher_model = None
    if args.use_teacher_distillation:
        if not args.teacher_checkpoint:
            raise ValueError("--use-teacher-distillation requires --teacher-checkpoint")
        teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
        teacher_args = teacher_ckpt.get("args", {})
        teacher_model = build_model(
            model_name=teacher_args.get("model", "dpfee_dual"),
            num_classes=int(teacher_ckpt["num_classes"]),
            width=int(teacher_args.get("width", args.width)),
            dropout=float(teacher_args.get("dropout", args.dropout)),
            vit_dim=int(teacher_args.get("vit_dim", args.vit_dim)),
            vit_depth=int(teacher_args.get("vit_depth", args.vit_depth)),
            vit_heads=int(teacher_args.get("vit_heads", args.vit_heads)),
            vit_patch_size=int(teacher_args.get("vit_patch_size", args.vit_patch_size)),
            in_channels=int(teacher_args.get("input_channels", args.input_channels)),
            attention=teacher_args.get("attention", "none"),
            use_edge_branch=bool(teacher_args.get("use_edge_branch", False)),
            edge_branch_type=teacher_args.get("edge_branch_type", "fixed"),
            image_size=int(teacher_ckpt.get("image_size", image_size)),
            use_component_head=bool(teacher_args.get("use_component_head", False)),
            component_dim=len(component_names),
            use_geometric_features=bool(teacher_args.get("use_geometric_features", False)),
            geo_feature_dim=int(teacher_args.get("geo_feature_dim", len(FEATURE_NAMES))),
            geo_mlp_hidden=int(teacher_args.get("geo_mlp_hidden", 64)),
            geo_dropout=float(teacher_args.get("geo_dropout", 0.1)),
            use_scratchness_head=bool(teacher_args.get("use_scratchness_head", False)),
            use_capsule_head=bool(teacher_args.get("use_capsule_head", False)),
            capsule_hard_class_count=len(capsule_hard_ids),
            capsule_dim=int(teacher_args.get("capsule_dim", 8)),
            capsule_routing_iters=int(teacher_args.get("capsule_routing_iters", 3)),
            dpfee_feature_map_size=int(teacher_args.get("dpfee_feature_map_size", 0)),
        ).to(device)
        teacher_state = teacher_ckpt.get("selected_model_state_dict")
        if teacher_state is None:
            teacher_state = teacher_ckpt.get("ema_model_state_dict")
        if teacher_state is None:
            teacher_state = teacher_ckpt["model_state_dict"]
        teacher_model.load_state_dict(teacher_state)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        print(
            f"[INFO] Teacher distillation enabled: {args.teacher_checkpoint}, "
            f"weight={args.distill_loss_weight}, T={args.distill_temperature}"
        )

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    ema = ModelEma(model, decay=args.ema_decay) if args.ema else None

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.03,
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "mask_loss": [],
        "mask_activation_mean": [],
    }

    best_val_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = out_dir / "best_model.pt"
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warmup_scale = float(epoch) / float(max(args.warmup_epochs, 1))
            for group in optimizer.param_groups:
                group["lr"] = float(args.lr) * warmup_scale
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            ema=ema,
            use_amp=use_amp,
            loss_name=args.loss,
            class_weights=class_weights_for_loss,
            focal_gamma=args.qfl_gamma if args.loss == "qfl" else args.focal_gamma,
            label_smoothing=args.label_smoothing,
            aux_loss_weight=args.aux_loss_weight,
            aux_focal_gamma=args.aux_focal_gamma,
            hard_sample_weights=hard_sample_weights,
            loc_class_idx=loc_class_idx,
            loc_loss_boost=args.loc_loss_boost,
            confusion_pairs=confusion_pairs,
            confusion_margin=args.confusion_margin,
            confusion_lambda=args.confusion_lambda if args.use_confusion_margin else 0.0,
            qfl_quality_min=args.qfl_quality_min,
            qfl_quality_max=args.qfl_quality_max,
            unet_loss_weight=args.unet_loss_weight if args.use_unet_structure_branch else 0.0,
            pseudo_mask_loss_weight=args.pseudo_mask_loss_weight if args.use_pseudo_mask else 0.0,
            component_loss_weight=args.component_loss_weight if args.use_component_head else 0.0,
            component_matrix=component_matrix if args.use_component_head else None,
            scratchness_loss_weight=args.scratchness_loss_weight if args.use_scratchness_head else 0.0,
            scratchness_start_epoch=args.scratchness_start_epoch,
            scratchness_hard_ids=scratchness_hard_ids,
            current_epoch=epoch,
            scratchness_thresholds={
                "anisotropy": args.scratchness_anisotropy_threshold,
                "skeleton": args.scratchness_skeleton_threshold,
                "aspect": args.scratchness_aspect_threshold,
            },
            geo_mean=geo_mean,
            geo_std=geo_std,
            capsule_loss_weight=args.capsule_loss_weight if args.use_capsule_head else 0.0,
            capsule_hard_ids=capsule_hard_ids,
            teacher_model=teacher_model,
            distill_loss_weight=args.distill_loss_weight if args.use_teacher_distillation else 0.0,
            distill_temperature=args.distill_temperature,
            aux_eval_weight=args.aux_eval_weight,
        )
        history["mask_loss"].append(getattr(model, "_last_mask_loss", None))
        history["mask_activation_mean"].append(getattr(model, "_last_mask_activation_mean", None))
        if scheduler is not None and args.scheduler == "cosine" and epoch > args.warmup_epochs:
            scheduler.step()

        use_ema_for_eval = ema is not None and epoch > args.ema_warmup_epochs
        eval_model = ema.module if use_ema_for_eval else model
        val_metrics, _, _ = evaluate(
            eval_model,
            val_loader,
            device,
            num_classes,
            class_names=class_names,
            criterion=criterion,
            use_amp=use_amp,
            use_tta=args.tta_val,
            aux_eval_weight=args.aux_eval_weight,
            expert_model=expert_model,
            expert_hard_ids=expert_hard_ids,
            expert_threshold=args.expert_threshold,
            capsule_hard_ids=capsule_hard_ids,
            capsule_fusion_mode="none",
        )
        if scheduler is not None and args.scheduler == "plateau":
            scheduler.step(val_metrics["macro_f1"])

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
        history["val_weighted_f1"].append(float(val_metrics["weighted_f1"]))

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ema_model_state_dict": ema.module.state_dict() if ema is not None else None,
                    "used_ema_for_selection": use_ema_for_eval,
                    "selected_model_state_dict": eval_model.state_dict(),
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "image_size": image_size,
                    "args": vars(args),
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_epoch": best_epoch,
                    "device_info": device_info,
                },
                best_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}.")
            break

    train_time_sec = time.perf_counter() - start_time
    epochs_ran = len(history["train_loss"])
    avg_epoch_time_sec = train_time_sec / max(epochs_ran, 1)

    print(f"\n[INFO] Best val macro-F1: {best_val_macro_f1:.4f}")
    print(f"[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Training time: {train_time_sec:.2f}s")
    print(f"[INFO] Avg epoch time: {avg_epoch_time_sec:.2f}s")
    print(f"[INFO] Saved best model to: {best_path}")
    best_pth_path = out_dir / "best_model.pth"
    best_pth_path.write_bytes(best_path.read_bytes())
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema.module.state_dict() if ema is not None else None,
            "class_names": class_names,
            "num_classes": num_classes,
            "image_size": image_size,
            "args": vars(args),
            "epoch": int(epochs_ran),
            "device_info": device_info,
        },
        out_dir / "last_model.pth",
    )

    # 载入最佳模型测试
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("selected_model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("ema_model_state_dict")
    if state_dict is None:
        state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)

    test_metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
        num_classes,
        class_names=class_names,
        criterion=criterion,
        use_tta=args.tta,
        use_amp=use_amp,
        aux_eval_weight=args.aux_eval_weight,
        expert_model=expert_model,
        expert_hard_ids=expert_hard_ids,
        expert_threshold=args.expert_threshold,
        capsule_hard_ids=capsule_hard_ids,
        capsule_fusion_mode=args.capsule_fusion_mode if args.use_capsule_head else "none",
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_recall = plot_per_class_recall(
        y_true,
        y_pred,
        class_names,
        out_dir / "per_class_recall.png",
    )
    mask_stats = {}
    if args.use_unet_structure_branch:
        mask_stats = save_unet_mask_visualizations(
            model,
            test_set,
            class_names,
            out_dir / "mask_visualizations",
            device,
            max_per_class=10,
        )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    per_class_metrics = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in class_names
    }

    with open(out_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for name, values in per_class_metrics.items():
            writer.writerow({"class": name, **values})

    with open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(history["train_loss"])):
            writer.writerow({
                "epoch": idx + 1,
                "train_loss": history["train_loss"][idx],
                "val_loss": history["val_loss"][idx],
                "val_accuracy": history["val_accuracy"][idx],
                "val_macro_f1": history["val_macro_f1"][idx],
                "val_weighted_f1": history["val_weighted_f1"][idx],
            })

    print("\n========== Test Metrics ==========")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n========== Classification Report ==========")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    # 保存图和结果
    plot_training_curves(history, out_dir / "training_curves.png")
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    results = {
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_epoch": int(best_epoch),
        "class_names": class_names,
        "normal_class_idx": normal_class_idx,
        "normal_class_name": class_names[normal_class_idx] if normal_class_idx is not None else None,
        "defect_class_names": [class_names[i] for i in defect_class_indices(class_names)],
        "per_class_recall": {
            class_names[i]: float(per_class_recall[i])
            for i in range(len(class_names))
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "history": history,
        "train_time_sec": float(train_time_sec),
        "avg_epoch_time_sec": float(avg_epoch_time_sec),
        "epochs_ran": int(epochs_ran),
        "use_cuda": bool(device_info["use_cuda"]),
        "gpu_name": device_info["gpu_name"],
        "cuda_version": device_info["cuda_version"],
        "amp": bool(use_amp),
        "device": str(device),
        "tta": args.tta,
        "ema_decay": float(args.ema_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "dual_head": bool(dual_head_enabled),
        "two_stage_normal_defect": bool(args.two_stage_normal_defect),
        "defect_loss_weight": float(args.defect_loss_weight),
        "use_component_head": bool(args.use_component_head),
        "component_loss_weight": float(args.component_loss_weight),
        "component_names": component_names,
        "component_mapping_path": args.component_mapping_path,
        "use_geometric_features": bool(args.use_geometric_features),
        "geo_feature_dim": int(args.geo_feature_dim),
        "geo_mlp_hidden": int(args.geo_mlp_hidden),
        "geo_dropout": float(args.geo_dropout),
        "use_scratchness_head": bool(args.use_scratchness_head),
        "scratchness_loss_weight": float(args.scratchness_loss_weight),
        "scratchness_start_epoch": int(args.scratchness_start_epoch),
        "scratchness_hard_classes": args.scratchness_hard_classes,
        "scratchness_inference": args.scratchness_inference,
        "use_teacher_distillation": bool(args.use_teacher_distillation),
        "teacher_checkpoint": args.teacher_checkpoint,
        "distill_loss_weight": float(args.distill_loss_weight),
        "distill_temperature": float(args.distill_temperature),
        "use_capsule_head": bool(args.use_capsule_head),
        "capsule_loss_weight": float(args.capsule_loss_weight),
        "capsule_hard_classes": args.capsule_hard_classes,
        "capsule_hard_ids": capsule_hard_ids,
        "capsule_fusion_mode": args.capsule_fusion_mode,
        "soft_pseudo": False,
        "temperature": None,
        "loc_recall": float(per_class_recall[class_names.index("Loc")]) if "Loc" in class_names else None,
        "scratch_recall": float(per_class_recall[class_names.index("Scratch")]) if "Scratch" in class_names else None,
        "edge_loc_recall": float(per_class_recall[class_names.index("Edge-Loc")]) if "Edge-Loc" in class_names else None,
        "donut_recall": float(per_class_recall[class_names.index("Donut")]) if "Donut" in class_names else None,
        "curriculum": False,
        "loss_type": args.loss_type,
        "qfl_gamma": float(args.qfl_gamma),
        "qfl_quality_source": args.qfl_quality_source,
        "qfl_quality_min": float(args.qfl_quality_min),
        "qfl_quality_max": float(args.qfl_quality_max),
        "use_geo_prior": bool(args.use_geo_prior),
        "use_hardclass_aug": bool(args.use_hardclass_aug),
        "hardclass_aug_prob": float(args.hardclass_aug_prob),
        "use_morph_aug": bool(args.use_morph_aug),
        "morph_prob": float(args.morph_prob),
        "use_scratch_aug": bool(args.use_scratch_aug),
        "scratch_aug_prob": float(args.scratch_aug_prob),
        "class_specific_aug": bool(args.class_specific_aug),
        "use_pseudo_mask": bool(args.use_pseudo_mask),
        "pseudo_mask_loss_weight": float(args.pseudo_mask_loss_weight),
        "freeze_dpfee": bool(args.freeze_dpfee),
        "freeze_unet": bool(args.freeze_unet),
        "dry_run_geo": bool(args.dry_run_geo),
        "enable_expert_gating": bool(args.enable_expert_gating),
        "expert_model_path": args.expert_model_path,
        "expert_threshold": float(args.expert_threshold),
        "gated_replaced": int(test_metrics.get("gated_replaced", 0)),
        "apply_human_corrections": args.apply_human_corrections,
        "human_corrections_count": int(len(human_corrections_applied)),
        "use_human_review": bool(args.use_human_review),
        "human_review_csv": args.human_review_csv,
        "human_review_stats": human_review_stats,
        "human_review_applied_stats": human_review_applied_summary,
        "human_review_filter": args.human_review_filter,
        "ignore_unlabeled_human_review": bool(args.ignore_unlabeled_human_review),
        "ambiguous_weight": float(args.ambiguous_weight),
        "relabel_weight": float(args.relabel_weight),
        "final_train_loss": float(history["train_loss"][-1]) if history["train_loss"] else None,
        "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
        "attention": args.attention,
        "use_edge_branch": bool(args.use_edge_branch),
        "edge_branch_type": args.edge_branch_type,
        "dpfee_feature_map_size": int(args.dpfee_feature_map_size),
        "use_confusion_margin": bool(args.use_confusion_margin),
        "confusion_margin": float(args.confusion_margin),
        "confusion_lambda": float(args.confusion_lambda),
        "use_clean_labels": bool(args.use_clean_labels),
        "cleanlab_mode": args.cleanlab_mode,
        "cleanlab_affected_samples": int(cleanlab_removed_or_weighted),
        "cleanlab_weight_normal": float(args.cleanlab_weight_normal),
        "cleanlab_weight_confusing": float(args.cleanlab_weight_confusing),
        "cleanlab_weight_strong": float(args.cleanlab_weight_strong),
        "use_unet_structure_branch": bool(args.use_unet_structure_branch),
        "unet_loss_weight": float(args.unet_loss_weight),
        "unet_entropy_weight": float(args.unet_entropy_weight),
        "freeze_dpfee_backbone": bool(args.freeze_dpfee_backbone),
        "mask_activation_mean": mask_stats.get("mask_activation_mean"),
        "mask_entropy": mask_stats.get("mask_entropy"),
        "cleanlab_class_counts_before": {
            class_names[i]: int(cleanlab_class_counts_before[i])
            for i in range(len(class_names))
        },
        "cleanlab_class_counts_after": {
            class_names[i]: int(cleanlab_class_counts_after[i])
            for i in range(len(class_names))
        },
        "train_class_counts": class_counts_dict(train_set.y, class_names),
        "val_class_counts": class_counts_dict(val_set.y, class_names),
        "test_class_counts": class_counts_dict(test_set.y, class_names),
        "args": vars(args),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(out_dir / "human_review_applied_stats.json", "w", encoding="utf-8") as f:
        json.dump(human_review_applied_summary, f, indent=2, ensure_ascii=False)
    if need_geo_features:
        with open(out_dir / "geometric_feature_stats.json", "w", encoding="utf-8") as f:
            json.dump(geo_stats, f, indent=2, ensure_ascii=False)
    if args.use_capsule_head:
        capsule_stats = {
            "use_capsule_head": True,
            "capsule_hard_classes": args.capsule_hard_classes,
            "capsule_hard_ids": capsule_hard_ids,
            "capsule_loss_weight": float(args.capsule_loss_weight),
            "capsule_routing_iters": int(args.capsule_routing_iters),
            "capsule_dim": int(args.capsule_dim),
            "capsule_fusion_mode": args.capsule_fusion_mode,
        }
        with open(out_dir / "capsule_stats.json", "w", encoding="utf-8") as f:
            json.dump(capsule_stats, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Saved outputs to:")
    print(f"  {out_dir / 'best_model.pt'}")
    print(f"  {out_dir / 'best_model.pth'}")
    print(f"  {out_dir / 'last_model.pth'}")
    print(f"  {out_dir / 'training_curves.png'}")
    print(f"  {out_dir / 'confusion_matrix.png'}")
    print(f"  {out_dir / 'per_class_recall.png'}")
    print(f"  {out_dir / 'per_class_metrics.csv'}")
    print(f"  {out_dir / 'train_log.csv'}")
    print(f"  {out_dir / 'config.json'}")
    print(f"  {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
