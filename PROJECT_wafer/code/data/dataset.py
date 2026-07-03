# data/dataset.py
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Minimal compatibility base class for wafer datasets."""

    pass


def _to_gray_wafer(x):
    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got {tuple(x.shape)}")
    if x.shape[0] == 1:
        return x[0].float()
    if x.shape[0] >= 3:
        values = torch.tensor([0.0, 0.5, 1.0], dtype=x.dtype, device=x.device).view(3, 1, 1)
        return (x[:3].float() * values.float()).sum(dim=0)
    return x.mean(dim=0).float()


def add_geometric_priors(wafer_map):
    """Append radial distance and local Sobel edge response channels.

    Input may be [C,H,W] or [N,C,H,W]. For one-hot 3-channel inputs, the edge
    response is computed on a scalar 0/1/2-like gray wafer proxy.
    """
    batched = wafer_map.ndim == 4
    x = wafer_map.float()
    if not batched:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected CHW or NCHW tensor, got {tuple(wafer_map.shape)}")

    n, _, h, w = x.shape
    ys = torch.arange(h, dtype=x.dtype, device=x.device).view(h, 1)
    xs = torch.arange(w, dtype=x.dtype, device=x.device).view(1, w)
    center_y = (h - 1) / 2.0
    center_x = (w - 1) / 2.0
    denom = max(center_x, center_y, 1.0)
    radial = torch.sqrt((ys - center_y) ** 2 + (xs - center_x) ** 2) / denom
    radial = radial.clamp(0.0, 1.0).expand(n, 1, h, w)

    gray = torch.stack([_to_gray_wafer(sample) for sample in x], dim=0).unsqueeze(1)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(gx.pow(2) + gy.pow(2))
    edge = edge / edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)

    out = torch.cat([x, radial, edge], dim=1)
    return out if batched else out[0]
