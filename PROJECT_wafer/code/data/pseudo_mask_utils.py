# -*- coding: utf-8 -*-

import numpy as np


def _to_label_map(wafer_map):
    arr = np.asarray(wafer_map)
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            return arr.argmax(axis=0).astype(np.int64)
        if arr.shape[-1] == 3:
            return arr.argmax(axis=-1).astype(np.int64)
        if arr.shape[0] == 1:
            arr = arr[0]
    if arr.max() <= 1.0:
        return np.rint(arr * 2.0).astype(np.int64).clip(0, 2)
    return np.rint(arr).astype(np.int64).clip(0, 2)


def _neighbors(mask):
    padded = np.pad(mask, 1, mode="constant")
    count = np.zeros_like(mask, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            count += padded[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
    return count


def _simple_skeleton(mask, iterations=24):
    mask = mask.astype(bool)
    skel = np.zeros_like(mask, dtype=bool)
    current = mask.copy()
    for _ in range(iterations):
        if not current.any():
            break
        n = _neighbors(current.astype(np.uint8))
        boundary = current & (n < 8)
        interior = current & ~boundary
        endpoints = current & (n <= 1)
        skel |= endpoints
        if not interior.any():
            skel |= current
            break
        current = interior
    return skel


def generate_pseudo_masks(wafer_map, output_size=64):
    labels = _to_label_map(wafer_map)
    if labels.shape != (output_size, output_size):
        raise ValueError(f"Expected {output_size}x{output_size}, got {labels.shape}")

    defect = labels > 0
    h, w = labels.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius_norm = radius / max(cx, cy)

    edge_band = radius_norm >= 0.78
    ring_band = (radius_norm >= 0.32) & (radius_norm <= 0.72)
    center_band = radius_norm <= 0.34

    skeleton = _simple_skeleton(defect)
    neighbor_count = _neighbors(defect.astype(np.uint8))
    thin = defect & (neighbor_count <= 3)

    # Sobel-like response on the discrete map; keep only responses on defects.
    gray = labels.astype(np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    edge_response = np.sqrt(gx * gx + gy * gy)
    linear = defect & (edge_response > 0)

    scratch_like = skeleton | thin | linear
    masks = np.stack(
        [
            defect,
            defect & edge_band,
            scratch_like,
            defect & (center_band | ring_band),
        ],
        axis=0,
    ).astype(np.float32)
    return masks

