from collections import deque

import numpy as np


FEATURE_NAMES = [
    "defect_area",
    "connected_components",
    "largest_component_area",
    "bbox_width",
    "bbox_height",
    "bbox_aspect_ratio",
    "skeleton_length",
    "skeleton_norm",
    "pca_lambda1",
    "pca_lambda2",
    "anisotropy",
    "major_axis_angle",
    "eccentricity",
    "edge_band_ratio",
    "mean_radius",
    "max_radius",
    "solidity",
    "hough_line_count",
]


def _to_discrete_map(wafer_map):
    arr = np.asarray(wafer_map)
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            return np.argmax(arr, axis=0).astype(np.float32)
        return arr[0].astype(np.float32)
    return arr.astype(np.float32)


def _component_sizes(mask):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    sizes = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            size = 0
            while q:
                cy, cx = q.popleft()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            sizes.append(size)
    return sizes


def _boundary_length(mask):
    if mask.sum() == 0:
        return 0.0
    padded = np.pad(mask.astype(np.float32), 1)
    center = padded[1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    boundary = (center > 0) & (neighbors < 4)
    return float(boundary.sum())


def extract_geometric_features(wafer_map):
    eps = 1e-6
    discrete = _to_discrete_map(wafer_map)
    mask = discrete > 0.5
    h, w = mask.shape
    area = float(mask.sum())
    if area <= 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    ys, xs = np.where(mask)
    sizes = _component_sizes(mask)
    connected_components = float(len(sizes))
    largest_component_area = float(max(sizes) if sizes else 0.0)

    bbox_width = float(xs.max() - xs.min() + 1)
    bbox_height = float(ys.max() - ys.min() + 1)
    bbox_aspect_ratio = max(bbox_width, bbox_height) / (min(bbox_width, bbox_height) + eps)

    boundary_len = _boundary_length(mask)
    skeleton_length = boundary_len * 0.5
    skeleton_norm = skeleton_length / np.sqrt(area + eps)

    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    centered = coords - coords.mean(axis=0, keepdims=True)
    if len(coords) > 1:
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        lambda1, lambda2 = float(eigvals[0]), float(eigvals[1])
        angle = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]) / np.pi)
    else:
        lambda1 = lambda2 = angle = 0.0
    anisotropy = lambda1 / (lambda2 + eps)
    eccentricity = np.sqrt(max(0.0, 1.0 - lambda2 / (lambda1 + eps))) if lambda1 > 0 else 0.0

    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_possible_radius = max(float(radius.max()), eps)
    defect_radius = radius[mask] / max_possible_radius
    mean_radius = float(defect_radius.mean())
    max_radius = float(defect_radius.max())
    edge_band = radius >= 0.78 * max_possible_radius
    edge_band_ratio = float((mask & edge_band).sum() / (area + eps))

    bbox_area = bbox_width * bbox_height
    solidity = area / (bbox_area + eps)

    # Kept as a placeholder feature; expensive Hough voting is intentionally omitted.
    hough_line_count = 0.0

    values = np.array(
        [
            area,
            connected_components,
            largest_component_area,
            bbox_width,
            bbox_height,
            bbox_aspect_ratio,
            skeleton_length,
            skeleton_norm,
            lambda1,
            lambda2,
            anisotropy,
            angle,
            eccentricity,
            edge_band_ratio,
            mean_radius,
            max_radius,
            solidity,
            hough_line_count,
        ],
        dtype=np.float32,
    )
    values[~np.isfinite(values)] = 0.0
    return values.astype(np.float32)
