from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk


def _kmeans_1d(values: np.ndarray, k: int = 3, max_iter: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Simple 1D k-means returning (labels, centers) with centers sorted ascending."""
    if values.ndim != 1:
        values = values.reshape(-1)
    values = values.astype(np.float32)

    # Initialize centers at percentiles
    percentiles = np.linspace(5, 95, k)
    centers = np.percentile(values, percentiles)

    for _ in range(max_iter):
        # Assign
        distances = np.abs(values[:, None] - centers[None, :])
        labels = distances.argmin(axis=1)
        new_centers = np.array([
            values[labels == i].mean() if np.any(labels == i) else centers[i] for i in range(k)
        ], dtype=np.float32)
        shift = np.abs(new_centers - centers).max()
        centers = new_centers
        if shift < tol:
            break

    # Ensure ascending order and relabel
    order = centers.argsort()
    centers = centers[order]
    relabel_map = {old: new for new, old in enumerate(order)}
    labels = np.vectorize(relabel_map.get)(labels)

    return labels, centers


def segment_tissues(normalized_image: sitk.Image, brain_mask: sitk.Image) -> Tuple[sitk.Image, Dict[str, int], np.ndarray]:
    """Segment CSF/GM/WM via 1D k-means on normalized intensities within brain mask.

    Returns:
      - label_image (0: CSF, 1: GM, 2: WM, -1 outside brain)
      - class_labels mapping
      - class_centers intensities
    """
    array = sitk.GetArrayFromImage(normalized_image).astype(np.float32)
    mask = sitk.GetArrayFromImage(brain_mask) > 0

    flat_vals = array[mask]
    if flat_vals.size < 100:
        # Not enough brain voxels; fallback to binary mask
        seg = -np.ones_like(array, dtype=np.int16)
        seg[mask] = 1
        label_img = sitk.GetImageFromArray(seg)
        label_img.CopyInformation(normalized_image)
        return label_img, {"csf": 0, "gm": 1, "wm": 2}, np.array([0.2, 0.5, 0.8])

    labels, centers = _kmeans_1d(flat_vals, k=3)

    seg = -np.ones_like(array, dtype=np.int16)
    seg_vals = np.zeros(flat_vals.shape[0], dtype=np.int16)
    seg_vals[:] = labels
    seg[mask] = seg_vals

    # Map ascending centers to CSF, GM, WM
    class_labels = {"csf": 0, "gm": 1, "wm": 2}

    label_img = sitk.GetImageFromArray(seg)
    label_img.CopyInformation(normalized_image)
    return label_img, class_labels, centers
