from __future__ import annotations

import os
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


def _choose_slices(num_slices: int, total: int) -> Iterable[int]:
    if total <= num_slices:
        return list(range(total))
    step = total / (num_slices + 1)
    return [int(round(step * (i + 1))) for i in range(num_slices)]


def _normalize_for_display(array: np.ndarray) -> np.ndarray:
    vmin, vmax = np.percentile(array, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1.0
    array = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    return array


def save_montage(image: sitk.Image, out_path: str, orientation: str = "axial", num_slices: int = 12, title: str = "") -> None:
    array = sitk.GetArrayFromImage(image).astype(np.float32)  # z, y, x
    array = _normalize_for_display(array)

    if orientation == "axial":  # z slices
        slices = _choose_slices(num_slices, array.shape[0])
        planes = [array[z, :, :] for z in slices]
    elif orientation == "coronal":  # y slices
        slices = _choose_slices(num_slices, array.shape[1])
        planes = [array[:, y, :] for y in slices]
    elif orientation == "sagittal":  # x slices
        slices = _choose_slices(num_slices, array.shape[2])
        planes = [array[:, :, x] for x in slices]
    else:
        raise ValueError("orientation must be one of: axial, coronal, sagittal")

    cols = 4
    rows = int(np.ceil(len(planes) / cols))

    plt.figure(figsize=(cols * 3, rows * 3))
    for i, plane in enumerate(planes, 1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(np.flipud(plane), cmap="gray")
        ax.axis("off")
    if title:
        plt.suptitle(title)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_segmentation_overlay(
    base_image: sitk.Image,
    seg_labels: sitk.Image,
    out_path: str,
    orientation: str = "axial",
    num_slices: int = 12,
    title: str = "Segmentation"
) -> None:
    base = sitk.GetArrayFromImage(base_image).astype(np.float32)
    base = _normalize_for_display(base)
    seg = sitk.GetArrayFromImage(seg_labels).astype(np.int16)

    if orientation == "axial":
        indices = _choose_slices(num_slices, base.shape[0])
        base_planes = [base[z, :, :] for z in indices]
        seg_planes = [seg[z, :, :] for z in indices]
    elif orientation == "coronal":
        indices = _choose_slices(num_slices, base.shape[1])
        base_planes = [base[:, y, :] for y in indices]
        seg_planes = [seg[:, y, :] for y in indices]
    elif orientation == "sagittal":
        indices = _choose_slices(num_slices, base.shape[2])
        base_planes = [base[:, :, x] for x in indices]
        seg_planes = [seg[:, :, x] for x in indices]
    else:
        raise ValueError("orientation must be one of: axial, coronal, sagittal")

    cols = 4
    rows = int(np.ceil(len(base_planes) / cols))

    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(cols * 3, rows * 3))
    for i, (b, s) in enumerate(zip(base_planes, seg_planes), 1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(np.flipud(b), cmap="gray")
        # Overlay CSF(0)=blue, GM(1)=orange, WM(2)=green
        for label_idx, alpha, color in [(0, 0.35, cmap(0)), (1, 0.35, cmap(1)), (2, 0.35, cmap(2))]:
            mask = (s == label_idx)
            if np.any(mask):
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[..., :3] = color[:3]
                overlay[..., 3] = (mask.astype(np.float32)) * alpha
                ax.imshow(np.flipud(overlay))
        ax.axis("off")
    if title:
        plt.suptitle(title)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()
