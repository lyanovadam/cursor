from __future__ import annotations

import json
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from .metrics import VolumeMetrics


def _save_slice_png(volume: np.ndarray, axis: int, out_path: str) -> None:
    idx = volume.shape[axis] // 2
    if axis == 0:
        sl = volume[idx, :, :]
    elif axis == 1:
        sl = volume[:, idx, :]
    else:
        sl = volume[:, :, idx]
    plt.figure(figsize=(5, 5))
    plt.imshow(np.rot90(sl), cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_report(
    output_dir: str,
    volume: np.ndarray,
    brain_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    metrics: VolumeMetrics,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Save slices with correct anatomical planes for (x, y, z) arrays
    # axial: slice along z (axis=2), coronal: along y (axis=1), sagittal: along x (axis=0)
    _save_slice_png(volume, 2, os.path.join(output_dir, "axial.png"))
    _save_slice_png(volume, 1, os.path.join(output_dir, "coronal.png"))
    _save_slice_png(volume, 0, os.path.join(output_dir, "sagittal.png"))

    # Save mask slices for quick QC
    _save_slice_png(brain_mask.astype(np.float32), 2, os.path.join(output_dir, "mask_axial.png"))
    _save_slice_png(brain_mask.astype(np.float32), 1, os.path.join(output_dir, "mask_coronal.png"))
    _save_slice_png(brain_mask.astype(np.float32), 0, os.path.join(output_dir, "mask_sagittal.png"))

    # Save JSON
    report = {
        "spacing_mm": {
            "x": spacing[0],
            "y": spacing[1],
            "z": spacing[2],
        },
        "voxel_volume_mm3": metrics.voxel_volume_mm3,
        "total_brain_volume_mm3": metrics.total_brain_volume_mm3,
        "mean_intensity": metrics.mean_intensity,
        "std_intensity": metrics.std_intensity,
        "snr_proxy": metrics.snr_proxy,
        "tissue_volumes_mm3": metrics.tissue_volumes_mm3,
    }

    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
