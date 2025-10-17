from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk


def _voxel_volume_ml(spacing_xyz_mm: Tuple[float, float, float]) -> float:
    vx_mm3 = float(spacing_xyz_mm[0] * spacing_xyz_mm[1] * spacing_xyz_mm[2])
    return vx_mm3 / 1000.0


def compute_metrics(image: sitk.Image, brain_mask: sitk.Image, tissue_labels: sitk.Image) -> Dict[str, object]:
    spacing = image.GetSpacing()
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    mask = sitk.GetArrayFromImage(brain_mask).astype(bool)
    labels = sitk.GetArrayFromImage(tissue_labels).astype(np.int16)

    voxel_ml = _voxel_volume_ml(spacing)

    num_voxels_total = int(array.size)
    num_voxels_brain = int(mask.sum())

    brain_volume_ml = float(num_voxels_brain * voxel_ml)

    # Volumes per tissue class
    num_csf = int(((labels == 0) & mask).sum())
    num_gm = int(((labels == 1) & mask).sum())
    num_wm = int(((labels == 2) & mask).sum())

    csf_volume_ml = float(num_csf * voxel_ml)
    gm_volume_ml = float(num_gm * voxel_ml)
    wm_volume_ml = float(num_wm * voxel_ml)

    # SNR estimation (mean brain / robust std of outside-brain background)
    brain_vals = array[mask]
    background_vals = array[~mask]
    if background_vals.size == 0:
        background_vals = array.flatten()
    bg_median = float(np.median(background_vals))
    mad = float(np.median(np.abs(background_vals - bg_median)))
    robust_std = 1.4826 * mad if mad > 0 else float(np.std(background_vals) + 1e-6)
    snr = float(np.mean(brain_vals) / (robust_std + 1e-6)) if brain_vals.size > 0 else 0.0

    metrics = {
        "dimensions_xyz_vox": list(map(int, image.GetSize())),
        "spacing_xyz_mm": [float(s) for s in spacing],
        "voxel_volume_ml": voxel_ml,
        "num_voxels_total": num_voxels_total,
        "num_voxels_brain": num_voxels_brain,
        "brain_volume_ml": brain_volume_ml,
        "csf_volume_ml": csf_volume_ml,
        "gm_volume_ml": gm_volume_ml,
        "wm_volume_ml": wm_volume_ml,
        "csf_fraction": float(csf_volume_ml / brain_volume_ml) if brain_volume_ml > 0 else 0.0,
        "gm_fraction": float(gm_volume_ml / brain_volume_ml) if brain_volume_ml > 0 else 0.0,
        "wm_fraction": float(wm_volume_ml / brain_volume_ml) if brain_volume_ml > 0 else 0.0,
        "snr_robust": snr,
        "mean_brain_intensity": float(np.mean(brain_vals)) if brain_vals.size > 0 else 0.0,
    }

    return metrics


def save_metrics_json(metrics: Dict[str, object], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
