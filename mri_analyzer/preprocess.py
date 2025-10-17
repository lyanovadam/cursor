from __future__ import annotations

from typing import Tuple

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize


def resample_isotropic(volume: np.ndarray, spacing: Tuple[float, float, float], target_spacing: float = 1.0) -> tuple[np.ndarray, tuple[float, float, float]]:
    sx, sy, sz = spacing
    zoom_x = sx / target_spacing
    zoom_y = sy / target_spacing
    zoom_z = sz / target_spacing

    new_shape = (
        max(1, int(round(volume.shape[0] * zoom_x))),
        max(1, int(round(volume.shape[1] * zoom_y))),
        max(1, int(round(volume.shape[2] * zoom_z))),
    )

    # order=1 (linear) for images
    resampled = resize(
        volume,
        new_shape,
        order=1,
        mode="edge",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    return resampled, (target_spacing, target_spacing, target_spacing)


def n4_bias_correction(volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    image = sitk.GetImageFromArray(volume.transpose(2, 1, 0))
    image = sitk.Cast(image, sitk.sitkFloat32)

    if mask is not None:
        mask_img = sitk.GetImageFromArray(mask.astype(np.uint8).transpose(2, 1, 0))
    else:
        # default mask: Otsu on whole volume to speed up N4
        otsu = sitk.OtsuThreshold(image, 0, 1)
        mask_img = sitk.Cast(otsu, sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(image, mask_img)

    corrected_np = sitk.GetArrayFromImage(corrected).transpose(2, 1, 0)
    return corrected_np.astype(np.float32)


def zscore_normalize(volume: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    if mask is not None:
        voxels = volume[mask > 0]
    else:
        voxels = volume.reshape(-1)
    mean = float(np.mean(voxels)) if voxels.size else 0.0
    std = float(np.std(voxels)) if voxels.size else 1.0
    std = max(std, eps)
    normalized = (volume - mean) / std
    return normalized.astype(np.float32)
