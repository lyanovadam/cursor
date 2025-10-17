from __future__ import annotations

from typing import Optional

import numpy as np
import SimpleITK as sitk


def resample_isotropic(image: sitk.Image, new_spacing_mm: float) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_spacing = (new_spacing_mm, new_spacing_mm, new_spacing_mm)
    new_size = [
        int(round(osz * osp / nsp)) for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(float(sitk.GetArrayFromImage(image).min()) if image.GetPixelID() != sitk.sitkUInt8 else 0)

    return resampler.Execute(image)


def _largest_component(binary: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(binary)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    if stats.GetNumberOfLabels() == 0:
        return binary
    largest_label = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
    largest = sitk.Equal(cc, int(largest_label))
    return sitk.Cast(largest, sitk.sitkUInt8)


def compute_brain_mask(image: sitk.Image) -> sitk.Image:
    """Compute a coarse brain mask using Otsu thresholding and morphology."""
    # Light smoothing to stabilize thresholding
    smoothed = sitk.CurvatureFlow(image1=image, timeStep=0.125, numberOfIterations=5)

    # Otsu thresholding into foreground/background
    otsu = sitk.OtsuThreshold(smoothed, 0, 1)
    otsu = sitk.Cast(otsu, sitk.sitkUInt8)

    # Morphological closing to fill small holes and gaps
    closed = sitk.BinaryMorphologicalClosing(otsu, (2, 2, 2))
    # Keep largest component
    largest = _largest_component(closed)
    # Fill holes in 3D
    filled = sitk.VotingBinaryHoleFilling(largest, radius=[2, 2, 2], majorityThreshold=1)

    return sitk.Cast(filled, sitk.sitkUInt8)


def n4_bias_field_correction(image: sitk.Image, brain_mask: Optional[sitk.Image] = None) -> sitk.Image:
    """Apply N4 bias field correction. If no mask provided, a mask is estimated."""
    if brain_mask is None:
        brain_mask = compute_brain_mask(image)

    # Shrink for speed
    shrink_factor = 2
    image_float = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.Cast(brain_mask, sitk.sitkUInt8)

    shrinker = sitk.ShrinkImageFilter()
    shrinker.SetShrinkFactors([shrink_factor] * 3)
    image_small = shrinker.Execute(image_float)
    mask_small = shrinker.Execute(mask)

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrected_small = n4.Execute(image_small, mask_small)

    # Compute bias field in full res and correct
    log_bias = n4.GetLogBiasFieldAsImage(image_float)
    corrected = image_float / sitk.Exp(log_bias)

    return sitk.Cast(corrected, image.GetPixelID())


def normalize_for_segmentation(image: sitk.Image, brain_mask: sitk.Image) -> sitk.Image:
    """Normalize intensities to [0, 1] within brain region using robust min/max."""
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    mask = sitk.GetArrayFromImage(brain_mask) > 0
    if mask.sum() == 0:
        # Fallback to global min/max
        vmin, vmax = np.percentile(array, [0.5, 99.5])
    else:
        brain_vals = array[mask]
        vmin, vmax = np.percentile(brain_vals, [1.0, 99.0])

    if vmax <= vmin:
        vmax = vmin + 1.0

    array = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    out = sitk.GetImageFromArray(array)
    out.CopyInformation(image)
    return sitk.Cast(out, sitk.sitkFloat32)
