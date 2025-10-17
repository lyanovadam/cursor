from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import SimpleITK as sitk


@dataclass
class ImageInfo:
    size_xyz: Tuple[int, int, int]
    spacing_xyz_mm: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    direction: Tuple[float, ...]


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_dicom_series(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(path)
        return bool(series_ids)
    except Exception:
        return False


def load_image(input_path: str) -> sitk.Image:
    """Load a 3D volume from a NIfTI file or a DICOM series directory.

    Returns a SimpleITK Image in LPS coordinate convention (native to ITK/SimpleITK).
    """
    if os.path.isdir(input_path):
        # DICOM series
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(input_path)
        if not series_ids:
            raise ValueError(f"No DICOM series found in directory: {input_path}")
        # Heuristic: pick the first series
        file_names = reader.GetGDCMSeriesFileNames(input_path, series_ids[0])
        reader.SetFileNames(file_names)
        image = reader.Execute()
        return image

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Assume image file readable by SITK (NIfTI, MHA, NRRD, etc.)
    try:
        image = sitk.ReadImage(input_path)
        return image
    except Exception as exc:
        raise RuntimeError(f"Failed to read image from {input_path}: {exc}") from exc


def save_image(image: sitk.Image, output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        ensure_directory(parent)
    sitk.WriteImage(image, output_path)


def get_image_info(image: sitk.Image) -> ImageInfo:
    size = tuple(int(s) for s in image.GetSize())
    spacing = tuple(float(s) for s in image.GetSpacing())
    origin = tuple(float(s) for s in image.GetOrigin())
    direction = tuple(float(d) for d in image.GetDirection())
    return ImageInfo(size_xyz=size, spacing_xyz_mm=spacing, origin_xyz=origin, direction=direction)


def image_to_numpy(image: sitk.Image) -> np.ndarray:
    """Return a numpy array in z, y, x order (SimpleITK default)."""
    return sitk.GetArrayFromImage(image)


def numpy_to_image(array_zyx: np.ndarray, reference_image: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array_zyx)
    image.CopyInformation(reference_image)
    return image
