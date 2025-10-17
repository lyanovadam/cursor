from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import nibabel as nib
try:
    import SimpleITK as sitk  # type: ignore
except Exception:  # pragma: no cover
    sitk = None  # lazy-checked in functions


def is_nifti(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def load_nifti(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = nib.load(path)
    data = np.asanyarray(img.get_fdata(dtype=np.float32))
    header = img.header
    zooms = header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return data, spacing


def load_dicom_series(directory: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if sitk is None:
        raise ImportError("SimpleITK is required for DICOM reading. Please install simpleitk.")
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)
    if not series_ids:
        raise ValueError(f"No DICOM series found in {directory}")
    if len(series_ids) > 1:
        # Choose the first series deterministically; users can pre-filter directory
        series_id = series_ids[0]
    else:
        series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(directory, series_id)
    reader.SetFileNames(file_names)
    image = reader.Execute()
    spacing = tuple(float(s) for s in image.GetSpacing())  # (sx, sy, sz)
    array = sitk.GetArrayFromImage(image)  # (z, y, x)
    # Convert to (x, y, z) with spacing mapping accordingly
    array = np.transpose(array, (2, 1, 0))
    return array.astype(np.float32, copy=False), (spacing[0], spacing[1], spacing[2])


def load_image(input_path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if os.path.isdir(input_path):
        return load_dicom_series(input_path)
    if is_nifti(input_path):
        return load_nifti(input_path)
    raise ValueError("Unsupported input. Provide a NIfTI file or a DICOM directory.")
