from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import ball, binary_closing, binary_opening, remove_small_objects
from scipy.ndimage import binary_fill_holes


def skull_strip(volume: np.ndarray, min_size_voxels: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    data = volume.astype(np.float32)
    # global Otsu thresholding as a simple baseline
    t = threshold_otsu(data)
    mask = data > t

    # morphological cleanup
    mask = remove_small_objects(mask, min_size=min_size_voxels)
    mask = binary_fill_holes(mask)

    selem = ball(2)
    mask = binary_opening(mask, footprint=selem)
    mask = binary_closing(mask, footprint=selem)

    stripped = data * mask.astype(np.float32)
    return stripped, mask.astype(np.uint8)
