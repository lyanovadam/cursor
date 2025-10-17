from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class VolumeMetrics:
    voxel_volume_mm3: float
    total_brain_volume_mm3: float
    mean_intensity: float
    std_intensity: float
    snr_proxy: float
    tissue_volumes_mm3: Dict[str, float]


def compute_voxel_volume(spacing: Tuple[float, float, float]) -> float:
    sx, sy, sz = spacing
    return float(sx * sy * sz)


def segment_tissues(volume: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    # KMeans with 3 clusters as rough CSF/GM/WM
    voxels = volume[brain_mask > 0].reshape(-1, 1)
    if voxels.size < 3:
        return np.zeros_like(volume, dtype=np.uint8)
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=0)
    labels = kmeans.fit_predict(voxels)

    # Map clusters by mean intensity order: lowest->CSF(1), mid->GM(2), high->WM(3)
    centers = kmeans.cluster_centers_.reshape(-1)
    order = np.argsort(centers)
    mapping = {order[0]: 1, order[1]: 2, order[2]: 3}

    seg = np.zeros_like(volume, dtype=np.uint8)
    seg[brain_mask > 0] = np.vectorize(lambda x: mapping[int(x)])(labels)
    return seg


def compute_metrics(volume: np.ndarray, brain_mask: np.ndarray, spacing: Tuple[float, float, float]) -> VolumeMetrics:
    voxel_vol = compute_voxel_volume(spacing)
    brain_voxels = int(np.count_nonzero(brain_mask))
    total_brain_volume = brain_voxels * voxel_vol

    masked = volume[brain_mask > 0]
    mean_intensity = float(np.mean(masked)) if masked.size else 0.0
    std_intensity = float(np.std(masked)) if masked.size else 0.0
    snr_proxy = float(mean_intensity / (std_intensity + 1e-6))

    seg = segment_tissues(volume, brain_mask)
    tissue_volumes = {
        "csf_mm3": float(np.sum(seg == 1) * voxel_vol),
        "gm_mm3": float(np.sum(seg == 2) * voxel_vol),
        "wm_mm3": float(np.sum(seg == 3) * voxel_vol),
    }

    return VolumeMetrics(
        voxel_volume_mm3=voxel_vol,
        total_brain_volume_mm3=total_brain_volume,
        mean_intensity=mean_intensity,
        std_intensity=std_intensity,
        snr_proxy=snr_proxy,
        tissue_volumes_mm3=tissue_volumes,
    )
