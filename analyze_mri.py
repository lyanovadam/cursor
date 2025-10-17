from __future__ import annotations

import os
import sys
import json
import click
import numpy as np

from mri_analyzer.io_utils import load_image
from mri_analyzer.preprocess import resample_isotropic, n4_bias_correction, zscore_normalize
from mri_analyzer.skull_strip import skull_strip
from mri_analyzer.metrics import compute_metrics
from mri_analyzer.report import save_report


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Path to NIfTI file or DICOM directory")
@click.option("--output", "output_dir", required=True, type=click.Path(), help="Output directory for report and images")
@click.option("--spacing", "target_spacing", default=1.0, type=float, show_default=True, help="Target isotropic spacing in mm")
@click.option("--no-n4", is_flag=True, help="Disable N4 bias field correction")
@click.option("--no-skull-strip", is_flag=True, help="Disable skull stripping (uses whole volume)")
@click.option("--just-report", is_flag=True, help="Skip preprocessing and skull stripping, just load and report")
def main(input_path: str, output_dir: str, target_spacing: float, no_n4: bool, no_skull_strip: bool, just_report: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)

    volume, spacing = load_image(input_path)

    if not just_report:
        # Resample to isotropic spacing
        volume, spacing = resample_isotropic(volume, spacing, target_spacing=target_spacing)

        # N4 bias correction
        if not no_n4:
            volume = n4_bias_correction(volume)

        # Skull stripping
        if not no_skull_strip:
            volume, brain_mask = skull_strip(volume)
        else:
            brain_mask = np.ones_like(volume, dtype=np.uint8)

        # Intensity normalization (z-score on brain voxels)
        volume = zscore_normalize(volume, brain_mask)
    else:
        brain_mask = np.ones_like(volume, dtype=np.uint8)

    metrics = compute_metrics(volume, brain_mask, spacing)

    save_report(output_dir, volume, brain_mask, spacing, metrics)

    # Also print summary to stdout
    summary = {
        "spacing_mm": spacing,
        "voxel_volume_mm3": metrics.voxel_volume_mm3,
        "total_brain_volume_mm3": metrics.total_brain_volume_mm3,
        "mean_intensity": metrics.mean_intensity,
        "std_intensity": metrics.std_intensity,
        "snr_proxy": metrics.snr_proxy,
        "tissue_volumes_mm3": metrics.tissue_volumes_mm3,
    }
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
