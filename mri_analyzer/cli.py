from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import SimpleITK as sitk

from .io_utils import ensure_directory, get_image_info, load_image, save_image
from .metrics import compute_metrics, save_metrics_json
from .preprocess import compute_brain_mask, n4_bias_field_correction, normalize_for_segmentation, resample_isotropic
from .segmentation import segment_tissues
from .viz import save_montage, save_segmentation_overlay
from .report import generate_html_report


def analyze(
    input_path: str,
    output_dir: str,
    isotropic_mm: Optional[float] = None,
    bias_correct: bool = True,
) -> int:
    ensure_directory(output_dir)

    print(f"[1/7] Loading image from: {input_path}")
    image = load_image(input_path)

    if isotropic_mm is not None and isotropic_mm > 0:
        print(f"[2/7] Resampling to isotropic spacing: {isotropic_mm} mm")
        image = resample_isotropic(image, isotropic_mm)
    else:
        print("[2/7] Skipping resampling (use --isotropic to enable)")

    print("[3/7] Computing brain mask")
    brain_mask = compute_brain_mask(image)
    save_image(brain_mask, os.path.join(output_dir, "brain_mask.nii.gz"))

    if bias_correct:
        print("[4/7] N4 bias field correction")
        corrected = n4_bias_field_correction(image, brain_mask)
    else:
        print("[4/7] Skipping bias correction")
        corrected = image
    save_image(corrected, os.path.join(output_dir, "corrected.nii.gz"))

    print("[5/7] Intensity normalization and tissue segmentation")
    normalized = normalize_for_segmentation(corrected, brain_mask)
    seg_labels, class_labels, centers = segment_tissues(normalized, brain_mask)
    save_image(seg_labels, os.path.join(output_dir, "segmentation.nii.gz"))

    print("[6/7] Computing metrics")
    metrics = compute_metrics(corrected, brain_mask, seg_labels)
    save_metrics_json(metrics, os.path.join(output_dir, "metrics.json"))

    print("[7/7] Generating visualizations and report")
    save_montage(corrected, os.path.join(output_dir, "montage_raw_axial.png"), orientation="axial", title="Исходные (аксиальные)")
    save_montage(brain_mask, os.path.join(output_dir, "montage_mask_axial.png"), orientation="axial", title="Маска (аксиальные)")
    save_segmentation_overlay(corrected, seg_labels, os.path.join(output_dir, "montage_seg_axial.png"), orientation="axial", title="Сегментация (аксиальные)")

    report_path = generate_html_report(output_dir, metrics)
    print(f"Report written to: {report_path}")

    # Also write a short summary to stdout
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Анализ MRI (DICOM/NIfTI): маска мозга, сегментация, метрики, отчёт.")
    p.add_argument("input", help="Путь к NIfTI файлу (.nii/.nii.gz) или папке с DICOM серией")
    p.add_argument("-o", "--out", default="mri_analysis_output", help="Папка для результатов")
    p.add_argument("--isotropic", type=float, default=None, help="Ресэмплинг к изотропному вокселю (мм)")
    p.add_argument("--no-bias-correct", action="store_true", help="Отключить N4 коррекцию неравномерности поля")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        return analyze(
            input_path=args.input,
            output_dir=args.out,
            isotropic_mm=args.isotropic,
            bias_correct=(not args.no_bias_correct),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
