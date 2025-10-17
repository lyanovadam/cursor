## MRI Analysis CLI

A small command-line tool to load brain MRI scans (NIfTI or DICOM), preprocess (resample, N4 bias correction, intensity normalization), skull-strip, compute basic metrics/segment tissues, and produce a JSON summary with PNG slice previews.

### Features
- Load NIfTI (`.nii`, `.nii.gz`) and DICOM series
- Resample to isotropic spacing
- N4 bias field correction (SimpleITK)
- Z-score intensity normalization
- Skull stripping via Otsu + morphological refinement
- Tissue segmentation (KMeans 3 classes: CSF/GM/WM approx.)
- Metrics: volume, mean/std, SNR proxy, tissue volumes
- Report: JSON file + axial/coronal/sagittal PNG slices

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage
```bash
python analyze_mri.py --input /path/to/scan --output reports/case001
```
- If `--input` is a directory, it will attempt to read a DICOM series.
- If it is a file ending with `.nii` or `.nii.gz`, it will load NIfTI.

### Notes
- This is a baseline tool and not for clinical use.
- For DICOM, ensure the directory contains a single coherent series.
