# Documentation

Pipeline stages:

1. `convert` — DICOM series → NIfTI (`.nii.gz`)
2. `segment` — nnU-Net v2 inference for teeth and mandibular canal
3. `report` — heuristic findings derived from segmentation masks
