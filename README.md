# dental-cbct-ai

CBCT dental imaging pipeline:

1. **DICOM → NIfTI** conversion
2. **nnU-Net** segmentation of teeth and the mandibular canal
3. **Heuristic findings report** generated from the segmentation masks

> ⚠️ Research use only — not a medical device. Outputs must be reviewed by a qualified clinician.

## Layout

```
src/dentalcbct/   package source
tests/            pytest tests
configs/          pipeline configuration (YAML)
docs/             documentation
```

## Setup (Windows / PowerShell)

```powershell
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -e ".[dev]"          # core + test tools
pip install -e ".[seg]"          # adds nnunetv2 (install a matching PyTorch build first)
```

## Data policy

Patient data and model weights are **never** committed. `.gitignore` excludes
`*.dcm`, `*.nii`, `*.nii.gz`, `*.xfm`, `data/`, `workdir/`, `models/`, `nnUNet_*` folders,
`*.zip`, `*.pth`, and `.venv/`. Keep inputs in `data/` and intermediates in `workdir/`.
