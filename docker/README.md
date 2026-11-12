# Docker (GPU)

Base image: `pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime`, with dcm2niix, nnunetv2 2.8.1,
SimpleITK, nibabel, pydicom, scipy, matplotlib, and this package (`pip install -e .`).

Requirements: Docker Desktop with the WSL2 backend and an NVIDIA driver that supports CUDA 12.6
or newer.

## Build

Run from the project root, in PowerShell:

```powershell
docker build -t "dental-cbct-ai:latest" .
```

No patient data or model weights go into the image. `.dockerignore` keeps them out of the
build context, and they are mounted at run time instead.

## Mounts

| Container path | Host path               | Mode       | Purpose                                   |
|----------------|-------------------------|------------|-------------------------------------------|
| `/data`        | patient DICOM folder    | read-only  | input scans                               |
| `/models`      | `.\models`              | read-write | `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` |
| `/work`        | `.\workdir`             | read-write | NIfTI conversions, masks, reports         |

Inside the container, these environment variables are set:
`nnUNet_raw=/models/nnUNet_raw`, `nnUNet_preprocessed=/models/nnUNet_preprocessed`,
`nnUNet_results=/models/nnUNet_results`.

## Run

Quote every Windows path, because the project path contains spaces:

```powershell
$Project = "C:\Users\hnich\Desktop\New folder\dental-cbct-ai"
$Patient = "D:\CBCT\patient_001"   # your DICOM folder

New-Item -ItemType Directory -Force "$Project\models", "$Project\workdir" | Out-Null

docker run --rm -it --gpus all --shm-size=2g `
  -v "${Patient}:/data:ro" `
  -v "${Project}\models:/models" `
  -v "${Project}\workdir:/work" `
  "dental-cbct-ai:latest"
```

Note: write `${Patient}` with braces in `"${Patient}:/data:ro"`. Without them, PowerShell
reads `$Patient:` as a scope-qualified variable name.

Example commands inside the container:

```bash
dcm2niix -z y -o /work /data                  # DICOM -> NIfTI
nnUNetv2_predict -i /work/in -o /work/seg -d <DATASET_ID> -c 3d_fullres -f 0
```

## Verify the GPU

```powershell
docker run --rm --gpus all "dental-cbct-ai:latest" `
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True NVIDIA GeForce RTX 3050 6GB Laptop GPU`.

`--shm-size=2g` is needed because nnU-Net's data loaders use shared memory, and Docker's
default of 64 MB causes crashes. On a 6 GB GPU, run inference with the
`nnUNet_def_n_proc` / `-npp` / `-nps` values kept low.
