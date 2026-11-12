# dental-cbct-ai — GPU image
# Base verified via `docker pull` on 2026-11-12. torch 2.12.1 satisfies nnunetv2 2.8.1's
# requirement (torch>=2.1.2, !=2.9.*). CUDA 12.6 supports the RTX 3050 (sm_86).
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# dcm2niix: DICOM -> NIfTI conversion. python3.12-venv: provides ensurepip for the venv below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends dcm2niix python3.12-venv \
 && rm -rf /var/lib/apt/lists/*

# The base image installs torch into the Ubuntu 24.04 system Python, which is marked
# EXTERNALLY-MANAGED (PEP 668), so pip won't install there. A venv with
# --system-site-packages can still import the base image's CUDA torch, and pip can
# install into it.
RUN python -m venv --system-site-packages /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Pin torch to the version in the base image so no dependency can swap it for
# a CPU or different-CUDA build.
RUN python -c "import torch; print('torch==' + torch.__version__)" > /tmp/constraints.txt

# Third-party deps first, in their own layer, so source edits don't trigger a reinstall
RUN pip install -c /tmp/constraints.txt \
        nnunetv2==2.8.1 \
        SimpleITK \
        nibabel \
        pydicom \
        scipy \
        matplotlib \
        pyyaml

# nnU-Net v2 folders. All live under /models, which is mounted at run time.
ENV nnUNet_raw=/models/nnUNet_raw \
    nnUNet_preprocessed=/models/nnUNet_preprocessed \
    nnUNet_results=/models/nnUNet_results

WORKDIR /app
# Project source only. Patient data and weights are excluded by .dockerignore and
# are never copied into the image; they're mounted at /data, /models, /work.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install -c /tmp/constraints.txt -e .

VOLUME ["/data", "/models", "/work"]
CMD ["bash"]
