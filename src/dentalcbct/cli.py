"""dentalcbct command-line interface."""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dentalcbct import __version__

MIN_PYTHON = (3, 11)
NNUNET_ENV_VARS = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
# (import name, distribution name on PyPI)
LIBRARIES = (
    ("SimpleITK", "SimpleITK"),
    ("nibabel", "nibabel"),
    ("pydicom", "pydicom"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        return f"[{'OK' if self.ok else 'FAIL':>4}] {self.name:<20} {self.detail}"


# --------------------------------------------------------------------------- checks


def check_python() -> Check:
    ok = sys.version_info[:2] >= MIN_PYTHON
    detail = f"{platform.python_version()} ({sys.executable})"
    if not ok:
        detail += f" — need >= {'.'.join(map(str, MIN_PYTHON))}"
    return Check("python", ok, detail)


def check_torch() -> list[Check]:
    try:
        torch = import_module("torch")
    except Exception as exc:  # ImportError, or a broken CUDA/DLL load
        return [Check("torch", False, f"not importable: {exc}")]

    checks = [Check("torch", True, f"{torch.__version__} (built for CUDA {torch.version.cuda})")]
    if not torch.cuda.is_available():
        checks.append(Check("cuda", False, "torch.cuda.is_available() is False"))
        return checks

    checks.append(Check("cuda", True, f"available, {torch.cuda.device_count()} device(s)"))
    try:
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        checks.append(Check("gpu", True, name))
        checks.append(
            Check("vram", True, f"{free / 2**30:.2f} GiB free / {total / 2**30:.2f} GiB total")
        )
    except Exception as exc:
        checks.append(Check("gpu", False, f"could not query device 0: {exc}"))
    return checks


def _dist_version(dist: str) -> str | None:
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def check_package(import_name: str, dist_name: str) -> Check:
    try:
        import_module(import_name)
    except Exception as exc:
        return Check(import_name, False, f"not importable: {exc}")
    return Check(import_name, True, _dist_version(dist_name) or "installed (version unknown)")


def check_dcm2niix() -> Check:
    exe = shutil.which("dcm2niix")
    if exe is None:
        return Check("dcm2niix", False, "not found on PATH")
    try:
        # dcm2niix exits non-zero (3) even for -v, so parse output and ignore the code.
        proc = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check("dcm2niix", False, f"{exe} failed to run: {exc}")
    match = re.search(r"v\d+\.\d+\.\d+", proc.stdout + proc.stderr)
    if match is None:
        return Check("dcm2niix", False, f"{exe} ran but printed no version")
    return Check("dcm2niix", True, f"{match.group(0)} ({exe})")


def check_env_dir(var: str) -> Check:
    value = os.environ.get(var)
    if not value:
        return Check(var, False, "environment variable not set")
    if Path(value).is_dir():
        return Check(var, True, f"{value} (exists)")
    return Check(var, False, f"{value} (folder does not exist)")


def run_checks() -> list[Check]:
    checks = [check_python(), *check_torch()]
    checks.append(check_package("nnunetv2", "nnunetv2"))
    checks.append(check_dcm2niix())
    checks.extend(check_package(imp, dist) for imp, dist in LIBRARIES)
    checks.extend(check_env_dir(var) for var in NNUNET_ENV_VARS)
    return checks


# --------------------------------------------------------------------------- commands


def cmd_doctor(_args: argparse.Namespace) -> int:
    checks = run_checks()
    for c in checks:
        print(c.line())
    failed = sum(not c.ok for c in checks)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dentalcbct",
        description="CBCT dental pipeline: DICOM -> NIfTI -> nnU-Net -> findings report",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    doctor = sub.add_parser("doctor", help="check the environment (GPU, packages, nnU-Net paths)")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
