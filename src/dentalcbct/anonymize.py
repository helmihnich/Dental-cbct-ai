"""Copy a DICOM folder with identifying tags blanked and private tags removed.

Safety rules:
- The input tree is only ever opened for reading.
- The output folder must be new or empty, and must not be inside the input folder.
- Output files get sequential names (000001.dcm, ...), because input file and folder
  names often contain patient names or IDs.
- Nothing about individual files is reported, not even paths, because they may contain
  PHI. pydicom warnings and log messages, which can quote tag values, are suppressed
  while files are processed.
- Pixel data and every non-listed element (geometry, UIDs, dates, ...) are written
  back byte-for-byte as read. Pixel data is never decoded.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import pydicom
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

# Elements blanked (set to an empty value) wherever they occur, including inside sequences.
BLANK_KEYWORDS: frozenset[str] = frozenset(
    {
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "OtherPatientIDs",
        "InstitutionName",
        "InstitutionAddress",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
        "AccessionNumber",
        "StudyID",
    }
)

MEDIA_STORAGE_DIRECTORY_UID = "1.2.840.10008.1.3.10"  # DICOMDIR


class AnonymizeError(Exception):
    """Invalid input/output arguments. Messages never contain file contents."""


@dataclass
class Counts:
    read: int = 0
    written: int = 0
    skipped: int = 0


def _blank_callback(dataset: Dataset, elem: DataElement) -> None:
    if elem.keyword in BLANK_KEYWORDS:
        elem.value = ""


def anonymize_dataset(ds: Dataset) -> None:
    """Blank BLANK_KEYWORDS and remove private tags in place, recursing into sequences."""
    ds.walk(_blank_callback)
    ds.remove_private_tags()


def _read_dicom(path: Path) -> Dataset | None:
    """Return the dataset, or None if the file isn't a usable DICOM image/object."""
    try:
        return pydicom.dcmread(path)
    except InvalidDicomError:
        pass  # No 'DICM' preamble. It could still be a raw (preamble-less) DICOM file.
    except Exception:
        return None
    try:
        ds = pydicom.dcmread(path, force=True)
        # force=True "parses" any file at all. Only accept it if it looks like a real
        # composite instance.
        if "SOPClassUID" in ds and "SOPInstanceUID" in ds:
            return ds
    except Exception:
        pass
    return None


def _is_complete(ds: Dataset) -> bool:
    """Reject partially read files. pydicom does not raise on truncation: a file cut
    after its header comes back as an empty dataset, and one cut inside the pixel data
    comes back with short PixelData. Neither case produces a warning."""
    try:
        if "SOPClassUID" not in ds or "SOPInstanceUID" not in ds:
            return False
        if "Rows" not in ds or "Columns" not in ds:
            return True  # not an image (e.g. structured report). Nothing more to check.
        if "PixelData" not in ds:
            return False
        meta = getattr(ds, "file_meta", None)
        ts = meta.get("TransferSyntaxUID") if meta is not None else None
        if ts is not None and ts.is_compressed:
            return len(ds.PixelData) > 0  # encapsulated: exact size unknown
        bits = (
            int(ds.Rows) * int(ds.Columns) * int(ds.get("NumberOfFrames", 1) or 1)
            * int(ds.get("SamplesPerPixel", 1)) * int(ds.BitsAllocated)
        )
        return len(ds.PixelData) >= (bits + 7) // 8
    except Exception:
        return False


def _is_dicomdir(ds: Dataset) -> bool:
    meta = getattr(ds, "file_meta", None)
    return meta is not None and meta.get("MediaStorageSOPClassUID") == MEDIA_STORAGE_DIRECTORY_UID


def _validate_paths(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    src = input_dir.resolve()
    dst = output_dir.resolve()
    if not src.is_dir():
        raise AnonymizeError("input folder does not exist or is not a folder")
    if dst == src or dst.is_relative_to(src):
        raise AnonymizeError("output folder must not be the input folder or inside it")
    if src.is_relative_to(dst):
        raise AnonymizeError("output folder must not contain the input folder")
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise AnonymizeError("output folder must not exist yet or be empty")
    return src, dst


def anonymize_folder(input_dir: Path, output_dir: Path) -> Counts:
    """Anonymize every DICOM file under input_dir (recursively) into output_dir."""
    src, dst = _validate_paths(Path(input_dir), Path(output_dir))
    files = sorted(p for p in src.rglob("*") if p.is_file())
    counts = Counts()

    pydicom_logger = logging.getLogger("pydicom")
    was_disabled = pydicom_logger.disabled
    pydicom_logger.disabled = True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dst.mkdir(parents=True, exist_ok=True)
            for path in files:
                counts.read += 1
                ds = _read_dicom(path)
                # DICOMDIR is an index of patient records, not image data.
                if ds is None or _is_dicomdir(ds) or not _is_complete(ds):
                    counts.skipped += 1
                    continue
                out = dst / f"{counts.written + 1:06d}.dcm"
                try:
                    anonymize_dataset(ds)
                    # 'xb': never overwrite. enforce_file_format=False writes the file in
                    # the same format it was read (preamble or not, same transfer syntax).
                    with open(out, "xb") as fh:
                        ds.save_as(fh, enforce_file_format=False)
                except Exception:
                    counts.skipped += 1
                    out.unlink(missing_ok=True)  # drop a partial write; name is reused
                    continue
                counts.written += 1
    finally:
        pydicom_logger.disabled = was_disabled
    return counts
