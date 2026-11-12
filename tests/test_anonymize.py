"""Tests for dentalcbct.anonymize, using small synthetic DICOM files only."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from dentalcbct import cli
from dentalcbct.anonymize import (
    BLANK_KEYWORDS,
    AnonymizeError,
    anonymize_folder,
)

# Distinctive fake PHI so leaks are easy to detect anywhere in output/files.
PHI = {
    "PatientName": "Doe^Johnathan",
    "PatientID": "PID-998877",
    "PatientBirthDate": "19700101",
    "PatientAddress": "42 Secret Street",
    "PatientTelephoneNumbers": "555-0199",
    "OtherPatientIDs": "OTHER-ID-123",
    "InstitutionName": "Hidden Dental Clinic",
    "InstitutionAddress": "1 Clinic Road",
    "ReferringPhysicianName": "Ref^Doctor",
    "PerformingPhysicianName": "Perf^Doctor",
    "OperatorsName": "Op^Erator",
    "AccessionNumber": "ACC-55555",
    "StudyID": "STUDY-777",
}
PRIVATE_SECRET = "PRIVATE-SECRET-VALUE"
GEOMETRY = (
    "Rows", "Columns", "PixelSpacing", "SliceThickness", "ImagePositionPatient",
    "ImageOrientationPatient", "BitsAllocated", "BitsStored", "HighBit",
    "PixelRepresentation", "RescaleSlope", "RescaleIntercept",
)
KEPT = ("SOPClassUID", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID",
        "Modality", "StudyDate", "InstanceNumber")


def make_dataset(z: float = 0.0, seed: int = 0) -> Dataset:
    ds = Dataset()
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.1"
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.2"
    ds.Modality = "CT"
    ds.StudyDate = "20260101"
    ds.InstanceNumber = int(z) + 1
    for kw, value in PHI.items():
        setattr(ds, kw, value)

    ds.Rows, ds.Columns = 8, 6
    ds.PixelSpacing = [0.25, 0.3]
    ds.SliceThickness = 0.25
    ds.ImagePositionPatient = [-10.5, 20.25, z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    rng = np.random.default_rng(seed)
    ds.PixelData = rng.integers(-1000, 3000, size=(8, 6), dtype=np.int16).tobytes()

    # Private tags at top level and inside a sequence; PHI nested in a sequence too.
    ds.private_block(0x0011, "ACME CBCT", create=True).add_new(0x01, "LO", PRIVATE_SECRET)
    ds.add_new(0x00091010, "LO", PRIVATE_SECRET)
    item = Dataset()
    item.PatientName = PHI["PatientName"]
    item.PatientID = PHI["PatientID"]
    item.CodeValue = "KEEP-ME"
    item.private_block(0x0013, "ACME NESTED", create=True).add_new(0x02, "LO", PRIVATE_SECRET)
    ds.RequestAttributesSequence = Sequence([item])
    return ds


def write_dicom(path: Path, ds: Dataset, preamble: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if preamble:
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.save_as(path, enforce_file_format=True)
    else:  # raw dataset, no preamble/'DICM'/file meta (seen in some older exports)
        ds.save_as(path, implicit_vr=True, little_endian=True)
    return path


def tree_fingerprint(root: Path) -> dict[str, tuple[str, int]]:
    """Relative path -> (sha256, mtime_ns) for every file under root."""
    return {
        str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def case(tmp_path: Path) -> Path:
    """An input folder shaped like a messy export: PHI in folder names, nested dirs,
    files without extensions, a preamble-less file, a non-DICOM file, a DICOMDIR,
    and a truncated DICOM."""
    root = tmp_path / "in" / "Doe_Johnathan_PID-998877"
    write_dicom(root / "series1" / "IM0001", make_dataset(0.0, seed=1))
    write_dicom(root / "series1" / "IM0002.dcm", make_dataset(0.25, seed=2))
    write_dicom(root / "series1" / "deeper" / "x", make_dataset(0.5, seed=3))
    write_dicom(root / "raw_no_preamble", make_dataset(0.75, seed=4), preamble=False)

    (root / "notes.txt").write_text("Patient Doe^Johnathan PID-998877\n")
    (root / "empty.bin").write_bytes(b"")

    # Fake DICOMDIR: a file whose media storage class is the DICOMDIR UID. Written with
    # enforce_file_format=False, because True rewrites the class from SOPClassUID.
    dicomdir = make_dataset(9.0)
    dicomdir.file_meta = FileMetaDataset()
    dicomdir.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.1.3.10"
    dicomdir.file_meta.MediaStorageSOPInstanceUID = dicomdir.SOPInstanceUID
    dicomdir.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dicomdir.preamble = b"\x00" * 128
    dicomdir.save_as(root / "DICOMDIR", enforce_file_format=False)
    assert pydicom.dcmread(root / "DICOMDIR").file_meta.MediaStorageSOPClassUID == (
        "1.2.840.10008.1.3.10"
    )

    good = (root / "series1" / "IM0001").read_bytes()
    (root / "truncated").write_bytes(good[:200])  # valid preamble, body cut after the header
    (root / "truncated_pixels").write_bytes(good[:-10])  # pixel data cut short
    return root


def outputs(out: Path) -> list[Path]:
    return sorted(out.iterdir())


# --------------------------------------------------------------------------- behaviour


def test_counts_and_skips(case: Path, tmp_path: Path):
    counts = anonymize_folder(case, tmp_path / "out")
    # 4 good DICOMs (incl. preamble-less); skipped: notes.txt, empty.bin, DICOMDIR,
    # truncated, truncated_pixels
    assert (counts.read, counts.written, counts.skipped) == (9, 4, 5)
    assert len(outputs(tmp_path / "out")) == 4


def test_blanks_listed_tags_everywhere(case: Path, tmp_path: Path):
    anonymize_folder(case, tmp_path / "out")
    for f in outputs(tmp_path / "out"):
        ds = pydicom.dcmread(f, force=True)
        for kw in BLANK_KEYWORDS:
            assert kw in ds, f"{kw} should be kept as an empty element"
            assert ds[kw].value in ("", None) or str(ds[kw].value) == "", kw
        nested = ds.RequestAttributesSequence[0]
        assert str(nested.PatientName) == "" and nested.PatientID == ""
        assert nested.CodeValue == "KEEP-ME"


def test_removes_all_private_tags(case: Path, tmp_path: Path):
    anonymize_folder(case, tmp_path / "out")
    for f in outputs(tmp_path / "out"):
        ds = pydicom.dcmread(f, force=True)
        private = []
        ds.walk(lambda _d, e: private.append(e.tag) if e.tag.is_private else None)
        assert private == []


def test_no_phi_bytes_in_any_output_file(case: Path, tmp_path: Path):
    anonymize_folder(case, tmp_path / "out")
    for f in outputs(tmp_path / "out"):
        blob = f.read_bytes()
        for secret in [*PHI.values(), PRIVATE_SECRET]:
            assert secret.encode() not in blob, f"{secret!r} leaked into {f.name}"


def test_geometry_pixels_and_uids_unchanged(case: Path, tmp_path: Path):
    anonymize_folder(case, tmp_path / "out")
    originals = {}
    for p in case.rglob("*"):
        if p.name in {"IM0001", "IM0002.dcm", "x", "raw_no_preamble"}:
            ds = pydicom.dcmread(p, force=True)
            originals[ds.SOPInstanceUID] = ds
    assert len(originals) == 4

    for f in outputs(tmp_path / "out"):
        new = pydicom.dcmread(f, force=True)
        old = originals.pop(new.SOPInstanceUID)
        assert new.PixelData == old.PixelData  # raw bytes, never decoded/re-encoded
        for kw in GEOMETRY + KEPT:
            assert new[kw].value == old[kw].value, kw
        if "TransferSyntaxUID" in getattr(new, "file_meta", {}):  # raw file can't be decoded
            assert np.array_equal(new.pixel_array, old.pixel_array)
    assert originals == {}


def test_preserves_file_format(case: Path, tmp_path: Path):
    anonymize_folder(case, tmp_path / "out")
    preambles = sorted(f.read_bytes()[128:132] == b"DICM" for f in outputs(tmp_path / "out"))
    assert preambles == [False, True, True, True]  # the raw file stays raw


def test_input_folder_is_never_modified(case: Path, tmp_path: Path):
    before = tree_fingerprint(case)
    anonymize_folder(case, tmp_path / "out")
    assert tree_fingerprint(case) == before


def test_output_names_do_not_leak_input_names(case: Path, tmp_path: Path):
    anonymize_folder(case, tmp_path / "out")
    names = [f.name for f in outputs(tmp_path / "out")]
    assert names == ["000001.dcm", "000002.dcm", "000003.dcm", "000004.dcm"]


# --------------------------------------------------------------------------- path safety


def test_rejects_output_inside_input(case: Path):
    before = tree_fingerprint(case)
    with pytest.raises(AnonymizeError):
        anonymize_folder(case, case / "anon")
    with pytest.raises(AnonymizeError):
        anonymize_folder(case, case)
    assert not (case / "anon").exists()
    assert tree_fingerprint(case) == before


def test_rejects_output_containing_input(case: Path):
    with pytest.raises(AnonymizeError):
        anonymize_folder(case, case.parent)


def test_rejects_non_empty_output(case: Path, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing.dcm").write_bytes(b"keep")
    with pytest.raises(AnonymizeError):
        anonymize_folder(case, out)
    assert (out / "existing.dcm").read_bytes() == b"keep"


def test_rejects_missing_input(tmp_path: Path):
    with pytest.raises(AnonymizeError):
        anonymize_folder(tmp_path / "nope", tmp_path / "out")


def test_empty_existing_output_is_ok(case: Path, tmp_path: Path):
    (tmp_path / "out").mkdir()
    assert anonymize_folder(case, tmp_path / "out").written == 4


# --------------------------------------------------------------------------- CLI


def test_cli_prints_only_counts(case: Path, tmp_path: Path, capfd):
    code = cli.main(["anonymize", "--input", str(case), "--output", str(tmp_path / "out")])
    out, err = capfd.readouterr()  # capfd also catches anything written at fd level
    assert code == 0
    assert out.splitlines() == ["files read:    9", "files written: 4", "files skipped: 5"]
    assert err == ""
    for secret in [*PHI.values(), PRIVATE_SECRET, "Doe_Johnathan", "IM0001"]:
        assert secret not in out + err


def test_cli_bad_paths_exit_2_without_leaking(case: Path, capfd):
    code = cli.main(["anonymize", "--input", str(case), "--output", str(case / "o")])
    out, err = capfd.readouterr()
    assert code == 2 and out == ""
    assert "Doe_Johnathan" not in err and "error:" in err


def test_cli_no_dicom_found_exits_1(tmp_path: Path, capfd):
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.txt").write_text("hello")
    code = cli.main(["anonymize", "--input", str(tmp_path / "in"), "--output", str(tmp_path / "o")])
    assert code == 1
    assert "files written: 0" in capfd.readouterr().out


def test_cli_requires_both_args():
    with pytest.raises(SystemExit) as exc:
        cli.main(["anonymize", "--input", "x"])
    assert exc.value.code == 2


@pytest.mark.skipif(os.name != "nt", reason="read-only attribute semantics checked on Windows")
def test_works_when_input_files_are_read_only(case: Path, tmp_path: Path):
    import stat

    files = [p for p in case.rglob("*") if p.is_file()]
    for p in files:
        p.chmod(stat.S_IREAD)
    try:
        assert anonymize_folder(case, tmp_path / "out").written == 4
    finally:
        for p in files:
            p.chmod(stat.S_IREAD | stat.S_IWRITE)
