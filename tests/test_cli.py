"""Tests for the dentalcbct CLI. They don't need a GPU or nnU-Net installed."""

import re

import pytest

from dentalcbct import cli

LINE_RE = re.compile(r"^\[(  OK|FAIL)\] \S+")


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "doctor" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert cli.__version__ in capsys.readouterr().out


def test_unknown_command_errors():
    with pytest.raises(SystemExit) as exc:
        cli.main(["nope"])
    assert exc.value.code == 2


def test_doctor_prints_one_line_per_check(capsys):
    code = cli.main(["doctor"])
    out = capsys.readouterr().out
    check_lines = [l for l in out.splitlines() if l.startswith("[")]

    assert check_lines, out
    assert all(LINE_RE.match(l) for l in check_lines), out
    for name in ("python", "torch", "nnunetv2", "dcm2niix", "SimpleITK", "nibabel",
                 "pydicom", "scipy", "matplotlib", *cli.NNUNET_ENV_VARS):
        assert any(f"] {name} " in l for l in check_lines), f"missing {name}:\n{out}"
    failed = sum(l.startswith("[FAIL]") for l in check_lines)
    assert code == (1 if failed else 0)


def test_doctor_all_ok_returns_zero(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_checks", lambda: [cli.Check("a", True, "x")])
    assert cli.main(["doctor"]) == 0
    assert "[  OK] a" in capsys.readouterr().out


def test_doctor_any_fail_returns_one(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "run_checks", lambda: [cli.Check("a", True, "x"), cli.Check("b", False, "y")]
    )
    assert cli.main(["doctor"]) == 1
    assert "1/2 checks passed" in capsys.readouterr().out


def test_env_dir_unset(monkeypatch):
    monkeypatch.delenv("nnUNet_raw", raising=False)
    c = cli.check_env_dir("nnUNet_raw")
    assert not c.ok and "not set" in c.detail


def test_env_dir_missing_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "missing"))
    c = cli.check_env_dir("nnUNet_raw")
    assert not c.ok and "does not exist" in c.detail


def test_env_dir_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("nnUNet_raw", str(tmp_path))
    assert cli.check_env_dir("nnUNet_raw").ok


def test_dcm2niix_missing(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    c = cli.check_dcm2niix()
    assert not c.ok and "not found" in c.detail


def test_missing_package_fails():
    assert not cli.check_package("definitely_not_a_real_pkg_xyz", "definitely-not-real").ok
