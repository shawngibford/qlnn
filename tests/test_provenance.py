"""Tests for the provenance-recording helper used by training scripts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from quantum_liquid_neuralode.utils import write_provenance
from quantum_liquid_neuralode.utils.provenance import _PACKAGES


REPO_ROOT = Path(__file__).resolve().parents[1]


_REQUIRED_TOP_LEVEL_KEYS = {
    "git_commit",
    "git_dirty",
    "git_branch",
    "data_sha256",
    "data_path",
    "data_size_bytes",
    "python_version",
    "platform",
    "package_versions",
    "wall_clock_start_utc",
}


def test_write_provenance_creates_valid_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "fake_data.csv"
    csv_path.write_text("a,b,c\n1,2,3\n")

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    payload = write_provenance(output_dir, csv_path, REPO_ROOT)

    prov_path = output_dir / "provenance.json"
    assert prov_path.exists()

    loaded = json.loads(prov_path.read_text())
    assert loaded == payload

    # All required top-level keys are present.
    assert _REQUIRED_TOP_LEVEL_KEYS.issubset(loaded.keys())

    # Every declared package appears in the package_versions dict.
    versions = loaded["package_versions"]
    assert isinstance(versions, dict)
    for pkg in _PACKAGES:
        assert pkg in versions
        assert isinstance(versions[pkg], str)

    # Data hash is a 64-char sha256 hex string (file existed).
    assert isinstance(loaded["data_sha256"], str)
    assert len(loaded["data_sha256"]) == 64
    assert loaded["data_size_bytes"] == len(csv_path.read_bytes())

    # Wall clock is an ISO-ish string ending in Z.
    assert isinstance(loaded["wall_clock_start_utc"], str)
    assert loaded["wall_clock_start_utc"].endswith("Z")


def test_write_provenance_handles_missing_csv(tmp_path: Path) -> None:
    """Missing CSV must NOT crash a training run — fall back to 'unknown'."""
    csv_path = tmp_path / "does_not_exist.csv"
    assert not csv_path.exists()

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    payload = write_provenance(output_dir, csv_path, REPO_ROOT)

    assert payload["data_sha256"] == "unknown"
    assert payload["data_size_bytes"] == "unknown"
    # File still got written.
    assert (output_dir / "provenance.json").exists()


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=False)
        return True
    except (FileNotFoundError, OSError):
        return False


@pytest.mark.skipif(not _git_available(), reason="git not installed in test env")
def test_write_provenance_handles_no_git(tmp_path: Path) -> None:
    """A non-git working directory must yield git_commit='unknown', not crash."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x\n0\n")

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    # tmp_path is NOT a git repo.
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()

    payload = write_provenance(output_dir, csv_path, not_a_repo)

    assert payload["git_commit"] == "unknown"
    # branch and dirty should also degrade gracefully (not raise).
    assert payload["git_branch"] == "unknown"
    assert payload["git_dirty"] in ("unknown", True, False)
