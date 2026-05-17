"""Provenance / reproducibility hygiene helper.

Writes a `provenance.json` snapshot into a run's output directory so reviewers
can map any artifact back to:
  - the exact git commit that produced it (+ whether the worktree was dirty)
  - the exact data file by content hash
  - the Python / platform / package-version environment
  - a UTC wall-clock start time

Designed to be defensive: a failure to look up a single package version or to
run `git` must NEVER crash a 30-minute training run. Missing values fall back
to ``"unknown"``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Packages whose versions we record. Keep this list aligned with both the
# classical (torch / torchdiffeq) and quantum (jax / equinox / diffrax / optax /
# pennylane) stacks so a single provenance.json is meaningful for either.
_PACKAGES: tuple[str, ...] = (
    "torch",
    "torchdiffeq",
    "jax",
    "jaxlib",
    "equinox",
    "diffrax",
    "optax",
    "pennylane",
    "pandas",
    "scikit-learn",
)

# scikit-learn is imported as `sklearn`; everything else uses its PyPI name.
_IMPORT_ALIASES: dict[str, str] = {"scikit-learn": "sklearn"}


def _run_git(args: list[str], cwd: Path) -> str:
    """Run `git <args>` in *cwd*; return stripped stdout or ``"unknown"`` on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _git_commit(cwd: Path) -> str:
    return _run_git(["rev-parse", "HEAD"], cwd)


def _git_branch(cwd: Path) -> str:
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)


def _git_dirty(cwd: Path) -> bool | str:
    """Return True/False for dirtiness, or ``"unknown"`` if git couldn't be queried."""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    return bool(proc.stdout.strip())


def _package_version(name: str) -> str:
    """Return ``module.__version__`` for *name*, or ``"unknown"`` on any failure."""
    import_name = _IMPORT_ALIASES.get(name, name)
    try:
        module = importlib.import_module(import_name)
    except Exception:
        return "unknown"
    version = getattr(module, "__version__", None)
    if version is None:
        return "unknown"
    return str(version)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_provenance(output_dir: Path, csv_path: Path, repo_root: Path) -> dict[str, Any]:
    """Write ``<output_dir>/provenance.json`` and return the recorded dict.

    Parameters
    ----------
    output_dir
        Run-artifact directory; must already exist (caller's responsibility).
    csv_path
        Absolute path to the dataset CSV. If the file is missing, ``data_sha256``
        and ``data_size_bytes`` fall back to ``"unknown"`` rather than raising
        — provenance recording must not block training.
    repo_root
        Repo root used as ``cwd`` for ``git`` invocations. Falling back to
        ``"unknown"`` is acceptable if this isn't a git checkout.

    Returns
    -------
    dict
        The exact JSON payload written to disk (handy for tests / debugging).
    """
    output_dir = Path(output_dir)
    csv_path = Path(csv_path)
    repo_root = Path(repo_root)

    if csv_path.exists() and csv_path.is_file():
        data_sha = _hash_file(csv_path)
        data_size = int(csv_path.stat().st_size)
    else:
        data_sha = "unknown"
        data_size = "unknown"

    package_versions = {name: _package_version(name) for name in _PACKAGES}

    payload: dict[str, Any] = {
        "git_commit": _git_commit(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "git_branch": _git_branch(repo_root),
        "data_sha256": data_sha,
        "data_path": str(csv_path),
        "data_size_bytes": data_size,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "wall_clock_start_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


__all__ = ["write_provenance"]
