"""Tests for the P6 G4 dataset-hash gate.

`assert_dataset_hash(name)` must:

  1. accept on-disk PDE fields whose recomputed `_provenance_hash` still
     matches the SHA256 recorded in `manifest.json`,
  2. raise `DatasetHashMismatchError` when the .npz has drifted by even
     one value,
  3. warn-and-return when the requested system name is not in the
     manifest (allow new systems being onboarded before they are locked).

The tests build a self-contained miniature PDE artifact + manifest in
`tmp_path` so they do not touch `data/pde/`.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from quantum_liquid_neuralode.data_processing.pde_systems import (
    DatasetHashMismatchError,
    _provenance_hash,
    assert_dataset_hash,
)


def _write_fake_pde(tmp_path, name="fake_pde", *, perturb=False):
    """Emit a tiny pretend PDE npz + matching manifest entry in tmp_path.

    Returns (manifest_path, pde_dir, npz_path)."""
    pde_dir = tmp_path / "pde"
    pde_dir.mkdir()

    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    x = np.linspace(0.0, 2.0 * np.pi, 16, dtype=np.float64)
    U = rng.standard_normal((t.size, x.size)).astype(np.float64)
    params = {"nu": 0.1, "ic": "sin(x)"}

    sha = _provenance_hash(name, t, x, U, params)
    meta = {
        "name": name,
        "equation": "fake",
        "regime": "smooth_periodic",
        "params": params,
        "sha256": sha,
    }

    # Optionally perturb the saved field so its recomputed hash differs
    # from the manifest's recorded sha. The manifest still carries the
    # pre-perturbation sha — that is the whole point of the tamper test.
    U_saved = U.copy()
    if perturb:
        U_saved[0, 0] += 1e-9

    npz_path = pde_dir / f"{name}.npz"
    np.savez_compressed(
        npz_path,
        u=U_saved,
        x=x,
        t=t,
        u0=U_saved[0],
        meta_json=json.dumps(meta),
        invariants_json=json.dumps({}),
    )

    manifest_path = pde_dir / "manifest.json"
    manifest_path.write_text(json.dumps({name: meta}, indent=2))
    return manifest_path, pde_dir, npz_path


def test_happy_path_matches(tmp_path):
    mp, pd, _ = _write_fake_pde(tmp_path)
    # No raise, no warning on a clean match.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert_dataset_hash("fake_pde", manifest_path=mp, pde_dir=pd)


def test_tamper_raises_mismatch(tmp_path):
    mp, pd, npz = _write_fake_pde(tmp_path, perturb=True)
    with pytest.raises(DatasetHashMismatchError) as exc:
        assert_dataset_hash("fake_pde", manifest_path=mp, pde_dir=pd)
    msg = str(exc.value)
    assert "fake_pde" in msg
    assert str(npz) in msg
    assert "expected" in msg and "actual" in msg


def test_unknown_name_warns_and_returns(tmp_path):
    mp, pd, _ = _write_fake_pde(tmp_path)
    with pytest.warns(UserWarning, match="not in"):
        assert_dataset_hash("not_in_manifest",
                            manifest_path=mp, pde_dir=pd)


def test_real_manifest_systems_match():
    """All four real PDE artifacts under data/pde/ must currently match
    their manifest entries. This catches accidental drift in the
    committed fields and ensures the gate would be a no-op on the
    canonical repo state.

    Skip-when-absent: the `data/` symlink is per-worktree (gitignored)
    and may legitimately be missing in fresh clones / CI runs that
    don't need it. We skip the assertion rather than fail in that
    case — the test fires when the symlink is present.
    """
    import pytest as _pytest
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[1]
    manifest = repo_root / "data" / "pde" / "manifest.json"
    if not manifest.exists():
        _pytest.skip("data/pde/manifest.json absent; gate-on-presence "
                     "(typical in worktrees without the data symlink)")
    for name in ("burgers_smooth", "burgers_shock", "allen_cahn", "kdv"):
        assert_dataset_hash(name)
