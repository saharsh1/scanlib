"""The two public scan laws shipped with the package.

``scanlib.scanlaw`` reads the GOST forecasts at the Pleiades centre and at
ecliptic latitude +60 deg, both over the DR4 span, from the package data.
The scan angles, the epochs and the along-scan parallax factors must equal
the arrays scanlib-dev exported (``scanlaw/*`` of the reference cases) to
the last bit; the four reference scenes use the first 40 transits of the
Pleiades law. Each file ships with its README, both are declared as
package data, and the CSVs are scanlib-dev's files byte for byte (their
SHA-256 below).
"""
from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

import numpy as np
import pytest

from scanlib.scanlaw import BETA60, PLEIADES, load_parallax_factor, load_scan_law

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "tests" / "reference" / "reference_cases.npz"
LAWS = {"pleiades": (PLEIADES, 47), "beta60": (BETA60, 95)}
SHA256 = {"gost_pleiades_centre_dr4span.csv":
          "1a2b3c1cfa92646db374ed898a3f2108f47fb804fcea47a5b0f7f40c7645eba8",
          "gost_ecl_lon060_lat60_dr4span.csv":
          "76fb4dfd4fa4a39ed098d57978b99c712eab1fb9d0df38717a715e6c506e4aca"}


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


@pytest.mark.parametrize("name", LAWS)
def test_the_loaders_return_the_exported_arrays(ref, name):
    path, n = LAWS[name]
    psi, tau = load_scan_law(path)
    fpar = load_parallax_factor(path)
    assert psi.shape == tau.shape == fpar.shape == (n,)
    assert np.array_equal(psi, ref[f"scanlaw/{name}/psi"])
    assert np.array_equal(tau, ref[f"scanlaw/{name}/tau"])
    assert np.array_equal(fpar, ref[f"scanlaw/{name}/fpar"])


def test_the_reference_scenes_use_the_first_40_pleiades_transits(ref):
    psi, _ = load_scan_law(PLEIADES)
    assert np.array_equal(ref["static_pair/inputs/psi"], psi[:40])


def test_the_epochs_are_years_from_the_laws_mean_epoch():
    for path, _ in LAWS.values():
        _, tau = load_scan_law(path)
        assert abs(tau.mean()) < 1e-12
        assert 5.0 < tau.max() - tau.min() < 5.5           # within the DR4 span of 5.5 yr


@pytest.mark.parametrize("name", LAWS)
def test_each_law_is_package_data_with_its_readme(name):
    path, _ = LAWS[name]
    assert path.is_file() and (path.parent / "README.md").is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256[path.name]
    globs = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"]["setuptools"]["package-data"]["scanlib"]
    package = REPO / "src" / "scanlib"
    shipped = {p for g in globs for p in package.glob(g)}
    assert path in shipped and path.parent / "README.md" in shipped
