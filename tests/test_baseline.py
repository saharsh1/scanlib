"""The package's own byte-identity baseline.

A small synthetic scene is rendered and run through every algorithm, and
a handful of scalars drawn from the outputs are compared EXACTLY, with
``==``, against ``tests/_baseline.py``: a baseline is a record of what the
code did, and a tolerance would hide the drift it exists to catch. The
reference-case test compares against the stored reference cases with a
tolerance; this one pins this package to itself.

Exact equality is guaranteed only where the libraries round as they did
at the bless: ``ENVIRONMENT`` in ``_baseline.py`` records numpy, scipy and
the platform. The comparison is exact everywhere; a difference fails where
the baseline was recorded, and elsewhere is skipped with a reason that
names the value and both environments, ``tests/test_reference_cases.py``
guarding the numbers there with its tolerance.

A deliberate numeric change re-blesses the baseline in its own commit
and says what moved::

    python tests/test_baseline.py --bless
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy

from scanlib import (Instrument, backprojection_map, faststack,
                     filtered_backprojection, fit_joint,
                     joint_fisher, loglike_profiled, loglike_windows, map_peak,
                     scene_positions_al, simulate_window)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import _baseline  # noqa: E402
except ModuleNotFoundError:          # first bless: no baseline yet
    _baseline = None
BASELINE = getattr(_baseline, "BASELINE", {})
ENVIRONMENT = getattr(_baseline, "ENVIRONMENT", {})

INST, SIGMA = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18), 10.0


def environment() -> dict[str, str]:
    """The libraries and the platform whose rounding the baseline records."""
    return {"numpy": np.__version__, "scipy": scipy.__version__,
            "platform": f"{platform.system()}-{platform.machine()}"}


def skip_reason(recorded: dict[str, str], here: dict[str, str]) -> str | None:
    """``None`` where the baseline was recorded; elsewhere, why a difference is not a failure."""
    if recorded == here:
        return None
    name = lambda env: ", ".join(f"{k} {v}" for k, v in sorted(env.items()))  # noqa: E731
    return (f"the baseline was recorded with {name(recorded)}; this environment has "
            f"{name(here)}. Library builds round differently, so exact equality is "
            "guaranteed only there; tests/test_reference_cases.py guards the numbers "
            "here with a tolerance.")


def _scene():
    rng = np.random.default_rng(20261029)
    psi = np.mod(np.linspace(0.0, 3.0 * np.pi, 24) + rng.uniform(-0.1, 0.1, 24), 2 * np.pi)
    tau = np.linspace(-2.5, 2.5, 24)
    fpar = 0.7 * np.sin(2 * np.pi * tau + 0.4)
    theta = np.array([110.0, 70.0, 4.0, -3.0, -5.0])
    flux = np.array([1000.0, 300.0])
    eta = scene_positions_al(theta, psi, tau, fpar)
    windows = np.array([simulate_window(e, flux, INST, rng, SIGMA) for e in eta])
    det_eta = eta[:, 1] + rng.normal(0.0, 3.0, psi.size)
    return psi, tau, fpar, theta, flux, eta, windows, det_eta


def compute() -> dict[str, float]:
    psi, tau, fpar, theta, flux, eta, windows, det_eta = _scene()
    out = {}
    out["design_columns"] = float(INST.design_columns(eta[:3]).sum())
    sky, _, _ = backprojection_map(psi, det_eta, np.full(psi.size, 3.0), 200.0, 2.0)
    out["backprojection_map"] = float(sky.max())
    peak, width, _, _ = map_peak(psi, det_eta, np.full(psi.size, 3.0), 200.0, 1.0)
    out["map_peak"] = float(peak[0] + 1e3 * peak[1] + 1e6 * width)
    image, _ = filtered_backprojection(psi, windows, INST, 200.0, 4.0)
    out["filtered_backprojection"] = float(image.sum())
    out["filtered_backprojection_squares"] = float(np.square(image).sum())   # the sum sees no ramp-filter change
    stack, variance, n_used, _ = faststack(psi, windows, INST, 200.0, 4.0)
    out["faststack"] = float(np.nansum(stack) + np.nansum(variance) + n_used.sum())
    out["loglike_windows"] = loglike_windows(theta, flux, windows, psi, INST,
                                             SIGMA, tau, fpar)
    out["loglike_profiled"] = loglike_profiled(theta, windows, psi, INST,
                                               SIGMA, tau, fpar)
    _, f = loglike_profiled(theta, windows, psi, INST, SIGMA, tau, fpar, return_fluxes=True)
    out["loglike_profiled_fluxes"] = float(f.sum())         # ln L is guarded above
    out["joint_fisher"] = float(joint_fisher(theta, flux, psi, INST, SIGMA,
                                             tau, fpar).sum())
    fit = fit_joint(windows, psi, theta + [5.0, -5.0, 0.0, 0.0, 0.0], INST, SIGMA,
                    tau=tau, fpar=fpar)
    out["fit_joint_dynamic"] = float(fit["theta"].sum() + fit["chi2"])
    rng = np.random.default_rng(7)
    static_windows = np.array([simulate_window(e, flux, INST, rng, SIGMA)
                               for e in scene_positions_al(theta[:2], psi)])
    static = fit_joint(static_windows, psi, theta[:2] + [5.0, -5.0], INST, SIGMA)
    out["fit_joint"] = float(static["theta"].sum() + static["chi2"])
    return out


@pytest.mark.parametrize("case", sorted(BASELINE))
def test_output_is_bit_identical_to_the_baseline(case):
    got, expected = compute()[case], BASELINE[case]
    reason = skip_reason(ENVIRONMENT, environment())
    if got != expected and reason:
        pytest.skip(f"{case}: {expected!r} -> {got!r}; {reason}")
    assert got == expected, (
        f"{case}: {expected!r} -> {got!r}. A deliberate change re-blesses the "
        "baseline in its own commit and says what moved.")


def test_the_baseline_covers_every_case():
    assert set(BASELINE) == set(compute())


def test_the_baseline_names_the_environment_it_was_recorded_in():
    assert set(ENVIRONMENT) == {"numpy", "scipy", "platform"}
    assert all(isinstance(v, str) and v for v in ENVIRONMENT.values())


def test_another_environment_is_named_and_skipped_not_failed():
    here = {"numpy": "2.5.3", "scipy": "1.18.1", "platform": "Darwin-arm64"}
    assert skip_reason(here, dict(here)) is None
    other = dict(here, numpy="2.4.3")
    reason = skip_reason(here, other)
    assert "2.5.3" in reason and "2.4.3" in reason and "test_reference_cases.py" in reason


if __name__ == "__main__" and "--bless" in sys.argv:
    lines = ['"""Byte-identity baseline of the package\'s own outputs',
             '(``tests/test_baseline.py``). Compared exactly; a change here is a',
             'deliberate numeric change, re-blessed in its own commit."""',
             "from __future__ import annotations", "",
             "BASELINE: dict[str, float] = {"]
    lines += [f"    {k!r}: {v!r}," for k, v in sorted(compute().items())]
    lines += ["}", "", "ENVIRONMENT: dict[str, str] = {"]
    lines += [f"    {k!r}: {v!r}," for k, v in sorted(environment().items())]
    lines += ["}", ""]
    (HERE / "_baseline.py").write_text("\n".join(lines), encoding="utf-8")
    print("blessed", HERE / "_baseline.py")

