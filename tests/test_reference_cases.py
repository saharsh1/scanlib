"""The package against the reference cases.

``tests/reference/reference_cases.npz`` holds, for four synthetic scenes
(a static pair, a moving pair with proper motion and parallax, a crowded
field of four sources, and the static pair with the windows centred
``(3, -2)`` mas off the primary) over the first 40 transits of the public
Pleiades scan-law forecast, the inputs and the outputs of every shared
algorithm, exported on 2026-11-10 from the code this package was ported
from; ``reference_cases.md`` catalogs the arrays. The joint routines are stored twice, the primary
pinned at the reference position (``truth/theta``, ``fit``, ``loglike``)
and free (``truth/theta_free``, ``fit_free``, ``loglike_free``); on the
fourth scene the pinned fit is the wrong model and is stored as such.
Every shared function runs here on the stored inputs and must
reproduce the stored outputs to ``rtol=1e-9, atol=1e-12``. The fits
compare ``theta`` to the stored solution within 1e-3 of its stored 1-sigma
(two Nelder-Mead runs may differ in the last digits), and the rest of the
fit to the same tolerance as everything else. ``meta/`` records the
export, and ``reference_cases.md`` names its date.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scanlib import (Instrument, backprojection_map, direction_weights, faststack,
                     filtered_backprojection, fit_joint,
                     joint_fisher, loglike_profiled, loglike_windows,
                     render_window, scene_positions_al)
from scanlib.joint import joint_covariance

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "tests" / "reference" / "reference_cases.npz"
SCENES = ("static_pair", "moving_pair", "crowded_field", "offset_primary")
RTOL, ATOL = 1e-9, 1e-12
MAP_HALF, MAP_STEP = 300.0, 4.0          # the sky grid of every stored image


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


class Case:
    """One scene's inputs, read once from the file."""

    def __init__(self, ref, scene):
        g = lambda key: ref[f"{scene}/{key}"]
        self.get = g
        self.psi = g("inputs/psi")
        self.windows = g("inputs/windows")
        self.sigma = float(g("inputs/sigma"))
        self.eta0 = g("inputs/eta0")
        # the scene's instrument, rebuilt from the stored FWHM and grid
        grid = g("inputs/grid")
        self.inst = Instrument.gaussian(fwhm_mas=float(g("inputs/lsf_fwhm")),
                                        pitch_mas=float(grid[1] - grid[0]),
                                        n_samples=grid.size)
        assert np.array_equal(self.inst.sample_grid_mas, grid)
        assert float(g("inputs/lsf_sigma")) == float(g("inputs/lsf_fwhm")) / (
            2.0 * np.sqrt(2.0 * np.log(2.0)))
        self.moving = f"{scene}/inputs/tau" in ref
        self.tau = g("inputs/tau") if self.moving else None
        self.fpar = g("inputs/fpar") if self.moving else None
        self.motion = dict(tau=self.tau, fpar=self.fpar)


@pytest.fixture(params=SCENES)
def case(ref, request):
    return Case(ref, request.param)


class Model:
    """The stored arrays of one model of the primary: pinned or free."""

    def __init__(self, free):
        self.free = free
        self.theta = "truth/theta_free" if free else "truth/theta"
        self.fit = "fit_free" if free else "fit"
        self.loglike = "loglike_free" if free else "loglike"


@pytest.fixture(params=(False, True), ids=("pinned", "free"))
def model(request):
    return Model(request.param)


def close(got, expected):
    got, expected = np.asarray(got, dtype=float), np.asarray(expected, dtype=float)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=RTOL, atol=ATOL, equal_nan=True), (
        f"max difference {np.nanmax(np.abs(got - expected)):.3e}")


def test_the_file_records_its_export(ref):
    """``meta/`` holds the commit, the version and the date of the export, and
    the catalog names the date."""
    assert len(str(ref["meta/source_commit"])) >= 7
    assert str(ref["meta/source_version"])
    exported = str(ref["meta/exported"])
    table = (REPO / "tests" / "reference" / "reference_cases.md").read_text(encoding="utf-8")
    assert exported in table


def test_the_toy_instrument_is_the_exported_one_to_the_last_bit(ref):
    """``Instrument.gaussian(100, 59, 18, margins=TOY_MARGINS)`` is the
    exported instrument: every ``instrument/*`` array equal bit for bit."""
    from scanlib.instrument import TOY_MARGINS, TOY_MARGINS_SEED
    g = lambda key: ref[f"instrument/{key}"]
    assert (float(g("fwhm_mas")), float(g("pitch_mas")), int(g("n_samples"))) == (100.0, 59.0, 18)
    assert np.array_equal(TOY_MARGINS, g("margins"))
    assert TOY_MARGINS_SEED == int(ref["margins/null/seed"])
    inst = Instrument.gaussian(float(g("fwhm_mas")), float(g("pitch_mas")),
                               int(g("n_samples")), margins=TOY_MARGINS)
    for got, key in ((inst.sample_grid_mas, "sample_grid"), (inst.atom_eta_mas, "atom_eta"),
                     (inst.atoms, "atoms"), (inst.margins, "margins"),
                     (inst.eta_bounds, "eta_bounds"), (inst.merge_gap_mas, "merge_gap"),
                     (inst.resolution_mas, "fwhm_mas")):
        expected = g(key)
        got = np.asarray(got)
        assert got.shape == expected.shape and np.array_equal(got, expected), key
    # the stored scenes were rendered on the same grid with the same LSF width
    assert np.array_equal(inst.sample_grid_mas, ref["static_pair/inputs/grid"])


def test_every_catalogued_array_is_in_the_file(ref):
    table = (REPO / "tests" / "reference" / "reference_cases.md").read_text(encoding="utf-8")
    names = [line.split("`")[1] for line in table.splitlines()
             if line.startswith("| `")]
    assert len(names) >= 100 and set(names) == set(ref), (
        "the catalog and the file list different arrays")


def test_instrument_renders_the_windows_up_to_their_noise(case):
    """The stored windows are the truth rendered plus noise of the stored sigma."""
    eta_true, fluxes = case.get("truth/eta"), case.get("truth/fluxes")
    clean = np.array([render_window(eta_true[j] - case.eta0[j], fluxes, case.inst) for j in range(case.psi.size)])
    noise = case.windows - clean
    assert abs(noise.std() / case.sigma - 1.0) < 0.1
    assert abs(noise.mean()) < 0.1 * case.sigma


def test_scene_positions(case, model):
    eta = scene_positions_al(case.get(model.theta), case.psi,
                             free_primary=model.free, **case.motion)
    if model.free:
        close(eta, case.get("truth/eta"))
    else:
        # pinned, the primary sits at the reference position, where the
        # truth has it unless the windows miss it (offset_primary)
        assert np.all(eta[:, 0] == 0.0)
        close(eta[:, 1:], case.get("truth/eta")[:, 1:])


def test_backprojection_map(case):
    sky, axis, extent = backprojection_map(case.get("detections/psi"),
                                           case.get("detections/eta"),
                                           case.get("detections/sigma"),
                                           MAP_HALF, MAP_STEP)
    close(sky, case.get("backprojection/map"))
    close(axis, case.get("backprojection/axis"))
    close(extent, case.get("backprojection/extent"))


def test_inverse_radon(case):
    weights = direction_weights(case.psi)
    close(weights, case.get("radon/weights"))
    image, axis = filtered_backprojection(case.psi, case.windows, case.inst,
                                          MAP_HALF, MAP_STEP, weights=weights)
    close(image, case.get("radon/image"))
    close(axis, case.get("radon/axis"))


def test_faststack(case):
    image, variance, n_used, axis = faststack(case.psi, case.windows, case.inst,
                                              MAP_HALF, MAP_STEP,
                                              alpha=float(case.get("faststack/alpha")))
    close(image, case.get("faststack/image"))
    close(variance, case.get("faststack/variance"))
    assert np.array_equal(n_used, case.get("faststack/n_used"))
    close(axis, case.get("faststack/axis"))


def test_loglike(case, model):
    args = (case.windows, case.psi, case.inst, case.sigma)
    kw = dict(free_primary=model.free, **case.motion)
    fit, loglike = model.fit, model.loglike
    close(loglike_windows(case.get(model.theta), case.get("truth/fluxes"), *args, **kw),
          case.get(f"{loglike}/at_truth"))
    close(loglike_windows(case.get(f"{fit}/theta"), case.get(f"{fit}/flux"), *args, **kw),
          case.get(f"{loglike}/at_fit"))
    # the stored fit maximizes the profiled likelihood over the fluxes
    profiled = loglike_profiled(case.get(f"{fit}/theta"), *args, **kw)
    assert profiled >= float(case.get(f"{loglike}/at_fit")) - 1e-9


def test_profiled_fluxes_and_fisher_at_the_stored_solution(case, model):
    fit, kw = model.fit, dict(free_primary=model.free, **case.motion)
    theta, flux = case.get(f"{fit}/theta"), case.get(f"{fit}/flux")
    lnl, got_flux = loglike_profiled(theta, case.windows, case.psi, case.inst, case.sigma,
                                     return_fluxes=True, **kw)
    close(got_flux, flux)
    n = case.windows.size                          # ln L = -chi^2/2 - n ln(2 pi sigma^2)/2
    close(-2.0 * lnl - n * np.log(2.0 * np.pi * case.sigma ** 2), case.get(f"{fit}/chi2"))
    fisher = joint_fisher(theta, flux, case.psi, case.inst,
                          case.sigma, **kw)
    cov, sigma_theta = joint_covariance(fisher, theta.size)
    close(cov, case.get(f"{fit}/cov"))
    close(sigma_theta, case.get(f"{fit}/sigma_theta"))


def test_fit(case, model):
    stage = model.fit
    theta0 = case.get(f"{stage}/theta0")
    # one function for both: tau None on the static scenes, given on the moving one
    fit = fit_joint(case.windows, case.psi, theta0, case.inst, case.sigma,
                    free_primary=model.free, **case.motion)
    assert fit["converged"]
    pull = np.abs(fit["theta"] - case.get(f"{stage}/theta")) / case.get(f"{stage}/sigma_theta")
    assert np.all(pull <= 1e-3), f"theta moved by up to {pull.max():.2e} sigma"
    close(fit["flux"], case.get(f"{stage}/flux"))
    close(fit["cov"], case.get(f"{stage}/cov"))
    close(fit["sigma_theta"], case.get(f"{stage}/sigma_theta"))
    close(fit["chi2"], case.get(f"{stage}/chi2"))
    close(fit["loglike"], case.get(f"{model.loglike}/at_fit"))


def test_fit_joint_is_one_function_with_both_budgets(ref, monkeypatch):
    """Static fits run Nelder-Mead with budgets 4000 / 16000, moving ones with
    8000 / 32000; the moving twin is gone; a rate bound needs epochs."""
    import scanlib
    import scanlib.joint as joint
    assert not hasattr(scanlib, "fit_joint_dynamic") and not hasattr(joint, "fit_joint_dynamic")
    seen = []
    real = joint.minimize_restarts

    def spy(*args, **kwargs):
        seen.append((kwargs["maxiter_first"], kwargs["maxiter_restart"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(joint, "minimize_restarts", spy)
    for scene in ("static_pair", "moving_pair"):
        case = Case(ref, scene)
        fit_joint(case.windows, case.psi, case.get("fit/theta0"), case.inst, case.sigma,
                  **case.motion)
    assert seen == [(4000, 16000), (8000, 32000)]
    case = Case(ref, "static_pair")
    with pytest.raises(ValueError):
        fit_joint(case.windows, case.psi, case.get("fit/theta0"), case.inst, case.sigma,
                  rate_bound=50.0)
