"""The campaign: the windows of a scene over a scan law, simulated, then reduced.

``simulation.simulate_campaign`` renders every transit of a ``Scene``;
``reduction.reduce_campaign`` reduces windows that exist (simulated or real)
into one ``DetectionSet``, with no random number and no truth;
``simulation.harvest_campaign`` is the two in one call. Three parts.

THE EXPORT: the detection sets of the static and the moving pair of
scanlib-dev's reference cases (``<scene>/campaign/*``), harvested at the
export's seeds over the first 40 transits of the Pleiades law at S/N 100
with ``k_max = 2``, reproduced: every detection, its sigma and flux, the
margins and the census. THE RULES, ported from scanlib-dev's tests: the
harvest is one render then the pure reduction; the law carries the epochs
and the parallax factors onto the detections; the reduction is the ladder
and the delivery rule window by window; the census closes; mismatched
inputs and a moving scene without epochs are refused. THE INTERFACE of the
shared package: the instrument's margins (none passed, an instrument
without them refused), ``k_max`` required, the noise a scalar or a
``NoiseModel``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scanlib.instrument import TOY_MARGINS, Instrument, project_al, render_window
from scanlib.noise import NoiseModel
from scanlib.reduction import (SIGMA_DET_FALLBACK_MAS, DetectionSet, delivery_mask,
                               ladder_window, reduce_campaign)
from scanlib.scanlaw import PLEIADES, load_parallax_factor, load_scan_law
from scanlib.simulation import PRIMARY_FLUX, Scene, harvest_campaign, simulate_campaign

REFERENCE = Path(__file__).resolve().parent / "reference" / "reference_cases.npz"
RTOL, ATOL = 1e-9, 1e-12
INST = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18, margins=TOY_MARGINS)
SIGMA = PRIMARY_FLUX / 100.0                       # S/N 100

# the scenes and seeds of the export (scanlib-dev tools/export_reference_cases.py)
EXPORT = {"static_pair": (Scene.pair(rho_mas=120.0, pa_deg=35.0, q=0.3), False, 20261110),
          "moving_pair": (Scene.pair(rho_mas=150.0, pa_deg=-50.0, q=0.3, pm=(6.0, -4.0), dpi=-5.0),
                          True, 20261111)}

# a synthetic campaign wide enough that blended, censored and single-source windows occur
PSIS = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
TAUS = np.linspace(-2.5, 2.5, PSIS.size)
FPAR = 0.7 * np.sin(np.linspace(0.0, 4 * np.pi, PSIS.size))
SCENE = Scene.pair(200.0, 35.0, 0.4)
FAINT_SCENE = Scene.pair(200.0, 35.0, 0.08)
SEED = 11
DETECTIONS = ("psi", "eta", "flux", "sigma")
CENSUS = ("n_windows", "n_windows_with_detection", "n_censored_windows",
          "n_single_source_windows", "n_sigma_substituted")


def _with(margins):
    import dataclasses
    return dataclasses.replace(INST, margins=margins)


def _law(moving):
    psi, tau = load_scan_law(PLEIADES)
    fpar = load_parallax_factor(PLEIADES)
    return (psi[:40], tau[:40], fpar[:40]) if moving else psi[:40]


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


# ------------------------------------------------------------------ the export

@pytest.mark.parametrize("name", EXPORT)
def test_the_campaigns_of_the_export_are_harvested_as_dev(ref, name):
    scene, moving, seed = EXPORT[name]
    assert seed == int(ref[f"{name}/campaign/seed"])
    ds = harvest_campaign(scene, _law(moving), np.random.default_rng(seed), INST, SIGMA, k_max=2)
    g = lambda key: ref[f"{name}/campaign/{key}"]
    assert np.array_equal(ds.psi, g("psi"))
    if moving:
        assert np.array_equal(ds.tau, g("t")) and np.array_equal(ds.fpar, g("fpar"))
    else:
        assert ds.tau is None and ds.fpar is None
    for key in ("eta", "sigma", "flux"):
        np.testing.assert_allclose(getattr(ds, key), g(key), rtol=RTOL, atol=ATOL)
    assert np.array_equal(ds.margins, g("margins"))
    assert [getattr(ds, k) for k in CENSUS] == g("counts").tolist()


# ------------------------------------------------------------------ the rules

def test_the_harvest_is_one_render_then_the_pure_reduction():
    new = harvest_campaign(SCENE, PSIS, np.random.default_rng(SEED), _with([10.0]), 3.0,
                           k_max=2, return_windows=True)
    windows, sigma_noise, eta_true = simulate_campaign(SCENE, PSIS, np.random.default_rng(SEED),
                                                       _with([10.0]), 3.0)
    reduced = reduce_campaign(PSIS, windows, _with([10.0]), sigma_noise, k_max=2)
    assert isinstance(new, DetectionSet) and new.psi.size > 0
    for key in (*DETECTIONS, *CENSUS):
        assert np.array_equal(getattr(new, key), getattr(reduced, key)), key
    assert new.tau is None and new.fpar is None
    assert np.array_equal(new.margins, [10.0]) and reduced.windows is None
    assert np.array_equal(new.windows, windows) and windows.shape == (PSIS.size, 18)
    assert eta_true.shape == (PSIS.size, 2)
    assert np.array_equal(eta_true[:, 1], [project_al(*SCENE.positions[1], p) for p in PSIS])


def test_the_law_carries_the_epochs_and_the_parallax_factors_onto_the_detections():
    got = harvest_campaign(SCENE, (PSIS, TAUS, FPAR), np.random.default_rng(SEED),
                           _with([10.0]), 3.0, k_max=2)
    window_of = np.searchsorted(PSIS, got.psi)
    assert np.array_equal(PSIS[window_of], got.psi)
    assert np.array_equal(got.tau, TAUS[window_of]) and np.array_equal(got.fpar, FPAR[window_of])
    bare = harvest_campaign(SCENE, PSIS, np.random.default_rng(SEED), _with([10.0]), 3.0, k_max=2)
    assert np.array_equal(got.eta, bare.eta)                  # a static scene ignores the epochs


def test_the_reduction_is_the_ladder_and_the_delivery_rule_per_window():
    harvest = harvest_campaign(SCENE, PSIS, np.random.default_rng(SEED), _with([10.0]), 3.0,
                               k_max=2, return_windows=True)
    det_psi, det_eta, n_single = [], [], 0
    for psi_j, w in zip(PSIS, harvest.windows):
        eta, _flux, k, _ = ladder_window(w, _with([10.0]), 3.0, k_max=2)
        n_single += int(k == 1)
        keep = delivery_mask(eta)
        det_psi.extend([psi_j] * int(keep.sum()))
        det_eta.extend(eta[keep].tolist())
    assert np.array_equal(harvest.psi, det_psi) and np.array_equal(harvest.eta, det_eta)
    assert harvest.n_single_source_windows == n_single


def test_the_sigma_substitutions_are_counted():
    harvest = harvest_campaign(SCENE, PSIS, np.random.default_rng(SEED), _with([10.0]), 3.0,
                               k_max=2, return_windows=True)
    reduced = reduce_campaign(PSIS, harvest.windows, _with([10.0]), 3.0, k_max=2)
    assert np.all(np.isfinite(reduced.sigma))
    assert reduced.n_sigma_substituted == int(np.sum(reduced.sigma == SIGMA_DET_FALLBACK_MAS))


def test_the_census_counts_every_branch():
    harvest = harvest_campaign(FAINT_SCENE, PSIS, np.random.default_rng(SEED), _with([10.0]), 3.0,
                               k_max=2, return_windows=True)
    reduced = reduce_campaign(PSIS, harvest.windows, _with([10.0]), 3.0, k_max=2)
    assert reduced.n_windows == PSIS.size
    assert reduced.n_windows_with_detection + reduced.n_censored_windows == reduced.n_windows
    assert 0 < reduced.n_single_source_windows <= reduced.n_censored_windows
    assert reduced.n_windows_with_detection > 0
    assert reduced.psi.size >= reduced.n_windows_with_detection
    closed = reduce_campaign(PSIS, harvest.windows, _with([np.inf]), 3.0, k_max=2)
    assert closed.n_single_source_windows == closed.n_censored_windows == PSIS.size
    assert closed.psi.size == 0


def test_mismatched_inputs_are_refused():
    harvest = harvest_campaign(SCENE, PSIS, np.random.default_rng(SEED), _with([10.0]), 3.0,
                               k_max=2, return_windows=True)
    with pytest.raises(ValueError, match="scan angle"):
        reduce_campaign(PSIS[:-1], harvest.windows, _with([10.0]), 3.0, k_max=2)
    with pytest.raises(ValueError, match="tau"):
        reduce_campaign((PSIS, TAUS[:-1]), harvest.windows, _with([10.0]), 3.0, k_max=2)
    with pytest.raises(ValueError, match="zero point"):
        reduce_campaign(PSIS, harvest.windows, _with([10.0]), 3.0, k_max=2, eta0=np.zeros(3))


def test_a_moving_scene_needs_its_epochs_and_parallax_factors():
    moving = Scene.pair(200.0, 35.0, 0.4, pm=(5.0, -3.0))
    parallactic = Scene.pair(200.0, 35.0, 0.4, dpi=-2.0)
    assert SCENE.static and not moving.static and not parallactic.static
    with pytest.raises(ValueError, match="epoch"):
        simulate_campaign(moving, PSIS, np.random.default_rng(0), INST, SIGMA)
    with pytest.raises(ValueError, match="parallax"):
        simulate_campaign(parallactic, (PSIS, TAUS), np.random.default_rng(0), INST, SIGMA)
    with pytest.raises(ValueError, match="flux"):
        Scene([[0.0, 0.0], [1.0, 2.0]], [1.0])
    with pytest.raises(ValueError, match="pm"):
        Scene([[0.0, 0.0]], [1.0], pm=[[0.0, 0.0], [1.0, 1.0]])


def test_a_moving_scene_places_each_component_at_its_epoch():
    scene = Scene.pair(200.0, 35.0, 0.4, pm=(5.0, -3.0), dpi=-2.0)
    _w, _s, eta_true = simulate_campaign(scene, (PSIS, TAUS, FPAR), np.random.default_rng(0),
                                         INST, SIGMA)
    dra, ddec = scene.positions[1]
    expected = project_al(dra + 5.0 * TAUS, ddec - 3.0 * TAUS, PSIS) - 2.0 * FPAR
    np.testing.assert_allclose(eta_true[:, 1], expected, rtol=0.0, atol=1e-12)
    assert np.all(eta_true[:, 0] == 0.0)


# ------------------------------------------------------------------ the interface

def test_the_margins_are_the_instruments_and_k_max_is_required():
    bare = Instrument.gaussian()
    with pytest.raises(ValueError, match="calibrate_margins"):
        harvest_campaign(SCENE, PSIS, np.random.default_rng(SEED), bare, SIGMA, k_max=2)
    with pytest.raises(TypeError):
        harvest_campaign(SCENE, PSIS, np.random.default_rng(SEED), INST, SIGMA)
    with pytest.raises(TypeError):
        reduce_campaign(PSIS, np.zeros((PSIS.size, 18)), INST, SIGMA)
    two = harvest_campaign(SCENE, PSIS, np.random.default_rng(SEED), INST, SIGMA, k_max=2)
    three = harvest_campaign(SCENE, PSIS, np.random.default_rng(SEED), INST, SIGMA, k_max=3)
    assert np.array_equal(two.margins, TOY_MARGINS[:1]) and np.array_equal(three.margins, TOY_MARGINS)


def test_a_noise_model_draws_each_sample_with_its_own_sigma():
    model = NoiseModel(gain=2.0, read_noise=3.0, background=20.0)
    windows, sigma, _ = simulate_campaign(SCENE, PSIS, np.random.default_rng(SEED), INST, model)
    rng = np.random.default_rng(SEED)
    for j, psi_j in enumerate(PSIS):
        clean = render_window(SCENE.along_scan(psi_j), SCENE.fluxes, INST)
        assert np.array_equal(sigma[j], model.sigma(clean))
        assert np.array_equal(windows[j], clean + rng.normal(0.0, model.sigma(clean), clean.size))
    ds = harvest_campaign(SCENE, PSIS, np.random.default_rng(SEED), INST, model, k_max=2)
    ref = reduce_campaign(PSIS, windows, INST, sigma, k_max=2)
    for key in (*DETECTIONS, *CENSUS):
        assert np.array_equal(getattr(ds, key), getattr(ref, key)), key
    assert len(ds.sigma_noise) == PSIS.size
    with pytest.raises(ValueError, match="sigma_noise"):
        simulate_campaign(SCENE, PSIS, np.random.default_rng(SEED), INST, np.ones(PSIS.size))


@pytest.mark.parametrize("n_windows", [PSIS.size, INST.sample_grid_mas.size])
def test_a_list_of_per_window_sigmas_reads_as_the_same_array(n_windows):
    # J = 24 windows, and J = N = 18, where a list read as one sigma per sample would pass
    psis = PSIS[:n_windows]
    windows, _, _ = simulate_campaign(SCENE, psis, np.random.default_rng(SEED), INST, SIGMA)
    per_window = np.linspace(0.8, 1.2, n_windows) * SIGMA
    per_sample = np.repeat(per_window[:, None], INST.sample_grid_mas.size, axis=1)
    as_array = reduce_campaign(psis, windows, INST, per_window, k_max=2)
    for sigma_noise in (per_window.tolist(), tuple(per_window), per_sample):
        got = reduce_campaign(psis, windows, INST, sigma_noise, k_max=2)
        for key in (*DETECTIONS, *CENSUS):
            assert np.array_equal(getattr(got, key), getattr(as_array, key)), key
