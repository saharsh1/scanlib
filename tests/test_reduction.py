"""The per-window reduction: the nested ladder and its margins.

Three parts. THE EXPORT: every window of the four reference scenes reduced
by ``ladder_window`` on the toy instrument with ``TOY_MARGINS`` (k_max 3)
gives the stored order, positions, fluxes, chi^2, rung statistics,
Fisher sigmas and delivery mask (``<scene>/reduction/*``), and the first
200 windows of each rung's null from ``TOY_MARGINS_SEED`` give the stored rung
statistics (``margins/null/*``). THE RULES: the grid solve's active set is relative to its brightest atom and
links at one sample pitch; a per-sample sigma solves the whitened
problem and weights the Fisher matrix; the starting configurations are the ones the
rule names; the residual start keeps the ladder nested; the ladder accepts
``K + 1`` exactly when the rung statistic exceeds the margin, stops at
``k_max`` and never fits ``2K >= N`` parameters; the margins are the 99th
percentile of the ladder's own statistic on its null, rung by rung on the
generator's spawned streams; the instrument carries its margins and the
reduction refuses one without them; the delivery rule names the primary by
position. THE FREEZE (slow): at 2000 null windows per rung the frozen
margins lie inside the binomial interval of the 99th percentile.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import nnls
from scipy.stats import binom

from scanlib import reduction as R
from scanlib import simulation as SIM
from scanlib.instrument import TOY_MARGINS, TOY_MARGINS_N, TOY_MARGINS_SEED, Instrument, render_window
from scanlib.joint import joint_covariance
from scanlib.simulation import PRIMARY_FLUX, simulate_window, snr_from_peak_snr

REFERENCE = Path(__file__).resolve().parent / "reference" / "reference_cases.npz"
SCENES = ("static_pair", "moving_pair", "crowded_field", "offset_primary")
RTOL, ATOL = 1e-9, 1e-12
INST = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18, margins=TOY_MARGINS)
BARE = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18)
ATOM = INST.atom_eta_mas
I0 = int(np.argmin(np.abs(ATOM)))              # the atom at the origin
PITCH = float(np.max(np.diff(ATOM)))           # 7.375 mas
SIGMA = 10.0


def _with(margins, inst=INST):
    """The instrument with other margins."""
    return dataclasses.replace(inst, margins=margins)


def _window(eta, fluxes, seed, sigma=SIGMA):
    return simulate_window(eta, fluxes, INST, np.random.default_rng(seed), sigma)


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


# ------------------------------------------------------------------ the export

@pytest.mark.parametrize("scene", SCENES)
def test_every_window_of_the_reference_scenes_reduces_as_stored(ref, scene):
    g = lambda key: ref[f"{scene}/{key}"]
    windows, sigma = g("inputs/windows"), float(g("inputs/sigma"))
    for j, samples in enumerate(windows):
        eta, flux, k, diag = R.ladder_window(samples, INST, sigma, k_max=3)
        assert k == g("reduction/k")[j], f"window {j}"
        np.testing.assert_allclose(eta, g("reduction/eta")[j, :k], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(flux, g("reduction/flux")[j, :k], rtol=RTOL, atol=ATOL)
        n_fit, n_rung = diag["chi2"].size, diag["delta_chi2"].size
        assert np.all(np.isnan(g("reduction/chi2")[j, n_fit:]))
        np.testing.assert_allclose(diag["chi2"], g("reduction/chi2")[j, :n_fit], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(diag["delta_chi2"], g("reduction/delta_chi2")[j, :n_rung],
                                   rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(R.sigma_eta(eta, flux, INST, sigma),
                                   g("reduction/sigma_eta")[j, :k], rtol=RTOL, atol=ATOL)
        assert np.array_equal(R.delivery_mask(eta), g("reduction/delivered")[j, :k])
        assert not g("reduction/delivered")[j, k:].any()


def test_the_null_windows_of_both_rungs_are_the_stored_ones(ref):
    snr = snr_from_peak_snr(50.0, INST)
    assert snr == float(ref["margins/null/snr"])
    assert TOY_MARGINS_SEED == int(ref["margins/null/seed"])
    # the streams of calibrate_margins (k_max 3); the export spawned three, and the
    # first two children of a spawn do not depend on how many are spawned
    streams = np.random.default_rng(TOY_MARGINS_SEED).spawn(2)
    for rung in (1, 2):
        stat = SIM.rung_null_statistic(streams[rung - 1], rung, 200, INST, snr)
        np.testing.assert_allclose(stat, ref[f"margins/null/rung{rung}"], rtol=RTOL, atol=ATOL)


# ------------------------------------------------------------------ the grid solve

def test_active_atoms_are_relative_to_the_brightest_and_link_at_one_pitch():
    samples = render_window([ATOM[I0], ATOM[I0 + 8]], [PRIMARY_FLUX, 0.4 * PRIMARY_FLUX], INST)
    x, clusters = R.active_clusters(samples, INST)
    assert np.array_equal(x, nnls(INST.atoms, samples)[0])
    assert len(clusters) == 1 and set(clusters[0]) == {I0, I0 + 8}      # one pitch links
    assert clusters[0][0] == I0                                          # brightest first
    _x, apart = R.active_clusters(render_window([ATOM[I0], ATOM[I0 + 9]],
                                                [PRIMARY_FLUX, 0.4 * PRIMARY_FLUX], INST), INST)
    assert [c.tolist() for c in apart] == [[I0], [I0 + 9]]              # ranked by flux
    for scale in (1e-6, 1.0, 1e6):                                       # relative tolerance
        _x, c = R.active_clusters(scale * samples, INST)
        assert [a.tolist() for a in c] == [a.tolist() for a in clusters]
    assert R.active_clusters(np.zeros(INST.sample_grid_mas.size), INST)[1] == []


def test_a_per_sample_sigma_solves_the_whitened_problem():
    # sigma_n per sample: the samples and the design rows divided by sigma_n,
    # then ordinary NNLS; a scalar sigma is the unweighted problem
    rng = np.random.default_rng(7)
    s = SIGMA * (1.0 + rng.random(INST.sample_grid_mas.size))
    samples = simulate_window([0.0, 80.0], [PRIMARY_FLUX, 0.3 * PRIMARY_FLUX], INST, rng, s)
    eta = np.array([2.0, 85.0])
    cols = INST.design_columns(eta)
    assert R.profiled_chi2(eta, samples, INST, s) == nnls(cols / s[:, None], samples / s)[1] ** 2
    assert R.profiled_chi2(eta, samples, INST, SIGMA) == nnls(cols, samples)[1] ** 2
    x, _ = R.active_clusters(samples, INST, s)
    assert np.array_equal(x, nnls(INST.atoms / s[:, None], samples / s)[0])
    e, flux = R.fit_positions(samples, eta, INST, s)
    assert np.array_equal(flux, nnls(INST.design_columns(e) / s[:, None], samples / s)[0])


def test_a_per_sample_sigma_weights_the_fisher_matrix():
    # sigma_eta inverts J^T diag(1 / sigma_n^2) J, J the model's Jacobian in the K
    # positions and the K fluxes; a scalar sigma is J^T J / sigma^2
    rng = np.random.default_rng(8)
    s = SIGMA * (1.0 + rng.random(INST.sample_grid_mas.size))
    eta, flux = np.array([0.0, 120.0]), np.array([PRIMARY_FLUX, 0.3 * PRIMARY_FLUX])
    jac = np.hstack([INST.design_jacobian(eta) * flux[None, :], INST.design_columns(eta)])
    assert np.array_equal(R.sigma_eta(eta, flux, INST, s),
                          joint_covariance(jac.T @ (jac / s[:, None] ** 2), 2)[1])
    assert np.array_equal(R.sigma_eta(eta, flux, INST, SIGMA),
                          joint_covariance(jac.T @ jac / SIGMA ** 2, 2)[1])

def test_the_residual_scan_is_the_clipped_matched_filter():
    samples = _window([0.0, 150.0], [PRIMARY_FLUX, 300.0], 3, 3.0)
    eta1, flux1 = R.fit_positions(samples, [0.0], INST)
    scan = R._residual_scan(samples, eta1, flux1, INST)
    resid = samples - INST.design_columns(eta1) @ flux1
    proj = INST.atoms.T @ resid
    assert np.allclose(scan, resid @ resid - np.clip(proj, 0.0, None) ** 2
                       / (INST.atoms ** 2).sum(0))
    assert abs(ATOM[np.argmin(scan)] - 150.0) < 2 * PITCH             # the companion
    assert np.allclose(R._residual_scan(samples, [], [], INST),
                       samples @ samples - np.clip(INST.atoms.T @ samples, 0, None) ** 2
                       / (INST.atoms ** 2).sum(0))


# ------------------------------------------------------------------ the starts rule

def _clusters(*atom_lists):
    return [np.asarray(a) for a in atom_lists]


def test_k1_starts_at_the_brightest_clusters_brightest_atom():
    starts = R._starting_positions(None, 1, _clusters([I0, I0 + 1], [I0 + 40]), None, INST)
    assert [(l, p.tolist()) for l, p in starts] == [("{1}", [ATOM[I0]])]


def test_k2_selections_are_the_multisets_of_the_two_brightest_clusters():
    samples = render_window([ATOM[I0], ATOM[I0 + 40]], [PRIMARY_FLUX, 300.0], INST)
    prev = (np.array([ATOM[I0] + 0.3]), np.array([PRIMARY_FLUX]))
    clusters = _clusters([I0, I0 + 1, I0 + 2], [I0 + 40, I0 + 41], [I0 - 60])
    starts = dict(R._starting_positions(samples, 2, clusters, prev, INST))
    assert set(starts) == {"{1,1}", "{1,2}", "{2,2}", "r"}
    assert starts["{1,1}"].tolist() == sorted([ATOM[I0], ATOM[I0 + 1]])
    assert starts["{1,2}"].tolist() == [ATOM[I0], ATOM[I0 + 40]]
    assert starts["{2,2}"].tolist() == [ATOM[I0 + 40], ATOM[I0 + 41]]
    short = dict(R._starting_positions(samples, 2, _clusters([I0], [I0 + 40]), prev, INST))
    assert set(short) == {"{1,2}", "r"}                  # a cluster short of atoms
    one = dict(R._starting_positions(samples, 2, _clusters([I0, I0 + 1]), prev, INST))
    assert set(one) == {"{1,1}", "r"}                    # one cluster only


def test_k3_selections_are_the_ten_multisets_of_three_clusters():
    samples = render_window([ATOM[I0], ATOM[I0 + 40], ATOM[I0 - 60]],
                            [PRIMARY_FLUX, 300.0, 200.0], INST)
    prev = (np.array([ATOM[I0] + 0.3, ATOM[I0 + 40] - 0.3]), np.array([PRIMARY_FLUX, 300.0]))
    big = _clusters([I0, I0 + 1, I0 + 2], [I0 + 40, I0 + 41, I0 + 42],
                    [I0 - 60, I0 - 61, I0 - 62])
    starts = R._starting_positions(samples, 3, big, prev, INST)
    assert {l for l, _ in starts} == {"{1,1,1}", "{1,1,2}", "{1,1,3}", "{1,2,2}", "{1,2,3}",
                                      "{1,3,3}", "{2,2,2}", "{2,2,3}", "{2,3,3}", "{3,3,3}", "r"}
    for _l, pos in starts:
        assert np.all(np.diff(pos) >= PITCH * (1 - 1e-9))
    assert np.allclose(np.sort(np.r_[prev[0], ATOM[I0 - 60]]), dict(starts)["r"], atol=PITCH)
    with pytest.raises(ValueError, match="residual"):
        R._starting_positions(samples, 2, big, None, INST)


def test_the_residual_start_avoids_atoms_within_a_pitch_of_a_fitted_position():
    samples = render_window([ATOM[I0]], [PRIMARY_FLUX], INST)
    prev = (np.array([ATOM[I0]]), np.array([PRIMARY_FLUX]))
    r = dict(R._starting_positions(samples, 2, _clusters([I0]), prev, INST))["r"]
    assert np.abs(r - ATOM[I0]).min() < 1e-9 and np.diff(r)[0] >= PITCH * (1 - 1e-9)


def test_identical_starts_are_kept_once():
    samples = render_window([ATOM[I0], ATOM[I0 + 40]], [PRIMARY_FLUX, 300.0], INST)
    prev = (np.array([ATOM[I0]]), np.array([PRIMARY_FLUX]))
    starts = R._starting_positions(samples, 2, _clusters([I0], [I0 + 40]), prev, INST)
    assert [l for l, _ in starts] == ["{1,2}"]           # the residual peak is cluster 2's atom


# ------------------------------------------------------------------ the fit

def test_fit_k_returns_sorted_positions_profiled_fluxes_and_the_residual_power():
    samples = _window([0.0, 150.0], [PRIMARY_FLUX, 300.0], 3, 3.0)
    eta, flux, chi2 = R.fit_k(samples, 2, INST)
    assert eta.shape == flux.shape == (2,) and np.all(np.diff(eta) > 0)
    assert np.allclose(flux, nnls(INST.design_columns(eta), samples)[0])
    resid = samples - INST.design_columns(eta) @ flux
    assert chi2 == pytest.approx(resid @ resid, rel=1e-12)
    assert np.allclose(eta, [0.0, 150.0], atol=5.0)
    with pytest.raises(ValueError, match="k"):
        R.fit_k(samples, 0, INST)


def test_the_fit_is_the_deepest_polished_start():
    samples = _window([0.0, 40.0], [PRIMARY_FLUX, 500.0], 5, 3.0)
    _x, clusters = R.active_clusters(samples, INST)
    prev = R.fit_k(samples, 1, INST)
    polished = [R._residual_power(*R.fit_positions(samples, pos, INST), samples, INST)
                for _l, pos in R._starting_positions(samples, 2, clusters, prev[:2], INST)]
    assert R.fit_k(samples, 2, INST, previous=prev[:2])[2] == min(polished)
    assert np.array_equal(R.fit_k(samples, 2, INST)[0],
                          R.fit_k(samples, 2, INST, previous=prev[:2])[0])


def test_the_residual_start_keeps_the_ladder_nested():
    rng = np.random.default_rng(2026)
    for i in range(40):
        samples, sigma = SIM._null_window(rng, 1 + i % 2, INST, 50.0)
        _e, _f, k, diag = R._ladder(samples, INST, sigma, np.full(2, -np.inf), 3)
        assert k == 3 and diag["chi2"].size == 3
        assert np.all(np.diff(diag["chi2"]) <= 1e-9 * diag["chi2"][:-1])
        assert np.all(diag["delta_chi2"] >= -1e-9)


def test_a_window_without_active_atoms_has_order_zero():
    dark = np.zeros(INST.sample_grid_mas.size)
    eta, flux, k, diag = R.ladder_window(dark, INST, SIGMA)
    assert k == 0 and eta.size == flux.size == 0
    assert diag["active_eta"].size == 0 and diag["chi2"].size == 0
    with pytest.raises(ValueError, match="active"):
        R.fit_k(dark, 1, INST)


# ------------------------------------------------------------------ the ladder

def test_the_ladder_accepts_exactly_above_the_margin():
    samples = _window([0.0, 120.0], [PRIMARY_FLUX, 300.0], 8)
    eta, flux, k, diag = R.ladder_window(samples, _with([10.0, np.inf]), SIGMA)
    assert k == 2 and eta.size == 2 and diag["delta_chi2"].size == 2
    d = float(diag["delta_chi2"][0])
    assert d == pytest.approx((diag["chi2"][0] - diag["chi2"][1]) / SIGMA ** 2)
    assert R.ladder_window(samples, _with([np.nextafter(d, -np.inf)]), SIGMA)[2] == 2
    one = R.ladder_window(samples, _with([d]), SIGMA)                  # strict ">"
    assert one[2] == 1 and one[0].size == 1 and np.array_equal(one[3]["delta_chi2"], [d])
    assert one[3]["n_starts"].tolist()[0] == 1


def test_a_single_star_stays_single_and_k_max_caps_the_climb():
    samples = _window([0.0], [PRIMARY_FLUX], 1, 3.0)
    eta, flux, k, diag = R.ladder_window(samples, _with([10.0, 10.0]), 3.0)
    assert k == 1 and abs(eta[0]) < 5.0
    capped = R.ladder_window(samples, INST, 3.0, k_max=1)
    assert capped[2] == 1 and capped[3]["delta_chi2"].size == 0
    forced = R._ladder(samples, INST, 3.0, np.full(2, -np.inf), 3)
    assert forced[2] == 3 and forced[3]["chi2"].size == 3


def test_the_triple_rung_finds_a_resolved_third_source():
    samples = _window([-200.0, 0.0, 200.0], [PRIMARY_FLUX, 400.0, 400.0], 2, 3.0)
    eta, _flux, k, _ = R.ladder_window(samples, _with([10.0, 10.0]), 3.0)
    assert k == 3 and np.allclose(eta, [-200.0, 0.0, 200.0], atol=10.0)
    assert R.ladder_window(samples, _with([10.0, 10.0]), 3.0, k_max=2)[2] == 2


def test_the_ladder_never_fits_more_parameters_than_samples():
    inst = INST.for_window(INST.sample_grid_mas[:5])                    # five samples
    samples = render_window([0.0], [PRIMARY_FLUX], inst)
    _e, _f, k, diag = R._ladder(samples, inst, SIGMA, np.full(2, -np.inf), 3)
    assert k == 2 and diag["chi2"].size == 2                            # 2K < 5 stops at K = 2


def test_the_ladder_reads_the_margins_of_its_instrument():
    samples = _window([0.0, 120.0], [PRIMARY_FLUX, 300.0], 8)
    full = R.ladder_window(samples, INST, SIGMA)
    assert np.array_equal(full[0], R._ladder(samples, INST, SIGMA, TOY_MARGINS, 3)[0])
    assert full[3]["chi2"].size <= len(INST.margins) + 1                # k_max None: every rung
    two = R.ladder_window(samples, INST, SIGMA, k_max=2)
    assert np.array_equal(two[0], R._ladder(samples, INST, SIGMA, TOY_MARGINS[:1], 2)[0])


@pytest.mark.parametrize("inst, kw, match", [
    (BARE, {}, "calibrate_margins"),
    (INST, dict(k_max=4), "calibrate_margins"),
    (INST, dict(k_max=0), "k_max"),
    (INST, dict(sigma_noise=0.0), "sigma_noise"),
])
def test_the_ladder_refuses_an_instrument_without_the_margins_it_needs(inst, kw, match):
    samples = render_window([0.0], [PRIMARY_FLUX], INST)
    sigma = kw.pop("sigma_noise", SIGMA)
    with pytest.raises(ValueError, match=match):
        R.ladder_window(samples, inst, sigma, **kw)


def test_default_margins_are_the_instruments():
    assert np.array_equal(R.default_margins(INST), TOY_MARGINS)
    assert np.array_equal(R.default_margins(INST, k_max=2), TOY_MARGINS[:1])
    assert R.default_margins(INST, k_max=1).size == 0
    with pytest.raises(ValueError, match="calibrate_margins"):
        R.default_margins(BARE)
    with pytest.raises(ValueError, match="calibrate_margins"):
        R.default_margins(INST, k_max=4)


# ------------------------------------------------------------------ the margins

def test_margin_calibration_is_deterministic_and_rung_by_rung():
    m2 = SIM.calibrate_margins(np.random.default_rng(3), BARE, 50.0, n=20, k_max=2)
    assert m2.shape == (1,)
    assert np.array_equal(m2, SIM.calibrate_margins(np.random.default_rng(3), BARE, 50.0, n=20, k_max=2))
    assert not np.array_equal(m2, SIM.calibrate_margins(np.random.default_rng(4), BARE, 50.0,
                                                      n=20, k_max=2))
    m3 = SIM.calibrate_margins(np.random.default_rng(3), BARE, 50.0, n=(20, 6))
    assert m3.shape == (2,) and m3[0] == m2[0]                          # rung 1 on its own stream
    for k_max in (1, 4):
        with pytest.raises(ValueError, match="k_max"):
            SIM.calibrate_margins(np.random.default_rng(3), BARE, 50.0, n=5, k_max=k_max)


def test_the_margin_is_the_ladders_own_statistic_on_its_null():
    n = 30
    got = SIM.calibrate_margins(np.random.default_rng(6), BARE, 50.0, n=n, k_max=2)
    rng = np.random.default_rng(6).spawn(1)[0]
    lo, hi = INST.sample_grid_mas.min(), INST.sample_grid_mas.max()
    stat = []
    for _ in range(n):
        eta = rng.uniform(lo, hi)
        samples = simulate_window([eta], [PRIMARY_FLUX], INST, rng, PRIMARY_FLUX / 50.0)
        stat.append(R._ladder(samples, INST, PRIMARY_FLUX / 50.0, [np.inf], 2)[3]["delta_chi2"][0])
    assert got[0] == np.percentile(stat, SIM.MARGIN_PERCENTILE)
    assert got[0] == np.percentile(SIM.rung_null_statistic(np.random.default_rng(6).spawn(1)[0],
                                                         1, n, BARE, 50.0), SIM.MARGIN_PERCENTILE)
    for rung in (0, 3):
        with pytest.raises(ValueError, match="rung"):
            SIM.rung_null_statistic(np.random.default_rng(6), rung, n, BARE, 50.0)


def test_the_pair_null_is_the_ensemble_inside_the_sampled_range():
    rng = np.random.default_rng(9)
    lo, hi = INST.sample_grid_mas.min(), INST.sample_grid_mas.max()
    for _ in range(50):
        state = rng.bit_generator.state
        sep = rng.uniform(*SIM.NULL_PAIR_SEPARATION_FWHM) * INST.resolution_mas
        q = rng.uniform(*SIM.NULL_PAIR_Q)
        centre = rng.uniform(lo + sep / 2, hi - sep / 2)
        noise = rng.normal(0.0, 1.0, INST.sample_grid_mas.size)
        rng.bit_generator.state = state
        samples, sigma = SIM._null_window(rng, 2, INST, 50.0)
        eta = [centre - sep / 2, centre + sep / 2]
        assert lo <= eta[0] and eta[1] <= hi and 0.1 <= q <= 1.0
        assert 0.5 * INST.resolution_mas <= sep <= 2.5 * INST.resolution_mas
        clean = render_window(eta, [PRIMARY_FLUX, q * PRIMARY_FLUX], INST)
        assert sigma == PRIMARY_FLUX / 50.0
        assert np.allclose(samples, clean + sigma * noise)


@pytest.mark.slow
@pytest.mark.parametrize("rung, n, cap", [(1, 300, 10), (2, 200, 7)])
def test_held_out_false_promotion_at_the_frozen_margins(rung, n, cap):
    """Fresh null windows cross the frozen margin at about the designed 1 %:
    the cap is 1 % plus 4 binomial sigmas."""
    stat = SIM.rung_null_statistic(np.random.default_rng([2026, 9, 27, rung]), rung, n, INST,
                                 snr_from_peak_snr(50.0, INST))
    assert cap >= n * (0.01 + 4.0 * np.sqrt(0.0099 / n)) - 1
    assert int((stat > TOY_MARGINS[rung - 1]).sum()) <= cap


@pytest.mark.slow
def test_the_frozen_margins_lie_in_the_binomial_interval_at_2000_windows():
    """The first 2000 null windows of each rung, from the freeze's own streams:
    the frozen margin (the 99th percentile of 20000) lies between the order
    statistics the binomial distribution of the rank allows at 99.9 %."""
    snr = snr_from_peak_snr(50.0, INST)
    streams = np.random.default_rng(TOY_MARGINS_SEED).spawn(2)
    n = 2000
    lo, hi = (int(v) for v in binom.interval(0.999, n, SIM.MARGIN_PERCENTILE / 100.0))
    assert TOY_MARGINS_N == (20000, 20000)
    for rung in (1, 2):
        stat = np.sort(SIM.rung_null_statistic(streams[rung - 1], rung, n, INST, snr))
        assert stat[lo - 1] <= TOY_MARGINS[rung - 1] <= stat[hi - 1], rung


# ------------------------------------------------------------------ the instrument per window

def test_for_window_spans_the_window_and_drops_the_margins():
    grid = np.arange(18) * 59.0 + 12345.0                 # an offset origin
    inst = INST.for_window(grid)
    assert inst.margins is None
    assert inst.atom_eta_mas.min() <= grid.min() - INST.resolution_mas + 1e-6
    assert inst.atom_eta_mas.max() >= grid.max() + INST.resolution_mas - 10.0
    assert np.diff(inst.atom_eta_mas).mean() < 59.0
    assert inst.atoms.shape == (grid.size, inst.atom_eta_mas.size)
    assert inst.eta_bounds[0] == pytest.approx(grid.min() - INST.resolution_mas)
    assert inst.merge_gap_mas == pytest.approx(INST.merge_gap_mas)
    with pytest.raises(ValueError, match="calibrate_margins"):
        R.ladder_window(render_window([12345.0], [PRIMARY_FLUX], inst), inst, SIGMA)


def test_the_window_origin_is_a_gauge_choice():
    truth = np.array([0.0, 155.0])
    fluxes = np.array([PRIMARY_FLUX, 0.35 * PRIMARY_FLUX])
    noise = np.random.default_rng(4).normal(0.0, 3.0, INST.sample_grid_mas.size)
    eta_a, flux_a = R.fit_positions(render_window(truth, fluxes, INST) + noise, truth + 20.0, INST)
    shift = 12345.0
    shifted = INST.for_window(INST.sample_grid_mas + shift)
    eta_b, flux_b = R.fit_positions(render_window(truth + shift, fluxes, shifted) + noise,
                                    truth + shift + 20.0, shifted)
    np.testing.assert_allclose(eta_b - shift, eta_a, atol=1e-3)
    np.testing.assert_allclose(flux_b, flux_a, rtol=1e-6)


def test_a_window_of_twelve_samples_reduces():
    inst = INST.for_window((np.arange(12) - 5.5) * 59.0)
    truth = np.array([0.0, 140.0])
    samples = render_window(truth, [PRIMARY_FLUX, 0.4 * PRIMARY_FLUX], inst) + \
        np.random.default_rng(9).normal(0.0, 3.0, 12)
    eta, flux, _chi2 = R.fit_k(samples, 2, inst)
    np.testing.assert_allclose(np.sort(eta), truth, atol=8.0)
    sig = R.sigma_eta(eta, flux, inst, 3.0)
    assert np.all(np.isfinite(sig)) and np.all(sig > 0.0)


# ------------------------------------------------------------------ delivery

def test_the_primary_is_the_component_nearest_the_origin():
    assert R.delivery_mask([-3.0, 40.0]).tolist() == [False, True]
    assert R.delivery_mask([120.0, -8.0, 60.0]).tolist() == [True, False, True]
    assert R.delivery_mask([50.0, 90.0], origin=80.0).tolist() == [True, False]
    assert R.delivery_mask([5.0]).tolist() == [False]
    assert R.delivery_mask([]).tolist() == []


def test_a_flux_swap_does_not_move_the_delivered_component():
    eta_fit, flux_fit = np.array([1.5, 44.0]), np.array([480.0, 1010.0])   # swapped
    assert eta_fit[R.delivery_mask(eta_fit)].tolist() == [44.0]
    assert np.delete(eta_fit, np.argmax(flux_fit)).tolist() == [1.5]
