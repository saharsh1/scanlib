"""The conventions, each pinned by a test: the projection's four cardinal
angles, the signs of the motion and parallax terms, the unit-area ridge,
the half-turn weights, and the likelihood's constant.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from scanlib import (Instrument, backprojection_map, direction_weights, filtered_backprojection,
                     gaussian_lsf_jacobian, joint_fisher, loglike_profiled,
                     loglike_windows, project_al,
                     render_window, scene_positions_al, simulate_window,
                     stacked_design)

INST = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18)
GRID = INST.sample_grid_mas


@pytest.mark.parametrize("psi, expected", [
    (0.0, 20.0),               # north: eta = ddec
    (np.pi / 2, 10.0),         # east: eta = dra*
    (np.pi, -20.0),            # south
    (3 * np.pi / 2, -10.0),    # west
])
def test_project_al_on_the_cardinal_angles(psi, expected):
    assert project_al(10.0, 20.0, psi) == pytest.approx(expected, abs=1e-12)


def test_motion_and_parallax_enter_with_positive_sign():
    psi, tau, fpar = np.array([0.0, np.pi / 2]), np.array([0.0, 2.0]), np.array([0.5, -0.5])
    theta = [10.0, 20.0, 3.0, -4.0, 6.0]           # dra*, ddec, mu_a, mu_d, dpi
    eta = scene_positions_al(theta, psi, tau, fpar)
    assert eta.shape == (2, 2) and np.all(eta[:, 0] == 0.0)
    assert eta[0, 1] == pytest.approx(20.0 + 6.0 * 0.5)              # tau 0, north
    assert eta[1, 1] == pytest.approx((10.0 + 3.0 * 2.0) + 6.0 * (-0.5))   # east at tau 2
    with pytest.raises(ValueError):
        scene_positions_al(theta[:3], psi, tau, fpar)
    with pytest.raises(ValueError):
        scene_positions_al(theta, psi, None, fpar)


def test_zero_points_shift_the_window_not_the_sky():
    psi = np.array([0.3])
    theta = [40.0, -30.0]
    A0 = stacked_design(theta, psi, INST)
    A1 = stacked_design(theta, psi, INST, eta0=[59.0])    # one sample further
    assert np.allclose(A1[:-1], A0[1:])       # the source sits one sample earlier in the window


def test_design_columns_integrate_to_one_flux_and_render_is_the_operator():
    fine = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=1.0, n_samples=4000)
    assert fine.design_columns([0.0]).sum() == pytest.approx(1.0, abs=1e-9)
    eta, flux = np.array([-30.0, 120.0]), np.array([1000.0, 200.0])
    assert np.array_equal(render_window(eta, flux, INST),
                          INST.design_columns(eta) @ flux)
    rng = np.random.default_rng(1)
    noisy = simulate_window(eta, flux, INST, rng, 10.0)
    assert noisy.shape == GRID.shape


def test_design_jacobian_is_the_derivative_with_respect_to_the_source_position():
    h = 1e-4
    numeric = (INST.design_columns([30.0 + h]) - INST.design_columns([30.0 - h])) / (2 * h)
    assert np.allclose(INST.design_jacobian([30.0]), numeric, atol=1e-9)


def test_each_ridge_has_unit_area_across_its_line():
    sky, axis, _ = backprojection_map([0.0], [40.0], [6.0], half_size=200.0, step=0.5)
    column = sky[:, 0]                      # a north scan: the ridge runs along dra*
    assert column.sum() * 0.5 == pytest.approx(1.0, abs=1e-6)
    assert axis[np.argmax(column)] == pytest.approx(40.0)


def test_direction_weights_sum_to_the_half_turn():
    w = direction_weights([0.5, 0.5 + np.pi, 2.0])
    assert w.sum() == pytest.approx(np.pi)
    # a direction and its reverse are one direction: together they own its share
    assert w[0] + w[1] == pytest.approx(direction_weights([0.5, 2.0])[0])
    assert np.allclose(direction_weights(np.linspace(0, np.pi, 9, endpoint=False)), np.pi / 9)


# a law with ties between unequal neighbour gaps: 0.3 and its reverse 0.3 + pi (equal modulo pi to
# within float error), and 0.9 twice
TIED = np.array([0.3, 0.9, 2.4, 0.3 + np.pi, 0.9])


def _weights_before(psi):
    """The rule without tie groups: each direction half the gap to each neighbour."""
    d = np.mod(psi, np.pi)
    order = np.argsort(d)
    gaps = np.diff(np.concatenate([d[order], d[order][:1] + np.pi]))
    w = np.empty_like(d)
    w[order] = 0.5 * (gaps + np.roll(gaps, 1))
    return w


def test_tied_directions_share_their_weight_equally_whatever_the_order():
    w = direction_weights(TIED)
    assert w.sum() == pytest.approx(np.pi, rel=1e-15)
    assert w[0] == w[3] and w[1] == w[4]                 # each tie split equally
    alone = direction_weights(np.array([0.3, 0.9, 2.4]))
    np.testing.assert_allclose([w[0] + w[3], w[1] + w[4], w[2]], alone, rtol=1e-15)
    for perm in ([4, 3, 2, 1, 0], [3, 1, 0, 4, 2], [1, 4, 3, 2, 0]):
        np.testing.assert_allclose(direction_weights(TIED[perm]), w[perm], rtol=1e-15)


def test_the_inverse_radon_image_does_not_depend_on_the_order_of_the_windows():
    # each window its own noise: two windows at one angle are two measurements, not one
    rng = np.random.default_rng(7)
    windows = np.array([simulate_window(scene_positions_al(np.array([60.0, -40.0]), p)[0],
                                        [1000.0, 300.0], INST, rng, 10.0) for p in TIED])
    image, _ = filtered_backprojection(TIED, windows, INST, half=200.0, step=4.0)
    perm = [4, 3, 2, 1, 0]                               # the two windows at 0.9 rad swap places
    again, _ = filtered_backprojection(TIED[perm], windows[perm], INST, half=200.0, step=4.0)
    np.testing.assert_allclose(again, image, rtol=0.0, atol=1e-12 * np.abs(image).max())


def test_a_law_without_ties_keeps_its_weights_bit_for_bit():
    for psi in (np.random.default_rng(3).uniform(0.0, 2 * np.pi, 47),
                np.linspace(0.0, np.pi, 9, endpoint=False) + 0.01):
        assert np.array_equal(direction_weights(psi), _weights_before(psi))


def test_loglike_is_the_gaussian_log_density_with_its_constant():
    rng = np.random.default_rng(2)
    psi = np.linspace(0.0, np.pi, 6, endpoint=False)
    theta, flux = np.array([80.0, -40.0]), np.array([1000.0, 250.0])
    eta = scene_positions_al(theta, psi)
    windows = np.array([simulate_window(e, flux, INST, rng, 10.0) for e in eta])
    model = (stacked_design(theta, psi, INST) @ flux).reshape(windows.shape)
    expected = norm.logpdf(windows, loc=model, scale=10.0).sum()
    assert loglike_windows(theta, flux, windows, psi, INST, 10.0) == pytest.approx(expected)
    # a per-sample sigma: the same expression, sample by sample
    s = rng.uniform(8.0, 12.0, windows.shape)
    expected = norm.logpdf(windows, loc=model, scale=s).sum()
    assert loglike_windows(theta, flux, windows, psi, INST, s) == pytest.approx(expected)
    # profiling maximizes over the fluxes, on both paths
    for sigma in (10.0, s):
        assert loglike_profiled(theta, windows, psi, INST, sigma) >= \
            loglike_windows(theta, flux, windows, psi, INST, sigma)
    with pytest.raises(ValueError):
        loglike_windows(theta, flux, windows, psi, INST, -1.0)


def test_the_profiled_fluxes_are_non_negative_and_the_fisher_flux_block_is_the_design():
    rng = np.random.default_rng(3)
    psi = np.linspace(0.0, np.pi, 8, endpoint=False)
    theta, flux = np.array([60.0, 90.0]), np.array([1000.0, 0.0])      # a companion of no flux
    eta = scene_positions_al(theta, psi)
    windows = np.array([simulate_window(e, flux, INST, rng, 10.0) for e in eta])
    lnl, f = loglike_profiled(theta, windows, psi, INST, 10.0, return_fluxes=True)
    assert np.all(f >= 0.0) and f[0] == pytest.approx(1000.0, rel=0.02)
    assert lnl == loglike_profiled(theta, windows, psi, INST, 10.0)        # one solve, one value
    import scanlib.joint
    assert not hasattr(scanlib.joint, "profiled_fluxes")                  # one public name
    fisher = joint_fisher(theta, [1000.0, 300.0], psi, INST, 10.0)
    A = stacked_design(theta, psi, INST)
    assert np.allclose(fisher[2:, 2:], A.T @ A / 100.0)
    assert np.allclose(fisher, fisher.T)


def test_the_instrument_is_frozen_and_its_margins_are_read_only():
    from dataclasses import FrozenInstanceError
    from scanlib import Instrument
    from scanlib.instrument import TOY_MARGINS
    inst = Instrument.gaussian(margins=TOY_MARGINS)
    assert np.array_equal(inst.atoms, inst.design_columns(inst.atom_eta_mas))
    sigma_lsf = 100.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert np.array_equal(inst.design_jacobian([12.0]), gaussian_lsf_jacobian(
        inst.sample_grid_mas[:, None] - 12.0, sigma_lsf, 59.0))
    with pytest.raises(ValueError):
        inst.margins[0] = 1.0
    with pytest.raises(FrozenInstanceError):
        inst.margins = None
    assert Instrument.gaussian().margins is None
    for bad in ([], [11.0, -1.0], [[11.0]]):
        with pytest.raises(ValueError):
            Instrument.gaussian(margins=bad)


def test_a_line_through_the_peak_is_kept_and_one_that_misses_it_is_not():
    from scanlib.backprojection import peak_consistency
    psi = np.array([0.0, np.pi / 2, 1.0, 2.0])
    peak = (30.0, -20.0)
    eta = project_al(*peak, psi) + np.array([0.0, 2.9, 3.1, 40.0])   # misses [mas]
    sigma = np.array([1.0, 1.0, 1.0, 1.0])
    keep, distance = peak_consistency(psi, eta, sigma, peak, width=0.5)
    assert np.allclose(distance, [0.0, 2.9, 3.1, 40.0])
    assert keep.tolist() == [True, True, False, False]              # within 3 sigma
    keep, _ = peak_consistency(psi, eta, sigma, peak, width=5.0)    # the width floors it
    assert keep.tolist() == [True, True, True, False]
