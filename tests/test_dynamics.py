"""The fits to the detection set: the motion fits, the cleaning, the static fit.

The detection set ``(psi, tau, eta, sigma, fpar)`` of a campaign is the
compressed form of its windows; ``dynamics`` fits the linear relative motion
(and the parallax difference) to it, or the position alone. THE EXPORT: the
motion fit and the cleaning of the moving pair's detection set
(``moving_pair/dynamics/*``) reproduced, including the bias the blend leaves
in it (about 2 sigma in each parameter; the joint fit to the windows removes
it). THE RULES, ported from scanlib-dev's tests: the fits recover an injected
motion; the cleaning converges, returns the fit on its keep set and drops a
displaced detection; on Figure 3's scene (a) at seed 14 it removes the one
detection off the track in two passes; with ``static`` it is one pool, one
prune and one fit; it takes no truth. THE ORDER: the map detects, a circle
of ``select_radius`` about its peak selects, the Huber fit of the motion to
the selected detections refines, and every detection is then tested against
the fitted track; the Huber parameters reach that first fit; a detection
outside the circle returns when it is on the track; with two companions the
seed is one of them. The static fit and the polar elements
follow their equations.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from scanlib import dynamics as D
from scanlib.backprojection import map_peak, peak_consistency
from scanlib.instrument import TOY_MARGINS, Instrument, project_al
from scanlib.joint import polar_elements
from scanlib.scanlaw import PLEIADES, load_parallax_factor, load_scan_law
from scanlib.simulation import PRIMARY_FLUX, Scene, harvest_campaign

REFERENCE = Path(__file__).resolve().parent / "reference" / "reference_cases.npz"
RTOL, ATOL = 1e-9, 1e-12
INST = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18, margins=TOY_MARGINS)
SIGMA = PRIMARY_FLUX / 100.0


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


def _harvest(seed, a0, d0, mua, mud, q=0.3):
    """The detection set of a companion drifting by (mua, mud) over 16 angles and 5 yr."""
    psi = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    tau = np.linspace(-2.5, 2.5, psi.size)
    scene = Scene([[0.0, 0.0], [a0, d0]], [PRIMARY_FLUX, PRIMARY_FLUX * q],
                  pm=[[0.0, 0.0], [mua, mud]], dpi=[0.0, 0.0])
    ds = harvest_campaign(scene, (psi, tau, np.zeros_like(tau)), np.random.default_rng(seed),
                          INST, SIGMA, k_max=3)
    return ds.psi, ds.tau, ds.eta, ds.sigma, ds.fpar


# ------------------------------------------------------------------ the export

def test_the_motion_fits_of_the_moving_pair_are_devs(ref):
    g = lambda key: ref[f"moving_pair/{key}"]
    psi, tau, eta, sigma, fpar = (g(f"campaign/{k}") for k in ("psi", "t", "eta", "sigma", "fpar"))
    theta, cov, _ = D.fit_motion(psi, tau, eta, sigma, fpar, parallax=True)
    np.testing.assert_allclose(theta, g("dynamics/fit_motion_theta"), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(cov, g("dynamics/fit_motion_cov"), rtol=RTOL, atol=ATOL)
    theta, cov, keep, info = D.clean_detection_set(psi, tau, eta, sigma, fpar, parallax=True)
    np.testing.assert_allclose(theta, g("dynamics/clean_theta"), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(cov, g("dynamics/clean_cov"), rtol=RTOL, atol=ATOL)
    assert np.array_equal(keep, g("dynamics/clean_keep"))
    assert info["iterations"] == int(g("dynamics/clean_iterations"))
    # the blend's bias, recorded as it is: about 2 sigma off the truth
    pull = (g("dynamics/fit_motion_theta") - g("truth/theta")) / np.sqrt(np.diag(g("dynamics/fit_motion_cov")))
    assert 1.5 < np.abs(pull[:4]).min() and np.abs(pull).max() < 3.0


# ------------------------------------------------------------------ the motion fits

def test_the_fits_recover_an_injected_relative_motion():
    a0, d0, mua, mud = 150.0, -90.0, 4.0, -6.0
    p, t, e, s, f = _harvest(3, a0, d0, mua, mud)
    assert p.size >= 8 and f.shape == p.shape
    theta, cov, e_corr = D.fit_motion(p, t, e, s)
    truth = np.array([a0, d0, mua, mud])
    assert np.abs((theta - truth) / np.sqrt(np.diag(cov))).max() < 5.0
    assert e_corr.shape == e.shape
    assert D.motion_design(p, t, f, False).shape == (p.size, 4)
    assert D.motion_design(p, t, f, True).shape == (p.size, 5)
    th_rob, _cov_rob, weights, _ = D.fit_motion_robust(p, t, e, s)
    assert weights.shape == p.shape and np.all(weights <= 1.0)
    th_all, _, th_refit, _, keep, _ = D.fit_mark_refit(p, t, e, s)
    assert np.array_equal(th_all, theta) and keep.shape == p.shape
    for alt in (th_rob, th_refit):
        assert np.abs((alt - truth) / np.sqrt(np.diag(cov))).max() < 5.0


def test_a_parallax_fit_needs_the_parallax_factors():
    p, t, e, s, _f = _harvest(3, 150.0, -90.0, 4.0, -6.0)
    with pytest.raises(ValueError, match="fpar"):
        D.fit_motion(p, t, e, s, parallax=True)


# ------------------------------------------------------------------ the cleaning

def test_the_cleaning_converges_and_drops_a_displaced_detection():
    a0, d0, mua, mud = 150.0, -90.0, 4.0, -6.0
    p, t, e, s, _f = _harvest(3, a0, d0, mua, mud)
    theta, cov, keep, info = D.clean_detection_set(p, t, e, s)
    assert info["converged"] and 1 <= info["iterations"] <= 5
    assert keep.dtype == bool and len(info["keep"]) == len(info["n_kept"]) == info["iterations"]
    th_keep, cov_keep, _ = D.fit_motion(p[keep], t[keep], e[keep], s[keep])
    assert np.array_equal(theta, th_keep) and np.array_equal(cov, cov_keep)
    assert np.abs((theta - np.array([a0, d0, mua, mud])) / np.sqrt(np.diag(cov))).max() < 5.0
    e_out = e.copy()
    e_out[0] += 60.0
    _, _, keep_out, info_out = D.clean_detection_set(p, t, e_out, s)
    assert info_out["converged"] and not keep_out[0] and keep_out[1:].all()
    assert info_out["n_removed"][-1] == 1


def test_the_cleaning_reproduces_figure_3s_keep_set():
    """Scene (a) of Figure 3 at seed 14, the ladder capped at two, S/N 100,
    over the Pleiades law: 34 detections; the one off the injected track by
    more than three sigma is removed, in two passes, and the relative proper
    motion moves from (3.68, 9.45) to (3.22, 8.97) mas/yr for an injected
    (3, 9). The track only names the removed detection; the cleaning sees
    neither it nor the seed."""
    mu_a, mu_d = 3.0, 9.0
    scene = Scene.pair(200.0, 110.0, 0.3, pm=(mu_a, mu_d), dpi=0.0)
    a0, d0 = scene.positions[1]
    psi, tau = load_scan_law(PLEIADES)
    ds = harvest_campaign(scene, (psi, tau, load_parallax_factor(PLEIADES)),
                          np.random.default_rng(14), INST, SIGMA, k_max=2)
    p, t, e, s = ds.psi, ds.tau, ds.eta, ds.sigma
    assert p.size == 34
    theta, cov, keep, info = D.clean_detection_set(p, t, e, s)
    assert info["converged"] and info["iterations"] == 2 and (~keep).sum() == 1
    off_track = np.abs(e - project_al(a0 + mu_a * t, d0 + mu_d * t, p)) > 3.0 * s
    assert np.array_equal(~keep, off_track)
    theta_all, _, _ = D.fit_motion(p, t, e, s)
    assert np.round(theta_all[2:], 2).tolist() == [3.68, 9.45]
    assert np.round(theta[2:], 2).tolist() == [3.22, 8.97]


def test_the_static_cleaning_is_one_pool_one_prune_one_fit():
    p, _t, e, s, _f = _harvest(5, 150.0, -90.0, 0.0, 0.0)
    theta, cov, keep, info = D.clean_detection_set(p, None, e, s, static=True)
    assert theta.shape == (2,) and cov.shape == (2, 2) and info["converged"]
    peak, width, _, _ = map_peak(p, e, s)
    keep_hand, _ = peak_consistency(p, e, s, peak, width)
    assert np.array_equal(keep, keep_hand)
    design = D.motion_design(p, np.zeros_like(p))[:, :2]
    theta_hand, cov_hand = D.wls(design[keep], e[keep], 1.0 / s[keep] ** 2)
    assert np.array_equal(theta, theta_hand) and np.array_equal(cov, cov_hand)
    f_syn = 0.7 * np.sin(np.linspace(0.0, 4 * np.pi, p.size))
    theta_p, cov_p, _, _ = D.clean_detection_set(p, None, e, s, f_syn, static=True, parallax=True)
    assert theta_p.shape == (3,) and cov_p.shape == (3, 3)
    with pytest.raises(ValueError, match="fpar"):
        D.clean_detection_set(p, None, e, s, static=True, parallax=True)


def test_the_cleaning_takes_no_truth():
    params = list(inspect.signature(D.clean_detection_set).parameters)
    assert params == ["psi", "tau", "eta", "sigma", "fpar", "parallax", "static", "select_radius",
                      "huber_delta", "huber_iter", "k_sigma", "origin_zone", "half_size", "step",
                      "max_iter"]


# ------------------------------------------------------------------ the cleaning's order:
# the map detects, a circle selects, the sinogram fit refines and cleans

def _track(a0, d0, mua, mud, n=24, sig=1.0):
    """Exact detections of one moving source over ``n`` angles and 5 yr, each of error ``sig``."""
    psi = np.linspace(0.0, 2 * np.pi, n, endpoint=False) + 0.1
    tau = np.linspace(-2.5, 2.5, n)[np.random.default_rng(0).permutation(n)]
    return psi, tau, project_al(a0 + mua * tau, d0 + mud * tau, psi), np.full(n, sig)


def test_the_first_fit_takes_only_the_lines_within_the_radius_of_the_static_peak():
    p, t, e, s, _f = _harvest(3, 150.0, -90.0, 4.0, -6.0)
    theta, _cov, keep, info = D.clean_detection_set(p, t, e, s, select_radius=30.0)
    peak, width, _, _ = map_peak(p, e, s)                      # the static map of the detections as measured
    _, distance = peak_consistency(p, e, s, peak, width)
    assert np.array_equal(info["peak"], peak) and info["width"] == width
    assert np.array_equal(info["selected"], distance <= 30.0)
    assert info["n_selected"] == int((distance <= 30.0).sum())
    assert len(info["keep"]) == len(info["n_kept"]) == len(info["theta"]) == info["iterations"]


def test_with_two_companions_the_seed_is_one_of_them():
    """A moving companion and a static one of the same flux over the Pleiades law, each window
    reduced to two sources at most, four noise draws: a motion fitted first to every detection
    mixes the two sources and ends 44 to 371 mas (position plus two years of motion) from both;
    fitted to the lines within the radius of the map's peak, it ends at one of them."""
    law = (*load_scan_law(PLEIADES), load_parallax_factor(PLEIADES))
    truths = np.array([[*Scene.pair(200.0, 110.0, 0.3).positions[1], 3.0, 9.0], [-150.0, 180.0, 0.0, 0.0]])
    scene = Scene([[0.0, 0.0], truths[0, :2], truths[1, :2]], PRIMARY_FLUX * np.array([1.0, 0.3, 0.3]),
                  pm=[[0.0, 0.0], truths[0, 2:], truths[1, 2:]])
    for seed in (107, 111, 112, 121):
        ds = harvest_campaign(scene, law, np.random.default_rng(seed), INST, SIGMA, k_max=2)
        theta, _cov, _keep, _info = D.clean_detection_set(ds.psi, ds.tau, ds.eta, ds.sigma)
        d = truths - theta
        assert min(np.hypot(d[:, 0], d[:, 1]) + 2.0 * np.hypot(d[:, 2], d[:, 3])) < 15.0, (seed, theta)


def test_a_line_outside_the_radius_returns_when_it_is_on_the_fitted_track():
    p, t, e, s = _track(150.0, -90.0, 6.0, 18.0)               # 19 mas/yr: up to 47 mas from the reference position
    theta, _cov, keep, info = D.clean_detection_set(p, t, e, s, select_radius=30.0)
    assert info["n_selected"] < p.size and keep.all() and info["converged"]
    np.testing.assert_allclose(theta, [150.0, -90.0, 6.0, 18.0], atol=1e-6)
    e_far = e.copy()
    e_far[0] += 80.0                                           # one line off the track stays out
    _, _, keep_far, _ = D.clean_detection_set(p, t, e_far, s, select_radius=30.0)
    assert not keep_far[0] and keep_far[1:].all()


def test_the_huber_parameters_reach_the_first_fit():
    p, t, e, s = _track(150.0, -90.0, 1.0, -2.0)
    e = e.copy()
    e[3] += 12.0                                               # inside the radius, 12 sigma off the track
    first = lambda **kw: D.clean_detection_set(p, t, e, s, select_radius=30.0, **kw)[3]
    info = first()
    sel = info["selected"]
    assert sel[3]
    plain, _, _ = D.fit_motion(p[sel], t[sel], e[sel], s[sel])
    huber, _, _, _ = D.fit_motion_robust(p[sel], t[sel], e[sel], s[sel])
    assert np.array_equal(info["theta"][0], huber) and not np.allclose(huber, plain, rtol=1e-6)
    assert np.array_equal(first(huber_delta=1e9)["theta"][0], plain)      # no detection beyond delta: the plain fit
    assert np.array_equal(first(huber_iter=0)["theta"][0], plain)         # no reweighting pass: the plain fit
    loose, _, _, _ = D.fit_motion_robust(p[sel], t[sel], e[sel], s[sel], delta=3.0)
    assert np.array_equal(first(huber_delta=3.0)["theta"][0], loose)


# ------------------------------------------------------------------ a design that does not determine the motion

MOTION = np.array([120.0, -80.0, 3.0, -2.0])           # (dra*, ddec, mu_a*, mu_d) [mas, mas/yr]
PSI_GOOD = np.linspace(0.0, np.pi, 20, endpoint=False) + 0.1
TAU_GOOD = np.linspace(-2.5, 2.5, 20)


def _exact(psi, tau):
    """Noise-free detections of MOTION, error 0.5 mas each: (eta, sigma)."""
    return D.motion_design(psi, tau) @ MOTION, np.full(psi.size, 0.5)


@pytest.mark.parametrize("psi, tau", [(PSI_GOOD, np.full(20, 1.0)),
                                      (PSI_GOOD, 2.0 + np.linspace(0.0, 1e-6, 20)),
                                      (np.full(20, 0.7), TAU_GOOD)],
                         ids=["one epoch", "a span of 1e-6 yr", "one scan direction"])
def test_a_motion_fit_refuses_a_design_that_does_not_determine_every_parameter(psi, tau):
    # before: a singular-matrix error, or a wrong fit with variances of 1e13 to 1e14 mas^2
    eta, sigma = _exact(psi, tau)
    with pytest.raises(ValueError, match="do not determine every parameter"):
        D.wls(D.motion_design(psi, tau), eta, 1.0 / sigma ** 2)
    for fit in (D.fit_motion, D.fit_mark_refit, D.fit_motion_robust):
        with pytest.raises(ValueError, match="do not determine every parameter"):
            fit(psi, tau, eta, sigma)


def test_the_robust_fit_refuses_a_sandwich_with_too_few_detections_inside_the_band():
    # 3 exact detections and 5 far outliers: 3 inside the Huber band for 4 parameters
    psi, tau = PSI_GOOD[::2][:8], TAU_GOOD[::2][:8]
    eta, sigma = _exact(psi, tau)
    eta[3:] += np.array([1.0, -1.0, 1.0, -1.0, 1.0]) * 1000.0
    with pytest.raises(ValueError, match="Huber band"):
        D.fit_motion_robust(psi, tau, eta, sigma)


def test_a_good_design_is_fitted_bit_for_bit_as_before():
    eta, sigma = _exact(PSI_GOOD, TAU_GOOD)
    X, w = D.motion_design(PSI_GOOD, TAU_GOOD), 1.0 / sigma ** 2
    theta, cov = D.wls(X, eta, w)
    Aw = X * w[:, None]
    cov_before = np.linalg.inv(X.T @ Aw)
    assert np.array_equal(cov, cov_before) and np.array_equal(theta, cov_before @ (Aw.T @ eta))


# ------------------------------------------------------------------ the static fit and the polar elements

def test_the_static_fit_is_the_weighted_least_squares_of_the_projection():
    p, _t, e, s, _f = _harvest(5, 150.0, -90.0, 0.0, 0.0)
    theta, cov = D.fit_detection_lsq(p, e, s)
    design = np.column_stack([np.sin(p), np.cos(p)])
    theta_hand, cov_hand = D.wls(design, e, 1.0 / s ** 2)
    np.testing.assert_allclose(theta, theta_hand, rtol=1e-12)
    np.testing.assert_allclose(cov, cov_hand, rtol=1e-12)
    assert D.fit_detection_lsq(p[:2], e[:2], s[:2]) == (None, None)


def test_the_polar_elements_follow_the_delta_method():
    theta, cov = np.array([3.0, 4.0]), np.array([[0.04, 0.01], [0.01, 0.09]])
    rho, sig_rho, pa, sig_pa = polar_elements(theta, cov)
    assert rho == 5.0 and pa == pytest.approx(np.degrees(np.arctan2(3.0, 4.0)))
    g_rho, g_phi = theta / 5.0, np.array([4.0, -3.0]) / 25.0
    assert sig_rho == pytest.approx(np.sqrt(g_rho @ cov @ g_rho))
    assert sig_pa == pytest.approx(np.degrees(np.sqrt(g_phi @ cov @ g_phi)))
    assert polar_elements(np.array([-1.0, 0.0]), np.eye(2))[2] == pytest.approx(270.0)
