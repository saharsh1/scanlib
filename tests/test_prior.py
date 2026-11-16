"""A Gaussian prior on chosen parameters of the joint fit, and its switch.

A source whose position is known from another measurement (a catalogue, a
deeper image) can carry a Gaussian prior ``prior=(index, mean, sigma)`` on
entries of ``theta``. The fit then minimises ``chi^2(theta) + sum_i ((theta_i
- m_i) / s_i)^2``, the covariance is the inverse of the Fisher matrix plus
``diag(1 / s_i^2)`` at those entries, and ``chi2_prior`` reports the prior's
term at the solution. Each test names what it rejects:
  1. ``prior=None`` (the default) moving any number of the fit (the switch is off);
  2. a prior that does not pull: a narrow one must pin its entry, a wide one
     must leave the free fit;
  3. a covariance or a ``chi2_prior`` that is not the stated formula, on the
     scalar and the per-sample noise path;
  4. a prior silently applied to the wrong entries.
"""
from __future__ import annotations

import numpy as np
import pytest

from scanlib import Instrument, fit_joint, joint_fisher, scene_positions_al, simulate_window
from scanlib.instrument import TOY_MARGINS
from scanlib.joint import joint_covariance

INST = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18, margins=TOY_MARGINS)
PSI = np.linspace(0.0, np.pi, 24, endpoint=False) + 0.1
TRUTH = np.array([110.0, 70.0])
FLUX = np.array([1000.0, 300.0])
SIGMA = 10.0
SEED = TRUTH + np.array([8.0, -6.0])


@pytest.fixture(scope="module")
def windows():
    rng = np.random.default_rng(20261111)
    return np.array([simulate_window(e, FLUX, INST, rng, SIGMA) for e in scene_positions_al(TRUTH, PSI)])


def test_no_prior_is_the_plain_fit_bit_for_bit(windows):
    plain = fit_joint(windows, PSI, SEED, INST, SIGMA)
    off = fit_joint(windows, PSI, SEED, INST, SIGMA, prior=None)
    for key in ("theta", "flux", "cov", "sigma_theta"):
        assert np.array_equal(plain[key], off[key]), key
    assert plain["chi2"] == off["chi2"] and off["chi2_prior"] == 0.0


def test_a_narrow_prior_pins_its_entry_and_a_wide_one_leaves_the_free_fit(windows):
    free = fit_joint(windows, PSI, SEED, INST, SIGMA)
    pinned = fit_joint(windows, PSI, SEED, INST, SIGMA, prior=([0], [100.0], [1e-4]))
    assert abs(pinned["theta"][0] - 100.0) < 1e-3
    assert pinned["sigma_theta"][0] < 2e-4                    # the prior's width rules
    wide = fit_joint(windows, PSI, SEED, INST, SIGMA, prior=([0, 1], [0.0, 0.0], [1e6, 1e6]))
    np.testing.assert_allclose(wide["theta"], free["theta"], atol=1e-3 * free["sigma_theta"].max())
    np.testing.assert_allclose(wide["sigma_theta"], free["sigma_theta"], rtol=1e-6)


@pytest.mark.parametrize("per_sample", [False, True], ids=["scalar sigma", "per-sample sigma"])
def test_the_covariance_and_the_prior_term_are_the_stated_formulas(windows, per_sample):
    sigma = np.full(windows.shape, SIGMA) if per_sample else SIGMA
    prior = (np.array([1]), np.array([75.0]), np.array([2.0]))
    fit = fit_joint(windows, PSI, SEED, INST, sigma, prior=prior)
    fisher = joint_fisher(fit["theta"], fit["flux"], PSI, INST, sigma)
    augmented = fisher.copy()
    augmented[1, 1] += 1.0 / 2.0 ** 2
    cov, sig = joint_covariance(fisher, 2, prior)
    assert np.array_equal(cov, fit["cov"]) and np.array_equal(sig, fit["sigma_theta"])
    np.testing.assert_allclose(cov, np.linalg.inv(augmented), rtol=1e-12)
    assert fit["chi2_prior"] == pytest.approx(((fit["theta"][1] - 75.0) / 2.0) ** 2, rel=1e-12)
    assert np.array_equal(joint_covariance(fisher, 2)[0], joint_covariance(fisher, 2, None)[0])


@pytest.mark.parametrize("prior, match", [
    (([2], [0.0], [1.0]), "index"),                         # theta has 2 entries
    (([0, 0], [0.0, 0.0], [1.0, 1.0]), "distinct"),
    (([0.5], [0.0], [1.0]), "integer"),
    (([0], [0.0, 1.0], [1.0]), "equal length"),
    (([0], [0.0], [0.0]), "sigma"),
    (([0], [np.nan], [1.0]), "mean"),
])
def test_a_prior_on_the_wrong_entries_is_refused(windows, prior, match):
    with pytest.raises(ValueError, match=match):
        fit_joint(windows, PSI, SEED, INST, SIGMA, prior=prior)
    with pytest.raises(ValueError, match=match):
        joint_covariance(np.eye(4), 2, prior)
