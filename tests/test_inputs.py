"""What the package accepts, and what it refuses with a clear error.

Each test names the wrong behaviour it rejects:
  1. a noise sigma of one value per window, shape (J,), read against the
     sample axis when the number of windows equals the number of samples
     (a silently wrong likelihood);
  2. an instrument changed after it is built, through the caller's array or
     its own (the grid and the dictionary then disagree), and an instrument
     that cannot be compared or hashed;
  3. a per-window instrument on a grid of another pitch (the LSF keeps the
     old pitch, so a unit flux renders as 2);
  4. inputs of mismatched lengths cut short by ``zip`` instead of refused;
  5. a peak returned for a map that holds no detection;
  6. bad values (no sigma, a non-positive sigma, a flux per source missing,
     an infinite noise setting, a NaN scan angle) that crash deep inside or
     pass silently;
  7. one name for the noise sigma and one place for the instrument: every
     routine that takes ``inst`` and a noise sigma names it ``sigma_noise``
     and takes it after ``inst``.
None of these changes a number the package computes on valid input.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect

import numpy as np
import pytest

from scanlib import (Instrument, backprojection_map, direction_weights, faststack,
                     filtered_backprojection, fit_joint, joint_fisher, loglike_profiled,
                     loglike_windows, map_peak, scene_positions_al,
                     simulate_window)
from scanlib import reduction as R
from scanlib.instrument import TOY_MARGINS
from scanlib.noise import NoiseModel
from scanlib.radon import ramp_filter

INST = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18, margins=TOY_MARGINS)
THETA, FLUX = np.array([120.0, 80.0]), np.array([1000.0, 300.0])


def _scene(J, sig, seed=1):
    psi = np.linspace(0.0, np.pi, J, endpoint=False)
    rng = np.random.default_rng(seed)
    eta = scene_positions_al(THETA, psi)
    w = np.array([simulate_window(e, FLUX, INST, rng, s) for e, s in zip(eta, sig)])
    return psi, w


# ------------------------------------------------------------------ 1. one sigma per window

@pytest.mark.parametrize("J", [18, 7], ids=["J equals N", "J differs from N"])
def test_one_sigma_per_window_is_read_per_window(J):
    sig = np.linspace(5.0, 40.0, J)
    psi, w = _scene(J, sig)
    per_sample = np.repeat(sig[:, None], INST.sample_grid_mas.size, axis=1)
    for form in (sig, sig[:, None]):
        assert loglike_windows(THETA, FLUX, w, psi, INST, form) == \
            loglike_windows(THETA, FLUX, w, psi, INST, per_sample)
        assert loglike_profiled(THETA, w, psi, INST, form) == \
            loglike_profiled(THETA, w, psi, INST, per_sample)
        assert np.array_equal(loglike_profiled(THETA, w, psi, INST, form, return_fluxes=True)[1],
                              loglike_profiled(THETA, w, psi, INST, per_sample, return_fluxes=True)[1])
        assert np.array_equal(joint_fisher(THETA, FLUX, psi, INST, form),
                              joint_fisher(THETA, FLUX, psi, INST, per_sample))


def test_a_sigma_of_another_shape_is_refused():
    psi, w = _scene(7, np.full(7, 10.0))
    for bad in (np.ones(5), np.ones((7, 3)), np.ones((2, 18))):
        with pytest.raises(ValueError, match="sigma"):
            loglike_windows(THETA, FLUX, w, psi, INST, bad)


# ------------------------------------------------------------------ 2. a frozen instrument

def test_the_instrument_owns_read_only_copies_of_its_arrays():
    grid = (np.arange(18) - 8.5) * 59.0
    inst = Instrument(grid, INST.atom_eta_mas, INST.lsf, INST.dlsf_deta, 100.0)
    grid[0] = -9999.0
    assert inst.sample_grid_mas[0] == -501.5
    for name in ("sample_grid_mas", "atom_eta_mas", "atoms"):
        with pytest.raises(ValueError):
            getattr(inst, name)[0] = 0.0


def test_an_instrument_compares_by_identity_and_hashes():
    other = Instrument.gaussian(margins=TOY_MARGINS)
    assert INST == INST and INST != other
    assert len({INST, other}) == 2
    again = dataclasses.replace(INST, margins=[12.0, 11.0])
    assert np.array_equal(again.atoms, INST.atoms) and again.margins.tolist() == [12.0, 11.0]


# ------------------------------------------------------------------ 3. one window's instrument

def test_for_window_refuses_a_grid_of_another_pitch_or_an_irregular_grid():
    for grid in ((np.arange(18) - 8.5) * 30.0, (np.arange(12) - 5.5) * 58.9,
                 np.array([0.0, 59.0, 177.0, 236.0])):
        with pytest.raises(ValueError, match="pitch|regular"):
            INST.for_window(grid)
    shifted = INST.for_window(INST.sample_grid_mas + 12345.0)
    assert shifted.design_columns([12345.0]).sum() == INST.design_columns([0.0]).sum()


# ------------------------------------------------------------------ 4. mismatched lengths

def test_the_images_refuse_windows_that_do_not_match_the_angles_or_the_grid():
    psi, w = _scene(10, np.full(10, 10.0))
    for fn in (filtered_backprojection, faststack):
        with pytest.raises(ValueError, match="window"):
            fn(psi, w[:7], INST)
        with pytest.raises(ValueError, match="sample"):
            fn(psi, w[:, :10], INST)
    with pytest.raises(ValueError, match="weight"):
        filtered_backprojection(psi, w, INST, weights=np.ones(7))
    with pytest.raises(ValueError, match="sample"):
        ramp_filter(w[0, :10], INST.sample_grid_mas)


def test_the_likelihood_and_the_fit_refuse_windows_that_do_not_match_the_angles_or_the_grid():
    # a transposed (N, J) array holds the same J N samples: without the check it fits wrongly
    psi, w = _scene(3, np.full(3, 10.0))
    calls = (lambda w_: loglike_windows(THETA, FLUX, w_, psi, INST, 10.0),
             lambda w_: loglike_profiled(THETA, w_, psi, INST, 10.0),
             lambda w_: fit_joint(w_, psi, THETA, INST, 10.0))
    for call in calls:
        with pytest.raises(ValueError, match="one window is required per scan angle"):
            call(w.T)
        with pytest.raises(ValueError, match="one window is required per scan angle"):
            call(w[:2])
        with pytest.raises(ValueError, match="18 samples"):
            call(w[:, :10])


def test_the_map_refuses_detections_of_mismatched_lengths():
    with pytest.raises(ValueError, match="detection"):
        backprojection_map(np.zeros(3), np.zeros(2), np.ones(3))
    with pytest.raises(ValueError, match="detection"):
        backprojection_map(np.zeros(3), np.zeros(3), np.ones(2))


# ------------------------------------------------------------------ 5. an empty map

def test_a_map_without_detections_has_no_peak():
    with pytest.raises(ValueError, match="no local maximum"):
        map_peak(np.array([]), np.array([]), np.array([]))


# ------------------------------------------------------------------ 6. bad values

def test_a_missing_or_bad_noise_sigma_is_named():
    psi, w = _scene(7, np.full(7, 10.0))
    with pytest.raises(ValueError, match="sigma_noise"):
        fit_joint(w, psi, THETA, INST, None)
    with pytest.raises(ValueError, match="sigma_noise"):
        R.ladder_window(w[0], INST, None)
    for bad in (0.0, -10.0, np.nan):
        with pytest.raises(ValueError, match="sigma"):
            R.sigma_eta([0.0, 120.0], [1000.0, 300.0], INST, bad)


@pytest.mark.parametrize("name, bad", [(n, v) for n in ("fwhm_mas", "pitch_mas")
                                       for v in (0.0, -50.0, np.nan, np.inf)]
                         + [("n_samples", v) for v in (0, 1, -3, 2.5, np.nan)])
def test_a_gaussian_instrument_refuses_a_non_physical_setting(name, bad):
    # a zero pitch raised ZeroDivisionError; a zero or NaN width built an instrument of NaN columns
    with pytest.raises(ValueError, match=name):
        Instrument.gaussian(**{name: bad})


def test_the_smallest_gaussian_instruments_are_accepted():
    assert Instrument.gaussian(n_samples=2).sample_grid_mas.size == 2
    assert Instrument.gaussian(n_samples=np.int64(18)).sample_grid_mas.size == 18


@pytest.mark.parametrize("bad", [0.0, -4.0, np.nan, np.inf])
def test_the_map_refuses_a_non_physical_sigma(bad):
    # a negative sigma made a negative map, which map_peak then read as "no local maximum"
    psi, eta, sigma = np.array([0.1, 1.0, 2.0]), np.zeros(3), np.array([4.0, bad, 4.0])
    with pytest.raises(ValueError, match="sigma"):
        backprojection_map(psi, eta, sigma)
    with pytest.raises(ValueError, match="sigma"):
        map_peak(psi, eta, sigma)


def test_one_flux_per_source_is_required():
    psi, _w = _scene(7, np.full(7, 10.0))
    with pytest.raises(ValueError, match="flux"):
        R.sigma_eta([0.0, 120.0], [1000.0], INST, 10.0)
    for flux in (FLUX[:1], np.r_[FLUX, 5.0]):
        with pytest.raises(ValueError, match="flux"):
            joint_fisher(THETA, flux, psi, INST, 10.0)


def test_a_noise_model_refuses_infinite_or_non_scalar_settings():
    for kw in ({"read_noise": np.inf}, {"background": np.inf}, {"gain": np.inf},
               {"gain": np.nan}, {"gain": np.array([1.0, 2.0])}):
        with pytest.raises(ValueError):
            NoiseModel(**kw)


def test_a_nan_scan_angle_is_refused():
    with pytest.raises(ValueError, match="psi"):
        direction_weights(np.array([0.1, np.nan, 1.0]))


# ------------------------------------------------------------------ 7. one name, one place

def test_the_noise_sigma_has_one_name_and_comes_after_the_instrument():
    for name in ("instrument", "radon", "joint", "reduction", "simulation", "search", "thresholds"):
        mod = importlib.import_module(f"scanlib.{name}")
        for fname, fn in vars(mod).items():
            if fname.startswith("_") or not inspect.isfunction(fn) or fn.__module__ != mod.__name__:
                continue
            params = list(inspect.signature(fn).parameters)
            if "inst" not in params:
                continue
            assert "sigma" not in params, f"{name}.{fname} names the noise 'sigma'"
            if "sigma_noise" in params:
                assert params.index("inst") < params.index("sigma_noise"), f"{name}.{fname}"


# ------------------------------------------------------------------ 8. the covariance's test of conditioning

@pytest.mark.parametrize("scale", [1e-3, 1e6])
def test_the_covariance_does_not_depend_on_the_flux_units(scale):
    """The same scene in other flux units (fluxes and noise scaled together)
    has the same position errors; the plain condition number grows as the
    units squared and used to make them infinite at 1e6."""
    from scanlib.joint import joint_covariance
    psi, _w = _scene(7, np.full(7, 10.0))
    _c, ref = joint_covariance(joint_fisher(THETA, FLUX, psi, INST, 10.0), 2)
    _c, got = joint_covariance(joint_fisher(THETA, scale * FLUX, psi, INST, scale * 10.0), 2)
    np.testing.assert_allclose(got, ref, rtol=1e-6)
    eta = np.array([0.0, 120.0])
    np.testing.assert_allclose(R.sigma_eta(eta, scale * FLUX, INST, scale * 10.0),
                               R.sigma_eta(eta, FLUX, INST, 10.0), rtol=1e-6)


def test_a_degenerate_or_uninformed_parameter_still_has_an_infinite_error():
    from scanlib.joint import joint_covariance
    one = np.array([0.3])                                    # one scan angle: a line, not a point
    _c, s = joint_covariance(joint_fisher(THETA, FLUX, one, INST, 10.0), 2)
    assert np.all(np.isinf(s))
    assert np.all(np.isinf(R.sigma_eta([0.0, 120.0], [1000.0, 0.0], INST, 10.0)))   # no flux


# ------------------------------------------------------------------ 9. trial positions that cover the window

def test_the_reduction_refuses_trial_positions_that_miss_part_of_the_window():
    wide = Instrument.gaussian(n_samples=30)                 # samples to +-855.5 mas, trial to +-530
    samples = np.zeros(30)
    samples[25] = 100.0
    for call in (lambda: R.active_clusters(samples, wide),
                 lambda: R.fit_k(samples, 1, wide),
                 lambda: R.ladder_window(samples, dataclasses.replace(wide, margins=TOY_MARGINS), 10.0)):
        with pytest.raises(ValueError, match="for_window"):
            call()
    own = wide.for_window(wide.sample_grid_mas)              # trial positions over the window +- FWHM
    assert R.fit_k(samples, 1, own)[0].size == 1
