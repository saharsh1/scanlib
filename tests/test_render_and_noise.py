"""The rendering additions of ``scanlib.instrument`` and the noise module.

Two parts. First, the reference cases (``render/*``, ``noise/*`` of
``tests/reference/reference_cases.npz``) on the toy instrument
``Instrument.gaussian(100, 59, 18, margins=TOY_MARGINS)``: the fine model
curve, the zero points of 40 windows, the S/N bridge at a peak S/N of 50,
one noisy window and the per-sample sigma of Fig. 4's noise model, each
equal to the stored array to the last bit. Second, the contract of each
function: the forms ``sample_sigma``
reads and the ones it refuses, the counts-to-sigma recipe, the frozen
``NoiseModel``, the zero points' one uniform draw, and the bridge that
reproduces the historical sigmas of the peak convention bit for bit.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from scanlib.instrument import TOY_MARGINS, Instrument, render_curve, render_window
from scanlib.simulation import (PRIMARY_FLUX, random_zero_points, simulate_window,
                                snr_from_peak_snr)
from scanlib import noise
from scanlib.noise import NoiseModel, sample_sigma, sigma_from_counts

REFERENCE = Path(__file__).resolve().parent / "reference" / "reference_cases.npz"
INST = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18, margins=TOY_MARGINS)
PITCH = 59.0

# the sigmas of the peak convention (sigma = brightest sample of an isolated source of flux
# 1000 at eta = 0 on the toy grid, over the peak S/N)
OLD_SIGMA = {20.0: 21.772147322158506, 50.0: 8.708858928863402, 100.0: 4.354429464431701,
             300.0: 1.4514764881439004, 1000.0: 0.4354429464431701}


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


# ------------------------------------------------------------------ the export, bit for bit

def test_the_rendered_window_and_the_fine_curve(ref):
    eta, flux = ref["render/eta"], ref["render/flux"]
    assert np.array_equal(render_window(eta, flux, INST), ref["render/samples"])
    assert np.array_equal(render_curve(ref["render/eta_fine"], eta, flux, INST),
                          ref["render/curve"])


def test_the_noisy_window_is_one_draw_at_sigma_flux_over_snr(ref):
    sigma = PRIMARY_FLUX / 100.0
    assert sigma == float(ref["render/noisy_sigma"])
    noisy = simulate_window(ref["render/eta"], ref["render/flux"], INST,
                            np.random.default_rng(20261109), sigma)
    assert np.array_equal(noisy, ref["render/noisy_samples"])


def test_the_zero_points_and_the_bridge(ref):
    z = random_zero_points(np.random.default_rng(20261110), 40, INST)
    assert np.array_equal(z, ref["render/zero_points"])
    assert snr_from_peak_snr(50.0, INST) == float(ref["render/snr_from_peak_snr_50"])


def test_the_per_sample_sigma_of_fig4s_noise_model(ref):
    assert np.array_equal(ref["noise/counts"], ref["render/samples"])
    model = NoiseModel(gain=float(ref["noise/gain"]), read_noise=float(ref["noise/read_noise"]),
                       background=float(ref["noise/background"]))
    assert np.array_equal(model.sigma(ref["noise/counts"]), ref["noise/sigma"])


# ------------------------------------------------------------------ sample_sigma

RNG = np.random.default_rng(5)


def _windows(lengths):
    return [RNG.normal(size=n) for n in lengths]


def test_a_scalar_sigma_for_one_window_and_for_a_list():
    s, is_scalar = sample_sigma(2.5, np.zeros(7))
    assert is_scalar and s.shape == (7,) and np.all(s == 2.5)
    for sig in (2.5, np.array(2.5)):
        s, is_scalar = sample_sigma(sig, _windows([4, 6]))
        assert is_scalar and [len(x) for x in s] == [4, 6]
        assert all(np.all(x == 2.5) for x in s)


def test_a_sigma_per_window_and_per_sample():
    s, is_scalar = sample_sigma(np.array([1.0, 2.0, 3.0]), _windows([4, 5, 6]))
    assert not is_scalar
    for x, v, n in zip(s, (1.0, 2.0, 3.0), (4, 5, 6)):
        assert x.shape == (n,) and np.all(x == v)
    v = np.linspace(1.0, 2.0, 7)
    s, is_scalar = sample_sigma(v, np.zeros(7))
    assert not is_scalar and np.array_equal(s, v)
    m = RNG.uniform(1.0, 2.0, size=(3, 5))
    s, _ = sample_sigma(m, _windows([5, 5, 5]))
    assert all(np.array_equal(a, b) for a, b in zip(s, m))
    rag = [RNG.uniform(1, 2, size=n) for n in (4, 7, 2)]
    s, _ = sample_sigma(rag, _windows([4, 7, 2]))
    assert all(np.array_equal(a, b) for a, b in zip(s, rag))


def test_j_equal_to_n_is_read_per_window():
    v = np.array([1.0, 2.0, 3.0])
    s, _ = sample_sigma(v, _windows([3, 3, 3]))
    for x, c in zip(s, v):
        assert np.all(x == c)


def test_a_wrong_shape_or_a_bad_value_is_refused():
    ws = _windows([4, 5, 6])
    for bad in (np.ones(2), np.ones((3, 4)), np.ones((2, 3, 4)), [np.ones(4), np.ones(5)],
                [np.ones(4), np.ones(5), np.ones(7)]):
        with pytest.raises(ValueError, match="sigma must be a scalar"):
            sample_sigma(bad, ws)
    with pytest.raises(ValueError, match="sigma must be a scalar"):
        sample_sigma(np.ones(6), np.zeros(7))
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="sigma must be positive and finite"):
            sample_sigma(bad, _windows([4, 5]))
        with pytest.raises(ValueError, match="sigma must be positive and finite"):
            sample_sigma(np.array([1.0, bad]), _windows([4, 5]))


def test_sample_sigma_returns_float_copies():
    v = np.array([1, 2, 3])
    s, _ = sample_sigma(v, np.zeros(3))
    assert s.dtype == np.float64
    s[0] = 99.0
    assert v[0] == 1
    m = np.full((2, 3), 2.0)
    out, _ = sample_sigma(m, _windows([3, 3]))
    out[0][0] = 99.0
    assert m[0, 0] == 2.0


# ------------------------------------------------------------------ the counts-to-sigma recipe

def test_the_recipe_sums_shot_background_and_read_variances():
    s = sigma_from_counts(np.array([1000.0, 100.0, 10.0]), 1.0, 4.0, 20.0)
    np.testing.assert_allclose(s ** 2, [1036.0, 136.0, 46.0], rtol=1e-14)
    c = np.array([[1.0, 4.0], [9.0, 16.0]])
    s = sigma_from_counts(c)
    assert s.dtype == np.float64 and s.shape == (2, 2)
    np.testing.assert_allclose(s, np.sqrt(c), rtol=1e-14)
    c = np.array([100.0, 400.0])
    np.testing.assert_allclose(sigma_from_counts(c, gain=4.0) ** 2, c / 4.0, rtol=1e-14)


def test_negative_counts_are_clipped_to_the_floor():
    s = sigma_from_counts(np.array([-50.0, 0.0]), gain=2.0, read_noise=3.0, background=8.0)
    np.testing.assert_allclose(s, np.sqrt(8.0 / 2.0 + 9.0), rtol=1e-14)


def test_the_recipe_and_the_noise_model_refuse_bad_settings():
    for kw in ({"gain": 0.0}, {"gain": -1.0}, {"read_noise": -1.0}, {"background": -1.0}):
        with pytest.raises(ValueError):
            sigma_from_counts(np.ones(3), **kw)
        with pytest.raises(ValueError):
            NoiseModel(**kw)


def test_the_noise_model_is_the_recipe_frozen():
    m = NoiseModel(gain=2.0, read_noise=4.0, background=20.0)
    c = np.array([1000.0, 100.0, -5.0])
    assert np.array_equal(m.sigma(c), sigma_from_counts(c, 2.0, 4.0, 20.0))
    assert NoiseModel().gain == 1.0 and NoiseModel().read_noise == 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.gain = 3.0


# ------------------------------------------------------------------ zero points and the bridge

def test_the_zero_points_are_one_uniform_draw_times_the_pitch():
    z = random_zero_points(np.random.default_rng(11), 1000, INST)
    assert np.array_equal(z, np.random.default_rng(11).uniform(-0.5, 0.5, 1000) * PITCH)
    assert z.min() >= -PITCH / 2 and z.max() < PITCH / 2
    wide = Instrument.gaussian(fwhm_mas=200.0, pitch_mas=118.0, n_samples=18)
    assert np.array_equal(random_zero_points(np.random.default_rng(11), 1000, wide), 2.0 * z)


@pytest.mark.parametrize("peak_snr, sigma", sorted(OLD_SIGMA.items()))
def test_the_bridge_gives_the_historical_sigma_bit_for_bit(peak_snr, sigma):
    assert PRIMARY_FLUX / snr_from_peak_snr(peak_snr, INST) == sigma


def test_the_bridge_takes_the_instrument_of_the_old_regime():
    wide = Instrument.gaussian(fwhm_mas=102.0)
    old_sigma = float(render_window([0.0], [PRIMARY_FLUX], wide).max()) / 300.0
    assert PRIMARY_FLUX / snr_from_peak_snr(300.0, wide) == old_sigma


# ------------------------------------------------------------------ the Gaia-like model

@pytest.mark.parametrize("G", [14.0, 16.0, 18.0])
@pytest.mark.parametrize("background", [10.0, 100.0])
def test_the_gaia_like_model_is_the_hand_written_one_bit_for_bit(G, background):
    """The expressions the notebooks wrote by hand (Figure 4, Figure 6, the quickstart)."""
    n_e = 9 * 19_000.0 * 4.42 * 10.0 ** (-0.4 * (G - 15.0))
    gain = n_e / PRIMARY_FLUX
    by_hand = NoiseModel(gain=gain, read_noise=np.sqrt(9) * 4.3 / gain, background=9 * background / gain)
    assert noise.gaia_like(G, PRIMARY_FLUX, background=background) == by_hand
    assert noise.gaia_electrons(G) == n_e


def test_the_gaia_like_variance_in_electrons_is_signal_plus_nine_times_background_and_read():
    """sigma^2 [e-^2] = s + 9 (b + r^2), s the noise-free sample in electrons (the paper's noise equation)."""
    b = 10.0
    model = noise.gaia_like(16.0, PRIMARY_FLUX, background=b)
    counts = render_window(np.array([0.0, 75.0]), np.array([PRIMARY_FLUX, 0.2 * PRIMARY_FLUX]), INST)
    variance_e = (model.sigma(counts) * model.gain) ** 2
    np.testing.assert_allclose(variance_e, counts * model.gain + 9 * (b + 4.3 ** 2), rtol=1e-13)


def test_the_gaia_like_defaults_and_numbers():
    assert noise.gaia_like(16.0, PRIMARY_FLUX) == noise.gaia_like(16.0, PRIMARY_FLUX, background=10.0)
    assert round(noise.gaia_electrons(16.0), -3) == 301_000.0           # 3.0e5 e- per transit at G = 16
    assert noise.gaia_electrons(15.0) == noise.GAIA_N_CCD * noise.GAIA_ZERO_POINT_E_PER_S_G15 * noise.GAIA_T_CCD_S
    assert (noise.GAIA_N_CCD, noise.GAIA_ZERO_POINT_E_PER_S_G15, noise.GAIA_T_CCD_S,
            noise.GAIA_READ_NOISE_E_PER_CCD) == (9, 19_000.0, 4.42, 4.3)
    # data already in electrons: the flux of the source is its electrons, and the gain is one
    in_electrons = noise.gaia_like(16.0, noise.gaia_electrons(16.0))
    assert in_electrons.gain == 1.0 and in_electrons.background == 90.0
    assert in_electrons.read_noise == np.sqrt(9) * 4.3


@pytest.mark.parametrize("kw", [dict(G=np.nan, flux=1000.0), dict(G=16.0, flux=0.0), dict(G=16.0, flux=-1.0),
                                dict(G=16.0, flux=np.inf), dict(G=[16.0, 17.0], flux=1000.0),
                                dict(G=16.0, flux=1000.0, background=-1.0)])
def test_the_gaia_like_model_refuses_bad_settings(kw):
    with pytest.raises(ValueError):
        noise.gaia_like(**kw)


@pytest.mark.parametrize("G", [3.0, 8.0, 12.0, 14.0, 16.0, 18.0, 20.7, 23.0])
def test_the_gaia_like_model_holds_at_any_magnitude(G):
    """At every magnitude: the electrons follow Pogson's law, the variance is the paper's equation, the
    sigma is finite and positive on every sample, and the units of the data do not matter."""
    assert np.isclose(noise.gaia_electrons(G) / noise.gaia_electrons(G + 5.0), 100.0, rtol=1e-12)
    b = 100.0
    counts = render_window(np.array([0.0, 150.0]), np.array([PRIMARY_FLUX, 0.01 * PRIMARY_FLUX]), INST)
    model = noise.gaia_like(G, PRIMARY_FLUX, background=b)
    sigma = model.sigma(counts)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0.0)
    np.testing.assert_allclose((sigma * model.gain) ** 2, counts * model.gain + 9 * (b + 4.3 ** 2), rtol=1e-12)
    # the same scene in electrons (gain one): the same sigma in electrons, so the same signal-to-noise
    n_e = noise.gaia_electrons(G)
    in_electrons = noise.gaia_like(G, n_e, background=b)
    np.testing.assert_allclose(in_electrons.sigma(counts * n_e / PRIMARY_FLUX), sigma * model.gain, rtol=1e-12)
