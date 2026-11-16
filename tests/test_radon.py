"""The ramp filter of the inverse Radon transform: band-limited, whatever the refinement.

``ramp_filter`` filters a window's density on ``n_pad`` points and returns it
on a grid ``upsample`` times finer. The filtered projection is a trigonometric
interpolant of the padded spectrum: at the samples it must not depend on the
refinement, and everywhere it must equal the direct sum of that spectrum, in
which the Nyquist term of an even ``n_pad`` is one cosine. Before the
correction the Nyquist coefficient was copied whole into the longer inverse
transform, where it became a positive frequency with a negative partner, and
was counted twice.
"""
from __future__ import annotations

import numpy as np
import pytest

from scanlib.instrument import Instrument, render_window
from scanlib.radon import ramp_filter

INST = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18)
GRID = INST.sample_grid_mas


def _direct(samples, grid, pad_factor, t):
    """The filtered projection at positions ``t`` [mas] by the direct sum of the padded spectrum:
    (1/n) [2 sum_{0<k<n/2} |kappa_k| Re(X_k e^{2 pi i kappa_k (t - t0)}) + |kappa_{n/2}| X_{n/2} cos(pi (t - t0)/p)],
    the last term only for an even n; X_k the DFT of the density on n = pad_factor N points."""
    p = grid[1] - grid[0]
    n = pad_factor * grid.size
    x = np.zeros(n)
    x[:grid.size] = samples / p
    m = np.arange(n)
    out = np.zeros_like(t)
    for k in range(1, (n + 1) // 2):
        X = np.sum(x * np.exp(-2j * np.pi * k * m / n))
        kappa = k / (n * p)
        out += 2.0 * kappa * np.real(X * np.exp(2j * np.pi * kappa * (t - grid[0])))
    if n % 2 == 0:
        X = np.sum(x * np.cos(np.pi * m))
        out += (0.5 / p) * X.real * np.cos(np.pi * (t - grid[0]) / p)
    return out / n


SAMPLES = {
    "random": np.random.default_rng(1).normal(size=18),
    "a pair": render_window([0.0, 150.0], [1000.0, 300.0], INST),
}


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("pad_factor", [16, 1, 3], ids=["n_pad even (default)", "n_pad 18", "n_pad 54"])
def test_the_filtered_samples_do_not_depend_on_the_refinement(name, pad_factor):
    samples = SAMPLES[name]
    _, coarse = ramp_filter(samples, GRID, pad_factor=pad_factor, upsample=1)
    scale = np.abs(coarse).max()
    for upsample in (2, 3, 8):
        _, fine = ramp_filter(samples, GRID, pad_factor=pad_factor, upsample=upsample)
        np.testing.assert_allclose(fine[::upsample], coarse, rtol=0.0, atol=1e-12 * scale)


@pytest.mark.parametrize("pad_factor", [16, 1])
def test_an_odd_padded_length_has_no_nyquist_term_to_split(pad_factor):
    grid = (np.arange(7) - 3.0) * 59.0                     # N = 7: n_pad 112 even, 7 odd
    samples = np.random.default_rng(2).normal(size=7)
    _, coarse = ramp_filter(samples, grid, pad_factor=pad_factor, upsample=1)
    _, fine = ramp_filter(samples, grid, pad_factor=pad_factor, upsample=4)
    np.testing.assert_allclose(fine[::4], coarse, rtol=0.0, atol=1e-12 * np.abs(coarse).max())


def test_the_pure_nyquist_mode_is_not_doubled():
    # the audit's case: +-1 alternating on 4 samples, no padding
    samples, grid = np.array([1.0, -1.0, 1.0, -1.0]), np.arange(4.0)
    _, coarse = ramp_filter(samples, grid, pad_factor=1, upsample=1)
    _, fine = ramp_filter(samples, grid, pad_factor=1, upsample=8)
    np.testing.assert_allclose(coarse, [0.5, -0.5, 0.5, -0.5], atol=1e-15)
    np.testing.assert_allclose(fine[::8], coarse, atol=1e-15)


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("pad_factor, upsample", [(16, 8), (1, 8), (3, 2), (16, 1)])
def test_the_filtered_projection_is_the_direct_sum_of_its_spectrum(name, pad_factor, upsample):
    samples = SAMPLES[name]
    t, fine = ramp_filter(samples, GRID, pad_factor=pad_factor, upsample=upsample)
    expected = _direct(samples, GRID, pad_factor, t)
    np.testing.assert_allclose(fine, expected, rtol=0.0, atol=1e-11 * np.abs(expected).max())
