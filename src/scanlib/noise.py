"""The noise on window samples: reading a sigma, and the counts-to-sigma recipe.

THE NOISE MODEL. The noise on a window sample is Gaussian, independent from
sample to sample, with a known standard deviation sigma in the units of the
data (flux per sample). Sigma may be one number for all samples, one per
window, or one per sample; :func:`sample_sigma` reads each form.

THE RECIPE. When sigma follows from the signal, it is the sum of three
independent variances,

    sigma^2 = (max(counts, 0) + background) / gain + read_noise^2,

with ``counts`` the signal in data units, ``gain`` the electrons per data
unit (1 when the data are electrons), ``background`` the sky level in data
units and ``read_noise`` in data units. The first term is the shot noise of
signal and background (Poisson in electrons), the last the read noise. The
functions return sigma, not its square: an independent noise term the
recipe does not hold (dark current, a level that depends on the position)
adds its variance, ``sigma_total = sqrt(sigma^2 + sigma_extra^2)``, not its
sigma. A bias is an offset of the mean, not a noise term: it is removed
from the data, not added to sigma.

Assumptions: the Gaussian approximation to Poisson noise, valid for large
counts; the signal clipped at zero, so that the variance stays positive on a
noisy negative sample.

THE UNITS OF THE THREE SETTINGS. ``background`` and ``read_noise`` are in
data units, so they depend on ``gain``: with the background ``B_e`` and the
read noise ``R_e`` in electrons, ``background = B_e / gain`` and
``read_noise = R_e / gain``, and a change of the gain changes all three.
In electrons the recipe reads ``sigma_e^2 = max(s_e, 0) + B_e + R_e^2``,
with ``s_e = gain * counts`` and ``sigma_e = gain * sigma``.
:func:`gaia_like` takes its numbers in electrons and makes the three
settings together.

A GAIA-LIKE MODEL. :func:`gaia_like` gives the settings for a source of
magnitude G whose transit sums the crossings of ``GAIA_N_CCD`` CCDs into
one window:

    N_e(G) = GAIA_N_CCD * GAIA_ZERO_POINT_E_PER_S_G15 * GAIA_T_CCD_S
             * 10^(-0.4 (G - 15))                      [electrons per transit]
    sigma_e^2 = s_e + GAIA_N_CCD * (b + r^2),

with ``b`` the background in electrons per sample per CCD and ``r =
GAIA_READ_NOISE_E_PER_CCD``. It sets sample variances of the right size;
it is not a calibration of Gaia.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The count rate of a G = 15 source [electrons per second]:
#: 10**(0.4 * (25.6874 - 15)) = 18 835, rounded to 19 000. The G zero point,
#: 25.6874 +/- 0.0028 mag (Vega system, fluxes in electrons per second), is
#: from Riello et al. (2021, A&A 649, A3), Table 3.
GAIA_ZERO_POINT_E_PER_S_G15 = 19_000.0
#: The time of one CCD crossing [s], without a gate (Gaia Collaboration,
#: Prusti et al. 2016, A&A 595, A1, Sect. 3.3.2).
GAIA_T_CCD_S = 4.42
#: The CCD crossings of the astrometric field in one transit, AF1 to AF9
#: (Lindegren et al. 2018, A&A 616, A2, Sect. 5.5). Gaia reads one window per
#: CCD; the sum of the nine into one window is the paper's model.
GAIA_N_CCD = 9
#: The read noise of one CCD crossing [electrons per sample]: the measured
#: total detection noise of AF2-9, 4.326 +/- 0.648 electrons (Hambly et al.
#: 2018, A&A 616, A15, Table 1). AF1 is noisier (8.556 +/- 0.438 electrons);
#: the model uses the AF2-9 value for all nine crossings.
GAIA_READ_NOISE_E_PER_CCD = 4.3

_FORM = ("sigma must be a scalar, shape (J,) (one per window), shape (J, N) or a list of "
         "J arrays of lengths N_j (one per sample)")


def _positive(a):
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("sigma must be positive and finite")


def sample_sigma(sigma, windows):
    """The noise sigma of each sample of one window or of a list of windows.

    Parameters
    ----------
    sigma : float, array or list of arrays
        The noise sigma, flux per sample, positive and finite. A scalar: the
        same on every sample. For a list of J windows: shape (J,), one per
        window; shape (J, N) when every window has N samples, or a list of J
        arrays of lengths N_j, one per sample. For one window of N samples:
        shape (N,), one per sample. A (J,) array for J windows of N = J
        samples is read per window; pass (J, N) for one sigma per sample.
    windows : array or list of arrays
        One window (a 1-D array) or a list of J windows (1-D arrays of
        lengths N_j).

    Returns
    -------
    sigmas : array, shape (N,), or list of J arrays, shape (N_j,)
        float64 copies, in the form of ``windows``.
    is_scalar : bool
        True exactly when ``sigma`` is a scalar.

    Raises
    ------
    ValueError
        If a value is not positive and finite, or if the shape does not match
        the windows. Nothing else is broadcast.
    """
    one = isinstance(windows, np.ndarray) and windows.ndim == 1
    lens = [len(windows)] if one else [len(w) for w in windows]
    J = len(lens)
    if isinstance(sigma, (list, tuple)) and any(np.ndim(x) >= 1 for x in sigma):
        rows = [np.array(x, dtype=float) for x in sigma]
        if one or len(rows) != J or any(r.shape != (n,) for r, n in zip(rows, lens)):
            raise ValueError(_FORM + f"; got a list of {len(rows)} arrays")
        for r in rows:
            _positive(r)
        return rows, False
    s = np.array(sigma, dtype=float)
    _positive(s)
    if s.ndim == 0:
        out = [np.full(n, float(s)) for n in lens]
        return (out[0] if one else out), True
    if one:
        if s.shape != (lens[0],):
            raise ValueError(f"sigma must be a scalar or shape ({lens[0]},) for one "
                             f"window, got {s.shape}")
        return s, False
    if s.shape == (J,):
        return [np.full(n, v) for n, v in zip(lens, s)], False
    if s.ndim == 2 and s.shape[0] == J and all(n == s.shape[1] for n in lens):
        return [row.copy() for row in s], False
    raise ValueError(_FORM + f"; got shape {s.shape} for J = {J}")


def _check(gain, read_noise, background):
    for name, value in (("gain", gain), ("read_noise", read_noise), ("background", background)):
        if np.ndim(value) != 0 or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite scalar, not {value!r}")
    if not gain > 0.0:
        raise ValueError("gain must be positive")
    if not read_noise >= 0.0:
        raise ValueError("read_noise must be non-negative")
    if not background >= 0.0:
        raise ValueError("background must be non-negative")


def sigma_from_counts(counts, gain=1.0, read_noise=0.0, background=0.0):
    """The noise sigma of each sample from its signal: shot, background and read noise.

    ``sigma = sqrt((max(counts, 0) + background) / gain + read_noise**2)``,
    element by element.

    Parameters
    ----------
    counts : array_like
        The signal in the data's units (flux per sample), any shape.
    gain : float
        Electrons per data unit, positive; 1 when the data are electrons.
    read_noise : float
        The read noise in data units, non-negative.
    background : float
        The background level in data units, non-negative.

    Returns
    -------
    array
        float64, the shape of ``counts``, in data units. A negative count is
        clipped at zero, so sigma never falls below the floor
        ``sqrt(background / gain + read_noise**2)``. With the defaults it is
        pure Poisson noise in the data's units.

    Raises
    ------
    ValueError
        If a setting is not a finite scalar, ``gain`` is not positive, or
        ``read_noise`` or ``background`` is negative.
    """
    _check(gain, read_noise, background)
    c = np.maximum(np.asarray(counts, dtype=float), 0.0)
    return np.sqrt((c + background) / gain + read_noise ** 2)


@dataclass(frozen=True)
class NoiseModel:
    """The recipe of :func:`sigma_from_counts` as one frozen setting.

    A simulation that draws its noise with the sigma it reports holds one of
    these, and a fit that reads the same object weighs the samples with the
    same sigma.

    Parameters
    ----------
    gain : float
        Electrons per data unit, positive.
    read_noise : float
        The read noise in data units, non-negative.
    background : float
        The background level in data units, non-negative.

    Raises
    ------
    ValueError
        As :func:`sigma_from_counts`, when the model is built.
    """

    gain: float = 1.0
    read_noise: float = 0.0
    background: float = 0.0

    def __post_init__(self):
        _check(self.gain, self.read_noise, self.background)

    def sigma(self, counts):
        """The sigma of each sample of ``counts`` [data units], as :func:`sigma_from_counts`."""
        return sigma_from_counts(counts, self.gain, self.read_noise, self.background)


def gaia_electrons(G):
    """The electrons a source of magnitude ``G`` gives in one transit.

    ``N_e = GAIA_N_CCD * GAIA_ZERO_POINT_E_PER_S_G15 * GAIA_T_CCD_S *
    10**(-0.4 * (G - 15))``: all the light of the nine CCD crossings, none
    lost outside the window.

    Parameters
    ----------
    G : float
        The source's G magnitude.

    Returns
    -------
    float
        Electrons per transit; 3.0e5 at G = 16.

    Raises
    ------
    ValueError
        If ``G`` is not a finite scalar.
    """
    if np.ndim(G) != 0 or not np.isfinite(G):
        raise ValueError(f"G must be a finite scalar, not {G!r}")
    return GAIA_N_CCD * GAIA_ZERO_POINT_E_PER_S_G15 * GAIA_T_CCD_S * 10.0 ** (-0.4 * (G - 15.0))


def gaia_like(G, flux, background=10.0):
    """The Gaia-like noise model of a source of magnitude ``G`` and total flux ``flux``.

    THE UNITS. The windows may be in any flux units: ``flux`` says how large
    the source of magnitude ``G`` is in them, and so how many electrons one
    data unit is, ``gain = gaia_electrons(G) / flux``. The noise of a sample
    is computed in electrons and returned in the data's units:

        s_e     = gain * counts                    (the sample, in electrons)
        sigma_e = sqrt(s_e + 9 * (b + r^2))        (its noise, in electrons)
        sigma   = sigma_e / gain                   (the same noise, in data units)

    The signal-to-noise ratio of a sample, ``counts / sigma = s_e /
    sigma_e``, does not depend on the units; the magnitude sets it. Example,
    G = 16 and ``flux = 1000``: 3.0e5 electrons per transit, 301 electrons
    per data unit; the primary's peak sample of 550 units is 1.65e5
    electrons with sigma_e = 407, so sigma = 1.35 units (S/N 406); an empty
    sample has sigma_e = 16.0, so sigma = 0.053 units.

    In electrons the variance of a sample is

        sigma_e^2 = s_e + GAIA_N_CCD * (background + GAIA_READ_NOISE_E_PER_CCD^2),

    with ``s_e`` the noise-free sample: the shot noise of the signal, and the
    shot noise of the background and the read noise of each of the
    ``GAIA_N_CCD`` crossings summed into the window. The background level
    itself is taken as subtracted from the data. The model is returned in
    data units: ``read_noise = sqrt(GAIA_N_CCD) * GAIA_READ_NOISE_E_PER_CCD /
    gain`` and ``background = GAIA_N_CCD * background / gain``.

    The formula holds at any magnitude. The instrument it imitates does not:
    for a source brighter than about G = 12 Gaia shortens the exposure
    (gates) and the samples saturate, so there the model gives more
    electrons, and less noise, than Gaia has.

    Parameters
    ----------
    G : float
        The G magnitude of the reference source (the primary).
    flux : float
        The total flux of that source in the data's units, positive (its
        samples summed over a window that holds it); every other source
        scales with it. For data in electrons, ``gaia_electrons(G)``, which
        gives a gain of one.
    background : float
        The background in electrons per sample per CCD crossing,
        non-negative: 10 for a quiet sky.

    Returns
    -------
    NoiseModel
        The three settings in data units; ``.sigma(counts)`` is the sigma of
        each sample [data units], and ``.gain`` the electrons per data unit.

    Raises
    ------
    ValueError
        If ``G`` is not a finite scalar, ``flux`` is not a positive finite
        scalar, or ``background`` is negative.
    """
    if np.ndim(flux) != 0 or not np.isfinite(flux) or not flux > 0.0:
        raise ValueError(f"flux must be a positive finite scalar, not {flux!r}")
    gain = gaia_electrons(G) / flux
    return NoiseModel(gain=gain, read_noise=np.sqrt(GAIA_N_CCD) * GAIA_READ_NOISE_E_PER_CCD / gain,
                      background=GAIA_N_CCD * background / gain)
