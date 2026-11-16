"""Images of the windows: the inverse Radon transform and faststack.

A window is a projection of the scene onto its scan direction, blurred by
the LSF and sampled. The set of windows over many scan angles is a
sinogram, and two classical images can be made of it without a
point-source model:

- :func:`filtered_backprojection`, the inverse Radon transform: each
  window is ramp-filtered (:func:`ramp_filter`) and laid back along its own
  sky line, weighted by its share of the half-turn of directions
  (:func:`direction_weights`). The exact inverse of LSF-blurred projections
  is the scene convolved with the radially symmetric 2-D kernel whose
  projection on every direction is the 1-D LSF (for a Gaussian LSF of
  width sigma, the 2-D Gaussian of the same sigma), in flux per mas^2.
- :func:`faststack`, the robust stack of Harrison (2011, Exp. Astron. 31,
  1, Eqs. 1-4): every sky pixel collects the sample of each window whose
  footprint covers it, and a trimmed, down-weighted mean suppresses the
  samples the primary's light fell into.

Both are pictures: nothing is measured on them. The measurement is the
joint fit of :mod:`scanlib.joint`; the detector the method reads is the
back-projection map of :mod:`scanlib.backprojection`.

Conventions. ``psi`` [rad] counter-clockwise from north; a sky pixel at
``(dra*, ddec)`` [mas] has along-scan coordinate ``eta = dra* sin(psi) +
ddec cos(psi)`` in a window of angle ``psi``, and ``eta - eta0_j`` in
window j's own frame, ``eta0_j`` its zero point (``None``: 0). ``windows``
is the (J, N) array of samples, flux per sample, on the instrument's sample
grid ``grid = inst.sample_grid_mas`` [mas]. Images are returned on the square grid ``axis`` =
``arange(-half, half + 1e-9, step)`` [mas], rows = ddec, columns = dra*,
both increasing with index (``imshow(origin='lower')``; east-left is a
display choice, never baked into the arrays).
"""
from __future__ import annotations

import numpy as np

from .instrument import _pitch, project_al

# The window is undersampled at the LSF scale, so the ramp-filtered
# projection is evaluated on a grid RAMP_UPSAMPLE times finer by band-limited
# (sinc) interpolation of its zero-padded spectrum, never by linear
# interpolation between the samples.
RAMP_PAD_FACTOR = 16         # zero-padding of the spectrum, in window lengths
RAMP_UPSAMPLE = 8            # the fine grid of the filtered projection
DIRECTION_TIE_RAD = 1e-12    # directions closer than this, modulo pi, are one direction


def ramp_filter(samples, grid, pad_factor=RAMP_PAD_FACTOR, upsample=RAMP_UPSAMPLE):
    """The ramp-filtered projection of one window on a fine along-scan grid.

    The projection density ``samples / p`` [flux/mas], ``p`` the pitch, is
    Fourier transformed on ``pad_factor N`` points, its spectrum multiplied
    by the ramp ``|kappa|`` [cycles/mas], and the result inverse-transformed
    on a grid ``upsample`` times finer than the samples. With ``d kappa = 1 /
    (pad_factor N p)`` and numpy's ``1/n`` in ``irfft`` the Riemann sum
    reduces to ``upsample * irfft(rfft(density) |kappa|)`` with no other
    factor. The result is the trigonometric interpolant of the padded
    spectrum: for an even ``n_pad`` its Nyquist term is one cosine,
    ``X_{n/2} |kappa_{n/2}| cos(pi (t - t0) / p)``, so the Nyquist
    coefficient is halved before the longer inverse transform, where it is a
    positive frequency with a negative partner. The filtered values at the
    samples do not depend on ``upsample``.

    Parameters
    ----------
    samples : array, shape (N,)
        One window, flux per sample.
    grid : array, shape (N,)
        The sample centers, in mas, regular.
    pad_factor : int, default RAMP_PAD_FACTOR
        Zero-padding of the spectrum, in window lengths.
    upsample : int, default RAMP_UPSAMPLE
        The fine grid's refinement over the samples.

    Returns
    -------
    t_fine : array, shape (N upsample,)
        The fine grid, in mas, from ``grid[0]`` in steps of ``p / upsample``.
    q : array, shape (N upsample,)
        The filtered projection on it, in flux/mas^2.
    """
    grid = np.asarray(grid, dtype=float)
    pitch = _pitch(grid)
    n = grid.size
    samples = np.asarray(samples, dtype=float)
    if samples.shape != (n,):
        raise ValueError(f"one sample is required per grid point: {n} points, "
                         f"samples of shape {samples.shape}")
    n_pad = pad_factor * n
    density = samples / pitch
    spectrum = np.fft.rfft(density, n=n_pad)
    kappa = np.fft.rfftfreq(n_pad, d=pitch)
    filtered = spectrum * np.abs(kappa)
    if upsample > 1 and n_pad % 2 == 0:
        filtered[-1] *= 0.5                     # the Nyquist cosine: one half to +kappa, one to -kappa
    fine = upsample * np.fft.irfft(filtered, n=n_pad * upsample)
    t_fine = grid[0] + np.arange(n * upsample) * pitch / upsample
    return t_fine, fine[:n * upsample]


def direction_weights(psi):
    """The angular quadrature weight of each window on the half-turn.

    Position angle is defined modulo pi, so two windows 180 deg apart are
    repeat measurements of ONE direction; each direction owns half the gap to
    each of its neighbors on [0, pi), so the weights sum to pi (``pi / J``
    for a uniform pattern). Uneven scan laws get uneven weights, which is
    what makes the sum over windows a quadrature of the angular integral and
    not a vote. Directions equal modulo pi to within
    :data:`DIRECTION_TIE_RAD` (a window and its reverse, ``psi`` and
    ``psi + pi``, differ by float error after the modulo) are one direction:
    the group owns half the gap to each of its neighbors and splits that
    share equally, so the weights do not depend on the order of the windows.

    Parameters
    ----------
    psi : array, shape (J,)
        The scan angles, in rad.

    Returns
    -------
    array, shape (J,)
        The weight of each window, in rad; they sum to pi.
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    if not np.all(np.isfinite(psi)):
        raise ValueError("every scan angle psi must be finite")
    direction = np.asarray(np.mod(psi, np.pi), dtype=float)
    order = np.argsort(direction)
    ordered = direction[order]
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + np.pi]))
    share = 0.5 * (gaps + np.roll(gaps, 1))
    tie = gaps < DIRECTION_TIE_RAD                 # ordered[k] and the next are one direction
    if tie.all():
        share[:] = np.pi / share.size
    elif tie.any():
        start = int(np.flatnonzero(~tie)[-1]) + 1      # a group's first member, so none wraps
        roll = (np.arange(share.size) + start) % share.size
        group = np.concatenate([[0], np.cumsum(~tie[roll][:-1])])
        total = np.bincount(group, weights=share[roll])
        share[roll] = (total / np.bincount(group))[group]
    weights = np.empty_like(direction)
    weights[order] = share
    return weights


def _axis(half, step):
    return np.arange(-half, half + 1e-9, step)


def _windows(psi, windows, inst):
    """``(psi, windows)`` as arrays, one window of the instrument's samples per angle."""
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    windows = np.asarray(windows, dtype=float)
    if windows.ndim != 2 or windows.shape[0] != psi.size:
        raise ValueError(f"one window is required per scan angle: {psi.size} angles, "
                         f"windows of shape {windows.shape}")
    if windows.shape[1] != inst.sample_grid_mas.size:
        raise ValueError(f"each window must hold the instrument's "
                         f"{inst.sample_grid_mas.size} samples, not {windows.shape[1]}")
    return psi, windows


def _zero_points(eta0, n):
    if eta0 is None:
        return None
    eta0 = np.atleast_1d(np.asarray(eta0, dtype=float))
    if eta0.shape != (n,):
        raise ValueError("one zero point is required per window")
    return eta0


def filtered_backprojection(psi, windows, inst, half=300.0, step=4.0,
                            weights=None, eta0=None):
    """The inverse Radon transform of the windows.

    Each window is ramp-filtered (:func:`ramp_filter`) and laid back along
    its own sky line, weighted by :func:`direction_weights` (or ``weights``).
    The exact inverse of LSF-blurred projections is the scene convolved with
    the LSF, in flux per mas^2; the sampling band-limits it, the finite set
    of directions leaves streaks, and a source outside every window leaves a
    hole. Nothing is measured on it. Window j's sample n lies at the sky
    along-scan coordinate ``eta0_j + grid_n``, so its projection is laid back
    at ``eta - eta0_j``.

    Parameters
    ----------
    psi : array, shape (J,)
        The scan angles, in rad.
    windows : array, shape (J, N)
        The windows, flux per sample, on the instrument's sample grid.
    inst : Instrument
        The instrument (:class:`scanlib.instrument.Instrument`); only its
        sample grid ``grid = inst.sample_grid_mas`` [mas], regular, is used.
    half : float, default 300.0
        Half the side of the square sky grid, in mas.
    step : float, default 4.0
        The pixel size, in mas.
    weights : array, shape (J,), optional
        The angular weights, in rad; ``None`` uses :func:`direction_weights`.
    eta0 : array, shape (J,), optional
        The windows' zero points, in mas; ``None`` means every zero point 0.

    Returns
    -------
    image : array, shape (M, M)
        The image, in flux per mas^2; rows are ``ddec``, columns ``dra*``.
    axis : array, shape (M,)
        The grid ``arange(-half, half + 1e-9, step)``, in mas: it ends at
        ``half`` when ``step`` divides ``2 half``.

    Raises
    ------
    ValueError
        When ``eta0`` is given with a shape other than (J,).
    """
    psi, windows = _windows(psi, windows, inst)
    weights = direction_weights(psi) if weights is None else np.atleast_1d(
        np.asarray(weights, dtype=float))
    if weights.shape != psi.shape:
        raise ValueError(f"one weight is required per window: {psi.size} windows, "
                         f"weights of shape {weights.shape}")
    eta0 = _zero_points(eta0, psi.size)
    axis = _axis(half, step)
    x, y = np.meshgrid(axis, axis)
    image = np.zeros_like(x)
    for j, (psi_j, w_j, samples) in enumerate(zip(psi, weights, windows)):
        t_fine, q = ramp_filter(samples, inst.sample_grid_mas)
        t = project_al(x, y, psi_j)
        if eta0 is not None:
            t = t - eta0[j]                                  # the pixel in window j's frame
        image += w_j * np.interp(t, t_fine, q, left=0.0, right=0.0)
    return image, axis


def faststack(psi, windows, inst, half=300.0, step=4.0, alpha=0.05, eta0=None):
    """The faststack image of the windows (Harrison 2011, Eqs. 1-4).

    Every image pixel collects the sample of each window whose footprint
    contains the pixel center: the sample ``k`` with the pixel's along-scan
    coordinate ``eta = dra* sin(psi_j) + ddec cos(psi_j) - eta0_j`` inside
    ``[grid_k - p/2, grid_k + p/2)``, ``p`` the pitch. The ``M`` collected
    values are sorted, ``theta_1 <= ... <= theta_M``, and weighted

        w_k = 1                              k <= alpha M
        w_k = exp(1 - (theta_k / r0)^2)      k >  alpha M

    with ``r0`` the mean of the lowest ``max(1, floor(alpha M))`` values
    (Eq. 3). A sample far above the pixel's own low values, one the
    primary's light fell into because that scan ran along the line joining
    the pixel and the primary, is suppressed. The pixel's value is the
    weighted mean (Eq. 1) and its spread the weighted variance about it
    (Eq. 4).

    Where this departs from the paper, and why. (i) Harrison weights every
    sample by ``1 / n_k`` as well, ``n_k`` the number of image pixels the
    sample covers, so that a small footprint (a 2-D sky-mapper pixel)
    counts more than a long one (a 1-D astrometric-field sample). Here all
    samples are of one kind and the toy windows are projections with no
    across-scan extent, so ``n_k`` is one constant and cancels in Eqs. 1,
    3 and 4. Mixed window types would need it back. (ii) The paper writes
    ``k <= alpha N_w`` and leaves ``alpha N_w < 1`` undefined; here ``r0``
    rests on at least one sample. (iii) A sample ``contributes`` to a pixel
    when the pixel center lies within its area (the paper's rule); with no
    across-scan extent the area is the strip ``[grid_k - p/2, grid_k +
    p/2)`` across the whole sky, so a source anywhere across the scan
    enters every window. A real window's across-scan width, and the
    sources it censors, are not modeled (``notebooks/scene/limitations/01``).
    (iv) The paper's samples are counts on a background, all positive.
    Here the background is zero: without noise ``r0`` can be 0 in empty
    sky, where the ratio is undefined, and ``r0`` is then the smallest
    positive float, which suppresses every sample above it; with noise
    the lowest samples are the negative tail, ``r0`` is negative, and the
    weight, which depends on ``(theta_k / r0)^2`` alone, favors samples
    with ``|theta_k| < |r0|`` whatever their sign
    (``notebooks/scene/limitations/06``).

    The image is in the windows' units (flux per sample), with no
    point-source model and no LSF: a picture, nothing is measured on it.
    Stars within about two FWHM merge into a plateau; a companion inside
    one FWHM never leaves the primary's core.

    Parameters
    ----------
    psi : array, shape (J,)
        The scan angles, in rad.
    windows : array, shape (J, N)
        The windows, flux per sample, on the instrument's sample grid.
    inst : Instrument
        The instrument (:class:`scanlib.instrument.Instrument`); only its
        sample grid ``grid = inst.sample_grid_mas`` [mas], regular, is used.
    half : float, default 300.0
        Half the side of the square sky grid, in mas.
    step : float, default 4.0
        The pixel size, in mas.
    alpha : float, default 0.05
        The trimming fraction, in (0, 1).
    eta0 : array, shape (J,), optional
        The windows' zero points, in mas; ``None`` means every zero point 0.

    Returns
    -------
    image : array, shape (M, M)
        The faststack image, flux per sample; rows are ``ddec``, columns
        ``dra*``; NaN where no sample fell.
    variance : array, shape (M, M)
        The weighted variance of Eq. 4, (flux per sample)^2; NaN likewise.
    n_used : array of int, shape (M, M)
        The number of samples each pixel collected.
    axis : array, shape (M,)
        The grid ``arange(-half, half + 1e-9, step)``, in mas: it ends at
        ``half`` when ``step`` divides ``2 half``.

    Raises
    ------
    ValueError
        When ``alpha`` is outside (0, 1) or ``eta0`` is not one value per
        window.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), not {alpha!r}")
    psi, windows = _windows(psi, windows, inst)
    grid = inst.sample_grid_mas
    pitch = _pitch(grid)
    eta0 = _zero_points(eta0, psi.size)
    axis = _axis(half, step)
    x, y = np.meshgrid(axis, axis)
    values = np.full((psi.size,) + x.shape, np.nan)
    for j, (psi_j, samples) in enumerate(zip(psi, windows)):
        eta = project_al(x, y, psi_j)
        if eta0 is not None:
            eta = eta - eta0[j]                              # the pixel in window j's frame
        k = np.floor((eta - grid[0]) / pitch + 0.5).astype(int)
        inside = (k >= 0) & (k < grid.size)
        values[j][inside] = samples[k[inside]]
    theta = np.sort(values, axis=0)                          # ascending, NaN last
    n_used = np.sum(~np.isnan(theta), axis=0)
    rank = np.arange(psi.size)[:, None, None]
    valid = rank < n_used
    base = np.where(valid, 1.0, 0.0)
    t0 = np.where(valid, theta, 0.0)
    m = np.maximum(1, np.floor(alpha * n_used).astype(int))
    low = rank < m
    r0 = np.sum(np.where(low, t0 * base, 0.0), axis=0) / np.maximum(
        np.sum(np.where(low, base, 0.0), axis=0), np.finfo(float).tiny)
    r0 = np.where(r0 == 0.0, np.finfo(float).tiny, r0)
    with np.errstate(over="ignore"):
        w = np.where(low, base, base * np.exp(1.0 - (t0 / r0) ** 2))
    w = np.where(valid, w, 0.0)
    total = np.sum(w, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        image = np.sum(w * t0, axis=0) / total
        variance = np.sum(w * (t0 - image) ** 2, axis=0) / total
    empty = n_used == 0
    image[empty] = np.nan
    variance[empty] = np.nan
    return image, variance, n_used, axis
