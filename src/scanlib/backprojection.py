"""The back-projection map: detections pooled on the sky, and its peak.

A detection is one along-scan position delivered by one window: the
window's scan angle ``psi`` [rad], the measured along-scan coordinate
``eta`` [mas] in sky coordinates (the window's zero point already added
back), and its uncertainty ``sigma`` [mas]. It places the source on a line
of the sky,

    eta = dra* sin(psi) + ddec cos(psi),

and the map pools those lines: a source is where they cross. The map is a
DETECTOR. Its peaks are candidates and seeds; positions and fluxes come
from the forward fit of :mod:`scanlib.joint`, never from the map. Where
the detections come from (a per-window reduction, a catalog, a
simulation) is the caller's business.

Sky offsets ``(dra*, ddec)`` [mas] from the reference position; maps on
the square grid ``axis = arange(-half_size, half_size + 1e-9, step)``
[mas], rows = ddec, columns = dra*, both increasing with index
(``imshow(origin='lower')``).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .instrument import project_al


def backprojection_map(psi, eta, sigma, half_size=300.0, step=4.0):
    """Each detection's Gaussian sky-line ridge, summed on a sky grid.

    Detection i deposits on every pixel the ridge

        R_i(dra*, ddec) = exp(-d_i^2 / (2 sigma_i^2)) / (sqrt(2 pi) sigma_i),
        d_i = dra* sin(psi_i) + ddec cos(psi_i) - eta_i,

    Gaussian in the perpendicular distance ``d_i`` [mas] from its line (the
    line's normal is a unit vector, so ``d_i`` is a true distance) with width
    equal to THAT detection's along-scan uncertainty, and the ridges are
    summed: ``map = sum_i R_i``, in 1/mas. The sum, not the product, makes
    the map an additive detector rather than a likelihood: one wrong line
    moves its peak far less than it moves a product's, though among few
    lines a narrow wrong ridge (crest ``1 / (sqrt(2 pi) sigma_i)``) can
    outweigh their crossing. Its ridges are not inverse-variance weights:
    a crest scales as ``1 / sigma_i``, least squares weighs ``1 / sigma_i^2``.

    Each ridge has UNIT AREA across its line. A unit-peak ridge would
    integrate to ``sqrt(2 pi) sigma_i``, so a broad (less precise) detection
    would deposit MORE total mass than a tight one and the least informative
    measurements would dominate. Unit area gives every detection the same
    total mass and lets precision show as concentration: a tight detection is
    a tall narrow ridge that localizes hard, a broad one a low plateau that
    washes out. Where a source sits, ridges from all scan angles cross and
    the mass piles up.

    Parameters
    ----------
    psi : array, shape (D,)
        The scan angle of each detection's window, in rad.
    eta : array, shape (D,)
        The measured along-scan coordinate of each detection, in mas, in sky
        coordinates (the window's zero point already added back).
    sigma : array, shape (D,)
        The uncertainty of each ``eta``, in mas, positive and finite.
    half_size : float, default 300.0
        Half the side of the square sky grid, in mas.
    step : float, default 4.0
        The pixel size of the grid, in mas.

    Returns
    -------
    map : array, shape (M, M)
        The summed ridges, in 1/mas; rows are ``ddec``, columns ``dra*``.
    axis : array, shape (M,)
        The shared grid ``arange(-half_size, half_size + 1e-9, step)``, in
        mas: it ends at ``half_size`` when ``step`` divides ``2 half_size``.
    extent : tuple of 4 floats
        ``(axis[0], axis[-1], axis[0], axis[-1])`` for ``imshow(origin='lower')``.

    Raises
    ------
    ValueError
        When ``psi``, ``eta`` and ``sigma`` are not one value per detection,
        or a ``sigma`` is not positive and finite.
    """
    psi, eta, sigma = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (psi, eta, sigma))
    if not psi.shape == eta.shape == sigma.shape:
        raise ValueError(f"one psi, eta and sigma is required per detection: shapes "
                         f"{psi.shape}, {eta.shape}, {sigma.shape}")
    if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
        raise ValueError("every sigma must be positive and finite")
    axis = np.arange(-half_size, half_size + 1e-9, step)
    ra_grid, dec_grid = np.meshgrid(axis, axis)
    accumulator = np.zeros_like(ra_grid)
    for psi_i, eta_i, sig_i in zip(psi, eta, sigma):
        dist = project_al(ra_grid, dec_grid, psi_i) - eta_i
        accumulator += (np.exp(-0.5 * (dist / sig_i) ** 2)
                        / (np.sqrt(2.0 * np.pi) * sig_i))
    return accumulator, axis, (axis[0], axis[-1], axis[0], axis[-1])


def map_peak(psi, eta, sigma, half_size=300.0, step=1.0, origin_zone=30.0,
             peak_filter_pixels=7):
    """The most prominent peak of the pooled map outside the origin zone.

    Pools the detections with :func:`backprojection_map` and returns the
    highest local maximum farther than ``origin_zone`` from the origin: the
    primary sits at the origin and delivers no detection, so a peak in its
    zone is not a companion. A pixel is a local maximum when it is the
    largest of its ``peak_filter_pixels`` square neighborhood. With the peak
    comes the map's own width there: the rms half-width of the connected
    region above half the peak height across that region's narrow direction
    (the square root of the smaller eigenvalue of the region's coordinate
    covariance), the map's resolution at the peak and the floor below which
    no single detection's ``sigma`` is to be trusted. The map only
    localizes; the fit measures.

    Parameters
    ----------
    psi, eta, sigma : arrays, shape (D,)
        The detections, as in :func:`backprojection_map`.
    half_size : float, default 300.0
        Half the side of the square sky grid, in mas.
    step : float, default 1.0
        The pixel size of the grid, in mas.
    origin_zone : float or None, default 30.0
        Radius about the origin, in mas, inside which no peak is accepted;
        ``None`` excludes no zone, and the peak is then the global argmax,
        the first pixel in row-major order on a tie.
    peak_filter_pixels : int, default 7
        The side of the square neighborhood a local maximum must dominate.

    Returns
    -------
    peak : tuple of 2 floats
        ``(dra*, ddec)`` of the peak pixel, in mas.
    width : float
        The map's rms half-width at the peak across its narrow direction, in
        mas; 0 for a one-pixel region.
    map : array, shape (M, M)
        The pooled map, in 1/mas.
    axis : array, shape (M,)
        Its grid, in mas.

    Raises
    ------
    ValueError
        When the map has no local maximum of positive height outside the
        zone (no detection, or none whose ridge reaches past it), or as
        :func:`backprojection_map`.
    """
    sky, axis, _ = backprojection_map(psi, eta, sigma, half_size, step)
    ra, dec = np.meshgrid(axis, axis)
    local = (sky == ndimage.maximum_filter(sky, size=peak_filter_pixels)) & (sky > 0.0)
    if origin_zone is not None:
        local &= np.hypot(ra, dec) > origin_zone
    candidates = np.flatnonzero(local)
    if candidates.size == 0:
        raise ValueError("the pooled map has no local maximum outside the "
                         f"{origin_zone} mas origin zone")
    top = candidates[np.argmax(sky.ravel()[candidates])]
    labels = np.zeros(sky.shape, dtype=np.int32)
    ndimage.label(sky >= 0.5 * sky.ravel()[top], output=labels)
    region = labels == labels.ravel()[top]
    if region.sum() < 2:                       # a one-pixel peak has no width
        width = 0.0
    else:
        moments = np.cov(np.vstack([ra[region] - ra.ravel()[top],
                                    dec[region] - dec.ravel()[top]]))
        width = float(np.sqrt(max(np.linalg.eigvalsh(moments).min(), 0.0)))
    return (float(ra.ravel()[top]), float(dec.ravel()[top])), width, sky, axis


def peak_consistency(psi, eta, sigma, peak, width, k_sigma=3.0):
    """Which detections' lines pass through a peak of the map.

    The perpendicular distance of ``peak`` from detection i's line is
    ``|eta_i - project_al(dra*, ddec, psi_i)|``. A detection is consistent
    when that distance is within ``max(k_sigma sigma_i, width)``: ``k_sigma``
    of its own uncertainty, floored by the map's own width at the peak (a
    detection reporting an implausibly small ``sigma`` is judged against a
    physical radius, not its own number).

    Parameters
    ----------
    psi, eta, sigma : arrays, shape (D,)
        The detections, as in :func:`backprojection_map`.
    peak : tuple of 2 floats
        ``(dra*, ddec)`` of the peak, in mas.
    width : float
        The map's width at the peak, in mas (from :func:`map_peak`).
    k_sigma : float, default 3.0
        The number of its own sigmas a detection may miss the peak by.

    Returns
    -------
    keep : array of bool, shape (D,)
        ``True`` for the detections whose line passes through the peak.
    distance : array, shape (D,)
        The perpendicular distance of the peak from each line, in mas.
    """
    distance = np.abs(np.asarray(eta, dtype=float)
                      - project_al(peak[0], peak[1], np.asarray(psi, dtype=float)))
    return distance <= np.maximum(k_sigma * np.asarray(sigma, dtype=float), width), distance
