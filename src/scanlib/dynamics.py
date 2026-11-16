"""The fits to the detection set: the position, the relative motion, the cleaning.

A campaign's detection set ``{(psi_i, tau_i, eta_i, sigma_i, fpar_i)}``
(:class:`scanlib.reduction.DetectionSet`) is the compressed form of its
windows: one along-scan position per detection, with its Fisher error. Each
detection measures the projection of the companion's position at its epoch,

    eta_i = (dra* + mu_alpha* tau_i) sin(psi_i) + (ddec + mu_delta tau_i) cos(psi_i)
            + dpi fpar_i                                              [mas]

(:func:`motion_design`), with ``psi`` [rad] the scan angle, ``tau`` [yr]
the epoch from the reference epoch, ``(dra*, ddec)`` [mas] the position at
the reference epoch, ``(mu_alpha*, mu_delta)`` [mas/yr] the proper motion
relative to the primary and ``dpi`` [mas] the parallax difference against
the dimensionless along-scan parallax factor ``fpar``. Every fit here is
weighted by ``1 / sigma_i^2`` and uses no truth.

The fits: :func:`fit_detection_lsq`, the position alone; :func:`fit_motion`,
the position and the motion (and ``dpi`` with ``parallax=True``) by weighted
least squares; :func:`fit_mark_refit`, the same fit, then once more without
the detections more than ``n_sigma`` off; :func:`fit_motion_robust`, a Huber
M-estimator that weights such detections down smoothly. The detections of a
blended window carry the blend's bias, which these fits inherit; the joint
fit to the windows (:func:`scanlib.joint.fit_joint`) is the measurement, and
these fits seed it.

THE CLEANING (:func:`clean_detection_set`): the map detects, a circle
selects, the sinogram fit refines. The highest peak of the back-projection
map of the detections as measured (:func:`scanlib.backprojection.map_peak`)
is the candidate; the detections whose lines pass within ``select_radius``
of it are fitted first, by the Huber estimator; then every detection is
tested against the fitted track, the motion is refitted to the ones on it,
and the test repeats until the kept set does not change. A detection on the
fitted track at its own epoch is a line through the fitted position once
moved to the reference epoch (:func:`to_reference_epoch`), so no second map
is needed.
"""
from __future__ import annotations

import numpy as np

from .backprojection import map_peak, peak_consistency
from .instrument import project_al
from .joint import joint_covariance

#: The Huber tuning of :func:`fit_motion_robust`: 95 % efficiency for
#: Gaussian errors.
HUBER_DELTA = 1.345
#: The radius [mas] about the map's peak within which the cleaning selects the
#: detections of its first fit (:func:`clean_detection_set`).
SELECT_RADIUS_MAS = 30.0


def wls(design, y, w):
    """Weighted least squares: ``theta = (X^T W X)^-1 X^T W y`` and its covariance.

    Parameters
    ----------
    design : array, shape (D, P)
        The design matrix ``X``.
    y : array, shape (D,)
        The measurements.
    w : array, shape (D,)
        The weights, ``1 / sigma^2``.

    Returns
    -------
    theta : array, shape (P,)
    cov : array, shape (P, P)
        ``(X^T W X)^-1``.

    Raises
    ------
    ValueError
        When ``X^T W X`` has a zero diagonal entry or a condition number of
        its correlation matrix above :data:`scanlib.joint.COND_MAX`
        (:func:`scanlib.joint.joint_covariance`): the detections do not
        determine every parameter.
    """
    Aw = design * w[:, None]
    cov, _ = joint_covariance(design.T @ Aw, design.shape[1])
    if cov is None:
        raise ValueError("the detections do not determine every parameter: the design is "
                         "rank-deficient or ill-conditioned (one epoch, too short a span, or "
                         "one scan direction)")
    return cov @ (Aw.T @ y), cov


def motion_design(psi, tau, fpar=None, parallax=False):
    """The design matrix of the linear relative motion.

    The columns ``[sin psi, cos psi, tau sin psi, tau cos psi]``, then
    ``fpar`` with ``parallax=True``: the parameters ``(dra*, ddec, mu_alpha*,
    mu_delta[, dpi])`` in mas, mas/yr and mas.

    Parameters
    ----------
    psi, tau : arrays, shape (D,)
        The scan angles [rad] and the epochs [yr] of the detections.
    fpar : None or array, shape (D,)
        The along-scan parallax factors; required with ``parallax=True``.
    parallax : bool
        Add the parallax column.

    Returns
    -------
    array, shape (D, 4) or (D, 5)

    Raises
    ------
    ValueError
        With ``parallax=True`` and no ``fpar``.
    """
    if parallax and fpar is None:
        raise ValueError("parallax=True needs the along-scan parallax factor fpar of every "
                         "detection")
    cols = [np.sin(psi), np.cos(psi), tau * np.sin(psi), tau * np.cos(psi)]
    if parallax:
        cols.append(fpar)
    return np.column_stack(cols)


def to_reference_epoch(theta, psi, tau, eta, fpar=None, parallax=False):
    """The detections moved to the reference epoch: the fitted motion removed.

    ``eta - mu_alpha* tau sin psi - mu_delta tau cos psi [- dpi fpar]``, the
    parameters ``theta`` in the column order of :func:`motion_design`.

    Returns
    -------
    array, the shape of ``eta``, in mas.
    """
    corr = project_al(theta[2] * tau, theta[3] * tau, psi)
    if parallax:
        corr = corr + theta[4] * fpar
    return eta - corr


def fit_motion(psi, tau, eta, sigma, fpar=None, parallax=False):
    """The linear relative motion fitted to the detections by weighted least squares.

    Parameters
    ----------
    psi, tau, eta, sigma : arrays, shape (D,)
        The detections: scan angle [rad], epoch [yr], along-scan position
        [mas] and its error [mas].
    fpar : None or array, shape (D,)
        The along-scan parallax factors; required with ``parallax=True``.
    parallax : bool
        Fit the parallax difference as a fifth parameter.

    Returns
    -------
    theta : array, shape (4,) or (5,)
        ``(dra*, ddec, mu_alpha*, mu_delta[, dpi])`` in mas, mas/yr, mas.
    cov : array, shape (4, 4) or (5, 5)
        The formal covariance.
    eta_ref : array, shape (D,)
        The detections at the reference epoch (:func:`to_reference_epoch`).

    Raises
    ------
    ValueError
        With ``parallax=True`` and no ``fpar``, or when the detections do not
        determine every parameter (:func:`wls`).
    """
    X = motion_design(psi, tau, fpar, parallax)
    theta, cov = wls(X, eta, 1.0 / sigma ** 2)
    return theta, cov, to_reference_epoch(theta, psi, tau, eta, fpar, parallax)


def fit_mark_refit(psi, tau, eta, sigma, fpar=None, parallax=False, n_sigma=3.0):
    """The motion fitted to every detection, then once more without the outliers.

    (1) :func:`fit_motion` on every detection; (2) the detections whose
    residual exceeds ``n_sigma`` times their own error are marked; (3) one
    refit without them. Both fits are returned.

    Returns
    -------
    theta_all, cov_all : the fit on every detection.
    theta_refit, cov_refit : the fit on the kept ones.
    keep : bool array, shape (D,), the detections kept.
    eta_ref : array, shape (D,), every detection at the reference epoch by
        the refit.

    Raises
    ------
    ValueError
        As :func:`fit_motion`, for either fit.
    """
    theta0, cov0, _ = fit_motion(psi, tau, eta, sigma, fpar if parallax else None,
                                 parallax=parallax)
    resid = (eta - motion_design(psi, tau, fpar, parallax) @ theta0) / sigma
    keep = np.abs(resid) < n_sigma
    theta1, cov1, _ = fit_motion(psi[keep], tau[keep], eta[keep], sigma[keep],
                                 fpar[keep] if (parallax and fpar is not None) else None,
                                 parallax=parallax)
    return theta0, cov0, theta1, cov1, keep, to_reference_epoch(theta1, psi, tau, eta, fpar,
                                                                parallax)


def _huber_weights(r, delta):
    """Huber weights of scaled residuals ``r``: 1 inside ``delta``, ``delta/|r|`` out."""
    return np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-12))


def _huber_theta(X, eta, sigma, delta, n_iter):
    """The Huber estimate of a linear model ``eta = X theta``: iterated
    reweighted least squares from the plain fit, at most ``n_iter`` passes."""
    w0 = 1.0 / sigma ** 2
    theta, _ = wls(X, eta, w0)
    for _ in range(n_iter):
        r = (eta - X @ theta) / sigma
        new, _ = wls(X, eta, w0 * _huber_weights(r, delta))
        if np.allclose(new, theta, rtol=1e-10, atol=1e-12):
            return new
        theta = new
    return theta


def fit_motion_robust(psi, tau, eta, sigma, fpar=None, parallax=False, delta=HUBER_DELTA,
                      n_iter=25):
    """The linear relative motion by a Huber M-estimator.

    Each detection is weighted by the Huber function of its scaled residual
    ``r = (eta - model) / sigma``: full weight for ``|r| <= delta``,
    ``delta / |r|`` beyond. Iterated reweighted least squares from the plain
    fit, at most ``n_iter`` passes. The covariance is the sandwich
    ``A^-1 B A^-1`` of an M-estimator.

    Returns
    -------
    theta, cov : the fit and its sandwich covariance.
    weights : array, shape (D,), the final Huber weights, at most 1.
    eta_ref : array, shape (D,), the detections at the reference epoch.

    Raises
    ------
    ValueError
        As :func:`fit_motion`; or when the detections inside the Huber band
        do not determine every parameter, so ``A`` cannot be inverted.
    """
    X = motion_design(psi, tau, fpar, parallax)
    theta = _huber_theta(X, eta, sigma, delta, n_iter)
    r = (eta - X @ theta) / sigma
    weights = _huber_weights(r, delta)
    psi_h = np.clip(r, -delta, delta)
    dpsi = (np.abs(r) <= delta).astype(float)
    A = X.T @ (X * (dpsi / sigma ** 2)[:, None])
    B = X.T @ (X * ((psi_h ** 2) / sigma ** 2)[:, None])
    Ainv, _ = joint_covariance(A, A.shape[0])
    if Ainv is None:
        raise ValueError("the detections inside the Huber band do not determine every "
                         "parameter: the sandwich covariance is undefined")
    return theta, Ainv @ B @ Ainv, weights, to_reference_epoch(theta, psi, tau, eta, fpar,
                                                               parallax)


def fit_detection_lsq(psi, eta, sigma):
    """The position alone fitted to the detections: the static fit.

    Each detection is a Gaussian measurement of
    ``eta = dra* sin(psi) + ddec cos(psi)`` with error ``sigma``, fitted by
    weighted least squares. It inherits the blend's bias and the censoring
    of the detections.

    Parameters
    ----------
    psi, eta, sigma : arrays, shape (D,)
        The detections: scan angle [rad], along-scan position [mas], error
        [mas].

    Returns
    -------
    theta : array, shape (2,), or None
        ``(dra*, ddec)`` in mas.
    cov : array, shape (2, 2), or None
        Its covariance. Both ``None`` with fewer than 3 detections or a
        degenerate design (:func:`scanlib.joint.joint_covariance`).
    """
    psi = np.atleast_1d(psi)
    if psi.size < 3:
        return None, None
    A = np.column_stack([np.sin(psi), np.cos(psi)])
    Aw = A * (1.0 / np.atleast_1d(sigma) ** 2)[:, None]
    cov, _ = joint_covariance(A.T @ Aw, 2)
    if cov is None:
        return None, None
    return cov @ (Aw.T @ np.atleast_1d(eta)), cov


def clean_detection_set(psi, tau, eta, sigma, fpar=None, *, parallax=False, static=False,
                        select_radius=SELECT_RADIUS_MAS, huber_delta=HUBER_DELTA, huber_iter=25,
                        k_sigma=3.0, origin_zone=30.0, half_size=300.0, step=1.0, max_iter=5):
    """One companion's detection set, cleaned, blind: the map detects, a circle
    selects, the sinogram fit refines.

    1. DETECT. The detections as measured are pooled on the back-projection
       map; its highest peak outside the origin zone is the candidate, with
       the map's width ``w`` there (:func:`scanlib.backprojection.map_peak`).
    2. SELECT. The detections whose line passes within ``select_radius`` of
       the peak, ``|eta_i - (dra* sin psi_i + ddec cos psi_i)| <=
       select_radius``. The radius is wide: a moving source's lines do not
       cross at one point, and the peak is not its position at the reference
       epoch.
    3. REFINE. The motion (:func:`motion_design`) is fitted to the selected
       detections by the Huber estimator (:func:`fit_motion_robust`:
       ``huber_delta``, ``huber_iter``), which a few lines of another source
       inside the radius pull less than they pull least squares.
    4. CLEAN. EVERY detection is tested against the fitted track at its own
       epoch, ``|eta_i - X_i theta| <= max(k_sigma sigma_i, w)``; the motion
       is refitted by weighted least squares (:func:`fit_motion`) to the ones
       that pass; repeated until the set that passes is the set last fitted,
       or ``max_iter`` passes. A detection outside the radius returns when it
       lies on the track.

    The residual of step 4 is the distance of the detection's line, moved to
    the reference epoch with the fitted motion (:func:`to_reference_epoch`),
    from the fitted position: the test is that of a map of the corrected
    detections, without the map.

    One companion is assumed: the map's highest peak outside the origin zone
    is its crossing. With ``static=True`` the two motion columns are dropped,
    nothing moves the detections, and the cleaning is the detections whose
    line passes within ``max(k_sigma sigma_i, w)`` of the peak
    (:func:`scanlib.backprojection.peak_consistency`) and one fit to them;
    ``select_radius`` and the Huber parameters are not used.

    Parameters
    ----------
    psi, tau, eta, sigma : arrays, shape (D,)
        The detection set [rad, yr, mas, mas]; ``tau`` is ignored with
        ``static``.
    fpar : None or array, shape (D,)
        The along-scan parallax factors; required with ``parallax=True``.
    parallax : bool
        Fit the parallax difference as the last parameter.
    static : bool
        Fit the position alone (with ``dpi`` when ``parallax``).
    select_radius : float
        The radius [mas] of step 2 about the map's peak. It selects the
        detections of the first fit and enters nothing else.
    huber_delta : float
        The scaled residual beyond which the first fit weights a detection
        down (:data:`HUBER_DELTA`).
    huber_iter : int
        The cap on the first fit's reweighting passes; 0 gives the plain
        least-squares fit.
    k_sigma : float
        The band of step 4, in units of each detection's own sigma.
    origin_zone : float
        The radius [mas] about the origin in which no peak is taken.
    half_size, step : float
        The map's half side and pixel [mas] (:func:`scanlib.backprojection.map_peak`).
    max_iter : int
        The cap on the passes of step 4.

    Returns
    -------
    theta, cov
        The least-squares fit on the final kept set, in
        :func:`motion_design`'s column order (the motion entries absent
        with ``static``).
    keep : bool array, shape (D,)
        The final kept set.
    info : dict
        ``peak`` [(dra*, ddec) mas] and ``width`` [mas] of the map's peak,
        ``sky_map`` and ``axis``; ``selected`` (bool, shape (D,)) and
        ``n_selected``, step 2; ``iterations`` and ``converged``; per pass of
        step 4 the lists ``theta`` (the fit the pass tests against: the Huber
        fit at the first), ``keep``, ``distance`` [mas] (each detection's
        ``|eta_i - X_i theta|``), ``n_kept``, ``n_removed``; and ``eta_ref``
        [mas], every detection at the reference epoch by the returned fit.
        With ``static``: one pass, ``selected`` the kept set, ``distance``
        from the peak.

    Raises
    ------
    ValueError
        With ``parallax=True`` and no ``fpar``, with ``max_iter < 1``, or
        when the selection or the cut keeps no more detections than
        parameters or ones that do not determine every parameter
        (:func:`wls`).
    """
    p = np.asarray(psi, dtype=float)
    e = np.asarray(eta, dtype=float)
    s = np.asarray(sigma, dtype=float)
    f = None if fpar is None else np.asarray(fpar, dtype=float)
    if parallax and f is None:
        raise ValueError("parallax=True needs the along-scan parallax factor fpar of every "
                         "detection")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    t = np.zeros_like(p) if static else np.asarray(tau, dtype=float)
    design = motion_design(p, t, f, parallax=parallax)
    if static:
        design = np.delete(design, [2, 3], axis=1)
    n_par = design.shape[1]
    weight = 1.0 / s ** 2

    def enough(mask, what):
        if mask.sum() <= n_par:
            raise ValueError(f"{what} kept {int(mask.sum())} detections for "
                             f"{n_par} parameters: no fit is possible")

    def fit(mask):
        enough(mask, "the consistency cut")
        if static:
            return wls(design[mask], e[mask], weight[mask])
        theta, cov, _ = fit_motion(p[mask], t[mask], e[mask], s[mask],
                                   f[mask] if (parallax and f is not None) else None,
                                   parallax=parallax)
        return theta, cov

    peak, width, sky, axis = map_peak(p, e, s, half_size, step, origin_zone)      # 1. detect
    history = {k: [] for k in ("theta", "keep", "distance", "n_kept", "n_removed")}

    def record(theta, new, distance):
        for key, value in (("theta", theta), ("keep", new), ("distance", distance),
                           ("n_kept", int(new.sum())), ("n_removed", int((~new).sum()))):
            history[key].append(value)

    if static:
        keep, distance = peak_consistency(p, e, s, peak, width, k_sigma)
        theta, cov = fit(keep)
        record(theta, keep, distance)
        selected, converged = keep, True
    else:
        _, from_peak = peak_consistency(p, e, s, peak, width, k_sigma)
        selected = from_peak <= float(select_radius)                              # 2. select
        enough(selected, f"the radius of {float(select_radius):g} mas about the map's peak")
        theta = _huber_theta(design[selected], e[selected], s[selected],          # 3. refine
                             huber_delta, huber_iter)
        keep = cov = None
        converged = False
        for _ in range(max_iter):                                                 # 4. clean
            distance = np.abs(e - design @ theta)
            new = distance <= np.maximum(k_sigma * s, width)
            record(theta, new, distance)
            if keep is not None and np.array_equal(new, keep):
                converged = True
                break
            keep = new
            theta, cov = fit(keep)
    eta_ref = e - design[:, 2:] @ theta[2:]                    # to the reference epoch
    info = dict(iterations=len(history["theta"]), converged=converged, peak=peak, width=width,
                selected=selected, n_selected=int(selected.sum()), eta_ref=eta_ref,
                sky_map=sky, axis=axis, **history)
    return theta, cov, keep, info
