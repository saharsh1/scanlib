"""The window-level likelihood and the joint fit of all windows at once.

THE MODEL. J windows of N samples each; window j is taken at scan angle
``psi_j`` [rad, counter-clockwise from north] and epoch ``tau_j`` [yr from
the reference epoch]. The scene is a primary and K-1 companions, the
primary first. Source k sits at the sky offset ``(dra*_k, ddec_k)`` [mas]
from the reference position at the reference epoch and, in a moving scene,
drifts relative to it at the proper motion ``(mu_a_k, mu_d_k)`` [mas/yr]
with the parallax difference ``dpi_k`` [mas]. Its along-scan position in
window j is

    eta_kj = (dra*_k + mu_a_k tau_j) sin(psi_j)
             + (ddec_k + mu_d_k tau_j) cos(psi_j) + dpi_k fpar_j        [mas]

with ``fpar_j`` the window's dimensionless along-scan parallax factor
(:func:`scene_positions_al`). Window j's samples are

    d_j = A_j(Theta) f + eps_j,        eps_j ~ N(0, diag(s_j^2)),

``A_j`` the (N, K) design matrix of
:meth:`scanlib.instrument.Instrument.design_columns`
at the positions ``eta_j - eta0_j`` (``eta0_j`` the window's zero point,
the sky along-scan coordinate of its sample grid's origin), ``f >= 0`` the
K fluxes, primary first, constant across windows, in the windows' units
(flux per sample), and ``s_j`` the known noise sigma of each sample.

THE LIKELIHOOD. With the windows stacked (``d``, ``A`` and ``s`` of length
J N),

    ln L(Theta, f) = - 1/2 sum_i (d_i - (A f)_i)^2 / s_i^2
                     - 1/2 J N ln(2 pi) - sum_i ln s_i                  [nats]

(:func:`loglike_windows`), the normalizing constant included, so the value
is a log-likelihood and not a chi^2: comparable between models, and the
likelihood factor of a Bayesian evidence. The fluxes enter linearly and
are non-negative, so at every Theta they are profiled,

    f(Theta) = argmin_{f >= 0} |(d - A(Theta) f) / s|^2,

one stacked non-negative least squares, which leaves the profiled
likelihood ``ln L(Theta, f(Theta))`` (:func:`loglike_profiled`, which also
returns the fluxes ``f(Theta)``) over the 2(K-1) positions of a static scene or
the 4(K-1) or 5(K-1) motions of a moving one, the primary pinned; 2K, 4K
or 5K with the primary free.

THE PRIMARY, PINNED OR FREE. By default (``free_primary=False``) the
primary is pinned at the reference position, the origin of the sky
offsets: its offset, motion and parallax difference are 0, column 0 of
``eta`` is identically 0, and the parameter vector holds the companions'
K-1 blocks. ``free_primary=True`` gives the primary its own offset from the
reference position and, in a moving scene, its own residual proper motion
and parallax difference, as one more block IN FRONT of the companions', in
the same per-source order; the parameter vector then holds K blocks. Every
sky offset, the companions' included, stays measured from the reference
position, the windows' origin; a companion's offset from the primary is its
block minus the primary's. The windows' zero points ``eta0`` fix the frame,
so nothing is degenerate. A pinned primary cannot absorb an error of the
reference position and pushes it into the companions; a free one absorbs
it. Every routine takes the flag.

THE FIT. :func:`fit_joint`, static (``tau=None``) or moving (``tau``
given), maximizes the profiled likelihood, i.e. minimizes

    chi^2(Theta) = |(d - A(Theta) f(Theta)) / s|^2,

by Nelder-Mead restarted until a restart stops helping
(:func:`scanlib.optimize.minimize_restarts`), from a seed ``theta0`` that is initialization,
never data. The uncertainty is the inverse of the expected Fisher matrix
of the full model, positions (motions) then fluxes, at the solution
(:func:`joint_fisher`).

THE PARAMETER VECTOR is flat: ``(dra*_1, ddec_1, dra*_2, ddec_2, ...)``
for a static scene; ``(dra*_1, ddec_1, mu_a_1, mu_d_1[, dpi_1], ...)`` for
a moving one, the parallax entry present exactly when ``fpar`` is given.
With ``free_primary`` the primary's block ``(dra*_0, ddec_0[, mu_a_0,
mu_d_0[, dpi_0]])`` comes first.
Every routine takes ``tau`` (``None``: static) and ``fpar`` (``None``: no
parallax term; it requires ``tau``), and the windows' zero points ``eta0``
(``None``: all 0).

THE NOISE. ``sigma_noise`` [flux per sample] is a positive scalar, the
same on every sample, one value per window, shape (J,) or (J, 1), or one
value per sample, shape (J, N) (:func:`scanlib.noise.sample_sigma` reads
it). A scalar solves the
least squares unweighted (one common sigma does not move its minimizer),
chi^2 being the residual power over sigma^2; an array whitens it, design
and data divided by ``s``. Both paths return the same quantities.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .instrument import project_al
from .noise import sample_sigma
from .optimize import minimize_restarts

#: Positions are bounded to +-THETA_BOUND [mas], the region any window of
#: the toy instrument can constrain; real windows pass their own bound.
THETA_BOUND = 600.0

#: The condition number of the correlation matrix ``F_ij / sqrt(F_ii F_jj)``
#: above which a Fisher matrix is not inverted: the parameters are then
#: degenerate. The correlation matrix is dimensionless, so the test does not
#: depend on the units of the positions or the fluxes.
COND_MAX = 1e12


def _n_par(tau, fpar):
    """Parameters per source: 2 static, 4 moving, 5 with parallax."""
    if fpar is not None and tau is None:
        raise ValueError("a parallax term needs the epochs tau of the windows")
    return 2 if tau is None else (5 if fpar is not None else 4)


def _per_window(name, value, n):
    value = np.atleast_1d(np.asarray(value, dtype=float))
    if value.shape != (n,):
        raise ValueError(f"one {name} is required per window, shape ({n},)")
    return value


def scene_positions_al(theta, psi, tau=None, fpar=None, free_primary=False):
    """The along-scan position of every source in every window, on the sky.

        eta[j, k] = (dra*_k + mu_a_k tau_j) sin(psi_j)
                    + (ddec_k + mu_d_k tau_j) cos(psi_j) + dpi_k fpar_j

    Column 0 is the primary, identically 0 unless ``free_primary``. The zero
    points are not applied here; :func:`stacked_design` moves these sky
    positions into each window's frame.

    Parameters
    ----------
    theta : array, shape (P (K - 1),), or (P K,) with ``free_primary``
        The sources' parameters, flat, measured from the reference position:
        ``P = 2`` entries ``(dra*, ddec)`` per source when ``tau`` is
        ``None``, 4 ``(dra*, ddec, mu_a, mu_d)`` when ``tau`` is given, 5 with
        ``dpi`` when ``fpar`` is given too; mas, mas/yr and mas. The
        companions only, or the primary's block first with ``free_primary``.
    psi : array, shape (J,)
        The scan angles, in rad.
    tau : array, shape (J,), optional
        The epochs, in years from the reference epoch; ``None`` for a static
        scene.
    fpar : array, shape (J,), optional
        The dimensionless along-scan parallax factor of each window; requires
        ``tau``.
    free_primary : bool, default False
        False pins the primary at the reference position; True reads its
        block from the front of ``theta``.

    Returns
    -------
    array, shape (J, K)
        ``eta`` of every source in every window, in mas, primary first.

    Raises
    ------
    ValueError
        When ``theta`` does not hold a whole number of sources for the
        given ``tau`` and ``fpar``, when ``fpar`` is given without ``tau``, or
        when ``tau`` or ``fpar`` is not one value per window.
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n_par = _n_par(tau, fpar)
    if theta.size % n_par:
        raise ValueError(f"theta must hold {n_par} entries per source, "
                         f"got {theta.size}")
    blocks = theta.reshape(-1, n_par)
    first = 0 if free_primary else 1                 # column of the first block
    eta = np.zeros((psi.size, blocks.shape[0] + first))
    if blocks.shape[0] == 0:
        return eta
    d_ra, d_dec = blocks[:, 0][None, :], blocks[:, 1][None, :]
    if tau is not None:
        tau = _per_window("tau", tau, psi.size)
        d_ra = d_ra + blocks[:, 2][None, :] * tau[:, None]
        d_dec = d_dec + blocks[:, 3][None, :] * tau[:, None]
    moving = project_al(d_ra, d_dec, psi[:, None])
    if fpar is not None:
        moving = moving + blocks[:, 4][None, :] * _per_window("fpar", fpar, psi.size)[:, None]
    eta[:, first:] = moving
    return eta


def _in_windows(eta, eta0):
    """(J, K) sky positions -> each window's own frame, ``eta - eta0_j``."""
    if eta0 is None:
        return eta
    return eta - _per_window("zero point eta0", eta0, eta.shape[0])[:, None]


def stacked_design(theta, psi, inst, tau=None, fpar=None, eta0=None,
                   free_primary=False):
    """The design matrix of all windows, stacked window-major.

    Column k is the response of a unit flux of source k in every window:
    :meth:`scanlib.instrument.Instrument.design_columns` at the positions of
    :func:`scene_positions_al`, moved into each window's frame by its zero
    point. ``stacked_design(...) @ f`` is the J windows concatenated. This is
    the hot path of the fit, rebuilt at every evaluation.

    Parameters
    ----------
    theta : array
        The sources' parameters, flat, as in :func:`scene_positions_al`.
    psi : array, shape (J,)
        The scan angles, in rad.
    inst : Instrument
        The instrument (:class:`scanlib.instrument.Instrument`): the
        windows' sample grid and line-spread function.
    tau, fpar : arrays, shape (J,), optional
        Epochs [yr] and parallax factors, as in :func:`scene_positions_al`.
    eta0 : array, shape (J,), optional
        The windows' zero points, in mas; ``None`` means every zero point 0.
    free_primary : bool, default False
        Whether ``theta`` holds the primary's block, as in
        :func:`scene_positions_al`.

    Returns
    -------
    array, shape (J N, K)
        The stacked design, dimensionless (fraction of each source's flux in
        each sample), primary first.
    """
    eta = _in_windows(scene_positions_al(theta, psi, tau, fpar, free_primary), eta0)
    return inst.design_columns(eta)


def _as_windows(windows, psi, inst):
    """``windows`` as a float (J, N) array: one row per scan angle, the
    instrument's N samples per row. A transposed array of a square case,
    J = N, has the right shape and cannot be caught."""
    windows = np.asarray(windows, dtype=float)
    n_psi, n_samples = np.size(psi), inst.sample_grid_mas.size
    if windows.ndim != 2 or windows.shape[0] != n_psi:
        raise ValueError(f"one window is required per scan angle: {n_psi} angles, "
                         f"windows of shape {windows.shape}")
    if windows.shape[1] != n_samples:
        raise ValueError(f"each window must hold the instrument's {n_samples} samples, "
                         f"not {windows.shape[1]}")
    return windows


def _sample_sigma(sigma_noise, shape):
    """The (J N,) per-sample sigma of windows of ``shape`` (J, N): a scalar,
    one per window as (J,) or (J, 1), or one per sample as (J, N); checked
    positive and finite."""
    if sigma_noise is None:
        raise ValueError("sigma_noise is required: the noise sigma, flux per sample")
    s = np.asarray(sigma_noise, dtype=float)
    if s.ndim == 2 and s.shape[1] == 1:
        s = s[:, 0]                                 # (J, 1): one per window
    return np.concatenate(sample_sigma(s, np.zeros(shape))[0])


def _whitening(sigma_noise, shape):
    """``None`` for a scalar sigma (the unweighted path), else the (J N,)
    sigma vector that whitens the stacked problem."""
    if sigma_noise is None:
        return None
    s = _sample_sigma(sigma_noise, shape)
    return None if np.ndim(sigma_noise) == 0 else s


def _nnls(A, d, s):
    """``argmin_{f >= 0} |(d - A f) / s|^2``; ``s`` ``None`` is unweighted.
    Returns ``(f, rnorm)``, the residual 2-norm of the (whitened) problem."""
    if s is None:
        return nnls(A, d)
    return nnls(A / s[:, None], d / s)


def _fisher(G, sigma_noise, s):
    """The Fisher matrix ``G^T diag(1 / s^2) G`` of a model with Jacobian ``G``
    (rows the samples); ``G^T G / sigma^2`` for a scalar ``sigma_noise``, ``s``
    then unused."""
    if np.ndim(sigma_noise) == 0:
        return G.T @ G / float(sigma_noise) ** 2
    return G.T @ (G / s[:, None] ** 2)


def _loglike(resid, s):
    return float(-0.5 * np.sum((resid / s) ** 2)
                 - 0.5 * s.size * np.log(2.0 * np.pi) - np.sum(np.log(s)))


def loglike_windows(theta, flux, windows, psi, inst, sigma_noise,
                    tau=None, fpar=None, eta0=None, free_primary=False):
    """The log-likelihood of the windows at given positions and fluxes.

        ln L = -1/2 sum_i r_i^2 / s_i^2 - 1/2 J N ln(2 pi) - sum_i ln s_i

    with ``r = d - A(Theta) f`` the residual of every sample and ``s`` its
    noise sigma; the normalizing constant is included, so the value is a
    log-likelihood and not a chi^2. No positivity is imposed on ``flux``: a
    negative flux is a valid model evaluation.

    Parameters
    ----------
    theta : array
        The sources' parameters, flat, as in :func:`scene_positions_al`.
    flux : array, shape (K,)
        The fluxes, flux per sample, primary first.
    windows : array, shape (J, N)
        The windows, flux per sample: one row per scan angle, the
        instrument's N samples per row. A transposed array is refused,
        except when J = N, where its shape is the same.
    psi : array, shape (J,)
        The scan angles, in rad.
    inst : Instrument
        The instrument (:class:`scanlib.instrument.Instrument`): the
        windows' sample grid and line-spread function.
    sigma_noise : float or array
        The noise sigma, flux per sample: a positive scalar or an array that
        broadcasts to (J, N). Required.
    tau, fpar, eta0 : arrays, shape (J,), optional
        Epochs [yr], parallax factors and zero points [mas], as in
        :func:`stacked_design`.
    free_primary : bool, default False
        Whether ``theta`` holds the primary's block, as in
        :func:`scene_positions_al`.

    Returns
    -------
    float
        ``ln L`` in nats.

    Raises
    ------
    ValueError
        When ``windows`` is not (J, N), ``flux`` does not have K entries, or
        ``sigma_noise`` is not positive and finite or not one of the forms
        of the module docstring.
    """
    windows = _as_windows(windows, psi, inst)
    A = stacked_design(theta, psi, inst, tau, fpar, eta0, free_primary)
    flux = np.asarray(flux, dtype=float)
    if flux.shape != (A.shape[1],):
        raise ValueError(f"flux must have length K = {A.shape[1]}, got shape {flux.shape}")
    s = _sample_sigma(sigma_noise, windows.shape)
    return _loglike(windows.ravel() - A @ flux, s)


def loglike_profiled(theta, windows, psi, inst, sigma_noise,
                     tau=None, fpar=None, eta0=None, free_primary=False, return_fluxes=False):
    """The log-likelihood at given positions, maximized over the fluxes ``f >= 0``.

    The fluxes are one stacked non-negative least squares over all windows,
    ``f(Theta) = argmin_{f >= 0} |(d - A(Theta) f) / s|^2`` (unweighted for a
    scalar sigma, whose value does not move the minimum): linear in the data,
    one per source, the same in every window. The log-likelihood is
    :func:`loglike_windows` at them: for a scalar ``sigma_noise``
    ``-1/2 rnorm^2 / sigma^2 - 1/2 J N ln(2 pi sigma^2)``, ``rnorm`` the
    residual 2-norm, for an array the full expression at the whitened
    solution.

    Parameters
    ----------
    theta, windows, psi, inst, sigma_noise, tau, fpar, eta0, free_primary
        As in :func:`loglike_windows`; ``sigma_noise`` is required.
    return_fluxes : bool, default False
        Return the fluxes ``f(Theta)`` too, from the same solve.

    Returns
    -------
    float, or (float, array of shape (K,))
        The profiled ``ln L`` in nats; with ``return_fluxes``, also the
        fluxes, flux per sample, primary first.
    """
    windows = _as_windows(windows, psi, inst)
    s = _sample_sigma(sigma_noise, windows.shape)
    A = stacked_design(theta, psi, inst, tau, fpar, eta0, free_primary)
    d = windows.ravel()
    if np.ndim(sigma_noise) == 0:
        flux, rnorm = _nnls(A, d, None)
        lnl = float(-0.5 * rnorm ** 2 / float(sigma_noise) ** 2
                    - 0.5 * s.size * np.log(2.0 * np.pi) - np.sum(np.log(s)))
    else:
        flux, _ = _nnls(A, d, s)
        lnl = _loglike(d - A @ flux, s)
    return (lnl, flux) if return_fluxes else lnl


def joint_fisher(theta, flux, psi, inst, sigma_noise,
                 tau=None, fpar=None, eta0=None, free_primary=False):
    """The expected Fisher matrix of the full model at a solution.

    Parameters ordered as ``theta`` (the positions or motions, the primary's
    block first with ``free_primary``) then the K fluxes, primary first. The
    chain rule through the projection gives, for source c with a block in
    ``theta``, flux ``f_c`` and ``dL/deta`` the column of
    :meth:`scanlib.instrument.Instrument.design_jacobian`,

        dm/d dra*_c = f_c dL/deta sin(psi_j)    dm/d mu_a_c = f_c dL/deta tau_j sin(psi_j)
        dm/d ddec_c = f_c dL/deta cos(psi_j)    dm/d mu_d_c = f_c dL/deta tau_j cos(psi_j)
        dm/d dpi_c  = f_c dL/deta fpar_j        dm/d f_k    = L (the design column)

    and the Fisher matrix is ``F = G^T diag(1 / s^2) G`` with ``G`` the (J
    N, P) Jacobian, i.e. ``G^T G / sigma^2`` for a scalar ``sigma_noise``.
    Units: 1/mas^2 on the position block, 1/(flux per sample)^2 on the flux
    block, and so on entry by entry. ``eta0`` moves where each window
    samples the sources, not the derivatives.

    Parameters
    ----------
    theta : array, shape (P,)
        The sources' parameters, flat, as in :func:`scene_positions_al`.
    flux : array, shape (K,)
        The fluxes, flux per sample, primary first.
    psi : array, shape (J,)
        The scan angles, in rad.
    inst : Instrument
        The instrument (:class:`scanlib.instrument.Instrument`): the
        windows' sample grid and line-spread function.
    sigma_noise : float or array
        The noise sigma, flux per sample: a scalar or an array that
        broadcasts to (J, N).
    tau, fpar, eta0 : arrays, shape (J,), optional
        Epochs [yr], parallax factors and zero points [mas], as in
        :func:`stacked_design`.
    free_primary : bool, default False
        Whether ``theta`` holds the primary's block, as in
        :func:`scene_positions_al`; True adds the primary's columns.

    Returns
    -------
    array, shape (P + K, P + K)
        The expected Fisher matrix, parameters then fluxes.

    Raises
    ------
    ValueError
        When ``flux`` does not have K entries, or ``sigma_noise`` is not
        positive and finite or not one of the forms of the module docstring.
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    flux = np.atleast_1d(np.asarray(flux, dtype=float))
    eta_all = _in_windows(scene_positions_al(theta, psi, tau, fpar, free_primary),
                          eta0)                                     # (J, K)
    first = 0 if free_primary else 1                                # first source with a block
    taus = None if tau is None else _per_window("tau", tau, psi.size)
    fpars = None if fpar is None else _per_window("fpar", fpar, psi.size)
    if flux.size != eta_all.shape[1]:
        raise ValueError(f"one flux is required per source: {eta_all.shape[1]} sources, "
                         f"{flux.size} fluxes")
    s = _sample_sigma(sigma_noise, (psi.size, inst.sample_grid_mas.size))
    rows = []
    for j, psi_j in enumerate(psi):
        block = inst.design_columns(eta_all[j])                     # (N, K)
        dL = inst.design_jacobian(eta_all[j])                       # dm/d eta_k / f_k
        cols = []
        for k in range(first, eta_all.shape[1]):
            dm_deta = dL[:, k] * flux[k]
            cols.append(dm_deta * np.sin(psi_j))                    # d/d dra*_c
            cols.append(dm_deta * np.cos(psi_j))                    # d/d ddec_c
            if taus is not None:
                cols.append(dm_deta * taus[j] * np.sin(psi_j))      # d/d mu_a_c
                cols.append(dm_deta * taus[j] * np.cos(psi_j))      # d/d mu_d_c
            if fpars is not None:
                cols.append(dm_deta * fpars[j])                     # d/d dpi_c
        rows.append(np.column_stack(cols + [block]))                # + flux columns
    return _fisher(np.vstack(rows), sigma_noise, s)


def _as_prior(prior, n_theta):
    """``None``, or ``(index, mean, sigma)`` checked against the ``n_theta``
    entries of ``theta`` and returned as ``(index, mean, weight)``, ``weight =
    1 / sigma^2``: a prior applied to the wrong entry is worse than none."""
    if prior is None:
        return None
    index, mean, sigma = prior
    index = np.atleast_1d(np.asarray(index))
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    if not (index.ndim == mean.ndim == sigma.ndim == 1 and index.size == mean.size == sigma.size):
        raise ValueError("prior is (index, mean, sigma): three 1-D arrays of equal length")
    if index.size and not np.issubdtype(index.dtype, np.integer):
        raise ValueError("prior index must be integers")
    if np.any(index < 0) or np.any(index >= n_theta):
        raise ValueError(f"prior index must lie in 0..{n_theta - 1} (theta has {n_theta} entries)")
    if np.unique(index).size != index.size:
        raise ValueError("prior index entries must be distinct")
    if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
        raise ValueError("prior sigma must be positive and finite")
    if not np.all(np.isfinite(mean)):
        raise ValueError("prior mean must be finite")
    return index.astype(int), mean, 1.0 / sigma ** 2


def _prior_penalty(prior, theta):
    """``sum_i ((theta_i - m_i) / s_i)^2`` over the entries of a checked prior: a chi^2."""
    index, mean, weight = prior
    return float(np.sum(weight * (np.asarray(theta, dtype=float)[index] - mean) ** 2))


def joint_covariance(fisher, n_theta, prior=None):
    """The covariance and the 1-sigma errors from a Fisher matrix.

    With a Gaussian prior on entries of ``theta`` (:func:`fit_joint`), the
    matrix inverted is the Fisher matrix plus the prior's curvature,
    ``1 / sigma_i^2`` at ``(index_i, index_i)`` of the parameter block; the
    fluxes are never constrained.

    Parameters
    ----------
    fisher : array, shape (P + K, P + K)
        A Fisher matrix of :func:`joint_fisher`, parameters then fluxes.
    n_theta : int
        The number ``P`` of position (motion) parameters, the leading block.
    prior : None or (index, mean, sigma)
        As in :func:`fit_joint`; ``None`` adds nothing.

    Returns
    -------
    cov : array, shape (P + K, P + K) or None
        The inverse, in the same parameter order (mas^2 on the position
        block, (flux per sample)^2 on the flux block), or ``None`` when a
        parameter carries no information (a zero diagonal entry) or the
        condition number of the correlation matrix ``F_ij / sqrt(F_ii F_jj)``
        exceeds ``COND_MAX``: the test is free of the parameters' units.
    sigma_theta : array, shape (P,)
        The 1-sigma of each of the first ``n_theta`` parameters, in their own
        units; ``inf`` when ``cov`` is ``None``.

    Raises
    ------
    ValueError
        If the prior's entries are not distinct indices of ``theta``, or its
        arrays differ in length, or a sigma is not positive and finite.
    """
    return _covariance(fisher, n_theta, _as_prior(prior, n_theta))


def _covariance(fisher, n_theta, prior):
    """:func:`joint_covariance` for a prior already checked by :func:`_as_prior`."""
    if prior is not None:
        fisher = fisher.copy()
        fisher[prior[0], prior[0]] += prior[2]
    d = np.sqrt(np.diag(fisher))
    if not np.all(np.isfinite(d) & (d > 0.0)) or \
            np.linalg.cond(fisher / np.outer(d, d)) > COND_MAX:
        return None, np.full(n_theta, np.inf)
    cov = np.linalg.inv(fisher)
    return cov, np.sqrt(np.clip(np.diag(cov)[:n_theta], 0.0, None))



def polar_elements(theta, cov):
    """The separation and the position angle of a sky offset, with their errors.

    ``rho = sqrt(dra*^2 + ddec^2)`` and ``PA = atan2(dra*, ddec)``, from north
    through east, in [0, 360) degrees. The errors follow the delta method,
    ``sigma_x^2 = g^T C g`` with ``g`` the gradient of ``x`` with respect to
    ``(dra*, ddec)``: valid while ``rho`` is well above its error (near the
    origin the modulus is biased and the angle undefined).

    Parameters
    ----------
    theta : array, shape (2,)
        ``(dra*, ddec)``, in mas.
    cov : array, shape (2, 2)
        Its covariance, in mas^2 (the position block of
        :func:`joint_covariance`).

    Returns
    -------
    rho, sigma_rho : float
        In mas.
    pa, sigma_pa : float
        In degrees.
    """
    dra, ddec = theta
    rho = np.hypot(dra, ddec)
    phi = np.arctan2(dra, ddec)
    g_rho = np.array([dra, ddec]) / rho
    g_phi = np.array([ddec, -dra]) / rho ** 2
    sig_rho = np.sqrt(g_rho @ cov @ g_rho)
    sig_phi = np.sqrt(g_phi @ cov @ g_phi)
    return rho, sig_rho, np.rad2deg(phi) % 360.0, np.rad2deg(sig_phi)

def _fit(windows, psi, theta0, inst, sigma_noise, tau, fpar, eta0,
         free_primary, bounds, maxiter_first, maxiter_restart, prior=None):
    windows = _as_windows(windows, psi, inst)
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    _sample_sigma(sigma_noise, windows.shape)                   # required, and checked
    s = _whitening(sigma_noise, windows.shape)
    d = windows.ravel()
    prior = _as_prior(prior, theta0.size)
    # the residual power is in flux^2 for a scalar sigma, so the dimensionless
    # prior term is scaled to it and chi2 is divided by it; on the whitened
    # path both are dimensionless
    scale = 1.0 if s is not None else float(sigma_noise) ** 2

    def chi2(theta):
        A = stacked_design(theta, psi, inst, tau, fpar, eta0, free_primary)
        if prior is None:
            return _nnls(A, d, s)[1] ** 2
        return _nnls(A, d, s)[1] ** 2 + scale * _prior_penalty(prior, theta)

    res, n_restarts, converged = minimize_restarts(
        chi2, theta0, bounds=bounds, xatol=1e-4,
        maxiter_first=maxiter_first, maxiter_restart=maxiter_restart)
    theta = np.asarray(res.x)
    flux, rnorm = _nnls(stacked_design(theta, psi, inst, tau, fpar, eta0, free_primary), d, s)
    fisher = joint_fisher(theta, flux, psi, inst, sigma_noise, tau, fpar, eta0,
                          free_primary)
    cov, sigma_theta = _covariance(fisher, theta.size, prior)
    return dict(theta=theta, flux=flux, cov=cov, sigma_theta=sigma_theta,
                chi2=rnorm ** 2 / scale,
                chi2_prior=0.0 if prior is None else _prior_penalty(prior, theta),
                loglike=loglike_windows(theta, flux, windows, psi, inst,
                                        sigma_noise, tau, fpar, eta0, free_primary),
                n_restarts=n_restarts, converged=converged)


def fit_joint(windows, psi, theta0, inst, sigma_noise, tau=None, fpar=None,
              eta0=None, free_primary=False, theta_bound=THETA_BOUND, rate_bound=None,
              prior=None):
    """The joint maximum-likelihood fit of a scene to all windows, static or moving.

    Minimizes the profiled ``chi^2(Theta)`` of the module docstring over the
    sources' parameters by :func:`scanlib.optimize.minimize_restarts` from
    the seed ``theta0``, with ``xatol`` 1e-4. A static scene (``tau=None``)
    is fitted over the positions with budgets of 4000 and 16000 iterations;
    a moving one over the positions, proper motions and, with ``fpar``, the
    parallax differences, with budgets of 8000 and 32000.

    Parameters
    ----------
    windows : array, shape (J, N)
        The windows, flux per sample: one row per scan angle, the
        instrument's N samples per row. A transposed array is refused,
        except when J = N, where its shape is the same.
    psi : array, shape (J,)
        The scan angles, in rad.
    theta0 : array, shape (P (K - 1),), or (P K,) with ``free_primary``
        The seed, flat, as in :func:`scene_positions_al`: ``P = 2`` static,
        ``(dra*, ddec)`` per source in mas; ``P = 4`` moving, ``(dra*, ddec,
        mu_a, mu_d)`` in mas and mas/yr; ``P = 5`` with ``fpar``, the
        parallax difference ``dpi`` in mas last. The companions only, or the
        primary's block first with ``free_primary``. Initialization only,
        never data: take the companions from the peak of the back-projection
        map and the primary at 0.
    inst : Instrument
        The instrument (:class:`scanlib.instrument.Instrument`): the
        windows' sample grid and line-spread function.
    sigma_noise : float or array
        The noise sigma, flux per sample (module docstring).
    tau : array, shape (J,), optional
        The epochs, in years from the reference epoch; ``None`` fits a
        static scene.
    fpar : array, shape (J,), optional
        The along-scan parallax factor of each window; adds ``dpi`` to each
        source's block. Requires ``tau``.
    eta0 : array, shape (J,), optional
        The windows' zero points, in mas; ``None`` means every zero point 0.
    free_primary : bool, default False
        False pins the primary at the reference position; True fits its
        offset and, moving, its residual proper motion and parallax
        difference (module docstring).
    theta_bound : float, default THETA_BOUND
        Each position, the primary's included, is bounded to
        ``+-theta_bound``, in mas.
    rate_bound : float, optional
        Each proper motion, the primary's included, is bounded to
        ``+-rate_bound``, in mas/yr; ``None`` leaves it free. The parallax
        difference is always free. Only with ``tau``.
    prior : None or (index, mean, sigma)
        A Gaussian prior on entries of ``theta``, for a source whose position
        is known from another measurement (a catalogue, a deeper image):
        three 1-D arrays of equal length, the distinct entries constrained,
        their means and widths in the units of those entries. The fit then
        minimizes ``chi^2(theta) + sum_i ((theta_i - mean_i) / sigma_i)^2``,
        and the covariance is the inverse of the Fisher matrix plus
        ``diag(1 / sigma_i^2)`` at those entries (:func:`joint_covariance`).
        The prior enters the likelihood, so it must come from a measurement
        the windows did not produce. ``None`` (the default) fits without one.

    Returns
    -------
    dict
        ``theta``, the fitted parameters, shaped as ``theta0``, in its
        units; ``flux`` (K,), flux per sample, primary first; ``cov``, the
        inverse Fisher matrix of the parameters then the fluxes
        (:func:`joint_covariance`), or ``None`` if ill-conditioned;
        ``sigma_theta``, the 1-sigma of each parameter, shaped as
        ``theta0``; ``chi2``, the residual chi^2 at the solution;
        ``chi2_prior``, the prior's term at the solution (0 without one);
        ``loglike``, :func:`loglike_windows` at the solution, in nats;
        ``n_restarts`` and ``converged`` from the optimizer: ``converged``
        is True when a fresh restart could not improve the objective and
        ended inside its own budget (:func:`scanlib.optimize.minimize_restarts`);
        not a test that the minimum is global or away from a bound.

    Raises
    ------
    ValueError
        When ``windows`` is not (J, N), ``fpar`` or ``rate_bound`` is given
        without ``tau``, or the prior does not fit ``theta``
        (:func:`joint_covariance`).
    """
    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    n_par = _n_par(tau, fpar)
    if tau is None:
        if rate_bound is not None:
            raise ValueError("a rate bound needs the epochs tau of the windows")
        bounds = [(-theta_bound, theta_bound)] * theta0.size
        return _fit(windows, psi, theta0, inst, sigma_noise, None, None, eta0,
                    free_primary, bounds, 4000, 16000, prior)
    rate = (None, None) if rate_bound is None else (-float(rate_bound), float(rate_bound))
    bounds = []
    for _ in range(theta0.size // n_par):
        bounds += [(-theta_bound, theta_bound)] * 2 + [rate] * 2 + [(None, None)] * (n_par - 4)
    return _fit(windows, psi, theta0, inst, sigma_noise, tau, fpar, eta0,
                free_primary, bounds, 8000, 32000, prior)
