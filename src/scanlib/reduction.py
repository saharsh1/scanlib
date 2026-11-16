"""The per-window reduction: how many sources one window holds, and where.

One window is ``N`` samples of ``K`` non-negative point sources of known
line-spread function plus white Gaussian noise of known per-sample sigma
``sigma_n``, in the window's own frame (positions ``eta`` in mas, fluxes
and samples in flux per sample). The reduction chooses ``K`` and measures
the positions, blind to the scene.

THE FIT (:func:`fit_k`). A ``K``-source fit minimises the profiled residual
power

    chi^2(eta_1..K) = min_{f >= 0} | d - A(eta) f |^2                [flux^2]

over the positions, the fluxes solved by non-negative least squares (NNLS)
at each trial set (:func:`profiled_chi2`); ``A = inst.design_columns(eta)``.
The minimisation starts from a few configurations read off the NNLS solve on
every trial position of the instrument (:func:`active_clusters`): its active
positions linked into clusters, the ``K`` brightest clusters selected with
repetition, plus the ``(K - 1)``-source fit with the position where an added
positive source most reduces its residual. Each start is polished by bounded
Nelder-Mead (:func:`fit_positions`) and the lowest chi^2 is kept.

THE LADDER (:func:`ladder_window`). Fit ``K = 1, 2, ...`` and accept ``K + 1``
when the rung statistic

    D_K = [chi^2(K) - chi^2(K + 1)] / sigma_n^2                     [1]

exceeds the rung's margin ``Delta_K``; the first rung not climbed gives the
window's order. The margins belong to the instrument (``inst.margins``).
Each is the 99th percentile of ``D_K`` on simulated windows that hold
exactly ``K`` sources, reduced by this same ladder
(:func:`scanlib.simulation.rung_null_statistic`,
:func:`scanlib.simulation.calibrate_margins`): one window in a hundred promotes falsely per
rung. Non-negativity clips an added source's flux at zero and the fit takes
a minimum over positions, so the null of ``D_K`` is calibrated, not read from
a chi^2 table.

DELIVERY (:func:`delivery_mask`). The window is centred on the primary's
predicted position, so of ``K >= 2`` components the one nearest the origin
is the primary and every other is delivered as a detection; a ``K = 1``
window delivers nothing. Position, not flux, names the primary: in a close
blend the fitted fluxes can swap between the components while their
positions do not.

UNCERTAINTIES (:func:`sigma_eta`). The along-scan 1-sigma of each position
from the inverse Fisher matrix of the selected model.

THE NOISE. ``sigma`` is a positive scalar, the same on every sample, or an
``(N,)`` array, one per sample. With an array every solve runs whitened (the
samples and the design rows divided by sigma), chi^2 is dimensionless and
``D_K`` is the difference of the whitened chi^2; the fluxes stay in data
units.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from .joint import _fisher, _nnls, joint_covariance
from .noise import sample_sigma
from .optimize import minimize_restarts

#: A trial position of the NNLS solve is ACTIVE above this fraction of the
#: brightest one's flux: relative, so the rule is the same at every flux
#: scale (the solve's exact zeros are the inactive set; the tolerance guards
#: against round-off in the active-set solve).
ACTIVE_FLUX_REL = 1e-9


# --- the continuous fit and its uncertainties ------------------------------------

def _whitening(sigma, samples):
    """The per-sample sigma array of a window, or ``None`` for the unweighted path.

    ``None`` and a scalar are the unweighted path (one common sigma does not
    move a least-squares minimum); a scalar is still checked. An ``(N,)``
    array is checked against the window and returned as float64.
    """
    if sigma is None or np.ndim(sigma) == 0:
        if sigma is not None:
            sample_sigma(sigma, np.asarray(samples, dtype=float))
        return None
    return np.asarray(sample_sigma(sigma, np.asarray(samples, dtype=float))[0], dtype=float)


def profiled_chi2(eta_trial, samples, inst, sigma_noise=None):
    """The residual power at trial positions, the fluxes profiled by NNLS.

    ``chi^2 = min_{f >= 0} | d - A(eta) f |^2``, ``A = inst.design_columns(eta)``.

    Parameters
    ----------
    eta_trial : array, shape (K,)
        Trial positions, in mas, in the window's frame.
    samples : array, shape (N,)
        The window, flux per sample.
    inst : Instrument
        The instrument: its sample grid and LSF.
    sigma_noise : None, float or array, shape (N,)
        ``None`` or a scalar: the unweighted problem. An array: the problem
        on ``samples / sigma_noise`` with the design rows divided by it.

    Returns
    -------
    float
        In flux^2, or dimensionless with a per-sample ``sigma_noise``.
    """
    cols = inst.design_columns(np.atleast_1d(eta_trial))
    return _nnls(cols, np.asarray(samples, dtype=float), _whitening(sigma_noise, samples))[1] ** 2


def fit_positions(samples, eta_init, inst, sigma_noise=None):
    """The positions that minimise :func:`profiled_chi2`, from one start.

    Bounded Nelder-Mead inside ``inst.eta_bounds``, restarted until a
    restart stops helping (:func:`scanlib.optimize.minimize_restarts`,
    ``xatol`` 1e-4 mas, budgets 2000 and 8000 iterations). The profiled
    surface has kinks where a component meets the NNLS boundary (a zero
    flux); the simplex can stop there, and a restart escapes. The polished
    chi^2 is never above the start's.

    Parameters
    ----------
    samples : array, shape (N,)
        The window, flux per sample.
    eta_init : array, shape (K,)
        The starting positions, in mas.
    inst : Instrument
        The instrument.
    sigma_noise : None, float or array, shape (N,)
        As :func:`profiled_chi2`.

    Returns
    -------
    eta : array, shape (K,)
        The positions, in mas, sorted.
    flux : array, shape (K,)
        The NNLS fluxes at those positions, flux per sample.
    """
    eta_init = np.atleast_1d(np.asarray(eta_init, dtype=float))
    s = _whitening(sigma_noise, samples)
    eta_lo, eta_hi = inst.eta_bounds
    res, _n_restarts, _converged = minimize_restarts(
        lambda e: profiled_chi2(e, samples, inst, s), eta_init,
        bounds=[(eta_lo, eta_hi)] * eta_init.size,
        xatol=1e-4, maxiter_first=2000, maxiter_restart=8000)
    eta = np.sort(np.atleast_1d(res.x))
    return eta, _nnls(inst.design_columns(eta), np.asarray(samples, dtype=float), s)[0]


def sigma_eta(eta, flux, inst, sigma_noise):
    """The along-scan 1-sigma of each position, from the inverse Fisher matrix.

    The Jacobian of the model with respect to the ``K`` positions and the
    ``K`` fluxes is ``J = [inst.design_jacobian(eta) * f, inst.design_columns(eta)]``;
    the Fisher matrix is ``J^T J / sigma_noise^2``, or
    ``J^T diag(1/sigma_noise^2) J`` for a per-sample sigma, inverted by
    :func:`scanlib.joint.joint_covariance`: a degenerate model (by the
    condition number of its correlation matrix) or a source of zero flux
    gives infinite sigmas.

    Parameters
    ----------
    eta : array, shape (K,)
        The positions, in mas.
    flux : array, shape (K,)
        The fluxes, flux per sample, one per position.
    inst : Instrument
        The instrument.
    sigma_noise : float or array, shape (N,)
        The noise sigma, flux per sample, positive and finite.

    Returns
    -------
    array, shape (K,)
        The sigmas of the positions, in mas.

    Raises
    ------
    ValueError
        If ``flux`` is not one value per position, or ``sigma_noise`` is not
        positive and finite.
    """
    eta = np.atleast_1d(eta)
    flux = np.atleast_1d(flux)
    if flux.shape != eta.shape:
        raise ValueError(f"one flux is required per position: {eta.size} positions, "
                         f"{flux.size} fluxes")
    s = np.asarray(sample_sigma(sigma_noise, np.zeros(inst.sample_grid_mas.size))[0], dtype=float)
    if eta.size == 0:
        return np.array([])
    jac = np.hstack([inst.design_jacobian(eta) * flux[None, :], inst.design_columns(eta)])
    return joint_covariance(_fisher(jac, sigma_noise, s), eta.size)[1]


def _residual_power(eta, flux, samples, inst, sigma=None):
    """``|samples - model|^2`` [flux^2], or ``|(samples - model) / sigma|^2``."""
    resid = samples - inst.design_columns(np.atleast_1d(eta)) @ np.atleast_1d(flux)
    if sigma is not None:
        resid = resid / sigma
    return float(resid @ resid)


# --- where the light is: the solve on every trial position -----------------------

def _atom_pitch(inst):
    """The step of the trial positions [mas] (7.375 mas on the toy, a sample over 8)."""
    return float(np.max(np.diff(inst.atom_eta_mas)))


def _sample_pitch(inst):
    """The step of the sample grid [mas]."""
    return float(np.mean(np.diff(inst.sample_grid_mas)))


def active_clusters(samples, inst, sigma_noise=None):
    """The NNLS solve on every trial position, and its clusters of active ones.

    ``x = nnls(inst.atoms, samples)`` puts non-negative flux on the trial
    positions with no threshold; a position is ACTIVE above
    :data:`ACTIVE_FLUX_REL` times the brightest one's flux. Active positions
    closer than one sample pitch are LINKED into one cluster: closer than
    that the instrument has not resolved them into separate patches of
    light, so they describe one blend.

    Parameters
    ----------
    samples : array, shape (N,)
        The window, flux per sample.
    inst : Instrument
        The instrument: its trial positions ``atom_eta_mas`` and ``atoms``.
    sigma_noise : None, float or array, shape (N,)
        As :func:`profiled_chi2`; an array whitens the solve.

    Returns
    -------
    x : array, shape (G,)
        The flux on each trial position, flux per sample.
    clusters : list of int arrays
        Indices into ``inst.atom_eta_mas``, the cluster of the largest
        summed flux first, each cluster's positions brightest first (a tie
        keeps the grid order). Empty when the solve puts no light.

    Raises
    ------
    ValueError
        If the trial positions do not cover the sample grid: a source there
        would have no starting configuration (``inst.for_window`` builds
        trial positions over the window).
    """
    lo, hi = float(inst.atom_eta_mas.min()), float(inst.atom_eta_mas.max())
    if lo > inst.sample_grid_mas.min() or hi < inst.sample_grid_mas.max():
        raise ValueError(f"the trial positions ({lo:.1f} to {hi:.1f} mas) do not cover the "
                         f"samples ({inst.sample_grid_mas.min():.1f} to "
                         f"{inst.sample_grid_mas.max():.1f} mas): use "
                         f"inst.for_window(inst.sample_grid_mas)")
    samples = np.asarray(samples, dtype=float)
    x = _nnls(inst.atoms, samples, _whitening(sigma_noise, samples))[0]
    peak = float(x.max()) if x.size else 0.0
    if not peak > 0.0:
        return x, []
    act = np.flatnonzero(x > ACTIVE_FLUX_REL * peak)
    eta = inst.atom_eta_mas
    link = _sample_pitch(inst) * (1.0 + 1e-9)
    clusters, run = [], []
    for i in act:
        if run and eta[i] - eta[run[-1]] > link:
            clusters.append(np.asarray(run))
            run = []
        run.append(int(i))
    clusters.append(np.asarray(run))
    clusters = [c[np.argsort(-x[c], kind="stable")] for c in clusters]
    return x, sorted(clusters, key=lambda c: -x[c].sum())


def _residual_scan(samples, eta, flux, inst, sigma=None):
    """The misfit [flux^2] of one added source at each trial position.

    The residual ``r`` of the model ``(eta, flux)`` projected on each trial
    position's unit-flux column ``a``, the added flux clipped at zero:
    ``|r|^2 - max(a . r, 0)^2 / |a|^2``. Its minimum marks where a
    ``K + 1``-th source helps most; an empty model scans the window itself.
    With a per-sample ``sigma`` the residual and the columns are whitened.
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    resid = np.asarray(samples, dtype=float)
    if eta.size:
        resid = resid - inst.design_columns(eta) @ np.atleast_1d(flux)
    atoms = inst.atoms
    if sigma is not None:
        resid = resid / sigma
        atoms = atoms / sigma[:, None]
    proj = atoms.T @ resid
    norm2 = np.einsum("ng,ng->g", atoms, atoms)
    return float(resid @ resid) - np.clip(proj, 0.0, None) ** 2 / norm2


# --- the starting configurations --------------------------------------------------

def _starting_positions(samples, k, clusters, previous, inst, sigma=None):
    """The starting configurations of a ``k``-source fit.

    ``clusters`` as :func:`active_clusters` returns them; ``previous`` the
    ``(k - 1)``-source fit ``(positions, fluxes)`` (``None`` for ``k = 1``).
    Returns ``[(label, positions [mas]), ...]``, positions sorted and inside
    ``inst.eta_bounds``, a repeated configuration kept once.

    - THE CLUSTER SELECTIONS. Every unordered selection with repetition of
      ``k`` clusters from the ``k`` brightest; a cluster selected ``m``
      times gives its ``m`` brightest positions, and a selection is skipped
      when a cluster has fewer. Labelled by the clusters' ranks: ``"{1,1}"``
      the two brightest positions of the brightest cluster, ``"{1,2}"`` the
      brightest of each of the two brightest. The repeated selections carry
      the BLENDS, components that the solve has not resolved into separate
      clusters (without ``"{1,1}"`` a sixth of the pairs at 0.4 FWHM and
      flux ratio 0.2 end in a shallower minimum).
    - THE RESIDUAL START (``"r"``, ``k >= 2``). The ``(k - 1)`` fit's
      positions plus the trial position where a positive added source most
      reduces its residual (:func:`_residual_scan`), at least one trial step
      from every fitted position. It carries the SEPARATED sources, and it
      makes the ladder NESTED: its profiled chi^2 is at most the ``(k - 1)``
      fit's (the previous fluxes with a zero added flux are feasible) and the
      polish never rises, so ``chi^2(k) <= chi^2(k - 1)`` and no rung
      statistic is negative.

    Every start has its positions at least one trial step apart. There are at
    most ``binom(2k - 1, k) + 1`` starts: 4 at ``k = 2``, 11 at ``k = 3``.
    """
    atom = inst.atom_eta_mas
    lo, hi = inst.eta_bounds
    starts = []
    top = clusters[:k]
    for pick in combinations_with_replacement(range(len(top)), k):
        counts = np.bincount(pick, minlength=len(top))
        if any(top[c].size < m for c, m in enumerate(counts)):
            continue
        pos = np.concatenate([atom[top[c][:m]] for c, m in enumerate(counts) if m])
        label = "{" + ",".join(str(c + 1) for c in pick) + "}"
        starts.append((label, np.sort(pos)))
    if k >= 2:
        if previous is None:
            raise ValueError("the residual start needs the (k - 1)-source fit")
        eta_prev = np.atleast_1d(np.asarray(previous[0], dtype=float))
        scan = _residual_scan(samples, eta_prev, previous[1], inst, sigma)
        pitch = _atom_pitch(inst) * (1.0 - 1e-9)
        admissible = ((atom >= lo) & (atom <= hi)
                      & (np.abs(atom[:, None] - eta_prev[None, :]) >= pitch).all(axis=1))
        if admissible.any():
            scan = np.where(admissible, scan, np.inf)
            starts.append(("r", np.sort(np.r_[eta_prev, atom[int(np.argmin(scan))]])))
    out = []
    for label, pos in starts:
        pos = np.clip(pos, lo, hi)
        if not any(np.allclose(pos, seen, rtol=0.0, atol=1e-9) for _, seen in out):
            out.append((label, pos))
    return out


def _solve(samples, k, inst, previous, clusters, sigma=None):
    """The ``k``-source fit: ``(positions, fluxes, chi^2, number of starts)``."""
    if clusters is None:
        clusters = active_clusters(samples, inst, sigma)[1]
    if k >= 2 and previous is None:
        previous = _solve(samples, k - 1, inst, None, clusters, sigma)[:2]
    starts = _starting_positions(samples, k, clusters, previous, inst, sigma)
    if not starts:
        raise ValueError(f"no starting configuration for {k} sources: the "
                         f"solve has too few active trial positions")
    best, best_chi2 = None, np.inf
    for _label, pos in starts:
        eta, flux = fit_positions(samples, pos, inst, sigma)
        chi2 = _residual_power(eta, flux, samples, inst, sigma)
        if chi2 < best_chi2:
            best, best_chi2 = (eta, flux), chi2
    if best is None:
        raise ValueError("no start reached a finite chi^2")
    return best[0], best[1], best_chi2, len(starts)


def fit_k(samples, k, inst, *, previous=None, sigma_noise=None):
    """The maximum-likelihood fit of ``k`` sources to one window, blind.

    :func:`profiled_chi2` minimised over the ``k`` positions inside
    ``inst.eta_bounds``: every starting configuration (the cluster selections
    of :func:`active_clusters` and the residual start built on ``previous``)
    polished by :func:`fit_positions`, the lowest chi^2 kept. Uses no truth.

    Parameters
    ----------
    samples : array, shape (N,)
        The window, flux per sample.
    k : int
        The number of sources, positive.
    inst : Instrument
        The instrument.
    previous : None or (array, array)
        The ``(k - 1)``-source fit ``(positions, fluxes)``; ``None`` computes
        it with this same rule.
    sigma_noise : None, float or array, shape (N,)
        As :func:`profiled_chi2`.

    Returns
    -------
    eta : array, shape (k,)
        The positions, in mas, sorted.
    flux : array, shape (k,)
        The profiled fluxes, flux per sample.
    chi2 : float
        The residual power of the fit, in flux^2, or dimensionless with a
        per-sample ``sigma_noise``.

    Raises
    ------
    ValueError
        If ``k`` is not positive, or the window holds no light.
    """
    if k < 1:
        raise ValueError(f"k must be a positive number of sources, not {k!r}")
    samples = np.asarray(samples, dtype=float)
    s = _whitening(sigma_noise, samples)
    eta, flux, chi2, _n = _solve(samples, k, inst, previous, None, s)
    return eta, flux, chi2


# --- the ladder ------------------------------------------------------------------

def default_margins(inst, k_max=None):
    """The margins the ladder uses on ``inst``: ``inst.margins[:k_max - 1]``.

    Parameters
    ----------
    inst : Instrument
        The instrument; it must carry its margins.
    k_max : None or int
        The highest order the ladder may reach; ``None`` is
        ``len(inst.margins) + 1``.

    Returns
    -------
    array, shape (k_max - 1,)
        A copy of the margins of the rungs ``1 -> 2`` to ``k_max - 1 -> k_max``.

    Raises
    ------
    ValueError
        If the instrument has no margins or fewer than ``k_max - 1``: they
        are a property of the sample grid, the LSF and the trial positions,
        and :func:`scanlib.simulation.calibrate_margins` computes them.
    """
    if inst.margins is None:
        raise ValueError("the instrument has no margins: calibrate them with "
                         "scanlib.simulation.calibrate_margins(rng, inst, snr) and "
                         "build the instrument with margins=...")
    n = inst.margins.size
    k_max = n + 1 if k_max is None else int(k_max)
    if k_max < 1:
        raise ValueError(f"k_max must be at least 1, not {k_max!r}")
    if k_max - 1 > n:
        raise ValueError(f"climbing to k_max = {k_max} needs {k_max - 1} margins and "
                         f"the instrument has {n}: calibrate them with "
                         f"scanlib.simulation.calibrate_margins(rng, inst, snr, k_max={k_max})")
    return inst.margins[:k_max - 1].copy()


def _ladder(samples, inst, sigma_noise, margins, k_max):
    """The ladder with explicit margins (any real values, one per rung)."""
    if sigma_noise is None or (np.ndim(sigma_noise) == 0
                               and not (np.isfinite(sigma_noise) and sigma_noise > 0.0)):
        raise ValueError(f"sigma_noise must be finite and positive, not {sigma_noise!r}")
    samples = np.asarray(samples, dtype=float)
    s = _whitening(sigma_noise, samples)
    _x, clusters = active_clusters(samples, inst, s)
    active = np.sort(np.concatenate(clusters)) if clusters else np.array([], int)
    chi2s, deltas, n_starts = [], [], []
    diag = dict(active_eta=inst.atom_eta_mas[active], chi2=np.array(chi2s),
                delta_chi2=np.array(deltas), n_starts=np.array(n_starts, int))
    if not clusters:
        return np.array([]), np.array([]), 0, diag
    eta, flux, chi2, n_s = _solve(samples, 1, inst, None, clusters, s)
    chi2s.append(chi2)
    n_starts.append(n_s)
    k = 1
    while k < k_max and 2 * (k + 1) < samples.size:
        eta_n, flux_n, chi2_n, n_s = _solve(samples, k + 1, inst, (eta, flux), clusters, s)
        chi2s.append(chi2_n)
        n_starts.append(n_s)
        if s is not None:
            deltas.append(chi2 - chi2_n)
        else:
            deltas.append((chi2 - chi2_n) / sigma_noise ** 2)
        if not deltas[-1] > margins[k - 1]:
            break
        eta, flux, chi2, k = eta_n, flux_n, chi2_n, k + 1
    diag.update(chi2=np.array(chi2s), delta_chi2=np.array(deltas),
                n_starts=np.array(n_starts, int))
    return np.atleast_1d(eta), np.atleast_1d(flux), k, diag


def ladder_window(samples, inst, sigma_noise, *, k_max=None):
    """The order and the fit of one window by the nested ladder, blind.

    The ``K = 1`` fit is :func:`fit_k`; rung ``K`` fits ``K + 1`` sources,
    the residual start built on the ``K`` fit, and accepts them when

        D_K = [chi^2(K) - chi^2(K + 1)] / sigma_noise^2 > Delta_K,

    ``Delta_K = inst.margins[K - 1]``. The ladder stops at the first rung not
    climbed, at ``k_max``, or before a ``2(K + 1) >= N`` parameter fit. A
    window whose solve puts no light returns ``K = 0``. Consumes no random
    numbers.

    Parameters
    ----------
    samples : array, shape (N,)
        The window, flux per sample.
    inst : Instrument
        The instrument, with its margins (:func:`default_margins`).
    sigma_noise : float or array, shape (N,)
        The noise sigma, flux per sample: a scalar, or one per sample (every
        solve then runs whitened, and ``D_K`` is the difference of the
        whitened chi^2).
    k_max : None or int
        The highest order; ``None`` is ``len(inst.margins) + 1``.

    Returns
    -------
    eta : array, shape (K,)
        The positions of the selected fit, in mas, sorted.
    flux : array, shape (K,)
        Its fluxes, flux per sample.
    k : int
        The selected order ``K``.
    diag : dict
        ``active_eta``, the active trial positions [mas]; ``chi2``, the
        residual power of every fit made, ``K = 1, 2, ...`` (flux^2 for a
        scalar sigma, dimensionless for a per-sample one); ``delta_chi2``,
        ``D_K`` of every rung evaluated; ``n_starts``, the configurations
        polished per fit.

    Raises
    ------
    ValueError
        If the instrument lacks the margins (:func:`default_margins`), or
        ``sigma_noise`` is not positive and finite.
    """
    margins = default_margins(inst, k_max)
    return _ladder(samples, inst, sigma_noise, margins, margins.size + 1)


def delivery_mask(eta, origin=0.0):
    """Which components of a reduced window are delivered as detections.

    The window is centred on the primary's predicted position ``origin``
    [mas], so the component nearest it is the primary and every other is
    delivered; a window of fewer than two components delivers nothing. The
    primary is named by position, not by flux. A tie keeps the first
    component as the primary.

    Parameters
    ----------
    eta : array, shape (K,)
        The fitted positions, in mas, in the window's frame.
    origin : float
        The primary's predicted position in the window's frame, in mas.

    Returns
    -------
    bool array, shape (K,)
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    mask = np.zeros(eta.size, bool)
    if eta.size >= 2:
        mask[:] = True
        mask[int(np.argmin(np.abs(eta - origin)))] = False
    return mask


# --- the margins -----------------------------------------------------------------


# --- a campaign of windows -----------------------------------------------------

#: The along-scan error [mas] given to a detection whose Fisher sigma is not
#: finite (a degenerate window), so that its line still has a width on the
#: map; the typical scatter of the toy's detections at S/N 100.
SIGMA_DET_FALLBACK_MAS = 10.0


@dataclass(frozen=True, eq=False)
class DetectionSet:
    """The detections of a campaign, and the census of its windows.

    One entry per delivered detection, ordered by window, in the arrays
    ``psi`` [rad] (its window's scan angle), ``tau`` [yr] (its window's
    epoch from the reference epoch), ``eta`` [mas] (the fitted along-scan
    position, on the sky), ``sigma`` [mas] (its Fisher error, or
    :data:`SIGMA_DET_FALLBACK_MAS`), ``flux`` (the fitted flux, flux per
    sample) and ``fpar`` (its window's along-scan parallax factor); ``tau``
    and ``fpar`` are ``None`` when the scan law did not carry them.

    The census counts windows: ``n_windows``; ``n_windows_with_detection``;
    ``n_censored_windows``, every window that delivered nothing;
    ``n_single_source_windows``, the censored windows the ladder reduced to
    one source (the others held no light); ``n_sigma_substituted``, the
    detections whose Fisher sigma was not finite.

    ``margins`` are the ladder margins the windows were reduced with,
    ``sigma_noise`` the noise sigma they carry (a scalar, or one (N,) array
    per window), ``windows`` the samples of every window when asked for
    (``None`` otherwise), and ``eta0`` the windows' zero points [mas]
    (``None`` for all 0).
    """

    psi: np.ndarray
    tau: np.ndarray | None
    eta: np.ndarray
    sigma: np.ndarray
    flux: np.ndarray
    fpar: np.ndarray | None
    n_windows: int
    n_windows_with_detection: int
    n_censored_windows: int
    n_single_source_windows: int
    n_sigma_substituted: int
    margins: np.ndarray
    sigma_noise: float | list
    windows: np.ndarray | None = None
    eta0: np.ndarray | None = None


def _zero_points(eta0, n):
    """The (n,) zero points [mas] as an array, or ``None``."""
    if eta0 is None:
        return None
    eta0 = np.atleast_1d(np.asarray(eta0, dtype=float))
    if eta0.shape != (n,):
        raise ValueError(f"one zero point is required per window: {n} windows, "
                         f"zero points of shape {eta0.shape}")
    return eta0


def _unpack_law(law):
    """``(psi, tau, fpar)`` of a scan law: ``(psi, tau)``, ``(psi, tau, fpar)``
    or ``psi`` alone, ``tau`` and ``fpar`` ``None`` when absent."""
    if isinstance(law, tuple):
        if len(law) not in (2, 3):
            raise ValueError("a scan law is psi, (psi, tau) or (psi, tau, fpar)")
        psi = np.asarray(law[0], dtype=float)
        tau = np.asarray(law[1], dtype=float)
        fpar = None if len(law) == 2 else np.asarray(law[2], dtype=float)
    else:
        psi, tau, fpar = np.asarray(law, dtype=float), None, None
    psi = np.atleast_1d(psi)
    for name, arr in (("tau", tau), ("fpar", fpar)):
        if arr is not None and np.atleast_1d(arr).shape != psi.shape:
            raise ValueError(f"one {name} is required per scan angle")
    return (psi, None if tau is None else np.atleast_1d(tau),
            None if fpar is None else np.atleast_1d(fpar))


def reduce_campaign(law, windows, inst, sigma_noise, *, k_max, eta0=None):
    """The detections of a campaign: every window reduced, blind to the scene.

    Per window, in the order given: :func:`ladder_window` up to ``k_max``,
    the delivery rule :func:`delivery_mask` (the component nearest the
    primary's place in the window is the primary, every other is a
    detection) and the Fisher error :func:`sigma_eta` of each detection. A
    window of one source, or of no light, delivers nothing. The windows may
    be simulated or real; the function draws no random number and never
    sees the truth.

    Parameters
    ----------
    law : array or tuple
        The scan law, one entry per window: ``psi`` [rad] alone,
        ``(psi, tau)`` or ``(psi, tau, fpar)``; the epochs [yr] and the
        parallax factors are carried onto the detections.
    windows : array, shape (J, N), or list of J arrays
        The windows, flux per sample, aligned with the law.
    inst : Instrument
        The instrument, with its margins (:func:`default_margins`).
    sigma_noise : float, array or list of arrays
        The noise sigma, flux per sample, in any form
        :func:`scanlib.noise.sample_sigma` reads: a scalar, one per window,
        or one per sample.
    k_max : int
        The highest order a window may take: 2 for a primary and at most one
        companion per window, 3 when a window can hold a third source.
    eta0 : None or array, shape (J,)
        The windows' zero points [mas]; ``None`` for all 0. Each window is
        reduced in its own frame, the primary taken nearest ``-eta0_j``, and
        the detections are delivered on the sky, at ``eta + eta0_j``.

    Returns
    -------
    DetectionSet
        The detections and the census; ``windows`` is ``None``.

    Raises
    ------
    ValueError
        If the law, the windows and the zero points do not match, or the
        instrument lacks the margins.
    """
    margins = default_margins(inst, k_max)
    psi, tau, fpar = _unpack_law(law)
    n_windows = psi.size
    if len(windows) != n_windows:
        raise ValueError(f"one scan angle is required per window: {n_windows} angles, "
                         f"{len(windows)} windows")
    eta0 = _zero_points(eta0, n_windows)
    det_psi, det_eta, det_flux, det_sigma, det_window = [], [], [], [], []
    n_with_detection = n_single = 0
    if not isinstance(sigma_noise, (list, tuple)) and np.ndim(sigma_noise) == 0:
        sig_w = [sigma_noise] * n_windows
    else:
        sig_w = list(sample_sigma(sigma_noise, list(windows))[0])
        sigma_noise = sig_w
    for j, (psi_j, samples) in enumerate(zip(psi, windows)):
        eta, flux, k, _diag = ladder_window(samples, inst, sig_w[j], k_max=k_max)
        n_single += int(k == 1)
        keep = delivery_mask(eta) if eta0 is None else delivery_mask(eta, origin=-eta0[j])
        if not keep.any():
            continue                                    # one source, or no light
        sig = sigma_eta(eta, flux, inst, sig_w[j])      # in the window's frame
        n_with_detection += 1
        det_psi.extend([psi_j] * int(keep.sum()))
        det_window.extend([j] * int(keep.sum()))
        det_eta.extend((eta[keep] if eta0 is None else eta[keep] + eta0[j]).tolist())
        det_flux.extend(flux[keep].tolist())
        det_sigma.extend(sig[keep].tolist())
    det_sigma = np.asarray(det_sigma)
    finite = np.isfinite(det_sigma)
    det_sigma = np.where(finite, det_sigma, SIGMA_DET_FALLBACK_MAS)
    det_window = np.asarray(det_window, dtype=int)
    return DetectionSet(
        psi=np.asarray(det_psi), tau=None if tau is None else tau[det_window],
        eta=np.asarray(det_eta), sigma=det_sigma, flux=np.asarray(det_flux),
        fpar=None if fpar is None else fpar[det_window],
        n_windows=n_windows, n_windows_with_detection=n_with_detection,
        n_censored_windows=n_windows - n_with_detection,
        n_single_source_windows=n_single, n_sigma_substituted=int((~finite).sum()),
        margins=margins, sigma_noise=sigma_noise, eta0=eta0)
