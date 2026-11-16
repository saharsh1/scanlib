"""The search's threshold Lambda_c: the look-elsewhere level, in closed form.

The search (:mod:`scanlib.search`) accepts a candidate when its score
``Lambda = 2 [ln L(s + 1 sources) - ln L(s)]`` exceeds the level that noise
alone would reach, for a candidate searched anywhere in the field, in a
fraction ``p`` of campaigns. That level follows from the instrument and the
scan law alone.

Under the null, with the primary at the window centre, the gain of a
candidate at a trial point ``theta`` (its sky position, and its motion when
the search is moving) is ``max(Z, 0)^2``, ``Z(theta)`` the candidate's
best-fit flux over its error: a standard Gaussian field over the searched
region, exactly and at any S/N. The non-negative flux of the joint fit is
that one-sided ``max(Z, 0)`` whenever the other fluxes sit away from zero.
The chance that the maximum of ``Z`` exceeds ``t`` is approximated by the
expected Euler characteristic of the excursion set (Adler 1981; Worsley
1994; the look-elsewhere formula of Gross & Vitells 2010),

    P(max Z > t) ~= E[chi(t)] = sum_d L_d rho_d(t),
    rho_d(t) = (2 pi)^{-(d+1)/2} He_{d-1}(t) exp(-t^2 / 2),

with ``L_d`` the Lipschitz-Killing curvatures of the searched region in the
field's own metric: the Fisher matrix of the unit-normalised candidate
template, made orthogonal to the primary's (:func:`search_curvatures`).
The expected Euler characteristic counts the components of the excursion
set; it is not the exceedance probability itself, and it approaches it as
``t`` grows: a high-threshold approximation, whose accuracy at the 1 %
level of the search is the simulation's (CHECKED, below).
:func:`lrt_threshold` solves ``sum_d L_d rho_d(t) = p`` and returns
``Lambda_c = t^2``.

THE REGION: the box ``|dra*|, |ddec| <= half_size`` (the search map's reach,
:data:`FIELD_HALF_MAS`) minus the primary's zone of radius ``origin_zone``,
and, for a moving search (epochs ``tau`` given), proper motions within
``+-mu_max`` per component (:data:`MU_MAX_MAS_YR`). A FREE PRIMARY (its
offset, and moving its residual motion, fitted in both fits) adds its
columns to the one the template is made orthogonal to; the field stays
Gaussian to first order in the primary's parameters.

THE NOISE: the curvatures are computed with every sample weighted
equally. That is exact for one noise sigma common to every sample (its
value cancels) and an approximation otherwise: unequal per-sample sigmas
change the field's metric, and with it the ``L_d``. In one test (a static
candidate at (120, 80) mas, the primary pinned, 40 even directions) the
area density of the metric is 1.53e-4 per mas^2 with equal sigmas and
5.65e-5 with 10 windows at sigma 1 and 30 at sigma 100. Figure 4's
per-sample noise is the one unequal model checked by simulation.

CHECKED against simulated noise-only campaigns: static 17.07 predicted
against 17.17 simulated (47 transits), 17.31 against 17.38 (95
transits). Moving within +-10 mas/yr on Figure 4's 47 transits
(``notebooks/paper/auxil/sim_lambda_c.ipynb``, 3200 campaigns per
configuration), the closed form gives 19.13; the campaigns that exceed
it are, with equal noise, 27 with the primary pinned (0.84 %, exact 95 %
interval 0.56-1.23 %) and 32 with it free (1.00 %, 0.68-1.41 %), and with
Figure 4's per-sample noise 25 and 24 (0.78 %, 0.51-1.15 %; 0.75 %,
0.48-1.11 %). Every rate is consistent with the nominal 1 % or less;
none is shown to be below it. A free primary moves the level by less
than 0.05 on the Pleiades law.

The formula is that of the first addition (the primary alone in the
projection) and is applied at every rung; it holds while a candidate
is not within about a PSF width of an accepted source. The per-window
ladder keeps its calibrated margins instead: 18 samples are too few for the
smooth-field approximation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

#: The nominal false-addition rate per rung of the search.
FALSE_RATE = 0.01
#: The searched range of a moving candidate's proper motion, per component
#: [mas/yr]; the moving fits of the search are bounded to it.
MU_MAX_MAS_YR = 10.0
#: The half side [mas] of the searched box: the search map's reach, the joint
#: fit's position bound.
FIELD_HALF_MAS = 600.0

_ETA_STEP_MAS = 0.25          # the fine along-scan grid of the template functions
_SKY_STEP_MAS = 3.0           # the sky grid of the area integral
_THRESHOLD_CACHE: dict = {}


def _template_functions(inst, reach):
    """The per-window 1-D functions of the candidate template on a fine
    along-scan grid: ||T||^2, L0.T, T.T', ||T'||^2, L0.T' (L0 the primary's
    template at the window centre, T' the derivative along the scan), and
    L0'.T, L0'.T' (L0' the primary's template derivative: a free primary's
    position columns)."""
    eta = np.arange(-reach, reach + _ETA_STEP_MAS, _ETA_STEP_MAS)
    T, Tp = inst.design_columns(eta), inst.design_jacobian(eta)
    L0 = inst.design_columns(np.array([0.0]))[:, 0]
    L0p = inst.design_jacobian(np.array([0.0]))[:, 0]
    fns = dict(h=np.einsum("nm,nm->m", T, T), c=L0 @ T,
               hp=np.einsum("nm,nm->m", T, Tp), hpp=np.einsum("nm,nm->m", Tp, Tp),
               cp=L0 @ Tp, s=L0p @ T, sp=L0p @ Tp)
    return eta, fns, float(L0 @ L0)


def _interp(f, eta, grid0):
    """``f`` on the fine grid starting at ``grid0``, interpolated at ``eta``
    (P, J); a 2-D ``f`` (J, M) holds one function per window."""
    idx = (eta - grid0) / _ETA_STEP_MAS
    i0 = np.clip(np.floor(idx).astype(int), 0, f.shape[-1] - 2)
    w = idx - i0
    if f.ndim == 2:
        cols = np.broadcast_to(np.arange(f.shape[0]), i0.shape)
        return f[cols, i0] * (1.0 - w) + f[cols, i0 + 1] * w
    return f[i0] * (1.0 - w) + f[i0 + 1] * w


def _shifted_template_functions(inst, reach, eta0):
    """:func:`_template_functions` for windows with zero points ``eta0``: the
    primary sits at ``-eta0_j`` in window j, so the cross terms are one
    function per window."""
    eta, fns, _l0l0 = _template_functions(inst, reach)
    T, Tp = inst.design_columns(eta), inst.design_jacobian(eta)
    L0 = inst.design_columns(-np.asarray(eta0, dtype=float)).T        # (J, N)
    L0p = inst.design_jacobian(-np.asarray(eta0, dtype=float)).T
    fns = dict(fns, c=L0 @ T, cp=L0 @ Tp, s=L0p @ T, sp=L0p @ Tp)
    return eta, fns, float(np.einsum("jn,jn->", L0, L0))


def _field_metric(eta_pj, V, fns, eta0, a0a0):
    """The (D, D) Fisher matrix of the unit-normalised candidate template,
    made orthogonal to the primary's, at each point: ``eta_pj`` (P, J) the
    candidate's along-scan positions, ``V`` (D, J) their derivatives with
    respect to the D parameters."""
    H = _interp(fns["h"], eta_pj, eta0).sum(1)
    C = _interp(fns["c"], eta_pj, eta0).sum(1)
    Hp, Cp, Hpp = (_interp(fns[k], eta_pj, eta0) for k in ("hp", "cp", "hpp"))
    dd = np.einsum("ij,kj,pj->pik", V, V, Hpp)
    a0 = np.einsum("ij,pj->pi", V, Cp)
    ad = np.einsum("ij,pj->pi", V, Hp)
    ww = H - C ** 2 / a0a0
    dw = dd - a0[:, :, None] * a0[:, None, :] / a0a0
    wd = ad - C[:, None] * a0 / a0a0
    return dw / ww[:, None, None] - wd[:, :, None] * wd[:, None, :] / ww[:, None, None] ** 2


def _primary_columns(inst, psi, tau, eta0):
    """A free primary's K columns, shared by all windows: its flux (the
    template L0_j) and its offset, and with ``tau`` its residual motion (the
    derivative L0'_j times sin psi_j, cos psi_j, tau_j sin psi_j, tau_j cos
    psi_j). Returns ``W`` (K, J), ``kind`` (K,), column k in window j being
    ``W[k, j]`` times L0_j (kind 0) or L0'_j (kind 1), and the inverse of the
    columns' (K, K) Gram matrix."""
    e0 = np.zeros(psi.size) if eta0 is None else -np.asarray(eta0, dtype=float)
    cols = (inst.design_columns(e0).T, inst.design_jacobian(e0).T)          # (J, N) each
    su, cu = np.sin(psi), np.cos(psi)
    dirs = [su, cu] + ([tau * su, tau * cu] if tau is not None else [])
    W = np.vstack([np.ones(psi.size)] + dirs)
    kind = np.array([0] + [1] * len(dirs))
    aa = [[np.einsum("jn,jn->j", cols[a], cols[b]) for b in (0, 1)] for a in (0, 1)]
    G = np.array([[np.sum(W[k] * W[m] * aa[kind[k]][kind[m]]) for m in range(len(W))]
                  for k in range(len(W))])
    return W, kind, np.linalg.inv(G)


def _field_metric_free(eta_pj, V, fns, eta0, W, kind, Gi):
    """:func:`_field_metric` with the template made orthogonal to a free
    primary's K columns (:func:`_primary_columns`)."""
    H = _interp(fns["h"], eta_pj, eta0).sum(1)
    Hp, Hpp = _interp(fns["hp"], eta_pj, eta0), _interp(fns["hpp"], eta_pj, eta0)
    x = {0: (_interp(fns["c"], eta_pj, eta0), _interp(fns["cp"], eta_pj, eta0)),
         1: (_interp(fns["s"], eta_pj, eta0), _interp(fns["sp"], eta_pj, eta0))}
    C = np.stack([x[kind[k]][0] @ W[k] for k in range(W.shape[0])], 1)                       # (P, K)
    dC = np.stack([np.einsum("ij,pj->pi", V * W[k][None, :], x[kind[k]][1])
                   for k in range(W.shape[0])], 1)                                          # (P, K, D)
    dd = np.einsum("ij,kj,pj->pik", V, V, Hpp)
    ad = np.einsum("ij,pj->pi", V, Hp)
    ww = H - np.einsum("pk,kl,pl->p", C, Gi, C)
    wd = ad - np.einsum("pk,kl,pli->pi", C, Gi, dC)
    dw = dd - np.einsum("pki,kl,plj->pij", dC, Gi, dC)
    return dw / ww[:, None, None] - wd[:, :, None] * wd[:, None, :] / ww[:, None, None] ** 2


def _rho(d, t):
    """The Euler-characteristic density of a Gaussian field in d dimensions."""
    if d == 0:
        return norm.sf(t)
    he = (1.0, t, t * t - 1.0, t ** 3 - 3.0 * t)[d - 1]
    return (2.0 * np.pi) ** (-(d + 1) / 2.0) * he * np.exp(-0.5 * t * t)


def search_curvatures(psi, inst, tau=None, *, mu_max=MU_MAX_MAS_YR, half_size=FIELD_HALF_MAS,
                      origin_zone=30.0, eta0=None, free_primary=False):
    """The Lipschitz-Killing curvatures ``(L_0, ..., L_D)`` of the searched region.

    The sky part A (the box of half side ``half_size`` minus the disk of
    radius ``origin_zone``): ``L_0 = 0``, ``L_1`` half the boundary length
    and ``L_2`` the area, both in the field's metric. With ``tau`` the region
    is A times the motion box ``|mu_i| <= mu_max``; its metric is close to
    block-diagonal, so ``L_j(A x M) = sum_i L_i(A) L_{j-i}(M)``, M a box whose
    sides in the metric come from the motion block averaged over the region
    (checked against direct integration within 10 %).

    Parameters
    ----------
    psi : array, shape (J,)
        The scan angles, in rad.
    inst : Instrument
        The instrument: the candidate's template.
    tau : None or array, shape (J,)
        The epochs [yr] of a moving search; ``None`` for a static one.
    mu_max : float
        The searched proper-motion range per component, in mas/yr.
    half_size, origin_zone : float
        The searched box's half side and the primary's zone radius, in mas.
    eta0 : None or array, shape (J,)
        The windows' zero points [mas]: the primary sits at ``-eta0_j`` in
        window j. ``None`` for all 0.
    free_primary : bool
        Make the template orthogonal to the primary's flux, offset and, with
        ``tau``, motion columns: the score of fits with the primary free.

    Returns
    -------
    tuple of 3 floats (static) or 5 floats (moving)
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    moving = tau is not None
    tau = None if tau is None else np.atleast_1d(np.asarray(tau, dtype=float))
    zero_points = eta0
    half = float(half_size)
    su, cu = np.sin(psi), np.cos(psi)
    tmax = 0.0 if tau is None else float(np.max(np.abs(tau)))
    reach = half * np.sqrt(2.0) + (mu_max * tmax * np.sqrt(2.0) if moving else 0.0) + 60.0
    if eta0 is None:
        eta0_grid, fns, l0l0 = _template_functions(inst, reach)
        eta0, a0a0 = float(eta0_grid[0]), psi.size * l0l0
        shift = 0.0
    else:
        shift = np.atleast_1d(np.asarray(eta0, dtype=float))
        if shift.shape != psi.shape:
            raise ValueError("one zero point is required per window")
        eta0_grid, fns, a0a0 = _shifted_template_functions(
            inst, reach + float(np.max(np.abs(shift))), shift)
        eta0 = float(eta0_grid[0])
        shift = shift[None, :]                     # candidate positions into the window frames
    V2 = np.stack([su, cu])
    if free_primary:
        W, kind, Gi = _primary_columns(inst, psi, tau, zero_points)

        def metric(e, Vd):
            return _field_metric_free(e, Vd, fns, eta0, W, kind, Gi)
    else:
        def metric(e, Vd):
            return _field_metric(e, Vd, fns, eta0, a0a0)

    ax = np.arange(-half, half + 1e-9, _SKY_STEP_MAS)          # L_2(A): the area in the metric
    X, Y = np.meshgrid(ax, ax)
    keep = np.hypot(X, Y) > origin_zone
    xs, ys = X[keep], Y[keep]
    area = 0.0
    for k in range(0, xs.size, 40_000):
        e = xs[k:k + 40_000, None] * su[None, :] + ys[k:k + 40_000, None] * cu[None, :] - shift
        lam = metric(e, V2)
        area += np.sqrt(np.clip(lam[:, 0, 0] * lam[:, 1, 1] - lam[:, 0, 1] ** 2, 0.0, None)).sum()
    LA2 = float(area * _SKY_STEP_MAS ** 2)

    def path_length(px, py):
        pts = np.stack([px, py], 1)
        seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
        mid = pts + 0.5 * seg
        lam = metric(mid[:, 0:1] * su[None, :] + mid[:, 1:2] * cu[None, :] - shift, V2)
        return float(np.sqrt(np.clip(np.einsum("pi,pik,pk->p", seg, lam, seg), 0.0, None)).sum())

    side = np.linspace(-half, half, 2400)
    outer = path_length(np.r_[side, np.full_like(side, half), side[::-1], np.full_like(side, -half)],
                        np.r_[np.full_like(side, -half), side, np.full_like(side, half), side[::-1]])
    ang = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
    inner = path_length(origin_zone * np.cos(ang), origin_zone * np.sin(ang))
    LA1 = 0.5 * (outer + inner)
    if not moving:
        return (0.0, LA1, LA2)
    V4 = np.stack([su, cu, tau * su, tau * cu])
    axc = np.arange(-half, half + 1e-9, 30.0)
    Xc, Yc = np.meshgrid(axc, axc)
    kc = np.hypot(Xc, Yc) > origin_zone
    mu = np.linspace(-mu_max, mu_max, 5)
    MA, MD = np.meshgrid(mu, mu)
    P = np.array([(x, y, a, d) for x, y in zip(Xc[kc], Yc[kc]) for a, d in zip(MA.ravel(), MD.ravel())])
    e = ((P[:, 0:1] + P[:, 2:3] * tau[None, :]) * su[None, :]
         + (P[:, 1:2] + P[:, 3:4] * tau[None, :]) * cu[None, :]) - shift
    lam_mu = np.mean(metric(e, V4), axis=0)
    m1, m2 = 2.0 * mu_max * np.sqrt(np.diag(lam_mu)[2:])
    LM1, LM2 = m1 + m2, m1 * m2
    return (0.0, LA1, LA2 + LA1 * LM1, LA2 * LM1 + LA1 * LM2, LA2 * LM2)


def lrt_threshold(psi, inst, tau=None, *, false_rate=FALSE_RATE, mu_max=MU_MAX_MAS_YR,
                  half_size=FIELD_HALF_MAS, origin_zone=30.0, eta0=None, free_primary=False):
    """The search's threshold ``Lambda_c`` on the score ``Lambda`` [noise units].

    ``Lambda_c = t^2`` with ``sum_d L_d rho_d(t) = false_rate``, the
    curvatures of :func:`search_curvatures`: approximately the level that a
    candidate searched anywhere in the region would reach on noise alone in
    a fraction ``false_rate`` of campaigns (the expected Euler
    characteristic, a high-threshold approximation), for the same noise
    weight on every sample (the module docstring: THE NOISE). It depends on
    the instrument, the scan law and the region only, not on the S/N. On the
    public Pleiades law: 17.1 static, 19.1 moving at 1 %. Cached per scan
    law and instrument.

    Parameters
    ----------
    psi, inst, tau, mu_max, half_size, origin_zone, eta0, free_primary
        As :func:`search_curvatures`; ``tau`` given makes the search moving.
    false_rate : float
        The false-addition rate per rung, in (0, 1).

    Returns
    -------
    float
        ``Lambda_c``, dimensionless.
    """
    if not 0.0 < float(false_rate) < 1.0:
        raise ValueError(f"false_rate must lie in (0, 1); got {false_rate!r}")
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    key = (tuple(np.round(psi, 12)),
           None if tau is None else tuple(np.round(np.asarray(tau, dtype=float), 12)),
           float(mu_max), float(half_size), float(origin_zone), inst,
           None if eta0 is None else tuple(np.round(np.asarray(eta0, dtype=float), 12)),
           bool(free_primary))
    if key not in _THRESHOLD_CACHE:
        _THRESHOLD_CACHE[key] = search_curvatures(psi, inst, tau, mu_max=mu_max, half_size=half_size,
                                                  origin_zone=origin_zone, eta0=eta0,
                                                  free_primary=free_primary)
    L = _THRESHOLD_CACHE[key]

    def excess(t):
        return sum(Ld * _rho(d, t) for d, Ld in enumerate(L)) - float(false_rate)

    root, _ = brentq(excess, 0.5, 15.0, full_output=True)
    return float(root) ** 2
