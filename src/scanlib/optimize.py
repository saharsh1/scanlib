"""The one optimizer of the package: restarted bounded Nelder-Mead.

The joint fits of :mod:`scanlib.joint` minimize a profiled chi^2, the
non-negative fluxes solved exactly at every trial position. That objective
is piecewise smooth: smooth where the set of non-zero fluxes is constant,
kinked where a flux enters or leaves it. Nelder-Mead has no convergence
theory on a kink; in practice the simplex collapses onto one, reports
success, and stops. :func:`minimize_restarts` is the rule that handles it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def minimize_restarts(fun, x0, bounds=None, xatol=1e-4, maxiter_first=4000,
                      maxiter_restart=16000, max_restarts=5, ftol_rel=1e-9):
    """Bounded Nelder-Mead, restarted until a restart stops helping.

    The profiled chi^2 is piecewise smooth: smooth where the set of non-zero
    fluxes is constant, kinked where a flux enters or leaves it. Nelder-Mead
    has no convergence theory on a kink: the simplex collapses onto one and
    stops, reporting success. So the rule is: minimize from ``x0``, then
    restart from the result with a fresh simplex, and stop when a restart
    improves the objective by no more than ``ftol_rel max(1, f_previous)``,
    or when ``max_restarts`` are spent. A fresh simplex at the same point is
    what escapes a collapsed one; the final non-improving restart is the
    evidence of convergence and is counted, provided it ended inside its own
    iteration budget at a finite value: a restart that runs out of iterations
    before it moves also improves nothing, and that is not convergence.
    Scipy's ``success`` flag is consulted for that alone (it says the
    budget was not exhausted); on a kink it is True whether or not the
    simplex collapsed, so it is never read as convergence. Nelder-Mead's
    absolute function tolerance is set per call to
    ``fatol = ftol_rel max(1, f(x0))``, reachable at any data scale (a fixed
    literal such as 1e-12 is below the resolution of a chi^2 of 1e4).

    Parameters
    ----------
    fun : callable
        The objective, ``fun(x) -> float``, with ``x`` an array of shape (P,).
    x0 : array, shape (P,)
        The starting point, in the parameters' own units.
    bounds : sequence of P (low, high) pairs, optional
        Bounds on each parameter; ``None`` in a pair means unbounded.
    xatol : float, default 1e-4
        The absolute simplex tolerance on the parameters, in their own units.
    maxiter_first : int, default 4000
        The iteration budget of the first minimization.
    maxiter_restart : int, default 16000
        The iteration budget of each restart.
    max_restarts : int, default 5
        The cap on restarts, so a pathological surface cannot loop forever.
    ftol_rel : float, default 1e-9
        The relative improvement below which a restart counts as confirming.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        The best result seen; ``res.x`` the minimizer, ``res.fun`` its value.
    n_restarts : int
        The number of restarts performed, the confirming one included.
    converged : bool
        ``True`` when the loop stopped because a restart could not improve
        and that restart ended inside its budget at a finite value;
        ``False`` when the restarts ran out while still improving, or the
        confirming restart ran out of iterations or ended non-finite.

    Raises
    ------
    ValueError
        When the objective is not finite at ``x0``.
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    f0 = float(fun(x0))
    if not np.isfinite(f0):
        raise ValueError(f"the objective is not finite at the start x0: {f0}")
    fatol = ftol_rel * max(1.0, f0)
    res = minimize(fun, x0, method="Nelder-Mead", bounds=bounds,
                   options={"xatol": xatol, "fatol": fatol, "maxiter": maxiter_first})
    lo = hi = None
    if bounds is not None:
        lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
        hi = np.array([np.inf if b[1] is None else b[1] for b in bounds])
    n_restarts, converged = 0, False
    for _ in range(max_restarts):
        prev_fun = float(res.fun)
        x_start = np.asarray(res.x, dtype=float)
        if bounds is not None:
            x_start = np.clip(x_start, lo, hi)   # float error can leave it a hair outside
        trial = minimize(fun, x_start, method="Nelder-Mead", bounds=bounds,
                         options={"xatol": xatol, "fatol": fatol,
                                  "maxiter": maxiter_restart})
        n_restarts += 1
        gain = prev_fun - float(trial.fun)
        if gain > 0.0:
            res = trial
        if gain <= ftol_rel * max(1.0, prev_fun):
            converged = bool(trial.success) and bool(np.isfinite(trial.fun))
            break
    return res, n_restarts, converged


