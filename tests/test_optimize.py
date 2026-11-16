"""``minimize_restarts``: what ``converged`` means.

``converged`` is True only when a restart could not improve the objective
AND that restart ended inside its own iteration budget with a finite value.
A restart that runs out of budget before it moves also "cannot improve", and
before this rule it was reported as convergence at a point that is not a
minimum. An objective that is not finite at the start is refused.
"""
from __future__ import annotations

import numpy as np
import pytest

from scanlib.optimize import minimize_restarts


def _parabola(x):
    """``(x + 5)^2`` summed: the minimum 0 at x = -5, the value 25 at x = 0 in one dimension
    (the audit's case: one Nelder-Mead iteration from 0 does not improve it at all)."""
    return float(np.sum((np.asarray(x) + 5.0) ** 2))


def test_a_restart_that_ran_out_of_budget_is_not_convergence():
    res, n_restarts, converged = minimize_restarts(_parabola, [0.0], maxiter_first=1,
                                                   maxiter_restart=1)
    assert not converged
    assert res.x[0] == 0.0 and res.fun == 25.0           # the start, not the minimum


def test_a_restart_inside_its_budget_confirms_the_minimum():
    res, n_restarts, converged = minimize_restarts(_parabola, [0.0, 0.0])
    assert converged and 1 <= n_restarts <= 5
    np.testing.assert_allclose(res.x, [-5.0, -5.0], atol=1e-3)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_an_objective_that_is_not_finite_at_the_start_is_refused(bad):
    with pytest.raises(ValueError, match="not finite"):
        minimize_restarts(lambda x: bad, [0.0])
