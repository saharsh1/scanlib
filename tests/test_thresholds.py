"""The search's threshold Lambda_c, in closed form, and its simulated null.

THE EXPORT: ``lrt_threshold`` on both public laws, static and moving, the
primary pinned and free (``thresholds/{pleiades,beta60}/*``), and the
maximum of the candidate field in three noise-only campaigns of
``field_null_max`` (``thresholds/null/*``), reproduced. THE RULES: the
closed form gives the levels checked against simulated noise (17.07 static and 19.13 moving on the Pleiades law,
17.31 static on the 95-transit law); a looser false rate lowers the level
and a wider motion range raises it; with the primary free the level moves by
less than 0.05; the free metric with the flux column alone is the pinned
metric; ``field_null_max`` reproduces the first campaign of the stored null
bit for bit. The cache of the thresholds keeps instruments apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scanlib import thresholds as T
from scanlib.instrument import TOY_MARGINS, Instrument
from scanlib.scanlaw import BETA60, PLEIADES, load_scan_law
from scanlib.simulation import field_null_max, random_zero_points

REFERENCE = Path(__file__).resolve().parent / "reference" / "reference_cases.npz"
RTOL = 1e-9
INST = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18, margins=TOY_MARGINS)
PSI40 = np.linspace(0.0, np.pi, 40, endpoint=False) + 0.1


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


# ------------------------------------------------------------------ the export

@pytest.mark.parametrize("tag, law", [("pleiades", PLEIADES), ("beta60", BETA60)])
def test_the_thresholds_of_the_export_are_reproduced(ref, tag, law):
    psi, tau = load_scan_law(law)
    for moving in (False, True):
        for free in (False, True):
            key = f"thresholds/{tag}/lrt_{'moving' if moving else 'static'}_{'free' if free else 'pinned'}"
            got = T.lrt_threshold(psi, INST, tau if moving else None, free_primary=free)
            assert got == pytest.approx(float(ref[key]), rel=RTOL), key


def test_the_simulated_null_of_the_export_is_reproduced(ref):
    psi, tau = load_scan_law(PLEIADES)
    seed = int(ref["thresholds/null/seed"])
    for free in (False, True):
        z = field_null_max(psi, tau, 3, np.random.default_rng(seed), INST, free_primary=free)
        np.testing.assert_allclose(z, ref[f"thresholds/null/max_z_{'free' if free else 'pinned'}"],
                                   rtol=RTOL)


# ------------------------------------------------------------------ the closed form

def test_the_closed_form_gives_the_levels_checked_against_simulation():
    psi, tau = load_scan_law(PLEIADES)
    assert T.lrt_threshold(psi, INST) == pytest.approx(17.07, abs=0.05)
    assert T.lrt_threshold(psi, INST, tau) == pytest.approx(19.13, abs=0.1)
    assert T.lrt_threshold(load_scan_law(BETA60)[0], INST) == pytest.approx(17.31, abs=0.05)
    assert T.lrt_threshold(psi, INST, false_rate=0.05) < T.lrt_threshold(psi, INST)
    assert T.lrt_threshold(psi, INST, tau, mu_max=20.0) > T.lrt_threshold(psi, INST, tau)
    with pytest.raises(ValueError, match="false_rate"):
        T.lrt_threshold(psi, INST, false_rate=1.5)
    assert len(T.search_curvatures(psi, INST)) == 3 and len(T.search_curvatures(psi, INST, tau)) == 5


def test_the_free_primary_moves_the_level_by_less_than_005():
    psi, tau = load_scan_law(PLEIADES)
    eta0 = random_zero_points(np.random.default_rng([20260929, 3]), psi.size, INST)
    assert T.lrt_threshold(psi, INST, free_primary=True) == pytest.approx(17.04, abs=0.01)
    assert T.lrt_threshold(psi, INST, tau, free_primary=True) == pytest.approx(19.12, abs=0.01)
    free, pinned = (T.lrt_threshold(psi, INST, tau, eta0=eta0, free_primary=f) for f in (True, False))
    assert free == pytest.approx(19.13, abs=0.01) and abs(free - pinned) < 0.05


def test_the_free_metric_with_the_flux_column_alone_is_the_pinned_metric():
    eta, fns, l0l0 = T._template_functions(INST, 900.0)
    e = np.random.default_rng(3).uniform(-500.0, 500.0, (50, PSI40.size))
    V = np.stack([np.sin(PSI40), np.cos(PSI40)])
    a0a0 = PSI40.size * l0l0
    pinned = T._field_metric(e, V, fns, float(eta[0]), a0a0)
    free = T._field_metric_free(e, V, fns, float(eta[0]), np.ones((1, PSI40.size)), np.array([0]),
                                np.array([[1.0 / a0a0]]))
    np.testing.assert_allclose(free, pinned, rtol=1e-10, atol=0.0)


def test_the_cache_keeps_instruments_apart():
    wide = Instrument.gaussian(fwhm_mas=140.0)
    assert T.lrt_threshold(PSI40, INST) != T.lrt_threshold(PSI40, wide)
    assert T.lrt_threshold(PSI40, wide) == T.lrt_threshold(PSI40, wide)


# ------------------------------------------------------------------ the simulated null

@pytest.mark.slow
def test_the_simulated_null_reproduces_the_stored_campaign():
    """The first noise-only campaign of seed 101 on the Pleiades law, moving,
    the primary free, uniform noise: the record of the stored check of
    Lambda_c, bit for bit; a sigma of ones is the uniform case."""
    psi, tau = load_scan_law(PLEIADES)
    taus = tau - tau.mean()
    z = field_null_max(psi, taus, 1, np.random.default_rng(101), INST, free_primary=True)
    assert z.tolist() == [2.3718388248777487]
    z1 = field_null_max(psi, taus, 1, np.random.default_rng(101), INST, free_primary=True,
                        sigma_noise=np.ones(18))
    assert np.array_equal(z1, z)
