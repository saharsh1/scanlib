"""The search: sources on the sky one at a time, each a likelihood-ratio test.

THE EXPORT: on the three static scenes of the reference cases
(``<scene>/search/*``), the campaign harvested with ``k_max = 3`` is
reproduced, and the search on its detections and windows, pinned and with
the primary free, scores every rung with the stored ``Lambda`` against the stored bar,
accepts the same candidates and keeps the same positions and fluxes. The
score is written as a likelihood ratio, ``Lambda = 2 [ln L(s + 1) -
ln L(s)]``; the export wrote the same number as a drop in chi^2.

THE CROWDED FIELD (four sources; in every window one companion within
0.44 FWHM of the primary along the scan, which a window does not resolve at
S/N 100). Its search arrays are the shared package's, re-blessed after the
correction of the line bookkeeping (the export recorded the search before it,
when the lines of a source were judged once, against its first fitted
position). The pool is now rebuilt after every rung against every accepted source at
its latest fitted position, and no line in it passes through an accepted
source. What remains is a known limit of the detections, not of the
bookkeeping: some of D's detections are biased beyond their errors, so they
propose D again, and the fit of that candidate takes unmodelled light
elsewhere. Pinned, the search finds B, C and D and two sources that do not
exist; free, D twice and B, and not C.

THE RULES: a pair yields one source near
the truth and the primary alone none; ``max_sources`` caps the search; a
number replaces the closed-form bar; a drifting companion is found once by
the moving score and several times by the static one; the assignment floor
is applied; two real companions 0.6 FWHM apart are both found; the default
bar is the closed-form ``Lambda_c``; a moving search needs the epoch of
every detection, and the lines of an accepted mover leave the pool against
its fitted track, so they do not propose it again; with the primary
free the default is unchanged, a centring error is absorbed, the accepted
position is the candidate's block, and every scoring fit gets the flag.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scanlib import search as S
from scanlib.backprojection import peak_consistency
from scanlib.instrument import TOY_MARGINS, Instrument, project_al, render_window
from scanlib.joint import fit_joint, loglike_profiled
from scanlib.simulation import PRIMARY_FLUX, Scene, harvest_campaign
from scanlib.thresholds import lrt_threshold

REFERENCE = Path(__file__).resolve().parent / "reference" / "reference_cases.npz"
RTOL, ATOL = 1e-9, 1e-12
INST = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18, margins=TOY_MARGINS)
SIGMA = PRIMARY_FLUX / 100.0

# the scenes and seeds of the export
PAIR = Scene.pair(rho_mas=120.0, pa_deg=35.0, q=0.3)
EXPORT = {
    "static_pair": (PAIR, 20261110),
    "crowded_field": (Scene(np.array([[0.0, 0.0], [90.0, 60.0], [-140.0, 30.0], [40.0, -170.0]]),
                            np.array([1.0, 0.3, 0.15, 0.5]) * PRIMARY_FLUX), 20261112),
    "offset_primary": (Scene(PAIR.positions + np.array([3.0, -2.0]), PAIR.fluxes), 20261113),
}

# a uniform 40-angle law for the mechanics
PSI = np.linspace(0.0, np.pi, 40, endpoint=False) + 0.1
TAUS = np.linspace(-2.5, 2.5, PSI.size)
POS_B = np.array([150.0 * np.sin(np.deg2rad(110.0)), 150.0 * np.cos(np.deg2rad(110.0))])
BAR = 12.0


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REFERENCE))


def _campaign(scene, seed, sigma=SIGMA, law=PSI):
    return harvest_campaign(scene, law, np.random.default_rng(seed), INST, sigma, k_max=3,
                            return_windows=True)


def _ladder(c, **kw):
    return S.map_ladder(c.psi, c.eta, c.sigma, c.windows, PSI, INST, c.sigma_noise, **kw)


@pytest.fixture(scope="module")
def pair():
    scene = Scene(np.array([[0.0, 0.0], POS_B]), np.array([PRIMARY_FLUX, 0.3 * PRIMARY_FLUX]))
    c = _campaign(scene, 1)
    return c, _ladder(c)


# ------------------------------------------------------------------ the export

@pytest.mark.parametrize("name", EXPORT)
def test_the_searches_of_the_export_are_reproduced(ref, name):
    scene, seed = EXPORT[name]
    g = lambda key: ref[f"{name}/search/{key}"]
    assert seed == int(g("seed"))
    psi = ref[f"{name}/inputs/psi"]
    ds = harvest_campaign(scene, psi, np.random.default_rng(seed), INST, SIGMA, k_max=3,
                          return_windows=True)
    for got, key in ((ds.psi, "det_psi"), (ds.eta, "det_eta"), (ds.sigma, "det_sigma"),
                     (ds.windows, "windows")):
        np.testing.assert_allclose(got, g(key), rtol=RTOL, atol=ATOL)
    assert ds.sigma_noise == float(g("sigma_noise"))
    for free, stage in ((False, ""), (True, "_free")):
        res = S.map_ladder(g("det_psi"), g("det_eta"), g("det_sigma"), g("windows"), psi, INST,
                           float(g("sigma_noise")), free_primary=free)
        scored = [r for r in res.log if "Lambda" in r]
        # 2 dlnL against the export's chi^2 drop: the same fits, ln L ~ -3700 nats cancelling, so an
        # absolute floor of about 1e-11 on a score that is zero (the crowded field's last rung)
        np.testing.assert_allclose([r["Lambda"] for r in scored], g(f"Lambda{stage}"), rtol=RTOL,
                                   atol=1e-9)
        np.testing.assert_allclose([r["bar"] for r in scored], g(f"bar{stage}"), rtol=RTOL)
        assert [r["verdict"] == "accepted" for r in scored] == g(f"accepted{stage}").tolist()
        np.testing.assert_allclose(res.positions, g(f"positions{stage}"), rtol=RTOL, atol=1e-9)
        np.testing.assert_allclose(res.fluxes, g(f"fluxes{stage}"), rtol=RTOL)


def _crowded_ladder(ref, free):
    g = lambda key: ref[f"crowded_field/search/{key}"]
    return S.map_ladder(g("det_psi"), g("det_eta"), g("det_sigma"), g("windows"),
                        ref["crowded_field/inputs/psi"], INST, float(g("sigma_noise")),
                        free_primary=free)


def _latest_sources(entry, free):
    """The accepted sources' positions in a rung's joint fit, the primary's block left out."""
    blocks = np.asarray(entry["fit"]["theta"], dtype=float).reshape(-1, 2)
    return blocks[1:] if free else blocks


@pytest.mark.parametrize("free, expected", [(False, ["D", "false", "false", "B", "C"]),
                                             (True, ["D", "false", "B"])])
def test_the_crowded_fields_known_limit(ref, free, expected):
    """The accepted sources, each named by the true companion within 15 mas
    of it or ``"false"``, at their positions in the last accepted joint fit.
    Pinned, that fit shares B's light between the second source (18.6 mas off,
    flux 103) and the fourth (9.3 mas, flux 200), and the third has flux 0;
    free, the second source sits at (250, -180) mas with flux 17. While each
    source kept its own rung's estimate the names were
    ``D, B, false, false, C`` and ``D, D, B``; before the correction of the
    line bookkeeping the pinned search stopped before C (``D, B, false,
    false``) and the free one after ``D, D`` (the export)."""
    truth = {"B": (90.0, 60.0), "C": (-140.0, 30.0), "D": (40.0, -170.0)}
    ladder = _crowded_ladder(ref, free)
    names = []
    for x in ladder.positions:
        name, d = min(((n, np.hypot(*(x - np.array(t)))) for n, t in truth.items()),
                      key=lambda nd: nd[1])
        names.append(name if d < 15.0 else "false")
    assert names == expected


@pytest.mark.parametrize("free", [False, True])
def test_no_line_in_the_pool_passes_through_an_accepted_source(ref, free):
    """After every rung, judged at the narrowest tolerance (k_sigma of the
    line's own error, floored by the assignment floor), no line of the pool
    the next rung sees passes through any accepted source at its position in
    the last joint fit."""
    g = lambda key: ref[f"crowded_field/search/{key}"]
    psi, eta, sigma = g("det_psi"), g("det_eta"), g("det_sigma")
    ladder = _crowded_ladder(ref, free)
    for before, entry in zip(ladder.log, ladder.log[1:]):
        pool = entry["pool"]
        for x in _latest_sources(before, free):
            through, _ = peak_consistency(psi[pool], eta[pool], sigma[pool], x,
                                          S.ASSIGN_FLOOR_MAS, k_sigma=3.0)
            assert not through.any(), f"rung {entry['rung']}: {through.sum()} lines through {x}"


# ------------------------------------------------------------------ the mechanics

def test_a_pair_yields_one_source_near_the_truth(pair):
    _c, ladder = pair
    assert isinstance(ladder, S.MapLadder) and ladder.n_sources == 1
    assert np.hypot(*(ladder.positions[0] - POS_B)) < 3.0
    assert 0.2 < ladder.fluxes[0] / PRIMARY_FLUX < 0.4
    first = ladder.log[0]
    assert first["verdict"] == "accepted" and first["Lambda"] > first["bar"]
    assert first["n_consistent"] >= 3 and first["pool"].all()
    assert "rung 0" in S.describe(ladder)


def test_the_primary_alone_yields_none():
    c = _campaign(Scene(np.array([[0.0, 0.0]]), np.array([PRIMARY_FLUX])), 3)
    ladder = _ladder(c)
    assert ladder.n_sources == 0 and ladder.positions.shape == (0, 2)
    assert all(entry["verdict"].startswith("stop") for entry in ladder.log)


def test_max_sources_caps_the_search(pair):
    c, _ = pair
    capped = _ladder(c, max_sources=0)
    assert capped.n_sources == 0 and capped.log == []


def test_a_number_replaces_the_closed_form_bar(pair):
    c, ladder = pair
    frozen = _ladder(c, bar=BAR)
    assert frozen.n_sources == 1 and np.allclose(frozen.positions, ladder.positions)
    assert all(e["bar"] == BAR for e in frozen.log if "bar" in e)
    for bad in (None, "bic"):
        with pytest.raises(ValueError, match="bar"):
            _ladder(c, bar=bad)


def test_the_default_bar_is_the_closed_form_threshold(pair):
    c, ladder = pair
    assert all(e["bar"] == lrt_threshold(PSI, INST) for e in ladder.log if "bar" in e)
    free = _ladder(c, free_primary=True)
    assert all(e["bar"] == lrt_threshold(PSI, INST, free_primary=True) for e in free.log if "bar" in e)


def test_the_moving_score_finds_a_drifting_companion_once():
    scene = Scene(np.array([[0.0, 0.0], POS_B]), np.array([PRIMARY_FLUX, 0.3 * PRIMARY_FLUX]),
                  pm=np.array([[0.0, 0.0], [5.0, -6.0]]))
    c = _campaign(scene, 7, sigma=PRIMARY_FLUX / 600.0, law=(PSI, TAUS))
    moving = _ladder(c, bar=20.0, tau=TAUS, det_tau=c.tau)
    assert moving.n_sources == 1 and np.hypot(*(moving.positions[0] - POS_B)) < 3.0
    assert [e["verdict"] for e in moving.log] == ["accepted"]      # its lines leave with it: no second candidate
    static = _ladder(c, bar=20.0)
    assert static.n_sources > 1                       # the drift spawns extra static sources
    free = _ladder(c, bar=20.0, tau=TAUS, det_tau=c.tau, free_primary=True)
    assert free.n_sources == 1 and np.hypot(*(free.positions[0] - POS_B)) < 3.0
    assert free.log[0]["fit"]["theta"].size == 8


def test_the_assignment_floor_is_applied():
    psi = PSI[:6]
    found = S.next_peak(psi, 100.0 * np.sin(psi), np.full(6, 0.01), assign_floor=2.0)
    assert found is not None and found[1] >= 2.0 and found[2].all()
    assert S.next_peak(np.array([]), np.array([]), np.array([])) is None


def test_two_close_real_companions_are_both_found():
    """Two companions 60 mas (0.6 FWHM) apart, 200 mas from the primary: the
    wider tolerance of an accepted source must not take its neighbour's
    lines."""
    b, c = np.array([200.0, 0.0]), np.array([200.0, 60.0])
    scene = Scene(np.array([[0.0, 0.0], b, c]), np.array([1.0, 0.3, 0.2]) * PRIMARY_FLUX)
    ladder = _ladder(_campaign(scene, 5))
    assert ladder.n_sources == 2
    found = _latest_sources(ladder.log[1], False)
    assert np.hypot(*(found[0] - b)) < 5.0 and np.hypot(*(found[1] - c)) < 5.0


def test_one_rungs_score_is_twice_the_log_likelihood_gain(pair):
    c, ladder = pair
    peak = ladder.log[0]["peak"]
    value, fit = S.lambda_rung(c.windows, PSI, INST, c.sigma_noise, [], peak)
    without = loglike_profiled(np.zeros(0), c.windows, PSI, INST, c.sigma_noise)
    with_candidate = fit_joint(c.windows, PSI, np.asarray(peak, dtype=float), INST, c.sigma_noise)
    assert value == 2.0 * (with_candidate["loglike"] - without)
    _l0, f0 = loglike_profiled(np.zeros(0), c.windows, PSI, INST, c.sigma_noise, return_fluxes=True)
    chi2_without = np.sum((c.windows.ravel() - np.tile(INST.design_columns([0.0]) @ f0,
                                                        PSI.size)) ** 2) / c.sigma_noise ** 2
    assert value == pytest.approx(chi2_without - with_candidate["chi2"], rel=1e-9)
    assert np.array_equal(fit["theta"], with_candidate["theta"])
    assert value == ladder.log[0]["Lambda"]


# ------------------------------------------------------------------ the free primary

def test_the_default_is_the_pinned_search_bit_for_bit(pair):
    c, _ = pair
    before, after = _ladder(c, bar=BAR), _ladder(c, bar=BAR, free_primary=False)
    assert np.array_equal(before.positions, after.positions)
    for a, b in zip(before.log, after.log):
        assert a["verdict"] == b["verdict"] and a.get("Lambda") == b.get("Lambda")
    assert before.log[0]["fit"]["theta"].size == 2


def test_the_free_primary_absorbs_a_centring_error():
    offset = np.array([3.0, 0.0])
    scene = Scene(np.array([offset, POS_B + offset]), np.array([PRIMARY_FLUX, 0.3 * PRIMARY_FLUX]))
    c = _campaign(scene, 11, sigma=PRIMARY_FLUX / 600.0)
    seed = POS_B + offset
    _lp, fit_pinned = S.lambda_rung(c.windows, PSI, INST, c.sigma_noise, [], seed)
    _lf, fit_free = S.lambda_rung(c.windows, PSI, INST, c.sigma_noise, [], seed, free_primary=True)
    assert fit_pinned["chi2"] - fit_free["chi2"] > 100.0
    assert fit_free["theta"].size == 4 and np.hypot(*(fit_free["theta"][:2] - offset)) < 1.0
    assert np.hypot(*(fit_free["theta"][2:] - fit_free["theta"][:2] - POS_B)) < 3.0


def test_the_accepted_position_is_the_candidates_block(pair):
    c, _ = pair
    lad = _ladder(c, bar=BAR, free_primary=True)
    assert lad.n_sources == 1 and np.hypot(*(lad.positions[0] - POS_B)) < 3.0
    assert np.array_equal(lad.log[0]["position"], lad.positions[0])
    assert lad.log[0]["fit"]["theta"].size == 4 and np.hypot(*lad.log[0]["fit"]["theta"][:2]) < 1.0


# ------------------------------------------------------------------ the returned sources (S1)

# two companions 60 mas apart, noise-free: the first rung's one-source fit absorbs both and is
# biased (about (200, 23) mas, flux 470); the two-source fit of the second rung separates them
S1_TRUE = np.array([[200.0, 0.0], [200.0, 60.0]])
S1_PSI = np.linspace(0.0, np.pi, 24, endpoint=False) + 0.03
S1_TAU = np.linspace(-2.5, 2.5, S1_PSI.size)


def _s1_ladder(bar=10.0, **kw):
    windows = np.array([render_window(np.r_[0.0, [project_al(*x, p) for x in S1_TRUE]],
                                      [1000.0, 300.0, 200.0], INST) for p in S1_PSI])
    det_eta = np.concatenate([project_al(*x, S1_PSI) for x in S1_TRUE])
    return S.map_ladder(np.tile(S1_PSI, 2), det_eta, np.ones(det_eta.size), windows, S1_PSI, INST,
                        1.0, bar=bar, **kw)


def _last_accepted(ladder, free, moving):
    """The companions' positions and fluxes in the last accepted rung's joint fit."""
    fit = [e for e in ladder.log if e.get("verdict") == "accepted"][-1]["fit"]
    blocks = np.asarray(fit["theta"], dtype=float).reshape(-1, 4 if moving else 2)
    return (blocks[1:] if free else blocks)[:, :2], np.asarray(fit["flux"], dtype=float)[1:]


@pytest.mark.parametrize("free", [False, True], ids=["pinned", "free"])
@pytest.mark.parametrize("moving", [False, True], ids=["static", "moving"])
def test_the_returned_sources_are_the_last_accepted_joint_fits(free, moving):
    ladder = _s1_ladder(free_primary=free, max_sources=2,
                        **(dict(tau=S1_TAU, det_tau=np.tile(S1_TAU, 2)) if moving else {}))
    assert ladder.n_sources == 2
    positions, fluxes = _last_accepted(ladder, free, moving)
    assert np.array_equal(ladder.positions, positions) and np.array_equal(ladder.fluxes, fluxes)
    np.testing.assert_allclose(ladder.positions, S1_TRUE, atol=0.5)
    np.testing.assert_allclose(ladder.fluxes, [300.0, 200.0], rtol=1e-3)
    # the first source's own rung, the biased one-source fit, stays in the log
    assert np.hypot(*(ladder.log[0]["position"] - S1_TRUE[0])) > 10.0


def test_a_rejected_last_rung_does_not_supply_the_sources():
    # the scores are about 1.9e6 and 2.4e4: at a bar of 1e5 the second candidate is rejected,
    # and its two-source fit must not replace the accepted one-source fit
    ladder = _s1_ladder(bar=1e5)
    assert [e["verdict"] == "accepted" for e in ladder.log] == [True, False]
    positions, fluxes = _last_accepted(ladder, False, False)
    assert np.array_equal(ladder.positions, positions) and np.array_equal(ladder.fluxes, fluxes)
    assert ladder.n_sources == 1 and np.array_equal(ladder.positions[0], ladder.log[0]["position"])


def test_no_accepted_source_returns_empty_arrays():
    ladder = _s1_ladder(bar=1e12)
    assert ladder.positions.shape == (0, 2) and ladder.fluxes.shape == (0,)


@pytest.mark.parametrize("moving", [False, True])
def test_every_scoring_fit_gets_the_flag(pair, monkeypatch, moving):
    c, _ = pair
    recorded = []
    real = S.fit_joint

    def spy(*a, **kw):
        recorded.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(S, "fit_joint", spy)
    det_tau = TAUS[np.searchsorted(PSI, c.psi)]               # each detection's window, hence its epoch
    _ladder(c, bar=BAR, free_primary=True, **(dict(tau=TAUS, det_tau=det_tau) if moving else {}))
    assert recorded and all(call.get("free_primary") is True for call in recorded)
    assert all((call.get("tau") is not None) == moving for call in recorded)


# ------------------------------------------------------------------ a mover's lines (S3)

# a mover A (8, -4) mas/yr and a faint static D, noise-free: A's detections at the ends of the
# campaign lie up to 22 mas from A's reference-epoch position along the scan
S3_PSI = np.linspace(0.0, 2 * np.pi, 40, endpoint=False) + 0.03
S3_TAU = np.linspace(-2.5, 2.5, 40)
S3_A, S3_MU, S3_D = np.array([180.0, 40.0]), np.array([8.0, -4.0]), np.array([-150.0, -120.0])
S3_SEEN = np.arange(0, 40, 5)                              # D is detected in 8 windows


def _s3_ladder(**kw):
    eta_a = project_al(S3_A[0] + S3_MU[0] * S3_TAU, S3_A[1] + S3_MU[1] * S3_TAU, S3_PSI)
    eta_d = project_al(*S3_D, S3_PSI)
    windows = np.array([render_window([0.0, a, d], [1000.0, 300.0, 60.0], INST)
                        for a, d in zip(eta_a, eta_d)])
    return S.map_ladder(np.r_[S3_PSI, S3_PSI[S3_SEEN]], np.r_[eta_a, eta_d[S3_SEEN]],
                        np.r_[np.full(40, 0.3), np.full(8, 4.0)], windows, S3_PSI, INST, 1.0,
                        tau=S3_TAU, bar=20.0, **kw)


def test_the_lines_of_an_accepted_mover_leave_the_pool():
    """Each line is compared with the source's fitted position at the line's own epoch (Eq. 17).
    Compared with the reference-epoch position, 21 of the mover's 40 lines stayed in the pool,
    proposed it again, and the fit of that candidate ran to the corner (600, -541) and passed the
    threshold (Lambda 184), while D was never found."""
    ladder = _s3_ladder(det_tau=np.r_[S3_TAU, S3_TAU[S3_SEEN]])
    assert ladder.log[1]["n_pool"] == S3_SEEN.size                 # after A, only D's lines are left
    assert ladder.n_sources == 2
    np.testing.assert_allclose(ladder.positions, [S3_A, S3_D], atol=1.0)


def test_a_moving_search_needs_the_epoch_of_every_detection():
    with pytest.raises(ValueError, match="det_tau"):
        _s3_ladder()                                               # tau without det_tau
    with pytest.raises(ValueError, match="det_tau"):
        _s3_ladder(det_tau=S3_TAU)                                 # 40 epochs for 48 detections
    c = _campaign(PAIR, 1)
    with pytest.raises(ValueError, match="det_tau"):
        _ladder(c, det_tau=np.zeros(c.eta.size))                   # det_tau in a static search


def test_a_line_is_judged_against_the_sources_track_and_its_error():
    """``_unassigned`` with the detections' epochs: the source is at ``x + mu tau_i`` for line i,
    and the variance of that position along the scan comes from the fit's 4x4 block, motion
    included: ``g_i = (sin, cos, tau sin, tau cos)``."""
    psi = np.array([0.3, 0.3, 1.2, 1.2, 2.0])
    tau = np.array([0.0, 2.5, -2.5, 2.5, 0.0])
    x, mu = np.array([100.0, -50.0]), np.array([6.0, -3.0])
    on_track = project_al(x[0] + mu[0] * tau, x[1] + mu[1] * tau, psi)
    eta = on_track + np.array([0.0, 0.0, 0.0, 5.0, 5.0])           # two lines 5 mas off the track
    fit = dict(theta=np.r_[x, mu], cov=np.diag([0.01, 0.01, 1.0, 1.0]))   # the motion known to 1 mas/yr
    kw = dict(moving=True, k_sigma=3.0)
    pool = S._unassigned(psi, eta, np.full(5, 0.3), fit, [2.0], det_tau=tau, **kw)
    # 5 mas off at tau = 2.5 is inside 3 sqrt(0.3^2 + 0.01 + 2.5^2) = 7.6 mas; at tau = 0 the band is 2 mas
    assert pool.tolist() == [False, False, False, False, True]
    # without the epochs the source is taken at its reference-epoch position: the lines at +-2.5 yr stay
    pool_ref = S._unassigned(psi, eta, np.full(5, 0.3), fit, [2.0], **kw)
    assert pool_ref.tolist() == [False, True, True, True, True]
