"""The search: sources on the sky one at a time, each a likelihood-ratio test.

The campaign-level counterpart of the per-window ladder
(:mod:`scanlib.reduction`): with ``s`` sources accepted so far, a candidate
is PROPOSED by the map of the detections not yet assigned to an accepted
source, SCORED by the joint fit of every window, and ACCEPTED by a
likelihood-ratio test of the scene with and without it. Per rung:

1. pool the unassigned detections; the highest peak of their map outside
   the primary's zone (:func:`scanlib.backprojection.map_peak`) is the
   candidate, and the detections whose line passes within ``k_sigma`` of
   their own error, floored by the peak's width, are its lines
   (:func:`scanlib.backprojection.peak_consistency`). Fewer than
   :data:`GATE_MIN_DETECTIONS` lines: stop;
2. the score

       Lambda_s = 2 [ln L(s + 1) - ln L(s)],

   the gain in the profiled log-likelihood of the joint fit to ALL windows
   (:func:`scanlib.joint.fit_joint`) when the candidate, seeded at the map's
   peak, is added to the ``s`` accepted sources; for Gaussian noise of known
   sigma it is the drop in chi^2, a pure number that does not depend on the
   noise level (:func:`lambda_rung`);
3. the candidate is accepted when ``Lambda_s`` exceeds the threshold
   ``Lambda_c`` that noise alone would reach in a fraction ``p`` of
   campaigns for a candidate searched anywhere in the field
   (:func:`scanlib.thresholds.lrt_threshold`; 17.1 static and 19.1 moving at
   ``p = 1 %`` on the public Pleiades law: a closed-form approximation for
   the same noise weight on every sample, checked by simulation for one
   per-sample noise model, :mod:`scanlib.thresholds`);
4. accepted: the joint fit's position is kept, the pool is rebuilt (below),
   and the search returns to 1.

THE POOL is rebuilt after every accepted rung from ALL the detections: a
detection leaves it when its line passes through any accepted source at
that source's position in the rung's joint fit, the latest and best
estimate (:func:`_unassigned`). Line i passes through source k when

    |eta_i - (dra*_k sin psi_i + ddec_k cos psi_i)|
        <= max(k_sigma sqrt(sigma_i^2 + s_ik^2), w_k),

with ``sigma_i`` the line's error, ``s_ik^2 = g_i^T C_k g_i`` the variance
of the source's position along the line's scan, ``g_i = (sin psi_i, cos
psi_i)`` and ``C_k`` the source's 2x2 position covariance from the fit, and
``w_k`` the map's width at the peak that proposed it (floored by
``assign_floor``). A fit whose covariance is ill-conditioned gives
``s_ik = 0``.

IN A MOVING SEARCH the source is taken where it was when the line was
measured: with ``tau_i`` the epoch of detection i (``det_tau``),

    |eta_i - ((dra*_k + mu_a*_k tau_i) sin psi_i + (ddec_k + mu_d_k tau_i) cos psi_i)|
        <= max(k_sigma sqrt(sigma_i^2 + s_ik^2), w_k),

``g_i = (sin psi_i, cos psi_i, tau_i sin psi_i, tau_i cos psi_i)`` and
``C_k`` the source's 4x4 block, motion and cross terms included. This is
the line moved to the reference epoch with the fitted motion
(:func:`scanlib.dynamics.to_reference_epoch`) and compared with the fitted
position. Compared with the reference-epoch position alone, the lines a
moving source left at the ends of the campaign stay in the pool and
propose it again.

A source fitted before its neighbours are in the model is off its
truth; judged against that first position alone, it would keep its most
precise lines in the pool, and they would propose it again.

KNOWN LIMIT: the detections. A window reduces to at most ``k_max`` sources
and does not resolve a faint companion close to the primary. The light of
a source left out of a window's model biases that window's other
detections beyond their stated errors, and no tolerance built from those
errors removes the biased lines of an accepted source. They propose it
again, and the fit of that candidate can move to unmodelled light
elsewhere and pass the threshold. Seen on a field of four sources, one
companion 0.24-0.44 FWHM from the primary along the scan in every window
at S/N 100 (a fourth rung of the window's ladder resolves it in 1 window
of 40): the search finds every companion with the primary pinned, and two
sources that do not exist; with the primary free it misses one companion.

The threshold prices a search over the whole field, so it does not depend
on how a candidate was proposed: the map only proposes, and the joint fit
seeded there measures. A blended companion's map peak can sit tens of mas
off its position (the per-window blend's pull).

TWO LESSONS OF A BRIGHT, MOVING SCENE. The score must use the model the
scene needs: scored with the static fit, a drifting companion leaves a
misfit that extra static sources absorb, so a moving scene is scored with
the moving fit (``tau`` given, every proper motion within ``+-mu_max``), and
its threshold prices the searched motion range as well. And the lines of an
accepted source are assigned with a floor on the radius (``assign_floor``):
at high S/N a window's error is a tenth of a mas while its systematic error
is larger, and ``3 sigma`` alone misses the source's own lines.

THE PRIMARY is pinned at the reference position in every scoring fit by
default, the case the closed-form threshold is derived for; with
``free_primary`` it has its own block in front of the others, with no prior,
and the threshold is the free primary's.

THE WINDOWS' ZERO POINTS ``eta0`` (one per window, [mas]) reach the score's
joint fits and the threshold; the detections are on the sky already.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backprojection import map_peak, peak_consistency
from .joint import fit_joint, loglike_profiled
from .thresholds import FALSE_RATE, FIELD_HALF_MAS, MU_MAX_MAS_YR, lrt_threshold

#: The detections a candidate needs to be testable: two lines always cross,
#: a third can disagree.
GATE_MIN_DETECTIONS = 3
#: The pixel [mas] of the map that proposes candidates.
MAP_STEP_MAS = 2.0
#: The least radius [mas] a line is judged against when it is assigned to an
#: accepted source: the map's pixel, a systematic floor under errors that
#: shrink below it at bright magnitudes.
ASSIGN_FLOOR_MAS = 2.0


@dataclass
class MapLadder:
    """The accepted sources and the record of every rung.

    ``positions`` (S, 2): the sky offsets [mas] of the accepted sources in
    the last accepted rung's joint fit, which fits all of them together,
    measured from the reference position (at the reference epoch in a moving
    search), in the order accepted, the primary's block left out;
    ``fluxes`` (S,), their fluxes in that fit, in the windows' units. A
    rejected rung's fit never supplies them. ``log``, one dict per rung:
    ``rung``, ``peak`` (the
    map's proposal), ``width`` [mas], ``n_pool``, ``n_consistent``, ``pool``
    (the mask of unassigned detections the rung saw) and, when the candidate
    was scored, ``Lambda``, ``bar``, ``fit`` (the joint fit with the
    candidate) and ``verdict`` (``"accepted"`` with ``position``, the
    candidate's estimate in its own rung, or the reason to stop).
    """

    positions: np.ndarray
    fluxes: np.ndarray
    log: list = field(default_factory=list)

    @property
    def n_sources(self):
        """The number of accepted sources."""
        return int(self.positions.shape[0])


def next_peak(psi, eta, sigma, *, origin_zone=30.0, k_sigma=3.0, half_size=FIELD_HALF_MAS,
              step=MAP_STEP_MAS, assign_floor=ASSIGN_FLOOR_MAS):
    """The candidate a pool of detections proposes.

    Parameters
    ----------
    psi, eta, sigma : arrays, shape (D,)
        The detections: scan angle [rad], along-scan position [mas], error
        [mas].
    origin_zone : float
        The primary's zone [mas], in which no peak is taken.
    k_sigma : float
        The consistency band in units of each detection's own sigma.
    half_size, step : float
        The map's half side and pixel [mas].
    assign_floor : float
        The least width [mas] of the peak.

    Returns
    -------
    None, or (peak, width, keep)
        ``None`` when the map has no peak; else the peak ``(dra*, ddec)``
        [mas], its width floored by ``assign_floor`` [mas], and the mask of
        the detections consistent with it.
    """
    try:
        peak, width, _sky, _axis = map_peak(psi, eta, sigma, half_size, step, origin_zone)
    except ValueError:                       # no local maximum outside the zone
        return None
    width = max(float(width), float(assign_floor))
    keep, _distance = peak_consistency(psi, eta, sigma, peak, width, k_sigma=k_sigma)
    return np.asarray(peak, dtype=float), width, keep


def lambda_rung(windows, psi, inst, sigma_noise, accepted, candidate, *, tau=None,
                mu_max=MU_MAX_MAS_YR, eta0=None, free_primary=False):
    """One rung's score: ``Lambda = 2 [ln L(accepted + candidate) - ln L(accepted)]``.

    Both models are fitted jointly to all windows (:func:`scanlib.joint.fit_joint`),
    the accepted sources seeded at their positions and the candidate at
    ``candidate``; the primary alone (no source accepted, the primary pinned)
    has no parameter and is evaluated by
    :func:`scanlib.joint.loglike_profiled`. With ``tau`` every source has its
    own linear motion, seeded at rest and bounded to ``+-mu_max``.

    Parameters
    ----------
    windows : array, shape (J, N)
        The windows, flux per sample.
    psi : array, shape (J,)
        The scan angles, in rad.
    inst : Instrument
        The instrument.
    sigma_noise : float or array
        The noise sigma, in any form of :mod:`scanlib.joint`.
    accepted : list of arrays, shape (2,)
        The accepted sources' positions [mas].
    candidate : array, shape (2,)
        The candidate's seed [mas].
    tau : None or array, shape (J,)
        The epochs [yr] of a moving search.
    mu_max : float
        The bound on every proper motion of a moving search, in mas/yr.
    eta0 : None or array, shape (J,)
        The windows' zero points [mas].
    free_primary : bool
        Give the primary its own block in front of the others, seeded at
        rest at the origin, with no prior.

    Returns
    -------
    Lambda : float
        The score, dimensionless.
    fit : dict
        The joint fit with the candidate (:func:`scanlib.joint.fit_joint`);
        its ``theta`` holds the candidate's block last.
    """
    positions = [np.asarray(x, dtype=float) for x in accepted]
    candidate = np.asarray(candidate, dtype=float)
    rest = np.zeros(2) if tau is not None else np.zeros(0)        # a block's motion, at rest
    blocks = [np.r_[x, rest] for x in positions]
    if free_primary:
        blocks = [np.zeros(2 + rest.size)] + blocks
    theta_s = np.concatenate(blocks) if blocks else np.zeros(0)
    kw = dict(tau=tau, eta0=eta0, free_primary=free_primary,
              rate_bound=mu_max if tau is not None else None)
    if theta_s.size:
        lnl_s = fit_joint(windows, psi, theta_s, inst, sigma_noise, **kw)["loglike"]
    else:                                            # the primary alone: nothing to fit
        lnl_s = loglike_profiled(theta_s, windows, psi, inst, sigma_noise, tau=tau, eta0=eta0)
    fit = fit_joint(windows, psi, np.concatenate([theta_s, candidate, rest]), inst,
                    sigma_noise, **kw)
    return float(2.0 * (fit["loglike"] - lnl_s)), fit


def _unassigned(psi, eta, sigma, fit, widths, *, det_tau=None, moving=False, free_primary=False,
                k_sigma=3.0):
    """The detections whose line passes through no accepted source.

    The accepted sources are the blocks of a joint fit (the module
    docstring, "The pool"): their positions [mas] and the covariances of
    those from ``fit['cov']`` (zero when it is ``None``). With ``det_tau`` in
    a moving fit each source is taken at the epoch of the detection,
    position plus motion times ``tau_i``, with its 4x4 block; without it, at
    the reference epoch, with its 2x2 position block.

    Parameters
    ----------
    psi, eta, sigma : arrays, shape (D,)
        The detections: scan angle [rad], along-scan position [mas], error
        [mas].
    fit : dict
        A joint fit (:func:`scanlib.joint.fit_joint`) of the accepted
        sources, the primary's block first with ``free_primary``.
    widths : sequence of float, length S
        The floor [mas] of each accepted source, in the fit's order.
    det_tau : None or array, shape (D,)
        The detections' epochs [yr from the reference epoch]; used only with
        ``moving``.
    moving : bool
        The fit is moving: four parameters per block.
    free_primary : bool
        The fit's first block is the primary's, which is not a source here.
    k_sigma : float
        The band in units of the combined error.

    Returns
    -------
    array of bool, shape (D,)
        ``True`` for the detections of the pool.
    """
    psi, eta, sigma = (np.asarray(a, dtype=float) for a in (psi, eta, sigma))
    n_par = 4 if moving else 2
    theta = np.asarray(fit["theta"], dtype=float)
    first = n_par if free_primary else 0
    starts = range(first, theta.size, n_par)
    if len(starts) != len(widths):
        raise ValueError(f"the fit has {len(starts)} sources; {len(widths)} widths given")
    g = np.stack([np.sin(psi), np.cos(psi)], axis=1)                      # (D, 2)
    if moving and det_tau is not None:                                    # the track at each line's epoch
        g = np.concatenate([g, g * np.asarray(det_tau, dtype=float)[:, None]], axis=1)   # (D, 4)
    n = g.shape[1]
    pool = np.ones(eta.size, dtype=bool)
    for i, w in zip(starts, widths):
        x = theta[i:i + n]
        c = np.zeros((n, n)) if fit["cov"] is None else np.asarray(fit["cov"])[i:i + n, i:i + n]
        s2 = np.einsum("di,ij,dj->d", g, c, g)
        band = np.maximum(k_sigma * np.sqrt(sigma ** 2 + s2), float(w))
        pool &= np.abs(eta - g @ x) > band
    return pool


def _candidate_position(fit, moving):
    """The candidate's fitted position: the last block of ``fit['theta']``."""
    theta = np.asarray(fit["theta"], dtype=float)
    return theta[-4:-2] if moving else theta[-2:]


def map_ladder(det_psi, det_eta, det_sigma, windows, psi, inst, sigma_noise, *, tau=None,
               det_tau=None, bar="lrt", false_rate=FALSE_RATE, max_sources=None, origin_zone=30.0,
               k_sigma=3.0, half_size=FIELD_HALF_MAS, step=MAP_STEP_MAS, mu_max=MU_MAX_MAS_YR,
               assign_floor=ASSIGN_FLOOR_MAS, eta0=None, free_primary=False):
    """The search: propose (map), score (joint fit), test (threshold), accept.

    Parameters
    ----------
    det_psi, det_eta, det_sigma : arrays, shape (D,)
        The campaign's detections [rad, mas, mas], on the sky.
    windows : array, shape (J, N)
        The campaign's windows, flux per sample.
    psi : array, shape (J,)
        Their scan angles, in rad.
    inst : Instrument
        The instrument.
    sigma_noise : float or array
        The windows' noise sigma: a scalar, one per window, or one per
        sample; the scoring fits weigh each sample by it.
    tau : None or array, shape (J,)
        The epochs [yr] of a moving search: every rung is scored with the
        moving fit, every proper motion within ``+-mu_max``, and the
        accepted positions are positions at the reference epoch.
    det_tau : None or array, shape (D,)
        The epoch [yr from the reference epoch] of every detection
        (:attr:`scanlib.reduction.DetectionSet.tau`); required with ``tau``,
        and not taken without it. The lines of an accepted source are
        removed against its fitted track at these epochs.
    bar : "lrt" or float
        The threshold every score must exceed: ``"lrt"``, the closed form
        :func:`scanlib.thresholds.lrt_threshold` at ``false_rate`` for this
        scan law, model and map (an approximation that assumes the same
        noise weight on every sample); or a number, for example a
        threshold calibrated by simulation.
    false_rate : float
        The false-addition rate per rung of ``bar="lrt"``.
    max_sources : None or int
        The cap on the accepted count.
    origin_zone, k_sigma, half_size, step, assign_floor
        As :func:`next_peak`.
    mu_max : float
        The searched proper-motion range [mas/yr].
    eta0 : None or array, shape (J,)
        The windows' zero points [mas].
    free_primary : bool
        Give the primary its own block in every scoring fit
        (:func:`lambda_rung`); the threshold is then the free primary's.

    Returns
    -------
    MapLadder

    Raises
    ------
    ValueError
        If ``bar`` is neither ``"lrt"`` nor a number; if ``tau`` is given
        without ``det_tau``, ``det_tau`` without ``tau``, or ``det_tau``
        does not hold one epoch per detection.
    """
    if isinstance(bar, str) and bar == "lrt":
        bar = lrt_threshold(psi, inst, tau, false_rate=false_rate, mu_max=mu_max,
                            half_size=half_size, origin_zone=origin_zone, eta0=eta0,
                            free_primary=free_primary)
    elif isinstance(bar, str) or bar is None or np.ndim(bar) != 0:
        raise ValueError(f"bar must be 'lrt' or a number; got {bar!r}")
    det_psi = np.atleast_1d(np.asarray(det_psi, dtype=float))
    det_eta = np.atleast_1d(np.asarray(det_eta, dtype=float))
    det_sigma = np.atleast_1d(np.asarray(det_sigma, dtype=float))
    if tau is None and det_tau is not None:
        raise ValueError("det_tau is the detections' epochs of a moving search; give tau as well, "
                         "or neither")
    if tau is not None:
        if det_tau is None:
            raise ValueError("a moving search (tau) needs det_tau, the epoch of every detection "
                             "(DetectionSet.tau): an accepted source's lines are removed against "
                             "its fitted track")
        det_tau = np.atleast_1d(np.asarray(det_tau, dtype=float))
        if det_tau.shape != det_eta.shape:
            raise ValueError(f"det_tau must hold one epoch per detection, shape {det_eta.shape}; "
                             f"got {det_tau.shape}")
    pool = np.ones(det_eta.size, dtype=bool)
    accepted, widths, log = [], [], []
    while pool.sum() >= GATE_MIN_DETECTIONS and (max_sources is None or len(accepted) < max_sources):
        found = next_peak(det_psi[pool], det_eta[pool], det_sigma[pool], origin_zone=origin_zone,
                          k_sigma=k_sigma, half_size=half_size, step=step,
                          assign_floor=assign_floor)
        if found is None:
            break
        peak, width, keep = found
        entry = dict(rung=len(accepted), peak=peak, width=width, n_pool=int(pool.sum()),
                     n_consistent=int(keep.sum()), pool=pool.copy())
        if keep.sum() < GATE_MIN_DETECTIONS:
            entry["verdict"] = "stop: fewer than 3 consistent lines"
            log.append(entry)
            break
        value, fit = lambda_rung(windows, psi, inst, sigma_noise, accepted, peak, tau=tau,
                                 mu_max=mu_max, eta0=eta0, free_primary=free_primary)
        entry.update(Lambda=value, bar=float(bar), fit=fit)
        if value <= float(bar):
            entry["verdict"] = "stop: below the threshold"
            log.append(entry)
            break
        position = _candidate_position(fit, tau is not None)
        entry.update(verdict="accepted", position=position)
        log.append(entry)
        accepted.append(position)
        widths.append(width)
        pool = _unassigned(det_psi, det_eta, det_sigma, fit, widths, det_tau=det_tau,
                           moving=tau is not None, free_primary=free_primary, k_sigma=k_sigma)
    return MapLadder(*_latest_sources(log, len(accepted), tau is not None, free_primary), log)


def _latest_sources(log, n_sources, moving, free_primary):
    """The accepted sources' positions (S, 2) and fluxes (S,) in the last
    accepted rung's joint fit: its blocks after the primary's, 2 or 4 entries
    each, and its fluxes after the primary's."""
    fits = [e["fit"] for e in log if e.get("verdict") == "accepted"]
    if not fits:
        return np.zeros((0, 2)), np.zeros(0)
    theta = np.asarray(fits[-1]["theta"], dtype=float)
    blocks = theta.reshape(-1, 4 if moving else 2)[1 if free_primary else 0:]
    assert blocks.shape[0] == n_sources
    return blocks[:, :2].copy(), np.asarray(fits[-1]["flux"], dtype=float)[1:].copy()


def describe(ladder):
    """One line per rung of a :class:`MapLadder`, for a notebook to print."""
    lines = []
    for e in ladder.log:
        line = (f"rung {e['rung']}: peak ({e['peak'][0]:6.0f}, {e['peak'][1]:6.0f}) "
                f"from {e['n_consistent']:2d} of {e['n_pool']:2d} lines")
        if "Lambda" in e:
            line += f";  Lambda {e['Lambda']:8.1f} vs threshold {e['bar']:5.1f}"
        line += f"  -> {e['verdict']}"
        if "position" in e:
            line += f"  at ({e['position'][0]:.1f}, {e['position'][1]:.1f})"
        lines.append(line)
    return "\n".join(lines)
