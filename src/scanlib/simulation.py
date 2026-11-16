"""The simulation: everything that knows the truth or draws random numbers.

The method (the atoms and the pipeline: :mod:`scanlib.instrument`,
:mod:`scanlib.joint`, :mod:`scanlib.reduction`, ...) never imports this
module, so it never sees a true position or a random number it did not
receive (``tests/test_import_closure.py`` holds the rule). This module
builds on the method: a simulated window is the method's own operator,
``inst.design_columns(eta) @ f``, plus Gaussian noise.

THE TOY REGIME. :data:`PRIMARY_FLUX` is the flux of the toy scenes' primary;
the S/N of a scene is the primary's total flux over the noise sigma per
sample, ``sigma = PRIMARY_FLUX / snr``, so S/N 100 is sigma 10.
:func:`snr_from_peak_snr` converts an S/N stated by the brightest sample.

ONE WINDOW (:func:`simulate_window`) and the windows' zero points at a
random sampling phase (:func:`random_zero_points`).

THE MARGINS' CALIBRATION (:func:`rung_null_statistic`,
:func:`calibrate_margins`). A margin of the per-window ladder
(:func:`scanlib.reduction.ladder_window`) is the 99th percentile of the
rung statistic on simulated windows that the lower model explains,
reduced by the ladder itself; the null windows are drawn here.

A CAMPAIGN. A :class:`Scene` is the true sky: the components' offsets from
the primary, their fluxes and their linear motion. :func:`simulate_campaign`
renders its window in every transit of a scan law; :func:`harvest_campaign`
renders them and reduces them with :func:`scanlib.reduction.reduce_campaign`,
which never sees the scene.

THE SEARCH'S NULL (:func:`field_null_max`): the maximum of the candidate
field in noise-only campaigns, the check of the closed-form threshold
:func:`scanlib.thresholds.lrt_threshold` by simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from .instrument import project_al, render_window
from .noise import NoiseModel
from .reduction import (_ladder, _unpack_law, _zero_points,   # _ladder: forced margins
                        default_margins, reduce_campaign)
from .thresholds import _ETA_STEP_MAS, FIELD_HALF_MAS, MU_MAX_MAS_YR

#: The flux of the toy scenes' primary, in arbitrary units (flux per sample
#: summed over the window): a companion of flux ratio ``q`` has flux
#: ``q * PRIMARY_FLUX``. The S/N of a toy scene is this flux over the noise
#: sigma per sample, so S/N 100 is sigma 10.
PRIMARY_FLUX = 1000.0
#: The largest order whose margins :func:`calibrate_margins` calibrates: the
#: rungs ``1 -> 2`` and ``2 -> 3``, the range in which the starting
#: configurations reach the optimum of an exhaustive search.
K_MAX = 3
#: The percentile of a rung's null that sets its margin: 1 % false promotion
#: per rung.
MARGIN_PERCENTILE = 99.0
#: The default number of null windows per rung behind a margin. At 600
#: windows the bootstrap error of a 99th percentile is about 0.5.
CALIBRATION_WINDOWS = 2000
#: The second rung's null: pairs that the two-source model explains, the
#: separation uniform in this range [FWHM], the flux ratio uniform in this
#: range, the centre uniform over the positions that keep both components in
#: the sampled range.
NULL_PAIR_SEPARATION_FWHM = (0.5, 2.5)
NULL_PAIR_Q = (0.1, 1.0)


def simulate_window(eta_sources, fluxes, inst, rng, sigma_noise):
    """One noisy window: :func:`render_window` plus Gaussian noise.

    The sigma is the caller's: it is what every fit must be handed with the
    window.

    Parameters
    ----------
    eta_sources : array, shape (K,)
        Along-scan positions of the sources in the window's frame, in mas.
    fluxes : array, shape (K,)
        The sources' fluxes, flux per sample.
    inst : Instrument
        The instrument: its sample grid and LSF.
    rng : numpy.random.Generator
        The random generator, consumed exactly once, by one
        ``rng.normal(0, sigma_noise, N)`` call.
    sigma_noise : float or array, shape (N,)
        The noise sigma, flux per sample: a scalar, the same on every sample,
        or one value per sample.

    Returns
    -------
    array, shape (N,)
        The noisy samples, flux per sample.
    """
    clean = render_window(eta_sources, fluxes, inst)
    return clean + rng.normal(0.0, sigma_noise, clean.size)


def random_zero_points(rng, n, inst):
    """The zero points of ``n`` windows at a uniform random sampling phase [mas].

    ``eta0_j = u_j p``, with ``u_j`` uniform on ``[-1/2, 1/2)`` and ``p`` the
    instrument's sample pitch (the mean step of its sample grid): the origin
    of window ``j``'s sample grid sits at the sky along-scan coordinate
    ``eta0_j``, so the samples fall at any phase about the reference
    position. The zero points are known and travel with the windows.

    Parameters
    ----------
    rng : numpy.random.Generator
        The random generator, consumed by one ``rng.uniform(-0.5, 0.5, n)``
        call. A stream of its own keeps the noise draws of a campaign as
        they are.
    n : int
        The number of windows.
    inst : Instrument
        The instrument: its sample pitch.

    Returns
    -------
    array, shape (n,)
        The zero points, in mas.
    """
    pitch = float(np.diff(inst.sample_grid_mas).mean())
    return rng.uniform(-0.5, 0.5, int(n)) * pitch


def snr_from_peak_snr(peak_snr, inst, reference_flux=PRIMARY_FLUX):
    """The S/N, total flux over sigma, that gives the sigma of a peak S/N.

    A peak S/N ``s_peak`` states the noise as the brightest sample of an
    isolated source of flux ``F`` at ``eta = 0`` over sigma,
    ``sigma = max_n F L(grid_n) / s_peak``. The package states it as the
    source's total flux over sigma, ``sigma = F / S``. This function returns
    ``S = F / sigma``, computed through sigma, so that
    ``reference_flux / snr_from_peak_snr(s_peak, inst, reference_flux)``
    equals that sigma exactly. On the toy instrument the brightest sample
    holds 43.5 % of the flux, so ``S`` is about 2.3 times ``s_peak``.

    Parameters
    ----------
    peak_snr : float
        The peak S/N.
    inst : Instrument
        The instrument the peak S/N refers to: its sample grid and LSF.
    reference_flux : float
        The source's flux ``F``, flux per sample summed over the window.

    Returns
    -------
    float
        The S/N ``S``, dimensionless.
    """
    sigma_peak = float(render_window([0.0], [reference_flux], inst).max()) / float(peak_snr)
    return float(reference_flux) / sigma_peak


def _null_window(rng, rung, inst, snr):
    """One window of rung ``rung``'s null: ``(samples, sigma)``.

    Rung 1: one source of ``PRIMARY_FLUX`` at a position uniform over the
    sampled range (first to last sample centre). Rung 2: sources of
    ``PRIMARY_FLUX`` and ``q PRIMARY_FLUX``, the separation uniform in
    :data:`NULL_PAIR_SEPARATION_FWHM` times ``inst.resolution_mas``, ``q``
    uniform in :data:`NULL_PAIR_Q`, the centre uniform over the positions
    that keep both in the sampled range. The geometry is drawn first, then
    the noise, one normal vector of sigma ``PRIMARY_FLUX / snr``.
    """
    lo = float(inst.sample_grid_mas.min())
    hi = float(inst.sample_grid_mas.max())
    if rung == 1:
        eta_true, fluxes = [rng.uniform(lo, hi)], [PRIMARY_FLUX]
    else:
        sep = rng.uniform(*NULL_PAIR_SEPARATION_FWHM) * inst.resolution_mas
        q = rng.uniform(*NULL_PAIR_Q)
        c_lo, c_hi = lo + 0.5 * sep, hi - 0.5 * sep
        centre = rng.uniform(min(c_lo, c_hi), max(c_lo, c_hi))
        eta_true = [centre - 0.5 * sep, centre + 0.5 * sep]
        fluxes = [PRIMARY_FLUX, q * PRIMARY_FLUX]
    sigma = float(PRIMARY_FLUX) / float(snr)
    return simulate_window(eta_true, fluxes, inst, rng, sigma), sigma


def rung_null_statistic(rng, rung, n, inst, snr):
    """Rung ``rung``'s statistic ``D_rung`` on ``n`` windows of its null.

    Each null window (:func:`_null_window`: one source for rung 1, a pair
    for rung 2) is reduced by the ladder itself on ``inst``, the rungs below
    forced open, and its ``D_rung`` is recorded; a window the ladder cannot
    climb (no active trial position) records 0. The statistic does not
    depend on ``snr`` while the LSF model is right; the draws do.

    Parameters
    ----------
    rng : numpy.random.Generator
        The random generator, consumed window by window: the geometry, then
        the noise.
    rung : int
        1 or 2.
    n : int
        The number of null windows.
    inst : Instrument
        The instrument; its margins are not used.
    snr : float
        The S/N of the null windows: the noise sigma is ``PRIMARY_FLUX / snr``.

    Returns
    -------
    array, shape (n,)
        ``D_rung`` of each window, dimensionless.
    """
    if rung not in range(1, K_MAX):
        raise ValueError(f"rungs run 1 to {K_MAX - 1}, not {rung!r}")
    forced = np.full(rung, np.inf)
    forced[:rung - 1] = -np.inf               # the rungs below are climbed
    stat = np.zeros(n)
    for i in range(n):
        samples, sigma = _null_window(rng, rung, inst, snr)
        _e, _f, _k, diag = _ladder(samples, inst, sigma, forced, rung + 1)
        if diag["delta_chi2"].size == rung:
            stat[i] = diag["delta_chi2"][-1]
    return stat


def calibrate_margins(rng, inst, snr, n=CALIBRATION_WINDOWS, *, k_max=K_MAX):
    """The ladder's margins on ``inst``: the 99th percentile of each rung's null.

    :func:`rung_null_statistic` on ``n`` windows of each rung's null (rung
    ``1 -> 2`` single sources anywhere in the sampled range, rung ``2 -> 3``
    the pairs), and its :data:`MARGIN_PERCENTILE`. A margin admits a false
    extra source in 1 % of the windows the lower model explains, per rung and
    per window. With the LSF model right it is a property of the sample
    grid, the LSF and the trial positions; on real data the null is the
    real isolated-star windows. Pass the result to the instrument,
    ``dataclasses.replace(inst, margins=calibrate_margins(rng, inst, snr))``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Rung ``r`` draws from the child stream ``rng.spawn(k_max - 1)[r - 1]``,
        so each rung is reproducible on its own.
    inst : Instrument
        The instrument.
    snr : float
        The S/N of the null windows (:func:`rung_null_statistic`).
    n : int or sequence of int
        The number of null windows: one for every rung, or one per rung.
    k_max : int
        2 or 3: the margins of the rungs up to ``k_max``.

    Returns
    -------
    array, shape (k_max - 1,)
        ``(Delta_1, ...)``, dimensionless.
    """
    if k_max not in range(2, K_MAX + 1):
        raise ValueError(f"margins exist for k_max = 2 to {K_MAX}, not {k_max!r}")
    n_rung = np.broadcast_to(np.asarray(n, dtype=int), (k_max - 1,))
    streams = rng.spawn(k_max - 1)
    return np.asarray([
        float(np.percentile(rung_null_statistic(streams[rung - 1], rung, int(n_rung[rung - 1]),
                                                inst, snr), MARGIN_PERCENTILE))
        for rung in range(1, k_max)])


# --- a campaign ------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Scene:
    """The true sky of a simulation: point sources about the primary.

    Row ``k`` of ``positions`` is component ``k``'s offset ``(dra*, ddec)``
    [mas] from the reference position at the reference epoch; row 0 is the
    primary, at ``(0, 0)`` by convention. ``fluxes`` are the K fluxes, in
    the units of :data:`PRIMARY_FLUX`. ``pm`` (K, 2) is each component's
    proper motion relative to the primary, ``(mu_alpha*, mu_delta)`` in
    mas/yr, and ``dpi`` (K,) its parallax difference from the primary in
    mas; ``None`` leaves the term out, and a scene with neither is static.
    In a transit of scan angle ``psi`` [rad], epoch ``tau`` [yr] and
    along-scan parallax factor ``fpar``, component ``k`` lies at

        eta_k = project_al(dra*_k + mu_alpha*_k tau, ddec_k + mu_delta_k tau, psi)
                + dpi_k fpar                                          [mas]

    The motion is linear: no orbit, no acceleration.

    Raises
    ------
    ValueError
        If the shapes do not match: positions (K, 2), fluxes (K,), pm (K, 2),
        dpi (K,).
    """

    positions: np.ndarray
    fluxes: np.ndarray
    pm: np.ndarray | None = None
    dpi: np.ndarray | None = None

    def __post_init__(self):
        positions = np.atleast_2d(np.asarray(self.positions, dtype=float))
        fluxes = np.atleast_1d(np.asarray(self.fluxes, dtype=float))
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must be (K, 2) sky offsets (dra*, ddec)")
        if fluxes.shape != (positions.shape[0],):
            raise ValueError("one flux is required per component")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "fluxes", fluxes)
        if self.pm is not None:
            pm = np.atleast_2d(np.asarray(self.pm, dtype=float))
            if pm.shape != positions.shape:
                raise ValueError("pm must be (K, 2), one relative proper motion per component")
            object.__setattr__(self, "pm", pm)
        if self.dpi is not None:
            dpi = np.atleast_1d(np.asarray(self.dpi, dtype=float))
            if dpi.shape != fluxes.shape:
                raise ValueError("dpi must be (K,), one parallax difference per component")
            object.__setattr__(self, "dpi", dpi)

    @classmethod
    def pair(cls, rho_mas, pa_deg, q, *, pm=None, dpi=None):
        """A primary of :data:`PRIMARY_FLUX` at the origin and one companion.

        Parameters
        ----------
        rho_mas : float
            The separation at the reference epoch, in mas.
        pa_deg : float
            The position angle, in degrees from north (+ddec) through east
            (+dra*): the companion lies at ``(rho sin PA, rho cos PA)``.
        q : float
            The companion's flux over the primary's.
        pm : None or (float, float)
            The companion's relative proper motion ``(mu_alpha*, mu_delta)``,
            in mas/yr.
        dpi : None or float
            The companion's parallax difference, in mas.

        Returns
        -------
        Scene
        """
        pa = np.deg2rad(pa_deg)
        positions = [[0.0, 0.0], [rho_mas * np.sin(pa), rho_mas * np.cos(pa)]]
        return cls(np.array(positions, dtype=float),
                   np.array([PRIMARY_FLUX, PRIMARY_FLUX * q], dtype=float),
                   pm=None if pm is None else np.array([[0.0, 0.0], list(pm)], dtype=float),
                   dpi=None if dpi is None else np.array([0.0, dpi], dtype=float))

    @property
    def static(self):
        """True when the scene has neither a proper motion nor a parallax difference."""
        return self.pm is None and self.dpi is None

    def along_scan(self, psi, tau=None, fpar=None):
        """The along-scan positions [mas] of the K components in one transit.

        Parameters
        ----------
        psi : float
            The transit's scan angle, in rad.
        tau : None or float
            Its epoch from the reference epoch, in yr; required when the
            scene has a proper motion.
        fpar : None or float
            Its along-scan parallax factor; required when the scene has a
            parallax difference.

        Returns
        -------
        array, shape (K,)
        """
        if self.pm is not None and tau is None:
            raise ValueError("a scene with a proper motion needs the epoch of every transit")
        if self.dpi is not None and fpar is None:
            raise ValueError("a scene with a parallax difference needs the along-scan "
                             "parallax factor of every transit")
        eta = np.empty(self.fluxes.size)
        for k, (dra, ddec) in enumerate(self.positions):
            if self.pm is not None:
                dra, ddec = dra + self.pm[k, 0] * tau, ddec + self.pm[k, 1] * tau
            eta[k] = project_al(dra, ddec, psi)
            if self.dpi is not None:
                eta[k] = eta[k] + self.dpi[k] * fpar
        return eta


def simulate_campaign(scene, law, rng, inst, sigma_noise, *, eta0=None):
    """The noisy windows of a scene in every transit of a scan law, with the truth.

    Transit ``j`` places every component of ``scene`` at
    ``scene.along_scan(psi_j, tau_j, fpar_j)`` and renders the window
    through :func:`simulate_window`, in the law's order.

    Parameters
    ----------
    scene : Scene
        The components, their fluxes and their motion.
    law : array or tuple
        ``psi`` [rad] alone, ``(psi, tau)`` or ``(psi, tau, fpar)``; it must
        carry ``tau`` for a scene with a proper motion and ``fpar`` for one
        with a parallax difference.
    rng : numpy.random.Generator
        Consumed once per window, one normal draw of its N samples.
    inst : Instrument
        The true instrument the photons pass through.
    sigma_noise : float or NoiseModel
        The noise: one sigma for every sample [flux per sample], or a
        :class:`scanlib.noise.NoiseModel` that gives each sample the sigma
        of its own noise-free counts.
    eta0 : None or array, shape (J,)
        The windows' zero points [mas]: a source at ``eta`` is rendered at
        ``eta - eta0_j`` in window ``j``. ``None`` for all 0.

    Returns
    -------
    windows : array, shape (J, N)
        The noisy windows, flux per sample.
    sigma : float or list of J arrays, shape (N,)
        The sigma they were drawn with.
    eta_true : array, shape (J, K)
        Every component's along-scan position on the sky, in mas.

    Raises
    ------
    ValueError
        If ``sigma_noise`` is neither a scalar nor a NoiseModel, or the law
        lacks what a moving scene needs.
    """
    model = isinstance(sigma_noise, NoiseModel)
    if not model and np.ndim(sigma_noise) != 0:
        raise ValueError("sigma_noise must be a scalar or a NoiseModel; a known sigma per "
                         "window or per sample is passed to the windows directly")
    psi, tau, fpar = _unpack_law(law)
    eta0 = _zero_points(eta0, psi.size)
    windows, truth, per_sample = [], [], []
    for j, psi_j in enumerate(psi):
        etas = scene.along_scan(psi_j, None if tau is None else tau[j],
                                None if fpar is None else fpar[j])
        in_window = etas if eta0 is None else etas - eta0[j]
        s = sigma_noise.sigma(render_window(in_window, scene.fluxes, inst)) if model else sigma_noise
        windows.append(simulate_window(in_window, scene.fluxes, inst, rng, s))
        truth.append(etas)
        per_sample.append(s)
    windows = np.array(windows, dtype=float).reshape(psi.size, inst.sample_grid_mas.size)
    eta_true = np.array(truth).reshape(psi.size, scene.fluxes.size)
    return windows, (per_sample if model else sigma_noise), eta_true


def harvest_campaign(scene, law, rng, inst, sigma_noise, *, k_max, return_windows=False,
                     eta0=None):
    """A scene simulated over a scan law and every window reduced: one call.

    :func:`simulate_campaign`, then :func:`scanlib.reduction.reduce_campaign`
    on the windows and the sigma they were drawn with. The scene enters the
    rendering only: no detection is selected by the true positions, so
    false detections survive at the ladder's calibrated rate.

    Parameters
    ----------
    scene, law, rng, inst, sigma_noise, eta0
        As :func:`simulate_campaign`; the reduction assumes the same ``inst``
        and ``eta0``.
    k_max : int
        The ladder's ceiling (:func:`scanlib.reduction.reduce_campaign`).
    return_windows : bool
        Keep the windows on the record.

    Returns
    -------
    DetectionSet

    Raises
    ------
    ValueError
        If the instrument lacks the margins for ``k_max``; nothing is drawn
        then.
    """
    default_margins(inst, k_max)                           # refuse before drawing
    windows, sigma, _eta_true = simulate_campaign(scene, law, rng, inst, sigma_noise, eta0=eta0)
    detections = reduce_campaign(law, windows, inst, sigma, k_max=k_max, eta0=eta0)
    return replace(detections, windows=windows) if return_windows else detections


# --- the search's null ----------------------------------------------------------

def field_null_max(psi, tau, n_campaigns, rng, inst, *, sigma_noise=None, free_primary=False,
                   mu_max=MU_MAX_MAS_YR, half_size=FIELD_HALF_MAS, origin_zone=30.0,
                   x_step=6.0, mu_step=3.0):
    """The maximum of the moving search's candidate field in noise-only campaigns.

    Each campaign draws whitened noise in every window of the law ``(psi,
    tau)`` (standard normal per sample), removes its projection on the
    primary's columns (its flux, and with ``free_primary`` also its offset and
    residual motion, linear in the primary's parameters), and correlates the
    remainder with the whitened candidate template made orthogonal to those
    columns. ``Z(x, mu)`` is that correlation over the template's norm: under
    the null a standard Gaussian field over the searched region, the box
    ``|x_i| <= half_size`` minus the disk ``|x| <= origin_zone`` [mas],
    times ``|mu_i| <= mu_max`` [mas/yr]. Its maximum is taken on a grid of
    ``x_step`` [mas] and ``mu_step`` [mas/yr] and refined by Nelder-Mead from
    the four best grid motions, kept only inside the region. A search
    exceeds the level ``u`` when ``max(Z, 0)^2 > u``. Assumed: the field is
    Gaussian to first order in the primary's parameters (exact for a pinned
    primary), the noise Gaussian and known; every window's zero point is 0.

    Parameters
    ----------
    psi, tau : arrays, shape (J,)
        The scan law: scan angles [rad] and epochs [yr].
    n_campaigns : int
        The number of noise-only campaigns.
    rng : numpy.random.Generator
        One standard-normal (J, N) draw per campaign.
    inst : Instrument
        The instrument: the templates.
    sigma_noise : None or array, shape (N,)
        The per-sample noise of a window that holds the primary alone, for a
        noise that follows the light; the templates are divided by it.
        ``None``: the same noise on every sample.
    free_primary : bool
        Remove the primary's offset and motion columns too.
    mu_max, half_size, origin_zone, x_step, mu_step : float
        The searched region and the grid of the maximum.

    Returns
    -------
    array, shape (n_campaigns,)
        The refined maximum of ``Z`` per campaign.
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    taus = np.atleast_1d(np.asarray(tau, dtype=float))
    half = float(half_size)
    J = psi.size
    su, cu = np.sin(psi), np.cos(psi)
    reach = half * np.sqrt(2) + mu_max * np.abs(taus).max() * np.sqrt(2) + 60
    eta_f = np.arange(-reach, reach + _ETA_STEP_MAS, _ETA_STEP_MAS)
    L0 = inst.design_columns(np.array([0.0]))[:, 0]
    L0p = inst.design_jacobian(np.array([0.0]))[:, 0]
    sig = np.ones_like(L0) if sigma_noise is None else np.asarray(sigma_noise, dtype=float)
    T = inst.design_columns(eta_f) / sig[:, None]                       # whitened template (N, M)
    A = [L0 / sig] + ([L0p / sig] if free_primary else [])
    V = np.stack([su, cu, taus * su, taus * cu])
    W = np.vstack([np.ones(J)] + ([V] if free_primary else []))         # column k = W[k, j] * A[kind[k]]
    kind = [0] + ([1] * 4 if free_primary else [])
    K = W.shape[0]
    h = np.einsum("nm,nm->m", T, T)
    c = [a @ T for a in A]
    aa = np.array([[a @ b for b in A] for a in A])
    G = np.array([[(W[k] * W[m]).sum() * aa[kind[k], kind[m]] for m in range(K)] for k in range(K)])
    Gi = np.linalg.inv(G)

    def interp(f, eta):
        idx = (eta - eta_f[0]) / _ETA_STEP_MAS
        i0 = np.clip(np.floor(idx).astype(int), 0, eta_f.size - 2)
        w = idx - i0
        return f[..., i0] * (1 - w) + f[..., i0 + 1] * w

    def norm2(e):
        ci = [interp(f, e) for f in c]
        C = np.stack([ci[kind[k]] @ W[k] for k in range(K)], 1)
        return interp(h, e).sum(1) - np.einsum("pk,kl,pl->p", C, Gi, C)

    xa = np.arange(-half, half + 1e-9, x_step)
    XX, YY = np.meshgrid(xa, xa)
    kx = np.hypot(XX, YY) > origin_zone
    xs, ys = XX[kx], YY[kx]
    mua = np.arange(-mu_max, mu_max + 1e-9, mu_step)
    MU = np.array([(a, b) for a in mua for b in mua])
    e_stat = xs[:, None] * su[None, :] + ys[:, None] * cu[None, :]
    off = (MU[:, 0:1] * taus[None, :]) * su[None, :] + (MU[:, 1:2] * taus[None, :]) * cu[None, :]
    jj = np.arange(J)[None, :]
    sqrt_ww = np.sqrt(np.clip(np.stack([norm2(e_stat + off[m][None, :]) for m in range(MU.shape[0])]),
                              1e-30, None))
    cols = np.stack([W[k][:, None] * A[kind[k]][None, :] for k in range(K)])     # (K, J, N)

    def z_point(p, g):
        e = ((p[0] + p[2] * taus) * su + (p[1] + p[3] * taus) * cu)[None, :]
        idx = (e[0] - eta_f[0]) / _ETA_STEP_MAS
        i0 = np.clip(np.floor(idx).astype(int), 0, eta_f.size - 2)
        w = idx - i0
        S = (g[np.arange(J), i0] * (1 - w) + g[np.arange(J), i0 + 1] * w).sum()
        return S / np.sqrt(max(float(norm2(e)[0]), 1e-30))

    def inside(p):
        return (abs(p[0]) <= half and abs(p[1]) <= half and np.hypot(p[0], p[1]) > origin_zone
                and abs(p[2]) <= mu_max and abs(p[3]) <= mu_max)

    max_z = np.empty(int(n_campaigns))
    for k in range(int(n_campaigns)):
        noise = rng.normal(size=(J, L0.size))
        b = Gi @ np.einsum("kjn,jn->k", cols, noise)
        g = (noise - np.einsum("k,kjn->jn", b, cols)) @ T                # (J, M)
        tops = []
        for m in range(MU.shape[0]):
            e = e_stat + off[m][None, :]
            idx = (e - eta_f[0]) / _ETA_STEP_MAS
            i0 = np.clip(np.floor(idx).astype(int), 0, eta_f.size - 2)
            w = idx - i0
            Z = (g[jj, i0] * (1 - w) + g[jj, i0 + 1] * w).sum(1) / sqrt_ww[m]
            i = int(np.argmax(Z))
            tops.append((Z[i], (xs[i], ys[i], MU[m, 0], MU[m, 1])))
        tops.sort(key=lambda v: -v[0])
        best = tops[0][0]
        for _z, p0 in tops[:4]:
            r = minimize(lambda p: -z_point(p, g), np.array(p0), method="Nelder-Mead",
                         options=dict(xatol=0.05, fatol=1e-4, maxiter=400))
            if inside(r.x):
                best = max(best, -r.fun)
        max_z[k] = best
    return max_z
