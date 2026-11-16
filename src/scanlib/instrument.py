"""The instrument: the along-scan projection, the line-spread function,
and one window rendered through them.

A window is the strip of N samples the instrument reads out along the
scan direction when a star crosses the focal plane. Its samples sit at the
along-scan coordinates ``grid`` [mas] in the window's own frame. A source
at sky offset ``(dra*, ddec)`` [mas] from the reference position (dra* =
d(ra) cos(dec)) is seen by a window of scan angle ``psi`` [rad,
counter-clockwise from north] at the along-scan coordinate

    eta = dra* sin(psi) + ddec cos(psi)                                  [mas]

(:func:`project_al`). A window whose sample grid's origin sits at the sky
along-scan coordinate ``eta0`` (its zero point) sees the source at
``eta - eta0`` in its own frame; the routines that combine windows take
``eta0`` with them (:mod:`scanlib.joint`, :mod:`scanlib.radon`).

A point source of flux ``f`` at ``eta`` puts into sample ``n`` the flux

    m_n = f L(grid_n - eta),

``L`` the line-spread function (LSF): the dimensionless fraction of the
source's flux that the sample collects. Fluxes and samples are in the same
units, flux per sample. The noise is Gaussian with a known sigma per
sample, in those units.

:class:`Instrument` holds everything the method needs to know about how
sky positions become samples: the sample grid, the LSF ``L`` and its
derivative with respect to the source position, the dictionary of trial
positions of the per-window reduction, the LSF width and the per-window
margins. :meth:`Instrument.gaussian` builds the Gaussian case,

    L(delta) = p exp(-delta^2 / (2 s^2)) / (sqrt(2 pi) s),

``s = FWHM / (2 sqrt(2 ln 2))`` [mas], integrated over one sample of pitch
``p`` by the midpoint rule (:func:`gaussian_lsf`,
:func:`gaussian_lsf_jacobian`); any other LSF is two callables passed to
:class:`Instrument`.

One operator does all the work: :meth:`Instrument.design_columns` turns
source positions into the samples a unit flux produces,
:meth:`Instrument.design_jacobian` is its derivative with respect to the
source position (the Fisher matrices flow through it), and
:func:`render_window` is the operator bound to a scene, and
:func:`scanlib.simulation.simulate_window` adds the noise to it. A fit and a
simulation are the same computation with different inputs.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


def project_al(dra, ddec, psi):
    """The along-scan coordinate ``eta = dra* sin(psi) + ddec cos(psi)`` [mas].

    A scan along north (``psi = 0``) measures ``ddec``, a scan along east
    (``psi = pi/2``) measures ``dra*``.

    Parameters
    ----------
    dra : float or array
        Offset in right ascension times cos(dec), ``dra*``, in mas from the
        reference position.
    ddec : float or array
        Offset in declination, in mas from the reference position.
    psi : float or array
        Scan angle in rad, counter-clockwise from north. The three arguments
        broadcast against each other.

    Returns
    -------
    float or array
        ``eta`` in mas, of the broadcast shape of the inputs.
    """
    return dra * np.sin(psi) + ddec * np.cos(psi)


def _fwhm_to_sigma(fwhm):
    """The Gaussian width ``sigma = FWHM / (2 sqrt(2 ln 2))``, same unit."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _sample_grid(n_samples, pitch):
    """``(n - (N - 1) / 2) pitch`` for n = 0 .. N-1 [mas]: a regular grid
    symmetric about zero, the window's own frame."""
    return (np.arange(n_samples) - (n_samples - 1) / 2.0) * pitch


def _pitch(grid):
    """The sample pitch [mas] of a regular grid; a ValueError otherwise."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be a 1-D array of at least two sample centers")
    pitch = float(grid[1] - grid[0])
    if pitch <= 0 or not np.allclose(np.diff(grid), pitch):
        raise ValueError("grid must be regular and increasing")
    return pitch


def gaussian_lsf(delta_eta, sigma_lsf, pitch):
    """The fraction of a source's flux in a sample centered ``delta_eta`` away.

    ``L(delta) = pitch exp(-delta^2 / (2 sigma_lsf^2)) / (sqrt(2 pi)
    sigma_lsf)``: the Gaussian density of width ``sigma_lsf`` times the
    sample pitch, the midpoint rule for the flux a sample of that width
    collects.

    Parameters
    ----------
    delta_eta : float or array
        ``grid_n - eta``, the sample center minus the source position, in
        mas; any shape.
    sigma_lsf : float
        The Gaussian LSF width, in mas.
    pitch : float
        The sample pitch, in mas.

    Returns
    -------
    float or array
        ``L``, dimensionless, the shape of ``delta_eta``.
    """
    density = np.exp(-0.5 * (delta_eta / sigma_lsf) ** 2) / (
        np.sqrt(2.0 * np.pi) * sigma_lsf)
    return pitch * density


def gaussian_lsf_jacobian(delta_eta, sigma_lsf, pitch):
    """``dL/d(eta)``, the derivative of :func:`gaussian_lsf` with respect to the
    SOURCE position, per unit flux.

    With ``delta = grid - eta``, ``d/d(eta) L(grid - eta) = -L'(delta) =
    (delta / sigma_lsf^2) L(delta)``.

    Parameters
    ----------
    delta_eta : float or array
        ``grid_n - eta``, in mas; any shape.
    sigma_lsf : float
        The Gaussian LSF width, in mas.
    pitch : float
        The sample pitch, in mas.

    Returns
    -------
    float or array
        The derivative, in 1/mas, the shape of ``delta_eta``.
    """
    return (delta_eta / sigma_lsf ** 2) * gaussian_lsf(delta_eta, sigma_lsf, pitch)


#: The per-window margins ``(Delta_1, Delta_2)`` of the toy instrument,
#: ``Instrument.gaussian(100, 59, 18)`` (dimensionless: the drop in residual
#: power over sigma_n^2). Rung ``k`` of the per-window ladder accepts a
#: ``k+1``-th source when its drop ``D_k`` exceeds ``Delta_k``; each margin
#: is the 99th percentile of ``D_k`` over 20000 simulated windows that hold
#: exactly ``k`` sources, at peak S/N 50, so one window in a hundred
#: promotes falsely per rung. They hold for this sample grid, this LSF, this
#: atom grid and the ladder's multi-start solver only, and only while the
#: LSF model is right. Pass them as ``Instrument.gaussian(...,
#: margins=TOY_MARGINS)``.
TOY_MARGINS = np.array([11.123287117782931, 10.766534123859318])
#: The seed of the simulation behind :data:`TOY_MARGINS`: the rung-``k``
#: null windows are drawn from the ``k``-th child stream of
#: ``np.random.default_rng(TOY_MARGINS_SEED)``, one independent stream per
#: rung, as :func:`scanlib.simulation.calibrate_margins` spawns them (the
#: first children of a spawn do not depend on how many are spawned).
TOY_MARGINS_SEED = 20260927
#: The number of null windows per rung behind :data:`TOY_MARGINS`.
TOY_MARGINS_N = (20000, 20000)


@dataclass(frozen=True, eq=False)
class Instrument:
    """One along-scan measurement configuration: where the samples sit, the
    line-spread function, and the dictionary of trial positions.

    The instrument owns the one operator of the package,
    :meth:`design_columns` (source positions to samples) and its derivative
    :meth:`design_jacobian`; every rendered window, design matrix and Fisher
    matrix is built from these two. It holds no noise model and no flux
    scale: those belong to an observation.

    Parameters
    ----------
    sample_grid_mas : array, shape (N,)
        The along-scan coordinates of the sample centers, in mas, in the
        window's own frame (the frame the scene's ``eta - eta0`` is in).
    atom_eta_mas : array, shape (G,)
        The dictionary of trial positions, in mas, in the same frame: the
        positions the per-window reduction tries for a source.
    lsf : callable
        ``lsf(delta_eta)``, ``delta_eta = grid_n - eta`` in mas, any shape:
        the fraction of a source's flux that sample ``n`` collects,
        dimensionless (integrated over the sample, not a curve value at its
        center). :func:`gaussian_lsf` is the Gaussian case.
    dlsf_deta : callable
        ``d lsf(grid_n - eta) / d eta``, the derivative with respect to the
        SOURCE position (``= -lsf'(delta)``), in 1/mas, same shapes.
        :func:`gaussian_lsf_jacobian` is the Gaussian case.
    resolution_mas : float
        The LSF full width, in mas. It sets the position bounds of the
        per-window fit (:attr:`eta_bounds`) and the separation below which
        two fitted components are one source (:attr:`merge_gap_mas`); no
        likelihood uses it.
    margins : None or array, shape (K_max - 1,)
        The per-window margins ``(Delta_1, ..., Delta_{K_max-1})`` of this
        instrument, positive and dimensionless (:data:`TOY_MARGINS`), or
        ``None``. They are a property of the grid, the LSF and the atom
        grid, so they travel with the instrument; a per-window ladder on it
        holds at most ``len(margins) + 1`` sources. Stored read-only.

    Attributes
    ----------
    atoms : array, shape (N, G)
        ``design_columns(atom_eta_mas)``: column ``g`` is the window of a
        unit-flux source at ``atom_eta_mas[g]``.

    The instrument is frozen: it holds read-only copies of its arrays, so the
    grid, the trial positions and ``atoms`` cannot drift apart. Two
    instruments compare equal only when they are the same object; build a
    variant with ``dataclasses.replace(inst, margins=...)``.
    """

    sample_grid_mas: np.ndarray
    atom_eta_mas: np.ndarray
    lsf: Callable[[np.ndarray], np.ndarray]
    dlsf_deta: Callable[[np.ndarray], np.ndarray]
    resolution_mas: float
    margins: np.ndarray | None = None
    atoms: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("sample_grid_mas", "atom_eta_mas"):
            a = np.array(getattr(self, name), dtype=float)        # a copy, not a view
            a.flags.writeable = False
            object.__setattr__(self, name, a)
        if self.margins is not None:
            margins = np.array(self.margins, dtype=float, ndmin=1)
            if margins.ndim != 1 or margins.size == 0 or not np.all(margins > 0):
                raise ValueError("margins must be a non-empty 1-D sequence of "
                                 "positive values (Delta_1, Delta_2, ...)")
            margins.flags.writeable = False
            object.__setattr__(self, "margins", margins)
        atoms = self.design_columns(self.atom_eta_mas)
        atoms.flags.writeable = False
        object.__setattr__(self, "atoms", atoms)

    @classmethod
    def gaussian(cls, fwhm_mas=100.0, pitch_mas=59.0, n_samples=18, margins=None):
        """A Gaussian LSF on a regular grid of samples.

        The samples sit at ``(n - (N - 1) / 2) pitch_mas``, symmetric about
        zero. The LSF is :func:`gaussian_lsf` of width ``sigma = fwhm_mas /
        (2 sqrt(2 ln 2))``, integrated over one sample of width
        ``pitch_mas`` by the midpoint rule; its derivative is
        :func:`gaussian_lsf_jacobian`. The trial positions run from -530
        mas in steps of ``pitch_mas / 8`` while they do not exceed 530 mas
        (to 524.625 mas at the default pitch), whatever the grid, and
        ``resolution_mas = fwhm_mas``.

        Parameters
        ----------
        fwhm_mas : float, default 100.0
            The full width at half maximum of the LSF, in mas.
        pitch_mas : float, default 59.0
            The sample pitch, in mas.
        n_samples : int, default 18
            The number of samples ``N``.
        margins : None or array, default None
            The instrument's per-window margins; :data:`TOY_MARGINS` for the
            defaults.

        Returns
        -------
        Instrument

        Raises
        ------
        ValueError
            When ``fwhm_mas`` or ``pitch_mas`` is not positive and finite, or
            ``n_samples`` is not an integer of at least 2.
        """
        for name, value in (("fwhm_mas", fwhm_mas), ("pitch_mas", pitch_mas)):
            if not (np.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if not (np.isfinite(n_samples) and n_samples == int(n_samples) and n_samples >= 2):
            raise ValueError(f"n_samples must be an integer of at least 2, got {n_samples}")
        sigma_mas = _fwhm_to_sigma(fwhm_mas)

        def lsf(delta_eta_mas):
            return gaussian_lsf(delta_eta_mas, sigma_mas, pitch_mas)

        def dlsf_deta(delta_eta_mas):
            return gaussian_lsf_jacobian(delta_eta_mas, sigma_mas, pitch_mas)

        return cls(sample_grid_mas=_sample_grid(n_samples, pitch_mas),
                   atom_eta_mas=np.arange(-530.0, 530.0 + 1e-9, pitch_mas / 8.0),
                   lsf=lsf, dlsf_deta=dlsf_deta, resolution_mas=fwhm_mas,
                   margins=margins)

    def _apply(self, kernel, eta_mas, sample_grid_mas):
        eta = np.atleast_1d(np.asarray(eta_mas, dtype=float))
        if eta.ndim == 2:
            if sample_grid_mas is not None:
                raise ValueError("2-D eta shares the instrument's own sample grid")
            grid = self.sample_grid_mas
            delta = grid[None, :, None] - eta[:, None, :]           # (J, N, K)
            return kernel(delta).reshape(eta.shape[0] * grid.size, eta.shape[1])
        if eta.ndim != 1:
            raise ValueError("eta_mas must be 1-D (K,) or 2-D (J, K)")
        grid = (self.sample_grid_mas if sample_grid_mas is None
                else np.asarray(sample_grid_mas, dtype=float))
        return kernel(grid[:, None] - eta[None, :])

    def design_columns(self, eta_mas, sample_grid_mas=None):
        """The operator source positions -> samples: the design matrix.

        Entry ``(n, k)`` is ``lsf(grid_n - eta_k)``, the fraction of source
        ``k``'s flux in sample ``n``. ``design_columns(eta) @ f`` is the
        noise-free window of fluxes ``f``.

        Parameters
        ----------
        eta_mas : array, shape (K,) or (J, K)
            Along-scan source positions, in mas, in the window's frame. 1-D:
            the K sources of one window. 2-D: J windows on this grid, row
            ``j`` the positions in window ``j``.
        sample_grid_mas : None or array, shape (M,)
            ``None`` uses the instrument's own samples; an array evaluates
            the same operator on another grid (a fine grid for a drawn
            profile). Only with 1-D ``eta_mas``.

        Returns
        -------
        array, shape (N, K) or (J N, K)
            Dimensionless (flux per sample per unit source flux). For 2-D
            input the J blocks of N rows are stacked window-major.
        """
        return self._apply(self.lsf, eta_mas, sample_grid_mas)

    def design_jacobian(self, eta_mas, sample_grid_mas=None):
        """``d design_columns / d eta``: each column's derivative with respect
        to its source's position, per unit flux.

        Entry ``(n, k)`` is ``dlsf_deta(grid_n - eta_k)``, in 1/mas, the
        derivative with respect to the SOURCE position. The model's
        derivative with respect to source ``k``'s position is ``f_k`` times
        column ``k``. Same shapes and stacking as :meth:`design_columns`.
        """
        return self._apply(self.dlsf_deta, eta_mas, sample_grid_mas)

    @property
    def eta_bounds(self):
        """``(grid_0 - resolution, grid_{N-1} + resolution)`` in mas: the
        window widened by one LSF width, the per-window fit's position bounds."""
        return (float(self.sample_grid_mas.min()) - self.resolution_mas,
                float(self.sample_grid_mas.max()) + self.resolution_mas)

    @property
    def merge_gap_mas(self):
        """``0.3 resolution`` in mas: two fitted components closer than this
        are one source."""
        return 0.3 * self.resolution_mas

    def for_window(self, sample_grid_mas, atom_step_mas=None, margin_mas=None):
        """The same LSF on another sample grid: the instrument of one window.

        Windows differ in their number of samples and in the origin of their
        grid, and each needs its own dictionary of trial positions. The new
        trial positions run from ``grid_min - margin_mas`` in steps of
        ``atom_step_mas`` up to ``grid_max + margin_mas``, so they cover the
        part of the sky the window can constrain. The margins are dropped:
        they hold for the grid they were calibrated on, and a new grid needs
        its own (:func:`scanlib.simulation.calibrate_margins`).

        The new grid must be regular with this instrument's pitch: the LSF is
        the fraction of a source's flux in one sample, integrated over a
        sample of that pitch, so a grid of another pitch is another
        instrument.

        Parameters
        ----------
        sample_grid_mas : array, shape (N,)
            The new window's sample centers, in mas, at least two, regular,
            with this instrument's pitch.
        atom_step_mas : None or float
            The step of the trial positions, in mas; ``None`` is the mean
            step of this instrument's.
        margin_mas : None or float
            How far the trial positions reach beyond the window, in mas;
            ``None`` is one LSF width, ``resolution_mas``.

        Returns
        -------
        Instrument
            With this instrument's ``lsf``, ``dlsf_deta`` and
            ``resolution_mas``, and no margins.

        Raises
        ------
        ValueError
            If the grid is not regular, or its pitch differs from this
            instrument's.
        """
        grid = np.asarray(sample_grid_mas, dtype=float)
        pitch, new = _pitch(self.sample_grid_mas), _pitch(grid)
        if not np.isclose(new, pitch, rtol=1e-9, atol=0.0):
            raise ValueError(f"the grid's pitch {new} mas differs from the instrument's "
                             f"{pitch} mas: build an instrument for that pitch")
        step = (float(np.diff(self.atom_eta_mas).mean()) if atom_step_mas is None
                else float(atom_step_mas))
        margin = self.resolution_mas if margin_mas is None else float(margin_mas)
        atoms = np.arange(grid.min() - margin, grid.max() + margin + 1e-9, step)
        return Instrument(sample_grid_mas=grid, atom_eta_mas=atoms, lsf=self.lsf,
                          dlsf_deta=self.dlsf_deta, resolution_mas=self.resolution_mas)


def render_window(eta_sources, fluxes, inst):
    """The noise-free window of point sources: ``inst.design_columns(eta) @ f``.

    Parameters
    ----------
    eta_sources : array, shape (K,)
        Along-scan positions of the sources in the window's frame, in mas.
    fluxes : array, shape (K,)
        The sources' fluxes, in the units of the samples (flux per sample).
    inst : Instrument
        The instrument: its sample grid and LSF.

    Returns
    -------
    array, shape (N,)
        The samples of the window, flux per sample.
    """
    return inst.design_columns(np.atleast_1d(eta_sources)) @ np.atleast_1d(
        np.asarray(fluxes, dtype=float))


def render_curve(eta_fine, eta_sources, fluxes, inst):
    """The model window on a fine grid: ``inst.design_columns(eta, eta_fine) @ f``.

    The operator of :func:`render_window` evaluated at the positions
    ``eta_fine`` instead of the instrument's samples. Each value is the flux
    that a sample of the instrument's pitch, centred at that position, would
    collect, so the curve passes through the window's samples and can be
    drawn over them.

    Parameters
    ----------
    eta_fine : array, shape (M,)
        The positions of the curve, in the window's frame, in mas.
    eta_sources : array, shape (K,)
        Along-scan positions of the sources in the window's frame, in mas.
    fluxes : array, shape (K,)
        The sources' fluxes, flux per sample.
    inst : Instrument
        The instrument: its LSF.

    Returns
    -------
    array, shape (M,)
        The curve, flux per sample.
    """
    return (inst.design_columns(eta_sources, sample_grid_mas=eta_fine)
            @ np.atleast_1d(fluxes))


