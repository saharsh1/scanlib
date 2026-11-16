"""scanlib: the method of one-dimensional scanning astrometry, its atoms and the paper's pipeline.

A scanning instrument reads out, each time a star crosses its focal plane,
a short strip of samples along the scan direction: a one-dimensional
projection of the sky around the star, blurred by the line-spread
function. Many such windows at many scan angles are the data. This
package holds the atoms that turn them into a scene, each a
function that takes arrays and returns arrays, with the equation, the
units, the frame and the sign conventions in its docstring:

- :mod:`scanlib.instrument`: the along-scan projection ``eta = dra*
  sin(psi) + ddec cos(psi)``, the :class:`~scanlib.instrument.Instrument`
  (sample grid, line-spread function, trial positions, margins) with its
  one operator ``design_columns``, and one window rendered through it
  without noise (``render_window``, and ``render_curve`` on a fine grid);
- :mod:`scanlib.noise`: the noise sigma of each sample, read from a
  scalar, one value per window or one per sample (``sample_sigma``), or
  computed from the signal by shot, background and read noise
  (``sigma_from_counts``, ``NoiseModel``), and a Gaia-like model from a
  magnitude (``gaia_like``);
- :mod:`scanlib.scanlaw`: two public scan laws, the GOST forecasts at the
  Pleiades centre and at ecliptic latitude +60 deg, shipped with the
  package (``load_scan_law``, ``load_parallax_factor``);
- :mod:`scanlib.backprojection`: the back-projection map of the
  detections, every detection's sky line as a unit-area Gaussian ridge,
  summed, and its peak (``backprojection_map``, ``map_peak``);
- :mod:`scanlib.radon`: the two images of the windows themselves, the
  inverse Radon transform and faststack (``filtered_backprojection``,
  ``faststack``);
- :mod:`scanlib.joint`: the window-level Gaussian log-likelihood with the
  fluxes profiled by non-negative least squares, and its maximization over
  the companions' positions and motions with the Fisher covariance
  (``loglike_windows``, ``fit_joint``, static with ``tau=None`` or moving).

THE WALK, one line per step::

    inst = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18)
    eta = scene_positions_al(theta_true, psi, tau)                 # where the sources fall
    windows = [simulate_window(e, flux, inst, rng, sigma) for e in eta]
    sky, axis, extent = backprojection_map(det_psi, det_eta, det_sigma)
    image, axis = filtered_backprojection(psi, windows, inst)
    stack, var, n, axis = faststack(psi, windows, inst)
    fit = fit_joint(windows, psi, theta0, inst, sigma, tau=tau)

Conventions: angles in rad, the scan angle ``psi`` counter-clockwise from
north; sky offsets ``(dra*, ddec)`` in mas from the reference position,
the primary's position; the along-scan coordinate ``eta`` in mas; each
window's zero point ``eta0`` (the sky along-scan coordinate of its sample
grid's origin, ``None`` for 0); epochs ``tau`` in years from the reference
epoch; proper motions in mas/yr; fluxes per sample in the windows' units;
Gaussian noise of known sigma. Parameter vectors are flat: ``(dra*_1,
ddec_1, ...)`` static, ``(dra*, ddec, mu_a, mu_d[, dpi])`` per companion
moving.

The paper's pipeline is built from these atoms in its own modules, imported
by name: :mod:`scanlib.reduction`, the per-window reduction (how many
sources one window holds, and where) and its detection set over a
campaign; :mod:`scanlib.dynamics`, the fits to the detection set (the
position, the relative motion, the cleaning on the map); :mod:`scanlib.search`,
the search on the sky by likelihood-ratio tests, with its threshold in
:mod:`scanlib.thresholds`. Everything that knows the truth or
draws random numbers lives in :mod:`scanlib.simulation`, which the method
never imports: one noisy window (``simulate_window``), the zero points at a
random sampling phase (``random_zero_points``), the toy regime and the bridge
from a peak S/N (``snr_from_peak_snr``), the margins' calibration, and a
campaign of a scene over a scan law (``Scene``, ``harvest_campaign``). One module reads a file:
:mod:`scanlib.scanlaw`, the two forecasts shipped with the package.

Every symbol, unit and array convention above is defined once, for the
package and the notebooks, in ``docs/conventions.md``.
"""
from __future__ import annotations

#: The package version, the single source: pyproject reads it from here.
__version__ = "0.1.0"

from .instrument import (Instrument, project_al, gaussian_lsf,
                         gaussian_lsf_jacobian, render_window, render_curve)
from .noise import sample_sigma, sigma_from_counts, NoiseModel, gaia_like, gaia_electrons
from .scanlaw import load_scan_law, load_parallax_factor
from .simulation import simulate_window, random_zero_points, snr_from_peak_snr
from .backprojection import backprojection_map, map_peak
from .radon import direction_weights, filtered_backprojection, faststack
from .joint import (scene_positions_al, stacked_design,
                    loglike_windows, loglike_profiled, joint_fisher,
                    fit_joint)

__all__ = [
    "Instrument", "project_al", "gaussian_lsf", "gaussian_lsf_jacobian",
    "render_window", "simulate_window", "render_curve",
    "random_zero_points", "snr_from_peak_snr",
    "sample_sigma", "sigma_from_counts", "NoiseModel", "gaia_like", "gaia_electrons",
    "load_scan_law", "load_parallax_factor",
    "backprojection_map", "map_peak",
    "direction_weights", "filtered_backprojection", "faststack",
    "scene_positions_al", "stacked_design",
    "loglike_windows", "loglike_profiled", "joint_fisher",
    "fit_joint",
]
