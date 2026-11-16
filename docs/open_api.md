# The open surface

**Contract.** A name is on the open surface when it is listed below, imports
from the module shown, carries a docstring that states its equation, units,
frame and sign conventions, and a test covers it, directly or through the
routine that calls it. Everything else (an
underscored helper, a module constant not listed) may change without notice.
`tests/test_import_closure.py` holds the package to this page.

**Conventions every row relies on.** Angles in radians; the scan angle ψ
counter-clockwise from north; the along-scan coordinate
η = Δα* sin ψ + Δδ cos ψ in mas; sky offsets (Δα*, Δδ) in mas relative to
the reference position, the primary's predicted position; each window's
**zero point** η₀ in mas, the along-scan sky coordinate of its sample grid's
origin, so that sample n lies at η₀ + gₙ and a source at η appears at η − η₀
in the window (every routine that combines windows takes `eta0`, one per
window, supplied with the windows; `None` is every zero point 0); epochs τ in years
from the reference epoch; proper motions in mas/yr; fluxes in the windows'
own units, flux per sample; Gaussian per-sample noise of known σ (one scalar, one per window, or one per sample).
Parameter vectors are flat: companion offsets `(Δα*₁, Δδ₁, Δα*₂, …)`, the
moving form per companion `(Δα*, Δδ, μα*, μδ[, Δϖ])`; with
`free_primary=True` the primary's own block comes first.

The symbols are defined once in `docs/conventions.md`.

## The instrument is an argument

Every routine that turns positions into samples takes the instrument `inst`,
an `Instrument` (sample grid, line-spread function and its derivative,
trial positions, LSF width, per-window margins), and nothing else describes
the instrument. There is no default instance: a notebook or a script
defines it once,

```python
from scanlib import Instrument
from scanlib.instrument import TOY_MARGINS

inst = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18,
                           margins=TOY_MARGINS)
```

and passes `inst` to every call. A different line-spread function is two
callables handed to `Instrument` (see `gaussian_lsf` and
`gaussian_lsf_jacobian` for the conventions they follow).

## Tiers

Each module is one of four tiers.

- **atom**: one algorithm, or the instrument, as a function of arrays. It
  imports numpy, scipy, the standard library and other atoms, and never a
  pipeline module.
- **data**: reads a file shipped with the package and returns arrays.
- **pipeline**: a stage of the paper's method (the per-window reduction, the
  campaign's detections, the dynamics, the search), composed of atoms.
- **simulation**: everything that knows the truth or draws random numbers to
  make data (a noisy window, the zero points, the margins' null windows,
  later the campaigns). It may import every other tier; no atom, data or
  pipeline module imports it, so the method never sees the truth.

`tests/test_import_closure.py` checks that every module has exactly one row
below, that every name of a row imports from its module and is defined
there only, that every name of `scanlib.__all__` has a row, that no atom
imports a pipeline module, and that no method module imports the
simulation.

## Modules

| module | tier | names | what it holds |
|---|---|---|---|
| `scanlib.instrument` | atom | `Instrument`, `TOY_MARGINS`, `TOY_MARGINS_SEED`, `TOY_MARGINS_N`, `project_al`, `gaussian_lsf`, `gaussian_lsf_jacobian`, `render_window`, `render_curve` | the along-scan projection; the instrument, its operator `design_columns` and its derivative `design_jacobian`, and the same LSF on another window's grid (`for_window`); the toy instrument's per-window margins; one window rendered without noise, on its samples and on a fine grid |
| `scanlib.noise` | atom | `sample_sigma`, `sigma_from_counts`, `NoiseModel`, `gaia_like`, `gaia_electrons` | the noise sigma of each sample, from a scalar, one value per window or one per sample; the counts-to-sigma recipe (shot, background and read noise) and its frozen setting; the Gaia-like setting of a source of magnitude G (its electrons per transit, nine CCD crossings, their background and read noise) |
| `scanlib.scanlaw` | data | `PLEIADES`, `BETA60`, `load_scan_law`, `load_parallax_factor` | two public GOST scan laws over the DR4 span, shipped with the package with their READMEs: the scan angles, the epochs and the along-scan parallax factors |
| `scanlib.reduction` | pipeline | `ACTIVE_FLUX_REL`, `profiled_chi2`, `fit_positions`, `sigma_eta`, `active_clusters`, `fit_k`, `default_margins`, `ladder_window`, `delivery_mask`, `SIGMA_DET_FALLBACK_MAS`, `DetectionSet`, `reduce_campaign` | the per-window reduction: the K-source fit from the starting configurations, the nested ladder on the instrument's margins, the delivery rule and the Fisher sigmas; the same reduction over a campaign of windows, into its detection set |
| `scanlib.dynamics` | pipeline | `HUBER_DELTA`, `SELECT_RADIUS_MAS`, `wls`, `motion_design`, `to_reference_epoch`, `fit_motion`, `fit_mark_refit`, `fit_motion_robust`, `fit_detection_lsq`, `clean_detection_set` | the fits to the detection set: the position alone, the linear relative motion (and the parallax difference) by weighted least squares, with one refit or Huber weights, and the cleaning on the map |
| `scanlib.simulation` | simulation | `PRIMARY_FLUX`, `K_MAX`, `MARGIN_PERCENTILE`, `CALIBRATION_WINDOWS`, `NULL_PAIR_SEPARATION_FWHM`, `NULL_PAIR_Q`, `simulate_window`, `random_zero_points`, `snr_from_peak_snr`, `rung_null_statistic`, `calibrate_margins`, `Scene`, `simulate_campaign`, `harvest_campaign`, `field_null_max` | the toy regime (the primary's flux, S/N as total flux over sigma, the bridge from a peak S/N); one noisy window; the zero points at a random sampling phase; the margins' calibration on each rung's null; the true sky of a simulation and its campaign over a scan law, rendered and reduced; the maximum of the search's candidate field in noise-only campaigns |
| `scanlib.thresholds` | pipeline | `FALSE_RATE`, `MU_MAX_MAS_YR`, `FIELD_HALF_MAS`, `search_curvatures`, `lrt_threshold` | the search's threshold Lambda_c in closed form: the Lipschitz-Killing curvatures of the searched region in the candidate field's metric and the expected Euler characteristic (a high-threshold approximation; equal noise weights on every sample), from the instrument and the scan law alone |
| `scanlib.search` | pipeline | `GATE_MIN_DETECTIONS`, `MAP_STEP_MAS`, `ASSIGN_FLOOR_MAS`, `MapLadder`, `next_peak`, `lambda_rung`, `map_ladder`, `describe` | the search: candidates proposed by the map of the unassigned detections, scored by the joint fit, Lambda = 2 Delta ln L, accepted above Lambda_c |
| `scanlib.backprojection` | atom | `backprojection_map`, `map_peak`, `peak_consistency` | the back-projection map of the detections and its peak |
| `scanlib.radon` | atom | `ramp_filter`, `direction_weights`, `filtered_backprojection`, `faststack` | the two images of the windows: the inverse Radon transform and faststack (Harrison 2011) |
| `scanlib.joint` | atom | `scene_positions_al`, `stacked_design`, `loglike_windows`, `loglike_profiled`, `joint_fisher`, `joint_covariance`, `polar_elements`, `fit_joint` | the window-level Gaussian log-likelihood with the fluxes profiled, its Fisher matrix, and the joint fit, static (`tau=None`) or moving |
| `scanlib.optimize` | atom | `minimize_restarts` | Nelder-Mead restarted until a restart stops helping |

## Rules every signature follows

- A routine that takes the instrument takes it as `inst`, and the noise
  sigma as `sigma_noise`, after `inst`. `tests/test_inputs.py` checks it.
- The noise sigma is a positive scalar, one value per window (shape `(J,)`
  or `(J, 1)`), or one per sample (shape `(J, N)`, or `(N,)` for one
  window); `scanlib.noise.sample_sigma` reads every form.
- Inputs that do not match (windows and angles, samples and grid, one flux
  per source, one value per detection) raise a `ValueError` that names them;
  nothing is cut short.
- An `Instrument` is frozen: read-only copies of its arrays, equal only to
  itself. The per-window reduction needs trial positions over the whole
  sample grid; `inst.for_window(grid)` builds them.
- A Fisher matrix is inverted when its correlation matrix has a condition
  number below `joint.COND_MAX` (1e12); otherwise the errors are infinite.
  The test does not depend on the units.
