# scanlib

The method of one-dimensional scanning astrometry: its atoms, and the
paper's pipeline built from them. A scanning
instrument such as Gaia reads out, each time a star crosses its focal
plane, a short strip of samples along the scan direction: a one-dimensional
projection of the sky around the star, blurred by the line-spread function.
Many such *windows* at many scan angles are the data. scanlib holds the
algorithms that turn them into a scene, each a function that takes arrays
and returns arrays, with the equation, the units, the frame and the sign
conventions in its docstring:

| | |
|---|---|
| `scanlib.instrument` | the along-scan projection η = Δα\* sin ψ + Δδ cos ψ; the `Instrument` (sample grid, line-spread function, trial positions, margins) with its one operator `design_columns`; one window rendered through it without noise (`render_window`, and `render_curve` on a fine grid) |
| `scanlib.noise` | the noise sigma of each sample, from a scalar, one value per window or one per sample (`sample_sigma`), or from the signal by shot, background and read noise (`sigma_from_counts`, `NoiseModel`); the Gaia-like setting for a magnitude G (`gaia_like`) |
| `scanlib.scanlaw` | two public GOST scan laws over the DR4 span, shipped with the package: the Pleiades centre (`PLEIADES`, 47 transits) and ecliptic latitude +60° (`BETA60`, 95 transits) (`load_scan_law`, `load_parallax_factor`) |
| `scanlib.reduction` | the per-window reduction: how many sources one window holds and where, by the nested ladder on the instrument's margins (`ladder_window`, `fit_k`, `sigma_eta`, `delivery_mask`), and over a campaign of windows into its detection set (`reduce_campaign`, `DetectionSet`) |
| `scanlib.thresholds` | the search's threshold Λ_c in closed form, from the instrument and the scan law (`lrt_threshold`) |
| `scanlib.search` | the search on the sky: candidates from the map, scored by the joint fit as Λ = 2 Δ ln L, accepted above Λ_c (`map_ladder`) |
| `scanlib.dynamics` | the fits to a campaign's detection set: the position (`fit_detection_lsq`), the relative motion and the parallax difference (`fit_motion`, `fit_mark_refit`, `fit_motion_robust`), and the cleaning (`clean_detection_set`: the map detects, a circle selects, the sinogram fit refines) |
| `scanlib.simulation` | everything that knows the truth or draws random numbers, which the method never imports: one noisy window (`simulate_window`), the zero points at a random sampling phase (`random_zero_points`), the toy regime (`PRIMARY_FLUX`, `snr_from_peak_snr`), the margins' calibration (`calibrate_margins`), a campaign of a `Scene` over a scan law (`simulate_campaign`, `harvest_campaign`), and the search's simulated null (`field_null_max`) |
| `scanlib.backprojection` | the back-projection map of the detections, each detection's sky line as a unit-area Gaussian ridge, summed, and its peak (`backprojection_map`, `map_peak`) |
| `scanlib.radon` | the two images of the windows themselves: the inverse Radon transform and faststack (Harrison 2011) (`filtered_backprojection`, `faststack`) |
| `scanlib.joint` | the window-level Gaussian log-likelihood with the fluxes profiled by non-negative least squares, and its maximization over the companions' positions and motions with the Fisher covariance (`loglike_windows`, `fit_joint`, static with `tau=None` or moving) |

From the detections to the joint fit there are two routes, one for a pair
(`clean_detection_set`) and one for a scene of several sources
(`map_ladder`); `docs/routes.md` sets them side by side, step by step.

The modules come in tiers (`docs/open_api.md`). The **atoms**
(`instrument`, `noise`, `backprojection`, `radon`, `joint`) each implement
one algorithm and know nothing of the paper's choices. The **pipeline**
(`reduction`, `dynamics`, `thresholds`, `search`) combines them as the paper
does; no atom imports it. `scanlaw` is data, and `simulation` holds
everything that knows the truth, which the method never imports. You bring
the windows as arrays. A colleague can read `joint.py` top to bottom and
find the likelihood.

## Install

With numpy and scipy already in your environment:

```
pip install "scanlib[notebooks] @ git+https://github.com/saharsh1/scanlib.git"
```

To follow and edit a clone instead:

```
git clone https://github.com/saharsh1/scanlib.git
pip install -e "./scanlib[notebooks]"
```

Two runtime dependencies, numpy and scipy. The `plots` extra adds
matplotlib and the `notebooks` extra matplotlib with Jupyter; the package
itself never draws. Then

```
jupyter notebook scanlib/notebooks/quickstart/1_sinogram_fit.ipynb
```

is the first of three quickstart notebooks, to be read in order
(`notebooks/quickstart/README.md`): the sinogram fit of one companion; a
moving companion, from the triangulation map to the seed and the joint fit;
and two companions, the search and the measurement. Each is a sequence of
short blocks: one call, written with its module (`dyn.wls`,
`joint.fit_joint`), the place in the paper that defines it, and what came
back. In `notebooks/scene/`,
tutorial 0 draws the three images of one scene and its joint fit, three
tutorials build each image step by step, and six limitation notebooks show
where the atoms' premises fail. Every limitation notebook has the same four parts: the premise and
the case that breaks it, then one section per atom in a fixed order, the
two images, the map of detections, the joint fit, each answering what the
atom assumes, what it returns and whether it warns, and a summary table
with the columns atom, effect, alarm, remedy. Every symbol, unit and array convention
they share with the package is defined once in `docs/conventions.md`. These notebooks (`quickstart/`,
`scene/` and `scene/limitations/`) are committed with their outputs, so GitHub shows them rendered:

| | |
|---|---|
| `notebooks/quickstart/1_sinogram_fit.ipynb` | one companion: from the 47 windows to the detections, and the linear fit of a sinusoid to them (the design matrix, weighted least squares, separation and position angle), at rest and with a proper motion; the Huber fit |
| `notebooks/quickstart/2_binary_backprojected_seed.ipynb` | a moving companion: the lines on the sky and their map, the pool of lines about the map's peak, the seed refined by the sinogram fit and the test against its track, the map refocused (Eq. 17), and the joint fit of all windows with its errors |
| `notebooks/quickstart/3_triple_ladder.ipynb` | the scene of the paper's Figure 4 (a companion inside one FWHM moving 8 mas/yr, and a neighbour): Λ_c, the search rung by rung (propose, score, test, assign) and in one call, the measurement, a catalogue prior, the companions relative to the primary, the checks, then the truth |
| `notebooks/scene/00_three_images_and_fit.ipynb` | one scene, forty windows, the three images and the joint fit |
| `notebooks/scene/01_inverse_radon.ipynb` | the inverse Radon transform step by step: the projection-slice theorem, the ramp filter, the half-turn weights, the image against the theory, and three ways the premise fails |
| `notebooks/scene/02_faststack.ipynb` | faststack step by step: the footprints, the plain mean, Harrison's four equations for one pixel, α, the sampling phase, and three ways the picture breaks |
| `notebooks/scene/03_triangulation_map.ipynb` | the triangulation map step by step: a detection as a line and a ridge, why unit area, the map built up, reading it with `map_peak`, and three ways it is ambiguous |
| `notebooks/scene/limitations/01_finite_window.ipynb` | the window's edge along the scan and its width across it: truncation modeled by the fit, clipped by the images; a source missing from some windows biases the fit and erases itself from faststack; why a flux-loss cut is worse |
| `notebooks/scene/limitations/02_scan_law.ipynb` | the largest gap and the anisotropy R of a set of directions, the error ellipse they shape, which windows resolve a close pair, faststack's α on clumped coverage, epochs tied to angles |
| `notebooks/scene/limitations/03_few_lines.ipynb` | one detection is a line, two cross anywhere with an error σ / sin Δψ, three can disagree; a close companion seen in one bundle, and the fit's ellipse as the directions narrow |
| `notebooks/scene/limitations/04_motion.ipynb` | the static fit of a mover and its χ² alarm, the images smeared, orbital curvature the straight line absorbs without a trace, a varying flux |
| `notebooks/scene/limitations/05_blend.ipynb` | inside a FWHM: the Fisher bound rises smoothly, one window's likelihood grows basins at low S/N, the window fit's tails, the joint fit's dependence on its seed |
| `notebooks/scene/limitations/06_model_errors.ipynb` | zero points, LSF width, noise σ and a missing neighbor, each put in error; how far each raises χ² against its scatter, and why χ² cannot name the input in error; noise in the two pictures |

## What it reproduces

`notebooks/paper/` reproduces the paper with the package alone: Figures 1
to 4, C.1 and C.2, the Gaia-like kernel of Sect. 6.2, and the two
thresholds, the per-window margins Λ_w (App. A) and the campaign threshold
Λ_c (Sect. 5.1). Each notebook asserts every number the paper quotes from
it; `notebooks/paper/README.md` lists them and the stores they read.
These notebooks are committed without outputs (`tests/test_paper_notebooks.py`
holds that): run them to see the figures and the asserted numbers.
Figures 5 and 6 (Sect. 6) are not part of the package.

## The walk

```python
import numpy as np
from scanlib import *

inst = Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18)   # 18 samples 59 mas apart, FWHM 100 mas
psi, tau = np.linspace(0, 3 * np.pi, 40) % (2 * np.pi), np.linspace(-2.5, 2.5, 40)   # a scan law
theta_true, flux = [120.0, 80.0, 4.0, -3.0], [1000.0, 300.0]     # a companion: offset [mas], drift [mas/yr]
eta = scene_positions_al(theta_true, psi, tau)                   # where each source falls in each window
rng = np.random.default_rng(0)
windows = np.array([simulate_window(e, flux, inst, rng, 10.0) for e in eta])   # noise sigma 10
image, axis = filtered_backprojection(psi, windows, inst)        # the inverse Radon transform
stack, variance, n_used, axis = faststack(psi, windows, inst)    # the robust stack
fit = fit_joint(windows, psi, [125.0, 75.0, 0.0, 0.0], inst, 10.0, tau=tau)   # the measurement
print(fit["theta"], fit["sigma_theta"], fit["loglike"])
```

With detections (a scan angle, an along-scan position and its uncertainty
per resolved window) `backprojection_map(psi, eta, sigma)` is the detector
whose peak seeds the fit. The notebooks in `notebooks/quickstart/` walk the
whole chain with the figures.

## Conventions

Every public function states them in its docstring. The ones that bite:

- angles in radians; the scan angle ψ counter-clockwise from north, so a
  scan along north measures Δδ and a scan along east measures Δα\*
- sky offsets (Δα\*, Δδ) in mas from the reference position, the primary's
  predicted position; by default the primary is pinned there and the
  parameter vectors hold the companions only; `free_primary=True` gives
  the primary its own offset (and, moving, its own residual motion), and
  every offset, the companions' included, stays measured from the reference
  position
- the along-scan coordinate η = Δα\* sin ψ + Δδ cos ψ in mas; each window's
  **zero point** η₀ is the sky along-scan coordinate of its sample grid's
  origin, so a source at η appears at η − η₀ in the window (`eta0=None`
  is every zero point 0)
- epochs τ in years from the reference epoch; proper motions in mas/yr;
  the parallax difference in mas, entering as Δϖ · f_par with the window's
  along-scan parallax factor
- fluxes per sample in the windows' own units; Gaussian noise of known σ,
  one scalar or one per sample
- flat parameter vectors: `(Δα*₁, Δδ₁, Δα*₂, …)` static, `(Δα*, Δδ, μα*,
  μδ[, Δϖ])` per source moving, the parallax entry present exactly when
  `fpar` is given; with `free_primary=True` the primary's block comes first
- **a seed is initialization, never data**: the map's peak starts the fit
  and carries no evidence

## Four laws, each a test

1. The numbers do not move by accident: `tests/test_baseline.py` compares
   the package's own outputs byte for byte, and
   `tests/test_reference_cases.py` compares against the development
   repository's export (below).
2. Conventions hold: the projection's cardinal angles, the signs of the
   motion and parallax terms, the unit-area ridge, the half-turn weights,
   the likelihood's constant (`tests/test_conventions.py`).
3. scanlib imports nothing outside itself, numpy, scipy and the standard
   library (`tests/test_import_closure.py`).
4. Every public name is the object it claims to be, with a docstring, and
   the source stays small (`tests/test_packaging.py`).

## Reference cases

The method was developed in a separate, private repository (`scanlib-dev`),
which this package replaces. The package is held to that code's outputs by
`tests/reference/reference_cases.npz` and `reference_cases.md`: four
synthetic scenes rendered by the development code over the first 40
transits of the public Gaia scan-law forecast for the Pleiades, with its
outputs of every algorithm here, the joint fit with the primary pinned and
free, and the stages of the paper's pipeline (the instrument, the scan
laws, the noise, the per-window reduction, the campaign, the dynamics, the
search and the thresholds).

**The reference file is the export of the development code at its commit
`b1c8848`** (exported 2026-11-10; the file's `meta/` arrays name it).
Fourteen of its arrays are this package's own, re-blessed after corrections
made here: the ten result arrays of `crowded_field/search` (the search's
line bookkeeping, and the search returning its last accepted fit's sources)
and the four `radon/image` arrays (the ramp filter's Nyquist term).
`reference_cases.md` says what moved.

## Citing

If you use scanlib, cite it with the metadata in
[`CITATION.cff`](CITATION.cff), and cite the published methods behind the
functions you use: faststack is Harrison (2011, Exp. Astron. 31, 1).

## License

MIT.
