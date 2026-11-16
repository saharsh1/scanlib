# Symbols, units and conventions

One list for the package, the tutorials and the limitation notebooks. Every
docstring and every notebook uses these names in these units. Angles are in
radians unless a degree sign is written; lengths in mas; fluxes per sample
in the windows' units.

## The sky and the scan

- **ψ**, the scan angle, counter-clockwise from north. A scan direction is
  defined modulo π: windows at ψ and ψ + π measure the same line.
- **(Δα\*, Δδ)**, the sky offsets of a source from the reference position,
  the primary's predicted position, in mas, Δα\* including the cos δ factor.
  By default the primary is pinned at the origin, so the offsets are also
  offsets from the primary. With `free_primary=True` the primary has its
  own offset, and a companion's offset from the primary is its own minus
  the primary's. In code: `dra`, `ddec`, and the flat vector `theta`.
- **η = Δα\* sin ψ + Δδ cos ψ**, the along-scan coordinate of a source in a
  window at angle ψ, in mas (`project_al`). A scan along north (ψ = 0)
  measures Δδ; a scan along east (ψ = π/2) measures Δα\*.
- **ζ = −Δα\* cos ψ + Δδ sin ψ**, the across-scan coordinate, in mas.
- **η₀**, a window's zero point: the sky along-scan coordinate of its sample
  grid's origin, in mas. A source at η appears at η − η₀ in that window.
  In code: `eta0`, one per window, `None` for 0.
- **ρ, φ**, the companion's separation, in mas or in units of the FWHM, and
  position angle, in degrees from north through east. The *pair axis* is the
  line from the primary to the companion. The notebooks use φ = 110°.

## The instrument

- **gₖ, p, N**, the sample centers along the scan, their pitch and the
  number of samples of a window, indexed k = 0, …, N − 1 with
  gₖ = g₀ + k p; **g₀** is the center of the first sample, `grid[0]`. The
  toy instrument has N = 18 and p = 59 mas, so g₀ = −501.5 mas and a window
  spans η from −501.5 to +501.5 mas. In code: `inst.sample_grid_mas`,
  `grid` in the notebooks.
  Sample k covers the along-scan interval [gₖ − p/2, gₖ + p/2), so the
  sample that holds a coordinate η, in the window's own frame, is
  k = ⌊(η − g₀)/p + ½⌋: the distance from the first center, counted in
  samples and rounded to the nearest. η is off the window when that k is
  outside 0 to N − 1. A window with a zero point η₀ reads a sky coordinate
  η at η − η₀ in its own frame first.
- **LSF, FWHM, σ_LSF**, the Gaussian line-spread function, its full width
  at half maximum and its standard deviation σ_LSF = FWHM / (2√(2 ln 2));
  the toy instrument has FWHM = 100 mas and σ_LSF = 42.5 mas. **L(gₖ − η)**
  is the fraction of a source's flux in sample k (`gaussian_lsf`,
  `inst.lsf`); **A**, the design matrix of those fractions, one column per
  source (`inst.design_columns`).
- **Flux**, per sample, in the windows' units; the notebooks give the
  primary 1000, so a sample centered on it holds about 550. **q**, the flux
  ratio companion / primary.
- **σ** (`sigma_noise`), the noise per sample, Gaussian and known: a scalar
  for every sample, or one value per window or per sample. **S/N**, the
  primary's total flux over σ: S/N 100 means σ = 10 in the notebooks.
- **Realistic noise** (`noise.gaia_like(G, flux)`): the windows stay in
  their flux units, and the magnitude G of the primary says how many
  electrons those units are (3.0 × 10⁵ electrons per transit at G = 16,
  so 301 electrons per unit when the primary is 1000). The noise of each
  sample is computed in electrons, σ_e² = s_e + 9 (b + r²) (the sample's
  own electrons, and the background b and read noise r of nine CCD
  crossings; the paper's Eq. 21), and returned in the windows' units,
  σ = σ_e / (electrons per unit). A sample's signal over its σ is the
  same in both units; the magnitude sets it. For data in electrons,
  `flux` is the source's electrons and one unit is one electron.
- **Windows**, J windows of N samples, the (J, N) array `windows`;
  *noise-free* from `render_window`, *noisy* from `simulate_window` with a
  seeded generator.
- **A sample's strip**, the band of sky one pitch wide, perpendicular to the
  scan, whose light the sample collected; unbounded across the scan in the
  toy instrument, which has no across-scan width.
- **The Gaia window**, for reference (Gaia Collaboration, Prusti et al.
  2016, Sect. 3.3.2): pixels of 58.9 mas along the scan and 176.8 mas
  across it; 18 samples along the scan for stars brighter than G = 16
  (±530 mas, the last sample center at ±501 mas), 12 samples for fainter
  stars (±353 mas); 12 pixels across the scan, ±1061 mas, binned into one
  row for stars fainter than G = 13. The toy instrument follows the bright
  case along the scan, with p = 59.0 mas, and has no across-scan width.

## Time and motion

- **τ**, the epoch of a window, in years from the reference epoch, over a
  span T. In code: `tau`, one per window, `None` for a static scene.
- **μ = (μα\*, μδ)**, a source's proper motion relative to the reference
  position, in mas/yr (relative to the primary when it is pinned; with
  `free_primary=True` the primary's own μ is its residual motion); **Δϖ** (`dpi`), the parallax difference, in mas, with the
  window's along-scan parallax factor **f_ϖ** (`fpar`); **a**, an
  acceleration, in mas/yr².
- **θ** (`theta`), the parameter vector of the fit, flat: (Δα\*, Δδ) per
  source for a static scene; (Δα\*, Δδ, μα\*, μδ) moving; (Δα\*, Δδ,
  μα\*, μδ, Δϖ) with parallax. Pinned (`free_primary=False`, the default),
  θ holds the K − 1 companions and the primary has no entry; free
  (`free_primary=True`), θ holds K blocks, the primary's first. **θ₀**
  (`theta0`), the seed: initialization only, never data.

## Detections and the map

- **A detection** (ψ, η̂, σ_η): one source's along-scan position in one
  window, on the sky, with **σ_η**, its Fisher 1σ in mas with the fluxes
  free. Two kinds appear in the notebooks:
  - *pipeline detections*, made from the windows by the per-window
    reduction (`reduce_campaign`, `scanlib.reduction`): each window is
    fitted with one, two or three sources, a source added only when it
    beats the instrument's margin Λ_w; the component nearest the primary's
    place in the window (−η₀) is the primary and is kept back
    (`delivery_mask`), every other component is delivered, with σ_η from
    the window's Fisher matrix (`sigma_eta`) or, where that matrix is
    singular, the fallback `SIGMA_DET_FALLBACK_MAS` = 10 mas (counted in
    `n_sigma_substituted`). A window reduced to one source delivers
    nothing: it is *censored*, and still enters the joint fit.
  - *teaching detections*, made in the scene tutorials and limitation
    notebooks as the truth plus one Gaussian draw of σ_η, where a window
    delivers one when σ_η is finite and ≤ 30 mas. This stand-in shows the
    map without the reduction; it is not the package's rule.
- **The map** (triangulation or back-projection map,
  `backprojection_map`): each detection's sky line as a Gaussian ridge of
  width σ_η and unit area, summed, in 1/mas. **`map_peak`** returns its
  highest local maximum outside the primary's zone and **the width**, the
  rms half-width of the region above half the peak across its narrow
  direction, in mas.

## The images

- **The inverse Radon transform** (`filtered_backprojection`): each window
  as a density in flux/mas, ramp-filtered, laid back along its line with
  its **direction weight** (`direction_weights`, each direction's share of
  the half-turn, in rad, summing to π); the image in flux/mas².
- **faststack** (`faststack`, Harrison 2011): at each sky pixel the sample
  of each window whose strip covers it, M samples sorted, **r₀** the mean
  of the lowest max(1, ⌊αM⌋), weights 1 for those and exp(1 − (θ/r₀)²) for
  the rest; the image in flux per sample. **α**, the trimming fraction.

## The fit and its diagnostics

- **ln L**, the window-level log-likelihood in nats, the normalizing
  constant included (`loglike_windows`); **the profiled likelihood**, the
  same with the fluxes solved by non-negative least squares at each trial θ
  (`loglike_profiled`; `return_fluxes=True` also returns the fluxes).
- **χ²**, the residual χ² of the fit at its solution; **dof**, its degrees
  of freedom, J N minus the number of fitted parameters, positions and
  fluxes; when the model and the noise are right, **χ²/dof** is about 1,
  with a 1σ scatter of √(2/dof) (0.037 for 80 windows of 18 samples), so it
  is read against that scatter, not against 1. The count holds while no
  parameter sits at a bound (a flux at zero, a position at the box's edge);
  a parameter held at a bound fits nothing, and the effective dof is larger.
- **Fisher matrix, Fisher σ, covariance, ellipse**: the expected Fisher
  matrix of the full model at the solution (`joint_fisher`), its inverse
  (`joint_covariance`), the square root of its diagonal, and the ellipse of
  the position block. The inverse of the Fisher matrix is the Cramér–Rao
  bound on the covariance of an unbiased estimator; the fit reports it at
  its own solution, a local estimate of its errors. It describes the fit
  while the likelihood is close to quadratic over the errors and no
  parameter sits at a bound; a blend's biased fit, a second basin or a flux
  at zero are outside it. Whether the errors cover the truth as often as
  they claim is measured over many draws (`paper/auxil/sim_coverage_fig3`,
  `sim_coverage_fig4`). **Pull**, (fit − truth) / Fisher σ.
- **converged** (`fit_joint`): `True` when a fresh restart of the
  optimizer could not improve the objective and ended inside its own
  iteration budget at a finite value (`scanlib.optimize`). It says the
  optimizer stopped at a minimum; not that the minimum is the global one,
  nor that no parameter sits at a bound.
- **Ellipses and regions**: two different objects, both drawn from a 2 × 2
  block C of the covariance. The **covariance ellipse at Mahalanobis radius
  k** is the set (x − x̂)ᵀ C⁻¹ (x − x̂) = k², its axes k times the square
  roots of C's eigenvalues; for two Gaussian parameters it encloses
  1 − e^{−k²/2} of the probability: 39.3 % (k = 1), 86.5 % (k = 2), 98.9 %
  (k = 3). The **joint confidence region** of two parameters is the set
  Δχ² ≤ 2.30, 6.18, 11.83, which encloses 68.3, 95.4, 99.7 %, the
  probabilities of the 1, 2, 3σ intervals of one parameter. The scene and
  limitation notebooks draw covariance ellipses ("kσ" = radius k); the
  paper's figures draw joint regions ("1, 2, 3σ" =
  68.3, 95.4, 99.7 %). A one-parameter σ, the square root of a diagonal
  entry, is a third object: the marginal error of that parameter alone.

## Scan laws

- **R = |⟨e^{2iψ}⟩|**, the anisotropy of a set of directions, 0 for an even
  set and 1 for one direction; **the largest gap**, the widest range of
  directions with no window, in degrees of the half-turn.
- **The laws of the notebooks**: *80 even directions* or *40 even
  directions*, spaced evenly over the half-turn; *the Gaia forecast*, the
  first 40 scan angles and epochs of the public GOST forecast for the
  Pleiades center, stored with the reference cases in `tests/reference/`.
- **The laws shipped with the package** (`scanlib.scanlaw`): the GOST
  forecasts over the DR4 span at the Pleiades center (`PLEIADES`, 47
  transits, largest gap 90°, R = 0.37) and at ecliptic latitude +60°
  (`BETA60`, 95 transits, largest gap 10°, R = 0.11). `load_scan_law`
  returns ψ and τ, the years from the mean of the transits' epochs;
  `load_parallax_factor` returns f_ϖ.

## Arrays

Flat parameter vectors as above. Sky images and maps are (M, M) arrays on
the grid `axis = arange(-half, half + 1e-9, step)` (it ends at `half` when `step` divides `2 half`), rows Δδ and columns
Δα\*, both increasing with index (`imshow(origin='lower')`); east to the
left is a display choice, never baked into the arrays. Detections are three
(D,) arrays `psi`, `eta`, `sigma`. The per-window arrays `psi`, `tau`,
`fpar`, `eta0` have shape (J,).
