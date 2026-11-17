# Reproducing the paper

One notebook per figure of the paper and one per simulation behind a number
of the paper. They depend on the package alone.

## The rules

- **Only the package.** A notebook imports `numpy`, `matplotlib` (with
  `matplotlib.patheffects`), `pathlib`, `scipy.stats`, `scipy.ndimage`,
  `scipy.interpolate` and `scipy.optimize`, and only the `scanlib` names
  listed under "The surface" below. A notebook in the design of
  `notebooks/quickstart/` imports the modules (`from scanlib import dynamics
  as dyn`) and writes every call with its module (`dyn.wls`); the others
  import the names (`from scanlib.dynamics import wls`). No helper modules,
  no `sys.path`, no private names. A plotting helper the figure needs is written in the
  notebook. `tests/test_paper_notebooks.py` enforces this.
- **One instrument, defined once.** The first code cell builds
  `inst = Instrument.gaussian(fwhm_mas=FWHM_MAS, pitch_mas=PITCH_MAS, n_samples=N_SAMPLES, margins=TOY_MARGINS)`
  and passes `inst` to every call that takes one. The noise is always an
  argument: S/N 100 is `sigma = PRIMARY_FLUX / 100`.
- **One capability per cell, nothing hidden.** The first cell states what
  the notebook reproduces, the numbers it asserts and how long it runs.
- **Simulations store, figures read.** A simulation too slow for every run
  (`auxil/sim_*.ipynb`) writes its arrays to `stored/` with `FULL = True`. By
  default it loads them and checks a cheap part against a fresh
  computation, which fails if a change to the package moves them. A figure
  notebook reads a store with `numpy.load` and nothing else.
- **The numbers are asserted.** Every number the paper quotes from a
  notebook is printed and asserted in it.
- **PDFs** go to `pdf/` (git-ignored) with `SAVE_PDF = True`.
- **Committed clean**, without outputs.

## The notebooks

| notebook | paper | asserts | run time | store |
|---|---|---|---|---|
| `fig1_per_window_reduction.ipynb` | Fig. 1, Sect. 2.2 | the ladder's fits and decisions on the two windows; the margins 11.1, 10.8 | < 1 s | — |
| `fig2_scans_to_sky.ipynb` | Fig. 2, Sects. 3, 4.4 | 29 detections of 47 windows, 18 censored, none rejected by the cleaning; the offset (189.8 ± 1.3, −69.0 ± 0.9) mas, q = 0.197 ± 0.003, the truth at Δχ² = 3.3; each convenience equal to its atoms | 2 s | — |
| `fig3_moving_companion.ipynb` | Fig. 3, Sects. 4.1–4.5 | (a) 34 detections, 13 censored, 1 rejected; the joint fit's proper motion (3.05 ± 0.56, 8.89 ± 0.37) mas/yr, q = 0.302; the null test Δϖ = −0.07 ± 0.96 mas; (b) 32 detections, 15 censored, 2 rejected, Δϖ = −7.45 ± 0.95 mas (−7.8σ); from the coverage store, the detection-set fit's bias and the window-level fit unbiased and 2–5 times tighter | 4 s | reads `stored/fig3_coverage.npz` |
| `fig4_contaminated_binary.ipynb` | Fig. 4, Sect. 5.2 | 3, 19, 25 windows with one, two, three sources; 69 detections; Λ_c = 19.1; C accepted first, then B, after which fewer than three lines are left; B at 75.0 ± 0.1 mas, 110.2° ± 0.2°, μ (−2.73 ± 0.10, −7.47 ± 0.09) mas/yr; C at 250.07 ± 0.05 mas, 300.00° ± 0.02°; the primary consistent with zero | 30 s | — |
| `figC1_inverse_radon.ipynb` | Fig. C.1, App. C | 95 and 47 transits; the companion over the spurious level on both laws; the map's peak within 5 mas; the hand-built inversion equal to `filtered_backprojection` | 3 s | — |
| `figC2_three_pictures.ipynb` | Fig. C.2, App. C | each picture over its value at A at B and C; their largest excursions beyond 1.5 FWHM; the hand-built inversion and faststack equal the package's | 1 s | — |

### Auxiliary (`auxil/`)

The notebooks behind the figures' numbers and inputs: the Gaia-like kernel and the five simulations.
The simulations write their stores to `notebooks/paper/stored/` (`../stored/` from here), where the
figures read them.

| notebook | paper | asserts | run time | store |
|---|---|---|---|---|
| `auxil/kernel_rowell_lsf.ipynb` | Sect. 6.2, the kernel of Fig. 5 | the Gaia-like kernel from the digitized FOV1 LSF of Rowell et al. (2021, Fig. 11), read from `data/rowell2021_lsf/` of the repository, as a non-Gaussian `Instrument` | 1 s | — |
| `auxil/sim_coverage_fig3.ipynb` | Sect. 4.5, the coverage of Fig. 3's errors | pull widths within 15 % of one; 1, 2, 3σ coverage within three binomial σ; the detection-set fit biased, the window-level fit unbiased | 10 s; `FULL` 9 min | `stored/fig3_coverage.npz` |
| `auxil/sim_coverage_fig4.ipynb` | Sect. 5.2, the coverage of Fig. 4's errors | 600 realizations of Fig. 4's measurement, primary free; B's pull widths within 4 % of one; coverage 0.693, 0.973, 1.000 | 5 s; `FULL` 14 min | `stored/fig4_coverage.npz` |
| `auxil/sim_lambda_c.ipynb` | Sects. 5.1, 5.2, the campaign threshold Λ_c | Λ_c = 19.13 for Figure 4; 3200 simulated null campaigns per configuration, every false rate at or below 1 % within its interval | 20 s; `FULL` 12 h | `stored/lambda_c_null.npz` |
| `auxil/sim_lambda_c_validation.ipynb` | Sect. 5.1, the closed form of Λ_c tested away from the paper's case | ten cases (the paper's, and nine that change the scan law, the searched region, the LSF width, or search at rest), 1600 simulated null campaigns each, uniform noise, primary pinned: at the formula's 1 % level every case's false rate agrees with 1 % within its interval, all together 0.98 % [0.83, 1.15]; at its 10 % level the rates are 6.8–9.2 % | 50 s; `FULL` 20 h | `stored/lambda_c_validation.npz` |
| `auxil/sim_margins_lambda_w.ipynb` | App. A, the margins Λ_w | Δ₁ = 11.1, Δ₂ = 10.8, equal to `TOY_MARGINS` to the last bit | 15 s; `FULL` 20–50 min | `stored/margin_nulls.npz` |

## Other environments

The store checks are exact. They hold on macOS arm64 with numpy 2.4.3 and
scipy 1.17.1, and with numpy 2.5.3 and scipy 1.18.1. Another numpy, scipy or
BLAS can round the last digits differently, and a fit behind an optimizer can
then stop at another point. If a store check fails by a small difference while
every assert on the paper's numbers holds, the cause is rounding. Please report
it with the versions and the difference.

## The surface

The `scanlib` names these notebooks use, by module; each is used by at
least one notebook.

- `scanlib.instrument`: `Instrument`, `project_al`, `render_curve`, `render_window`, `TOY_MARGINS`, `TOY_MARGINS_N`, `TOY_MARGINS_SEED`
- `scanlib.noise`: `gaia_like`, `gaia_electrons`, `GAIA_N_CCD`, `GAIA_READ_NOISE_E_PER_CCD`
- `scanlib.scanlaw`: `BETA60`, `load_parallax_factor`, `load_scan_law`, `PLEIADES`
- `scanlib.simulation`: `field_null_max`, `harvest_campaign`, `MARGIN_PERCENTILE`, `PRIMARY_FLUX`, `random_zero_points`, `rung_null_statistic`, `Scene`, `simulate_campaign`, `simulate_window`, `snr_from_peak_snr`
- `scanlib.reduction`: `active_clusters`, `delivery_mask`, `fit_k`, `fit_positions`, `ladder_window`, `SIGMA_DET_FALLBACK_MAS`, `sigma_eta`
- `scanlib.backprojection`: `backprojection_map`, `map_peak`, `peak_consistency`
- `scanlib.radon`: `direction_weights`, `faststack`, `filtered_backprojection`, `ramp_filter`
- `scanlib.joint`: `fit_joint`, `joint_covariance`, `joint_fisher`, `loglike_profiled`, `loglike_windows`, `polar_elements`, `scene_positions_al`, `stacked_design`, `THETA_BOUND`
- `scanlib.dynamics`: `clean_detection_set`, `fit_motion`, `fit_motion_robust`, `motion_design`, `to_reference_epoch`, `wls`
- `scanlib.search`: `describe`, `GATE_MIN_DETECTIONS`, `lambda_rung`, `map_ladder`, `next_peak`
- `scanlib.thresholds`: `lrt_threshold`, `search_curvatures`
- `scanlib.optimize`: `minimize_restarts`
