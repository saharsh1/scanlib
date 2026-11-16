# Quickstart

Three notebooks, to be read in order. Each one is a sequence of short blocks: one call, where the paper
defines it, and what came back. Every scanlib call carries the name of its module (`dyn.wls`,
`joint.fit_joint`), so each line can be found in the package. All data are synthetic. The notebooks are
committed with their outputs.

| notebook | subject | paper |
|---|---|---|
| [`1_sinogram_fit.ipynb`](1_sinogram_fit.ipynb) | one companion: from the windows to the detections, and the linear fit of a sinusoid to them, at rest and with a proper motion | Sects. 2, 3, 4.1, 4.2; Eqs. 8–12, 14–17 |
| [`2_binary_backprojected_seed.ipynb`](2_binary_backprojected_seed.ipynb) | a moving companion: the triangulation map, the pool of lines about its peak, the refined seed, and the joint fit of all windows | Sects. 3.3, 4.2, 4.3, 4.5; Eqs. 13, 15–18; Fig. 3 |
| [`3_triple_ladder.ipynb`](3_triple_ladder.ipynb) | two companions: the search, one source at a time (propose, score, test, assign), the threshold, and the measurement | Sect. 5; Fig. 4 |

They run in about 10 s, 20 s and 1 min.

## The modules, as the notebooks import them

`atom`: one algorithm, independent of the paper's choices. `pipeline`: the paper's way of combining them
([`docs/open_api.md`](../../docs/open_api.md)).

| name | module | role | calls used |
|---|---|---|---|
| `ins` | `scanlib.instrument` | atom | `Instrument.gaussian`, `TOY_MARGINS`, `project_al` |
| `noise` | `scanlib.noise` | atom | `gaia_like`, `gaia_electrons` |
| `scanlaw` | `scanlib.scanlaw` | data | `load_scan_law`, `load_parallax_factor`, `PLEIADES` |
| `sim` | `scanlib.simulation` | simulation only | `Scene`, `Scene.pair`, `simulate_campaign`, `random_zero_points` |
| `red` | `scanlib.reduction` | pipeline | `ladder_window`, `sigma_eta`, `delivery_mask`, `reduce_campaign` |
| `bp` | `scanlib.backprojection` | atom | `backprojection_map`, `map_peak`, `peak_consistency` |
| `dyn` | `scanlib.dynamics` | pipeline | `motion_design`, `wls`, `fit_detection_lsq`, `fit_motion`, `fit_motion_robust`, `to_reference_epoch`, `clean_detection_set` |
| `joint` | `scanlib.joint` | atom | `fit_joint`, `loglike_profiled`, `joint_fisher`, `joint_covariance`, `polar_elements` |
| `thr` | `scanlib.thresholds` | pipeline | `lrt_threshold` |
| `search` | `scanlib.search` | pipeline | `next_peak`, `lambda_rung`, `map_ladder`, `describe` |

## Your own windows

The calls take plain arrays. What each needs from real data:

| input | shape | units, frame |
|---|---|---|
| `windows` | (J, N) | flux per sample, the samples of each window along the scan, in the window's own frame |
| `sigma` | scalar, (J,) or (J, N) | flux per sample: the noise σ of each sample, in the units of `windows`. For a Gaia-like σ from a magnitude, `noise.gaia_like` (notebook 1, block 3) |
| `psi` | (J,) | rad, the scan angle of each window, counter-clockwise from north |
| `tau` | (J,) | yr from the reference epoch; needed for a moving scene |
| `fpar` | (J,) | the along-scan parallax factor; needed for a parallax difference |
| `eta0` | (J,) | mas, the sky position of each window's sample-grid origin, along the scan |
| `inst` | — | your sample grid and LSF (`ins.Instrument`), with margins calibrated for them (`sim.calibrate_margins`) |

Positions are offsets from the reference position (the primary's catalogue position at the reference
epoch). The order of the calls:

```python
det = red.reduce_campaign((psi, tau, fpar), windows, inst, sigma, k_max=3, eta0=eta0)    # notebook 1, block 5

# one companion (notebook 2): the seed from the detections, then the joint fit
seed, cov, keep, info = dyn.clean_detection_set(det.psi, det.tau, det.eta, det.sigma)
fit = joint.fit_joint(windows, psi, seed, inst, sigma, tau=tau, eta0=eta0)

# an unknown number of sources (notebook 3): the search; its last accepted fit is the measurement
ladder = search.map_ladder(det.psi, det.eta, det.sigma, windows, psi, inst, sigma,
                           tau=tau, det_tau=det.tau, eta0=eta0, free_primary=True)
fit = ladder.log[-1]["fit"]
```

## Where to read next

How each image and the map work, one at a time: the tutorials in [`../scene/`](../scene/). Where each
step's premise breaks: [`../scene/limitations/`](../scene/limitations/). The paper's figures and numbers:
[`../paper/`](../paper/). The two ways from the detections to the joint fit:
[`docs/routes.md`](../../docs/routes.md). Every equation is in the module docstrings.
