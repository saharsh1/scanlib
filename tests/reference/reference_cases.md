# Reference cases exported from scanlib-dev

Written by `tools/export_reference_cases.py` (its module docstring gives the
file layout, the scenes, the conventions and how the detections are made).
Load with `numpy.load`; every array is listed here with its shape, its unit
and the function that produced it. `tests/test_reference_cases_export.py`
checks that the file and this table agree.

**Re-blessed in the shared scanlib.** The ten result arrays of
`crowded_field/search` (`positions`, `fluxes`, `Lambda`, `bar`, `accepted`,
each pinned and `_free`) are this package's, after the correction of the
search's line bookkeeping; scanlib-dev's export recorded
the search before it. Pinned, the search now accepts C at the fifth rung
(Lambda 2078) where dev's stopped at the fifth rung (Lambda 0); with the
primary free it accepts B at the third rung (Lambda 31.3) where dev's fit
failed (Lambda -6001). Re-blessed again when the search came to
return the sources of its last accepted joint fit instead of each source's
own rung: `positions`, `fluxes`, `positions_free` and `fluxes_free` (the
rungs, `Lambda`, `bar` and `accepted`, are unchanged). Pinned, the five
sources move from (34.01, -166.96), (99.87, 61.85), (240.08, -193.10),
(75.35, -92.09), (-141.51, 29.55) mas with fluxes 532.93, 273.47, 19.25,
33.25, 145.65 to (40.61, -170.39), (78.42, 74.59), (536.33, -289.66),
(97.63, 54.63), (-141.51, 29.55) with 501.89, 102.67, 0.00, 200.35,
145.65; free, the three from (38.59, -163.93), (38.57, -163.87), (98.87,
55.47) with 542.03, 537.10, 314.20 to (43.83, -168.99), (249.62, -179.52),
(98.87, 55.47) with 503.19, 16.93, 314.20. The other 318 arrays are
unchanged. The four `radon/image` arrays (one per scene) are re-blessed after the
correction of the ramp filter's Nyquist term (the commit before this note's): the
images move by at most 6.1e-5, 5.4e-5, 9.6e-5 and 6.9e-5 flux/mas² (static_pair,
moving_pair, crowded_field, offset_primary), 0.08 %, 0.06 %, 0.13 % and 0.09 % of
their peaks. `moving_pair/dynamics/clean_iterations` is re-blessed after the cleaning's new order (the map's peak, the
detections within 30 mas of it, the Huber fit, then the test against the fitted track): 3 passes, was 2; the
cleaned fit, its covariance and the kept set (`clean_theta`, `clean_cov`, `clean_keep`) are unchanged.
Every array but these fifteen (the ten of `crowded_field/search` named
above, the four images and this count) is dev's, byte for byte; `meta/` still names the dev
commit of the export.

| array | shape | unit | produced by |
|---|---|---|---|
| `static_pair/inputs/psi` | `(40,)` | rad | scanlib.io.gost.load_scan_law(gost_pleiades_centre_dr4span.csv), first 40 |
| `static_pair/inputs/windows` | `(40, 18)` | flux per sample | scanlib.sky.campaign.simulate_campaign |
| `static_pair/inputs/sigma` | `()` | flux per sample | simulate_campaign (noise sigma, S/N convention) |
| `static_pair/inputs/eta0` | `(40,)` | mas | the windows' zero points, all 0 |
| `static_pair/inputs/grid` | `(18,)` | mas | scanlib.instrument.model.TOY.sample_grid_mas |
| `static_pair/inputs/lsf_fwhm` | `()` | mas | scanlib.instrument.model.PSF_FWHM_MAS (the Gaussian LSF of TOY) |
| `static_pair/inputs/lsf_sigma` | `()` | mas | scanlib.instrument.model.PSF_SIGMA_MAS = FWHM / (2 sqrt(2 ln 2)) |
| `static_pair/truth/positions` | `(2, 2)` | mas | Scene.positions (dra*, ddec), primary first |
| `static_pair/truth/fluxes` | `(2,)` | flux per sample | Scene.fluxes |
| `static_pair/truth/eta` | `(40, 2)` | mas | simulate_campaign (along-scan position of every component in every window) |
| `static_pair/truth/theta` | `(2,)` | mas; mas/yr; mas | the companions' parameters in the fit's flat order |
| `static_pair/detections/psi` | `(27,)` | rad | the window's scan angle, one row per resolved companion per window |
| `static_pair/detections/eta` | `(27,)` | mas | truth/eta of the companion plus a Gaussian draw of detections/sigma |
| `static_pair/detections/sigma` | `(27,)` | mas | scanlib.window.reduction.sigma_eta at the truth, companions |
| `static_pair/detections/resolved` | `(40, 1)` | bool | which (window, companion) delivered a detection (sigma finite and <= 30 mas) |
| `static_pair/backprojection/map` | `(151, 151)` | 1/mas (unit-area ridges summed) | scanlib.sky.pool.backprojection_map |
| `static_pair/backprojection/axis` | `(151,)` | mas | backprojection_map (rows = ddec, columns = dra*) |
| `static_pair/backprojection/extent` | `(4,)` | mas | backprojection_map (for imshow, origin='lower') |
| `static_pair/radon/weights` | `(40,)` | rad | scanlib.sky.skymap.direction_weights |
| `static_pair/radon/image` | `(151, 151)` | flux per mas^2 | scanlib.sky.skymap.filtered_backprojection |
| `static_pair/radon/axis` | `(151,)` | mas | filtered_backprojection (rows = ddec, columns = dra*) |
| `static_pair/faststack/alpha` | `()` | 1 | the trimming fraction of Harrison (2011) |
| `static_pair/faststack/image` | `(151, 151)` | flux per sample | scanlib.sky.skymap.faststack (NaN where no sample) |
| `static_pair/faststack/variance` | `(151, 151)` | (flux per sample)^2 | faststack, Eq. 4 |
| `static_pair/faststack/n_used` | `(151, 151)` | 1 | faststack, samples per pixel |
| `static_pair/faststack/axis` | `(151,)` | mas | faststack (rows = ddec, columns = dra*) |
| `static_pair/truth/theta_free` | `(4,)` | mas; mas/yr; mas | the parameters of the free fit's flat order, the primary's block first |
| `static_pair/fit/theta0` | `(2,)` | mas; mas/yr; mas | the seed: truth/theta with the positions shifted by (8, -6) mas, motions 0 |
| `static_pair/fit/theta` | `(2,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint() |
| `static_pair/fit/flux` | `(2,)` | flux per sample | scanlib.motion.joint.fit_joint() (primary first) |
| `static_pair/fit/cov` | `(4, 4)` | parameter units squared | scanlib.motion.joint.fit_joint() (positions then fluxes) |
| `static_pair/fit/sigma_theta` | `(2,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint() |
| `static_pair/fit/chi2` | `()` | 1 | scanlib.motion.joint.fit_joint() (profiled residual chi^2) |
| `static_pair/loglike/at_truth` | `()` | nats | scanlib.motion.likelihood.loglike_windows at truth/theta, truth/fluxes |
| `static_pair/loglike/at_fit` | `()` | nats | scanlib.motion.likelihood.loglike_windows at fit/theta, fit/flux |
| `static_pair/fit_free/theta0` | `(4,)` | mas; mas/yr; mas | the seed: the primary's block at 0, then fit/theta0 |
| `static_pair/fit_free/theta` | `(4,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint(free_primary=True) |
| `static_pair/fit_free/flux` | `(2,)` | flux per sample | scanlib.motion.joint.fit_joint(free_primary=True) (primary first) |
| `static_pair/fit_free/cov` | `(6, 6)` | parameter units squared | scanlib.motion.joint.fit_joint(free_primary=True) (positions then fluxes) |
| `static_pair/fit_free/sigma_theta` | `(4,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint(free_primary=True) |
| `static_pair/fit_free/chi2` | `()` | 1 | scanlib.motion.joint.fit_joint(free_primary=True) (profiled residual chi^2) |
| `static_pair/loglike_free/at_truth` | `()` | nats | scanlib.motion.likelihood.loglike_windows(free_primary=True) at truth/theta_free, truth/fluxes |
| `static_pair/loglike_free/at_fit` | `()` | nats | scanlib.motion.likelihood.loglike_windows(free_primary=True) at fit_free/theta, fit_free/flux |
| `moving_pair/inputs/psi` | `(40,)` | rad | scanlib.io.gost.load_scan_law(gost_pleiades_centre_dr4span.csv), first 40 |
| `moving_pair/inputs/tau` | `(40,)` | yr | load_scan_law, years from the law's mid-epoch |
| `moving_pair/inputs/fpar` | `(40,)` | 1 | scanlib.io.gost.load_parallax_factor |
| `moving_pair/inputs/windows` | `(40, 18)` | flux per sample | scanlib.sky.campaign.simulate_campaign |
| `moving_pair/inputs/sigma` | `()` | flux per sample | simulate_campaign (noise sigma, S/N convention) |
| `moving_pair/inputs/eta0` | `(40,)` | mas | the windows' zero points, all 0 |
| `moving_pair/inputs/grid` | `(18,)` | mas | scanlib.instrument.model.TOY.sample_grid_mas |
| `moving_pair/inputs/lsf_fwhm` | `()` | mas | scanlib.instrument.model.PSF_FWHM_MAS (the Gaussian LSF of TOY) |
| `moving_pair/inputs/lsf_sigma` | `()` | mas | scanlib.instrument.model.PSF_SIGMA_MAS = FWHM / (2 sqrt(2 ln 2)) |
| `moving_pair/truth/positions` | `(2, 2)` | mas | Scene.positions (dra*, ddec), primary first |
| `moving_pair/truth/fluxes` | `(2,)` | flux per sample | Scene.fluxes |
| `moving_pair/truth/pm` | `(2, 2)` | mas/yr | Scene.pm (mu_a, mu_d) relative to the primary |
| `moving_pair/truth/dpi` | `(2,)` | mas | Scene.dpi, parallax difference |
| `moving_pair/truth/eta` | `(40, 2)` | mas | simulate_campaign (along-scan position of every component in every window) |
| `moving_pair/truth/theta` | `(5,)` | mas; mas/yr; mas | the companions' parameters in the fit's flat order |
| `moving_pair/detections/psi` | `(31,)` | rad | the window's scan angle, one row per resolved companion per window |
| `moving_pair/detections/eta` | `(31,)` | mas | truth/eta of the companion plus a Gaussian draw of detections/sigma |
| `moving_pair/detections/sigma` | `(31,)` | mas | scanlib.window.reduction.sigma_eta at the truth, companions |
| `moving_pair/detections/resolved` | `(40, 1)` | bool | which (window, companion) delivered a detection (sigma finite and <= 30 mas) |
| `moving_pair/backprojection/map` | `(151, 151)` | 1/mas (unit-area ridges summed) | scanlib.sky.pool.backprojection_map |
| `moving_pair/backprojection/axis` | `(151,)` | mas | backprojection_map (rows = ddec, columns = dra*) |
| `moving_pair/backprojection/extent` | `(4,)` | mas | backprojection_map (for imshow, origin='lower') |
| `moving_pair/radon/weights` | `(40,)` | rad | scanlib.sky.skymap.direction_weights |
| `moving_pair/radon/image` | `(151, 151)` | flux per mas^2 | scanlib.sky.skymap.filtered_backprojection |
| `moving_pair/radon/axis` | `(151,)` | mas | filtered_backprojection (rows = ddec, columns = dra*) |
| `moving_pair/faststack/alpha` | `()` | 1 | the trimming fraction of Harrison (2011) |
| `moving_pair/faststack/image` | `(151, 151)` | flux per sample | scanlib.sky.skymap.faststack (NaN where no sample) |
| `moving_pair/faststack/variance` | `(151, 151)` | (flux per sample)^2 | faststack, Eq. 4 |
| `moving_pair/faststack/n_used` | `(151, 151)` | 1 | faststack, samples per pixel |
| `moving_pair/faststack/axis` | `(151,)` | mas | faststack (rows = ddec, columns = dra*) |
| `moving_pair/truth/theta_free` | `(10,)` | mas; mas/yr; mas | the parameters of the free fit's flat order, the primary's block first |
| `moving_pair/fit/theta0` | `(5,)` | mas; mas/yr; mas | the seed: truth/theta with the positions shifted by (8, -6) mas, motions 0 |
| `moving_pair/fit/theta` | `(5,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint_dynamic(parallax=True) |
| `moving_pair/fit/flux` | `(2,)` | flux per sample | scanlib.motion.joint.fit_joint_dynamic(parallax=True) (primary first) |
| `moving_pair/fit/cov` | `(7, 7)` | parameter units squared | scanlib.motion.joint.fit_joint_dynamic(parallax=True) (positions then fluxes) |
| `moving_pair/fit/sigma_theta` | `(5,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint_dynamic(parallax=True) |
| `moving_pair/fit/chi2` | `()` | 1 | scanlib.motion.joint.fit_joint_dynamic(parallax=True) (profiled residual chi^2) |
| `moving_pair/loglike/at_truth` | `()` | nats | scanlib.motion.likelihood.loglike_windows_dynamic at truth/theta, truth/fluxes |
| `moving_pair/loglike/at_fit` | `()` | nats | scanlib.motion.likelihood.loglike_windows_dynamic at fit/theta, fit/flux |
| `moving_pair/fit_free/theta0` | `(10,)` | mas; mas/yr; mas | the seed: the primary's block at 0, then fit/theta0 |
| `moving_pair/fit_free/theta` | `(10,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint_dynamic(parallax=True, free_primary=True) |
| `moving_pair/fit_free/flux` | `(2,)` | flux per sample | scanlib.motion.joint.fit_joint_dynamic(parallax=True, free_primary=True) (primary first) |
| `moving_pair/fit_free/cov` | `(12, 12)` | parameter units squared | scanlib.motion.joint.fit_joint_dynamic(parallax=True, free_primary=True) (positions then fluxes) |
| `moving_pair/fit_free/sigma_theta` | `(10,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint_dynamic(parallax=True, free_primary=True) |
| `moving_pair/fit_free/chi2` | `()` | 1 | scanlib.motion.joint.fit_joint_dynamic(parallax=True, free_primary=True) (profiled residual chi^2) |
| `moving_pair/loglike_free/at_truth` | `()` | nats | scanlib.motion.likelihood.loglike_windows_dynamic(free_primary=True) at truth/theta_free, truth/fluxes |
| `moving_pair/loglike_free/at_fit` | `()` | nats | scanlib.motion.likelihood.loglike_windows_dynamic(free_primary=True) at fit_free/theta, fit_free/flux |
| `crowded_field/inputs/psi` | `(40,)` | rad | scanlib.io.gost.load_scan_law(gost_pleiades_centre_dr4span.csv), first 40 |
| `crowded_field/inputs/windows` | `(40, 18)` | flux per sample | scanlib.sky.campaign.simulate_campaign |
| `crowded_field/inputs/sigma` | `()` | flux per sample | simulate_campaign (noise sigma, S/N convention) |
| `crowded_field/inputs/eta0` | `(40,)` | mas | the windows' zero points, all 0 |
| `crowded_field/inputs/grid` | `(18,)` | mas | scanlib.instrument.model.TOY.sample_grid_mas |
| `crowded_field/inputs/lsf_fwhm` | `()` | mas | scanlib.instrument.model.PSF_FWHM_MAS (the Gaussian LSF of TOY) |
| `crowded_field/inputs/lsf_sigma` | `()` | mas | scanlib.instrument.model.PSF_SIGMA_MAS = FWHM / (2 sqrt(2 ln 2)) |
| `crowded_field/truth/positions` | `(4, 2)` | mas | Scene.positions (dra*, ddec), primary first |
| `crowded_field/truth/fluxes` | `(4,)` | flux per sample | Scene.fluxes |
| `crowded_field/truth/eta` | `(40, 4)` | mas | simulate_campaign (along-scan position of every component in every window) |
| `crowded_field/truth/theta` | `(6,)` | mas; mas/yr; mas | the companions' parameters in the fit's flat order |
| `crowded_field/detections/psi` | `(48,)` | rad | the window's scan angle, one row per resolved companion per window |
| `crowded_field/detections/eta` | `(48,)` | mas | truth/eta of the companion plus a Gaussian draw of detections/sigma |
| `crowded_field/detections/sigma` | `(48,)` | mas | scanlib.window.reduction.sigma_eta at the truth, companions |
| `crowded_field/detections/resolved` | `(40, 3)` | bool | which (window, companion) delivered a detection (sigma finite and <= 30 mas) |
| `crowded_field/backprojection/map` | `(151, 151)` | 1/mas (unit-area ridges summed) | scanlib.sky.pool.backprojection_map |
| `crowded_field/backprojection/axis` | `(151,)` | mas | backprojection_map (rows = ddec, columns = dra*) |
| `crowded_field/backprojection/extent` | `(4,)` | mas | backprojection_map (for imshow, origin='lower') |
| `crowded_field/radon/weights` | `(40,)` | rad | scanlib.sky.skymap.direction_weights |
| `crowded_field/radon/image` | `(151, 151)` | flux per mas^2 | scanlib.sky.skymap.filtered_backprojection |
| `crowded_field/radon/axis` | `(151,)` | mas | filtered_backprojection (rows = ddec, columns = dra*) |
| `crowded_field/faststack/alpha` | `()` | 1 | the trimming fraction of Harrison (2011) |
| `crowded_field/faststack/image` | `(151, 151)` | flux per sample | scanlib.sky.skymap.faststack (NaN where no sample) |
| `crowded_field/faststack/variance` | `(151, 151)` | (flux per sample)^2 | faststack, Eq. 4 |
| `crowded_field/faststack/n_used` | `(151, 151)` | 1 | faststack, samples per pixel |
| `crowded_field/faststack/axis` | `(151,)` | mas | faststack (rows = ddec, columns = dra*) |
| `crowded_field/truth/theta_free` | `(8,)` | mas; mas/yr; mas | the parameters of the free fit's flat order, the primary's block first |
| `crowded_field/fit/theta0` | `(6,)` | mas; mas/yr; mas | the seed: truth/theta with the positions shifted by (8, -6) mas, motions 0 |
| `crowded_field/fit/theta` | `(6,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint() |
| `crowded_field/fit/flux` | `(4,)` | flux per sample | scanlib.motion.joint.fit_joint() (primary first) |
| `crowded_field/fit/cov` | `(10, 10)` | parameter units squared | scanlib.motion.joint.fit_joint() (positions then fluxes) |
| `crowded_field/fit/sigma_theta` | `(6,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint() |
| `crowded_field/fit/chi2` | `()` | 1 | scanlib.motion.joint.fit_joint() (profiled residual chi^2) |
| `crowded_field/loglike/at_truth` | `()` | nats | scanlib.motion.likelihood.loglike_windows at truth/theta, truth/fluxes |
| `crowded_field/loglike/at_fit` | `()` | nats | scanlib.motion.likelihood.loglike_windows at fit/theta, fit/flux |
| `crowded_field/fit_free/theta0` | `(8,)` | mas; mas/yr; mas | the seed: the primary's block at 0, then fit/theta0 |
| `crowded_field/fit_free/theta` | `(8,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint(free_primary=True) |
| `crowded_field/fit_free/flux` | `(4,)` | flux per sample | scanlib.motion.joint.fit_joint(free_primary=True) (primary first) |
| `crowded_field/fit_free/cov` | `(12, 12)` | parameter units squared | scanlib.motion.joint.fit_joint(free_primary=True) (positions then fluxes) |
| `crowded_field/fit_free/sigma_theta` | `(8,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint(free_primary=True) |
| `crowded_field/fit_free/chi2` | `()` | 1 | scanlib.motion.joint.fit_joint(free_primary=True) (profiled residual chi^2) |
| `crowded_field/loglike_free/at_truth` | `()` | nats | scanlib.motion.likelihood.loglike_windows(free_primary=True) at truth/theta_free, truth/fluxes |
| `crowded_field/loglike_free/at_fit` | `()` | nats | scanlib.motion.likelihood.loglike_windows(free_primary=True) at fit_free/theta, fit_free/flux |
| `offset_primary/inputs/psi` | `(40,)` | rad | scanlib.io.gost.load_scan_law(gost_pleiades_centre_dr4span.csv), first 40 |
| `offset_primary/inputs/windows` | `(40, 18)` | flux per sample | scanlib.sky.campaign.simulate_campaign |
| `offset_primary/inputs/sigma` | `()` | flux per sample | simulate_campaign (noise sigma, S/N convention) |
| `offset_primary/inputs/eta0` | `(40,)` | mas | the windows' zero points, all 0 |
| `offset_primary/inputs/grid` | `(18,)` | mas | scanlib.instrument.model.TOY.sample_grid_mas |
| `offset_primary/inputs/lsf_fwhm` | `()` | mas | scanlib.instrument.model.PSF_FWHM_MAS (the Gaussian LSF of TOY) |
| `offset_primary/inputs/lsf_sigma` | `()` | mas | scanlib.instrument.model.PSF_SIGMA_MAS = FWHM / (2 sqrt(2 ln 2)) |
| `offset_primary/truth/positions` | `(2, 2)` | mas | Scene.positions (dra*, ddec), primary first |
| `offset_primary/truth/fluxes` | `(2,)` | flux per sample | Scene.fluxes |
| `offset_primary/truth/eta` | `(40, 2)` | mas | simulate_campaign (along-scan position of every component in every window) |
| `offset_primary/truth/theta` | `(2,)` | mas; mas/yr; mas | the companions' parameters in the fit's flat order |
| `offset_primary/detections/psi` | `(27,)` | rad | the window's scan angle, one row per resolved companion per window |
| `offset_primary/detections/eta` | `(27,)` | mas | truth/eta of the companion plus a Gaussian draw of detections/sigma |
| `offset_primary/detections/sigma` | `(27,)` | mas | scanlib.window.reduction.sigma_eta at the truth, companions |
| `offset_primary/detections/resolved` | `(40, 1)` | bool | which (window, companion) delivered a detection (sigma finite and <= 30 mas) |
| `offset_primary/backprojection/map` | `(151, 151)` | 1/mas (unit-area ridges summed) | scanlib.sky.pool.backprojection_map |
| `offset_primary/backprojection/axis` | `(151,)` | mas | backprojection_map (rows = ddec, columns = dra*) |
| `offset_primary/backprojection/extent` | `(4,)` | mas | backprojection_map (for imshow, origin='lower') |
| `offset_primary/radon/weights` | `(40,)` | rad | scanlib.sky.skymap.direction_weights |
| `offset_primary/radon/image` | `(151, 151)` | flux per mas^2 | scanlib.sky.skymap.filtered_backprojection |
| `offset_primary/radon/axis` | `(151,)` | mas | filtered_backprojection (rows = ddec, columns = dra*) |
| `offset_primary/faststack/alpha` | `()` | 1 | the trimming fraction of Harrison (2011) |
| `offset_primary/faststack/image` | `(151, 151)` | flux per sample | scanlib.sky.skymap.faststack (NaN where no sample) |
| `offset_primary/faststack/variance` | `(151, 151)` | (flux per sample)^2 | faststack, Eq. 4 |
| `offset_primary/faststack/n_used` | `(151, 151)` | 1 | faststack, samples per pixel |
| `offset_primary/faststack/axis` | `(151,)` | mas | faststack (rows = ddec, columns = dra*) |
| `offset_primary/truth/theta_free` | `(4,)` | mas; mas/yr; mas | the parameters of the free fit's flat order, the primary's block first |
| `offset_primary/fit/theta0` | `(2,)` | mas; mas/yr; mas | the seed: truth/theta with the positions shifted by (8, -6) mas, motions 0 |
| `offset_primary/fit/theta` | `(2,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint() |
| `offset_primary/fit/flux` | `(2,)` | flux per sample | scanlib.motion.joint.fit_joint() (primary first) |
| `offset_primary/fit/cov` | `(4, 4)` | parameter units squared | scanlib.motion.joint.fit_joint() (positions then fluxes) |
| `offset_primary/fit/sigma_theta` | `(2,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint() |
| `offset_primary/fit/chi2` | `()` | 1 | scanlib.motion.joint.fit_joint() (profiled residual chi^2) |
| `offset_primary/loglike/at_truth` | `()` | nats | scanlib.motion.likelihood.loglike_windows at truth/theta, truth/fluxes |
| `offset_primary/loglike/at_fit` | `()` | nats | scanlib.motion.likelihood.loglike_windows at fit/theta, fit/flux |
| `offset_primary/fit_free/theta0` | `(4,)` | mas; mas/yr; mas | the seed: the primary's block at 0, then fit/theta0 |
| `offset_primary/fit_free/theta` | `(4,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint(free_primary=True) |
| `offset_primary/fit_free/flux` | `(2,)` | flux per sample | scanlib.motion.joint.fit_joint(free_primary=True) (primary first) |
| `offset_primary/fit_free/cov` | `(6, 6)` | parameter units squared | scanlib.motion.joint.fit_joint(free_primary=True) (positions then fluxes) |
| `offset_primary/fit_free/sigma_theta` | `(4,)` | mas; mas/yr; mas | scanlib.motion.joint.fit_joint(free_primary=True) |
| `offset_primary/fit_free/chi2` | `()` | 1 | scanlib.motion.joint.fit_joint(free_primary=True) (profiled residual chi^2) |
| `offset_primary/loglike_free/at_truth` | `()` | nats | scanlib.motion.likelihood.loglike_windows(free_primary=True) at truth/theta_free, truth/fluxes |
| `offset_primary/loglike_free/at_fit` | `()` | nats | scanlib.motion.likelihood.loglike_windows(free_primary=True) at fit_free/theta, fit_free/flux |
| `instrument/fwhm_mas` | `()` | mas | the Gaussian LSF's FWHM; also resolution_mas |
| `instrument/pitch_mas` | `()` | mas | the sample pitch |
| `instrument/n_samples` | `()` | 1 | samples per window |
| `instrument/margins` | `(2,)` | 1 (residual power / sigma_n^2) | scanlib.window.reduction.TOY_MARGINS[:2]: Delta_1, Delta_2 (App. A.2) |
| `instrument/sample_grid` | `(18,)` | mas | Instrument.gaussian(fwhm_mas, pitch_mas, n_samples, margins=TOY_MARGINS[:2]).sample_grid_mas |
| `instrument/atom_eta` | `(144,)` | mas | Instrument.gaussian(fwhm_mas, pitch_mas, n_samples, margins=TOY_MARGINS[:2]).atom_eta_mas (the trial positions) |
| `instrument/atoms` | `(18, 144)` | 1 (flux per sample per unit flux) | Instrument.gaussian(fwhm_mas, pitch_mas, n_samples, margins=TOY_MARGINS[:2]).atoms: A_nk = lsf(x_n - eta_k) |
| `instrument/eta_bounds` | `(2,)` | mas | Instrument.gaussian(fwhm_mas, pitch_mas, n_samples, margins=TOY_MARGINS[:2]).eta_bounds (window +- one FWHM) |
| `instrument/merge_gap` | `()` | mas | Instrument.gaussian(fwhm_mas, pitch_mas, n_samples, margins=TOY_MARGINS[:2]).merge_gap_mas (0.3 FWHM) |
| `scanlaw/pleiades/psi` | `(47,)` | rad | scanlib.io.gost.load_scan_law(gost_pleiades_centre_dr4span.csv) |
| `scanlaw/pleiades/tau` | `(47,)` | yr | load_scan_law, years from the law's mid-epoch |
| `scanlaw/pleiades/fpar` | `(47,)` | 1 | scanlib.io.gost.load_parallax_factor(gost_pleiades_centre_dr4span.csv) |
| `scanlaw/beta60/psi` | `(95,)` | rad | scanlib.io.gost.load_scan_law(gost_ecl_lon060_lat60_dr4span.csv) |
| `scanlaw/beta60/tau` | `(95,)` | yr | load_scan_law, years from the law's mid-epoch |
| `scanlaw/beta60/fpar` | `(95,)` | 1 | scanlib.io.gost.load_parallax_factor(gost_ecl_lon060_lat60_dr4span.csv) |
| `render/eta` | `(2,)` | mas | the rendered sources' along-scan positions |
| `render/flux` | `(2,)` | flux per sample | the rendered sources' fluxes |
| `render/samples` | `(18,)` | flux per sample | scanlib.instrument.render.render_samples(eta, flux, inst) |
| `render/eta_fine` | `(241,)` | mas | the fine grid of render_curve |
| `render/curve` | `(241,)` | flux per sample | scanlib.instrument.model.render_curve(eta_fine, eta, flux, inst) |
| `render/noisy_samples` | `(18,)` | flux per sample | scanlib.instrument.render.noisy_window(eta, flux, default_rng(20261109), inst) |
| `render/noisy_sigma` | `()` | flux per sample | noisy_window (the noise sigma, S/N 100) |
| `render/zero_points` | `(40,)` | mas | scanlib.instrument.render.random_zero_points(default_rng(20261110), 40, inst) |
| `render/snr_from_peak_snr_50` | `()` | 1 | scanlib.instrument.render.snr_from_peak_snr(50, inst) |
| `noise/gain` | `()` | e- per flux unit | Fig. 4: G = 16 primary, 9 CCDs x 4.42 s, 19 000 e-/s at G = 15 |
| `noise/read_noise` | `()` | flux per sample | Fig. 4: sqrt(9) x 4.3 e- / gain |
| `noise/background` | `()` | flux per sample | Fig. 4: 9 x 10 e- / gain |
| `noise/counts` | `(18,)` | flux per sample | render/samples |
| `noise/sigma` | `(18,)` | flux per sample | scanlib.instrument.noise.NoiseModel(gain, read_noise, background).sigma(counts) |
| `static_pair/reduction/k` | `(40,)` | 1 | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): the number of sources selected |
| `static_pair/reduction/eta` | `(40, 3)` | mas | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): positions, sorted; NaN beyond k |
| `static_pair/reduction/flux` | `(40, 3)` | flux per sample | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): fluxes; NaN beyond k |
| `static_pair/reduction/chi2` | `(40, 3)` | 1 (whitened) | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): residual power of the K = 1, 2, 3 fits made; NaN where not made |
| `static_pair/reduction/delta_chi2` | `(40, 2)` | 1 | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): D_K of the rungs evaluated; NaN where not evaluated |
| `static_pair/reduction/sigma_eta` | `(40, 3)` | mas | scanlib.window.reduction.sigma_eta(eta, flux, sigma, inst) |
| `static_pair/reduction/delivered` | `(40, 3)` | bool | scanlib.window.reduction.delivery_mask(eta) (origin 0) |
| `moving_pair/reduction/k` | `(40,)` | 1 | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): the number of sources selected |
| `moving_pair/reduction/eta` | `(40, 3)` | mas | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): positions, sorted; NaN beyond k |
| `moving_pair/reduction/flux` | `(40, 3)` | flux per sample | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): fluxes; NaN beyond k |
| `moving_pair/reduction/chi2` | `(40, 3)` | 1 (whitened) | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): residual power of the K = 1, 2, 3 fits made; NaN where not made |
| `moving_pair/reduction/delta_chi2` | `(40, 2)` | 1 | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): D_K of the rungs evaluated; NaN where not evaluated |
| `moving_pair/reduction/sigma_eta` | `(40, 3)` | mas | scanlib.window.reduction.sigma_eta(eta, flux, sigma, inst) |
| `moving_pair/reduction/delivered` | `(40, 3)` | bool | scanlib.window.reduction.delivery_mask(eta) (origin 0) |
| `crowded_field/reduction/k` | `(40,)` | 1 | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): the number of sources selected |
| `crowded_field/reduction/eta` | `(40, 3)` | mas | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): positions, sorted; NaN beyond k |
| `crowded_field/reduction/flux` | `(40, 3)` | flux per sample | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): fluxes; NaN beyond k |
| `crowded_field/reduction/chi2` | `(40, 3)` | 1 (whitened) | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): residual power of the K = 1, 2, 3 fits made; NaN where not made |
| `crowded_field/reduction/delta_chi2` | `(40, 2)` | 1 | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): D_K of the rungs evaluated; NaN where not evaluated |
| `crowded_field/reduction/sigma_eta` | `(40, 3)` | mas | scanlib.window.reduction.sigma_eta(eta, flux, sigma, inst) |
| `crowded_field/reduction/delivered` | `(40, 3)` | bool | scanlib.window.reduction.delivery_mask(eta) (origin 0) |
| `offset_primary/reduction/k` | `(40,)` | 1 | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): the number of sources selected |
| `offset_primary/reduction/eta` | `(40, 3)` | mas | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): positions, sorted; NaN beyond k |
| `offset_primary/reduction/flux` | `(40, 3)` | flux per sample | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): fluxes; NaN beyond k |
| `offset_primary/reduction/chi2` | `(40, 3)` | 1 (whitened) | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): residual power of the K = 1, 2, 3 fits made; NaN where not made |
| `offset_primary/reduction/delta_chi2` | `(40, 2)` | 1 | scanlib.window.reduction.ladder_window(inputs/windows[j], None, inputs/sigma, inst, k_max=3): D_K of the rungs evaluated; NaN where not evaluated |
| `offset_primary/reduction/sigma_eta` | `(40, 3)` | mas | scanlib.window.reduction.sigma_eta(eta, flux, sigma, inst) |
| `offset_primary/reduction/delivered` | `(40, 3)` | bool | scanlib.window.reduction.delivery_mask(eta) (origin 0) |
| `margins/null/seed` | `()` |  | TOY_MARGINS_SEED; rung r draws from default_rng(seed).spawn(3)[r - 1] |
| `margins/null/snr` | `()` | 1 | snr_from_peak_snr(50, inst): the regime the margins were frozen in |
| `margins/null/rung1` | `(200,)` | 1 | scanlib.window.reduction.rung_null_statistic(stream, 1, 200, inst, snr): D_1 of the first 200 null windows |
| `margins/null/rung2` | `(200,)` | 1 | scanlib.window.reduction.rung_null_statistic(stream, 2, 200, inst, snr): D_2 of the first 200 null windows |
| `static_pair/campaign/seed` | `()` |  | the campaign's generator seed |
| `static_pair/campaign/psi` | `(28,)` | rad | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261110), k_max=2, snr=100, inst).psi |
| `static_pair/campaign/eta` | `(28,)` | mas | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261110), k_max=2, snr=100, inst).eta |
| `static_pair/campaign/sigma` | `(28,)` | mas | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261110), k_max=2, snr=100, inst).sigma |
| `static_pair/campaign/flux` | `(28,)` | flux per sample | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261110), k_max=2, snr=100, inst).flux |
| `static_pair/campaign/margins` | `(1,)` | 1 | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261110), k_max=2, snr=100, inst).margins |
| `static_pair/campaign/counts` | `(5,)` | 1 | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261110), k_max=2, snr=100, inst): n_windows, n_windows_with_detection, n_censored_windows, n_single_source_windows, n_sigma_substituted |
| `moving_pair/campaign/seed` | `()` |  | the campaign's generator seed |
| `moving_pair/campaign/psi` | `(32,)` | rad | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261111), k_max=2, snr=100, inst).psi |
| `moving_pair/campaign/t` | `(32,)` | yr | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261111), k_max=2, snr=100, inst).t |
| `moving_pair/campaign/fpar` | `(32,)` | 1 | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261111), k_max=2, snr=100, inst).fpar |
| `moving_pair/campaign/eta` | `(32,)` | mas | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261111), k_max=2, snr=100, inst).eta |
| `moving_pair/campaign/sigma` | `(32,)` | mas | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261111), k_max=2, snr=100, inst).sigma |
| `moving_pair/campaign/flux` | `(32,)` | flux per sample | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261111), k_max=2, snr=100, inst).flux |
| `moving_pair/campaign/margins` | `(1,)` | 1 | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261111), k_max=2, snr=100, inst).margins |
| `moving_pair/campaign/counts` | `(5,)` | 1 | scanlib.sky.campaign.harvest_campaign(scene, law, default_rng(20261111), k_max=2, snr=100, inst): n_windows, n_windows_with_detection, n_censored_windows, n_single_source_windows, n_sigma_substituted |
| `moving_pair/dynamics/fit_motion_theta` | `(5,)` | mas; mas/yr; mas | scanlib.motion.dynamics.fit_motion(campaign/psi, t, eta, sigma, fpar, parallax=True): (dra*, ddec, mu_a, mu_d, dpi) |
| `moving_pair/dynamics/fit_motion_cov` | `(5, 5)` | parameter units squared | fit_motion, the formal covariance |
| `moving_pair/dynamics/clean_theta` | `(5,)` | mas; mas/yr; mas | scanlib.motion.dynamics.clean_detection_set(campaign/psi, t, eta, sigma, fpar, parallax=True): the fit on the kept detections |
| `moving_pair/dynamics/clean_cov` | `(5, 5)` | parameter units squared | scanlib.motion.dynamics.clean_detection_set(campaign/psi, t, eta, sigma, fpar, parallax=True) |
| `moving_pair/dynamics/clean_keep` | `(32,)` | bool | scanlib.motion.dynamics.clean_detection_set(campaign/psi, t, eta, sigma, fpar, parallax=True): the kept detections |
| `moving_pair/dynamics/clean_iterations` | `()` | 1 | scanlib.motion.dynamics.clean_detection_set(campaign/psi, t, eta, sigma, fpar, parallax=True): the passes run |
| `static_pair/search/seed` | `()` |  | the campaign's generator seed |
| `static_pair/search/det_psi` | `(28,)` | rad | harvest_campaign(scene, psi, default_rng(20261110), k_max=3, snr=100, inst, return_windows=True).psi |
| `static_pair/search/det_eta` | `(28,)` | mas | the same harvest, .eta |
| `static_pair/search/det_sigma` | `(28,)` | mas | the same harvest, .sigma |
| `static_pair/search/windows` | `(40, 18)` | flux per sample | the same harvest, .windows |
| `static_pair/search/sigma_noise` | `()` | flux per sample | the same harvest, .sigma_noise |
| `static_pair/search/positions` | `(1, 2)` | mas | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).positions (primary first) |
| `static_pair/search/fluxes` | `(1,)` | flux per sample | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).fluxes |
| `static_pair/search/Lambda` | `(1,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).log: Lambda of every scored rung, in order |
| `static_pair/search/bar` | `(1,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).log: the bar Lambda_c of each |
| `static_pair/search/accepted` | `(1,)` | bool | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).log: whether each scored rung accepted its candidate |
| `static_pair/search/positions_free` | `(1, 2)` | mas | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).positions (primary first) |
| `static_pair/search/fluxes_free` | `(1,)` | flux per sample | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).fluxes |
| `static_pair/search/Lambda_free` | `(1,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).log: Lambda of every scored rung, in order |
| `static_pair/search/bar_free` | `(1,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).log: the bar Lambda_c of each |
| `static_pair/search/accepted_free` | `(1,)` | bool | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).log: whether each scored rung accepted its candidate |
| `crowded_field/search/seed` | `()` |  | the campaign's generator seed |
| `crowded_field/search/det_psi` | `(80,)` | rad | harvest_campaign(scene, psi, default_rng(20261112), k_max=3, snr=100, inst, return_windows=True).psi |
| `crowded_field/search/det_eta` | `(80,)` | mas | the same harvest, .eta |
| `crowded_field/search/det_sigma` | `(80,)` | mas | the same harvest, .sigma |
| `crowded_field/search/windows` | `(40, 18)` | flux per sample | the same harvest, .windows |
| `crowded_field/search/sigma_noise` | `()` | flux per sample | the same harvest, .sigma_noise |
| `crowded_field/search/positions` | `(5, 2)` | mas | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).positions (primary first) |
| `crowded_field/search/fluxes` | `(5,)` | flux per sample | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).fluxes |
| `crowded_field/search/Lambda` | `(6,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).log: Lambda of every scored rung, in order |
| `crowded_field/search/bar` | `(6,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).log: the bar Lambda_c of each |
| `crowded_field/search/accepted` | `(6,)` | bool | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).log: whether each scored rung accepted its candidate |
| `crowded_field/search/positions_free` | `(3, 2)` | mas | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).positions (primary first) |
| `crowded_field/search/fluxes_free` | `(3,)` | flux per sample | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).fluxes |
| `crowded_field/search/Lambda_free` | `(4,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).log: Lambda of every scored rung, in order |
| `crowded_field/search/bar_free` | `(4,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).log: the bar Lambda_c of each |
| `crowded_field/search/accepted_free` | `(4,)` | bool | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).log: whether each scored rung accepted its candidate |
| `offset_primary/search/seed` | `()` |  | the campaign's generator seed |
| `offset_primary/search/det_psi` | `(26,)` | rad | harvest_campaign(scene, psi, default_rng(20261113), k_max=3, snr=100, inst, return_windows=True).psi |
| `offset_primary/search/det_eta` | `(26,)` | mas | the same harvest, .eta |
| `offset_primary/search/det_sigma` | `(26,)` | mas | the same harvest, .sigma |
| `offset_primary/search/windows` | `(40, 18)` | flux per sample | the same harvest, .windows |
| `offset_primary/search/sigma_noise` | `()` | flux per sample | the same harvest, .sigma_noise |
| `offset_primary/search/positions` | `(1, 2)` | mas | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).positions (primary first) |
| `offset_primary/search/fluxes` | `(1,)` | flux per sample | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).fluxes |
| `offset_primary/search/Lambda` | `(1,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).log: Lambda of every scored rung, in order |
| `offset_primary/search/bar` | `(1,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).log: the bar Lambda_c of each |
| `offset_primary/search/accepted` | `(1,)` | bool | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3).log: whether each scored rung accepted its candidate |
| `offset_primary/search/positions_free` | `(1, 2)` | mas | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).positions (primary first) |
| `offset_primary/search/fluxes_free` | `(1,)` | flux per sample | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).fluxes |
| `offset_primary/search/Lambda_free` | `(1,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).log: Lambda of every scored rung, in order |
| `offset_primary/search/bar_free` | `(1,)` | 1 | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).log: the bar Lambda_c of each |
| `offset_primary/search/accepted_free` | `(1,)` | bool | scanlib.search.map_ladder(det_psi, det_eta, det_sigma, windows, psi, sigma_noise, bar='lrt', inst, k_max=3, free_primary=True).log: whether each scored rung accepted its candidate |
| `thresholds/pleiades/lrt_static_pinned` | `()` | 1 | scanlib.search.lrt_threshold(psi, None, moving=False, free_primary=False, inst) on gost_pleiades_centre_dr4span.csv |
| `thresholds/pleiades/lrt_static_free` | `()` | 1 | scanlib.search.lrt_threshold(psi, None, moving=False, free_primary=True, inst) on gost_pleiades_centre_dr4span.csv |
| `thresholds/pleiades/lrt_moving_pinned` | `()` | 1 | scanlib.search.lrt_threshold(psi, tau, moving=True, free_primary=False, inst) on gost_pleiades_centre_dr4span.csv |
| `thresholds/pleiades/lrt_moving_free` | `()` | 1 | scanlib.search.lrt_threshold(psi, tau, moving=True, free_primary=True, inst) on gost_pleiades_centre_dr4span.csv |
| `thresholds/beta60/lrt_static_pinned` | `()` | 1 | scanlib.search.lrt_threshold(psi, None, moving=False, free_primary=False, inst) on gost_ecl_lon060_lat60_dr4span.csv |
| `thresholds/beta60/lrt_static_free` | `()` | 1 | scanlib.search.lrt_threshold(psi, None, moving=False, free_primary=True, inst) on gost_ecl_lon060_lat60_dr4span.csv |
| `thresholds/beta60/lrt_moving_pinned` | `()` | 1 | scanlib.search.lrt_threshold(psi, tau, moving=True, free_primary=False, inst) on gost_ecl_lon060_lat60_dr4span.csv |
| `thresholds/beta60/lrt_moving_free` | `()` | 1 | scanlib.search.lrt_threshold(psi, tau, moving=True, free_primary=True, inst) on gost_ecl_lon060_lat60_dr4span.csv |
| `thresholds/null/seed` | `()` |  | the generator seed of each configuration |
| `thresholds/null/max_z_pinned` | `(3,)` | 1 | scanlib.search.field_null_max(psi, tau, 3, default_rng(20261111), free_primary=False, inst) on gost_pleiades_centre_dr4span.csv: the largest search statistic per campaign |
| `thresholds/null/max_z_free` | `(3,)` | 1 | scanlib.search.field_null_max(psi, tau, 3, default_rng(20261111), free_primary=True, inst) on gost_pleiades_centre_dr4span.csv: the largest search statistic per campaign |
| `meta/scanlib_dev_commit` | `()` |  | git rev-parse HEAD at export |
| `meta/scanlib_dev_version` | `()` |  | scanlib.__version__ |
| `meta/exported` | `()` |  | the export date |
| `meta/seed` | `()` |  | the noise seed (SEED + scene index per scene) |
