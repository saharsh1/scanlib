# Changelog

scanlib follows semantic versioning. While the version is 0.x, a change to
the public surface (a name added, moved or removed, a convention changed) is
a MINOR bump; fixes and additions behind the same surface are PATCH bumps.
A change that moves a computed number re-blesses the baseline in its own
commit and says what moved.

## 0.1.0 — 2026-11-16

The first public release.

The method of one-dimensional scanning astrometry, as a package of flat
modules with two runtime dependencies, numpy and scipy.

- **The atoms.** `instrument` (the along-scan projection, the `Instrument`
  with its line-spread function and its one operator, one window rendered),
  `noise` (the noise sigma of each sample; the Gaia-like setting of a
  magnitude), `backprojection` (the map of the detections and its peak),
  `radon` (the inverse Radon transform and faststack) and `joint` (the
  window-level log-likelihood with profiled fluxes, the Fisher matrix, the
  joint fit of a static or a moving scene, with the primary pinned or free).
- **The paper's pipeline.** `reduction` (the per-window reduction by the
  nested ladder, and a campaign reduced to its detection set), `dynamics`
  (the fits to the detections and their cleaning), `thresholds` (the
  search's threshold in closed form) and `search` (the map ladder).
- **Data and simulation.** `scanlaw` (two public scan-law forecasts, shipped
  with the package) and `simulation` (everything that knows the truth or
  draws random numbers; no method module imports it).
- **Notebooks.** `notebooks/quickstart/` (three, in order), `notebooks/scene/`
  (the three images step by step, and six limitation notebooks) and
  `notebooks/paper/` (Figures 1 to 4, C.1 and C.2 of the paper, the
  Gaia-like kernel, and the simulations behind the paper's numbers).
- **Tests.** The reference cases of the development code
  (`tests/reference/`), a byte-for-byte baseline of the package's own
  outputs, the conventions, the import closure, and the paper notebooks.
