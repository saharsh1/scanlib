# From the detections to the measurement: the two routes

The per-window reduction gives the detections: one along-scan position per
resolved source and window, each a line on the sky. The joint fit of all
windows (`scanlib.joint.fit_joint`) is the measurement. Between the two the
package has two routes. Both start at the triangulation map and end at the
joint fit.

| | route 1: one companion | route 2: the search |
|---|---|---|
| for | a pair: the primary and one companion | a scene with an unknown number of sources |
| in the paper | Sect. 3–4; Figures 2, 3, C.1 | Sect. 5; Figures 4, 6 |
| the calls | `dynamics.clean_detection_set`, then `joint.fit_joint` | `search.map_ladder`, then `joint.fit_joint` |
| decides how many sources | no: one is assumed | yes: one at a time, each against Λ_c |

## Route 1: the map detects, a circle selects, the sinogram refines

`clean_detection_set(psi, tau, eta, sigma)`:

1. **Detect.** The map of the detections as measured; its highest peak
   outside the primary's zone (`origin_zone`, 30 mas), and the map's width
   `w` there.
2. **Select.** The detections whose line passes within `select_radius`
   (30 mas) of the peak. The circle is wide because the lines of a moving
   source do not cross at one point, and the peak is not its position at the
   reference epoch.
3. **Refine.** The sinogram fit, with the linear motion (and the parallax
   difference when asked), to the selected detections. This first fit is a
   Huber fit (`huber_delta`, `huber_iter`): a few lines of another source
   inside the circle pull it less than they pull least squares.
4. **Clean.** Every detection is tested against the fitted track at its own
   epoch, `|eta_i - X_i theta| <= max(k_sigma sigma_i, w)`. The motion is
   fitted again, by least squares, to the detections that pass. This repeats
   until the set that passes is the set last fitted.

The result (position, motion, covariance, the kept set) starts the joint
fit. It is not the measurement: detections from blended windows carry the
blend's bias, and the fit inherits it (Sect. 4.5 of the paper).

The residual of step 4 is the distance of the detection's line, moved to the
reference epoch with the fitted motion (Eq. 17, `to_reference_epoch`), from
the fitted position. So step 4 is the test "the line passes through the peak
of the map of the corrected detections", and that map is drawn (Figure 3)
but not computed. With `static=True` nothing moves the detections, and the
route is: the peak, the detections whose line passes through it, one fit.

## Route 2: propose, score, test, assign

`map_ladder(det_psi, det_eta, det_sigma, windows, psi, inst, sigma_noise,
tau=..., det_tau=...)`, per source:

1. **Propose.** The highest peak of the map of the detections not yet
   assigned to a source (the pool), crossed by at least three lines.
2. **Score.** The joint fit of all windows with the accepted sources and
   the candidate, the candidate started at the peak, at rest; Λ is twice
   the gain in log-likelihood over the fit without it.
3. **Test.** The candidate is accepted when Λ > Λ_c
   (`thresholds.lrt_threshold`); otherwise the search stops.
4. **Assign.** The pool is rebuilt from all the detections: a detection
   leaves it when its line passes through an accepted source, taken at the
   line's own epoch (position plus motion times `det_tau`), within
   `max(k_sigma sqrt(sigma_i^2 + s_i^2), w)`. This is Eq. 17 applied to the
   line. Without it, the lines a moving source left at the ends of the
   campaign stay in the pool and propose it again.

The search returns the sources of its last accepted joint fit. A final
joint fit can then measure the whole scene again; each source can start
there from its own sinogram fit (route 1 on the lines of that source's
rung). `notebooks/quickstart/3_triple_ladder.ipynb`, block 9, makes this
start for one source and compares it with the search's own fit.

## How the routes differ, and what may change

Route 2 has no sinogram fit between its proposal and its joint fit (steps 2
and 3 of route 1). We measured the effect of adding it (2026-11-13):

- For a well-measured source the sinogram fit is the better start. Figure
  4's B: the map's peak is 32 mas from the truth, the sinogram fit about
  1 mas. The joint fit ends at the same place from both.
- Near the detection limit a companion has few detections, blended and
  biased, and the sinogram start is not better. In a sample of 144 of
  Figure 6's injected campaigns the first accepted Λ was equal or lower
  from it wherever both starts led to a detection; it found the companion
  in 4 campaigns where the peak's start did not, and lost it in 3.

So route 2 keeps the peak as its start. A search that tries both starts
and keeps the fit with the higher likelihood would join the two routes: it
detects whatever either start detects (126 against 122 of the 144). It is
not in the package. It would move Figure 6's detection limit toward
fainter companions and needs its own check of the false-detection rate, so
it is a candidate for a later version; Figure 6 as published is the
conservative limit in this respect.

The same holds for the other choices of the search that the paper fixes
(the zone around the primary, the tolerances of step 4, the bound on the
motions): they are the fiducial implementation, and the exact detection
limits depend on them (Sect. 6.3).
