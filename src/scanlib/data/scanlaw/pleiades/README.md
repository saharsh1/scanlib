# The scan law at the Pleiades centre (GOST, public)

**Source.** ESA's Gaia Observation Forecast Tool (GOST,
https://gaia.esac.esa.int/gost/), version 22.4.3, a public query without
credentials, made on 2026-07-13.

**Query.** One sky position, the Pleiades centre, RA 56.75°, Dec +24.1167°
(03:47:00.0 +24:07:00.1), over the DR4 span 2014-07-25T10:32:00 to
2020-01-28T00:00:00 UTC; the single-position mode of the GOST servlet, the
result exported as CSV.

**File.** `gost_pleiades_centre_dr4span.csv`, 47 rows, one per predicted
focal-plane transit. The largest gap between scan directions on the
half-turn is 90°. Loaded as `scanlib.scanlaw.PLEIADES`. The columns (the
GOST public set; GOST Software User Manual, GAIA-CU1-UG-ESAC-JFH-005,
Table 3):

| column | meaning |
| --- | --- |
| `ra[rad]`, `dec[rad]` | the queried position |
| `ObservationTimeAtGaia[UTC]` | the transit time at the spacecraft (AF4) |
| `ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]` | the barycentric Julian date (TCB), the epoch `load_scan_law` reads |
| `CcdRow[1-7]` | the focal-plane CCD row |
| `zetaFieldAngle[rad]` | the across-scan field angle |
| `scanAngle[rad]` | the scan angle ψ (AF6), as defined in GAIA-LL-061 |
| `Fov` | the preceding or the following field of view |
| `parallaxFactorAlongScan`, `parallaxFactorAcrossScan` | the parallax factors |

**Convention.** GOST's `scanAngle[rad]` is the package's ψ, with
η = Δα* sin ψ + Δδ cos ψ, and `parallaxFactorAlongScan` has the package's
sign. The check: an along-scan parallax factor computed in the package's
convention agrees with GOST's column to an rms of 0.005, correlation +1.000.

**Caveats that GOST states.** These are forecast crossings, not guaranteed
observations: the data of a transit reach the ground with a probability of
about 80 %, so 47 predicted transits give about 38. Dead time, CCD gaps and
interruptions of operations are not modelled. Only 3 < G < 20 is observed
astrometrically.
