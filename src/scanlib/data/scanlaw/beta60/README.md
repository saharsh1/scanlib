# The scan law at ecliptic latitude +60° (GOST, public)

**Source.** ESA's Gaia Observation Forecast Tool (GOST,
https://gaia.esac.esa.int/gost/), a public query without credentials.

**Query.** One sky position, ecliptic longitude 60°, latitude +60°
(barycentric mean ecliptic; in ICRS RA 11.9248°, Dec +75.1960°), over the
DR4 span 2014-07-25T10:32:00 to 2020-01-28T00:00:00 UTC.

**File.** `gost_ecl_lon060_lat60_dr4span.csv`, 95 rows, one per predicted
focal-plane transit. The largest gap between scan directions on the
half-turn is 10°, against 90° at the Pleiades centre (`../pleiades/`), and
the directions are nearly isotropic, |⟨exp(2iψ)⟩| = 0.11 against 0.37.
Loaded as `scanlib.scanlaw.BETA60`. The columns, the convention and the
caveats are those of the Pleiades forecast (`../pleiades/README.md`).
