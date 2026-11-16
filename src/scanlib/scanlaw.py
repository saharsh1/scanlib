"""Two public scan laws: the GOST forecasts shipped with the package.

A scan law is, for one sky position, the list of the instrument's transits
over it: the scan angle psi of each transit [rad, counter-clockwise from
north, the psi of ``eta = dra* sin(psi) + ddec cos(psi)``], its epoch, and
its along-scan parallax factor ``f_par`` (the along-scan shift in mas per mas
of parallax). It depends on the sky position alone, so a forecast queried at
one pointing is a scan law for that pointing and nothing else.

The two files are forecasts of ESA's public Gaia Observation Forecast Tool
(GOST, https://gaia.esac.esa.int/gost/), queried without credentials over
the DR4 span (2014-07-25 to 2020-01-28):

- :data:`PLEIADES`, at the Pleiades centre (RA 56.75 deg, Dec +24.1167 deg):
  47 transits; the largest gap between scan directions on the half-turn is
  90 deg, and the directions are anisotropic, ``|<exp(2 i psi)>| = 0.37``.
- :data:`BETA60`, at ecliptic longitude 60 deg, latitude +60 deg (RA
  11.9248 deg, Dec +75.1960 deg): 95 transits; the largest gap is 10 deg and
  ``|<exp(2 i psi)>| = 0.11``, so the triangulation error ellipse is nearly
  round.

Each file's README, beside it, records the query and the columns. GOST's
``scanAngle[rad]`` is the package's psi directly, and its
``parallaxFactorAlongScan`` has the package's sign; both agree with an
along-scan parallax factor computed from the package's convention to an rms
of 0.005 (correlation +1.000). The forecasts are predicted crossings, not
guaranteed observations: GOST gives about 80 % for the data of a transit to
reach the ground. Numbers are parsed exactly, as the nearest double to the
decimal in the file.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent / "data" / "scanlaw"

#: The GOST forecast at the Pleiades centre over the DR4 span, 47 transits.
PLEIADES = _DATA / "pleiades" / "gost_pleiades_centre_dr4span.csv"
#: The GOST forecast at ecliptic latitude +60 deg over the DR4 span, 95 transits.
BETA60 = _DATA / "beta60" / "gost_ecl_lon060_lat60_dr4span.csv"


def _columns(path, names):
    """The named columns of a GOST CSV as float arrays, in the file's row order.

    GOST pads its header names with spaces; they are matched stripped.
    """
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    header = [c.strip() for c in rows[0]]
    index = [header.index(n) for n in names]
    return [np.array([float(r[i]) for r in rows[1:]]) for i in index]


def load_scan_law(path):
    """The scan angles and the epochs of a GOST scan law.

    Parameters
    ----------
    path : str or Path
        A GOST CSV: :data:`PLEIADES`, :data:`BETA60` or another forecast with
        the columns ``scanAngle[rad]`` and
        ``ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]``.

    Returns
    -------
    psi : array, shape (J,)
        The scan angle of each transit, in rad, counter-clockwise from north.
    tau : array, shape (J,)
        The epoch of each transit, in years of 365.25 days from the mean of
        the transits' barycentric epochs (TCB), the law's reference epoch.
    """
    psi, t_bary = _columns(path, ["scanAngle[rad]",
                                  "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]"])
    tau = (t_bary - t_bary.mean()) / 365.25
    return psi, tau


def load_parallax_factor(path):
    """The along-scan parallax factor of each transit of a GOST scan law.

    Parameters
    ----------
    path : str or Path
        A GOST CSV with the column ``parallaxFactorAlongScan``, as for
        :func:`load_scan_law`.

    Returns
    -------
    array, shape (J,)
        ``f_par``, dimensionless: the along-scan shift in mas per mas of
        parallax, in the row order of :func:`load_scan_law`. A source at
        parallax difference ``dpi`` [mas] from the reference position moves
        along the scan by ``f_par * dpi``.
    """
    return _columns(path, ["parallaxFactorAlongScan"])[0]
