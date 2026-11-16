"""Byte-identity baseline of the package's own outputs
(``tests/test_baseline.py``). Compared exactly; a change here is a
deliberate numeric change, re-blessed in its own commit."""
from __future__ import annotations

BASELINE: dict[str, float] = {
    'backprojection_map': 1.2459101946313353,
    'design_columns': 5.999697435510329,
    'faststack': 3454689.2460922445,
    'filtered_backprojection': 80.84785645075435,
    'filtered_backprojection_squares': 3.0425632995219907,
    'fit_joint': 553.6756864326574,
    'fit_joint_dynamic': 601.9787702578278,
    'joint_fisher': 7.6598055965821,
    'loglike_profiled': -1607.73363497081,
    'loglike_profiled_fluxes': 1299.0605183439475,
    'loglike_windows': -1608.088734997958,
    'map_peak': 7612867.237052452,
}

ENVIRONMENT: dict[str, str] = {
    'numpy': '2.5.3',
    'platform': 'Darwin-arm64',
    'scipy': '1.18.1',
}
