# The Gaia along-scan LSF against colour (Rowell et al. 2021, Fig. 11, digitized)

**Provenance.** Rowell, N., et al. 2021, *Gaia Early Data Release 3: Modelling and calibration of Gaia's point and
line spread functions*, A&A 649, A11, https://doi.org/10.1051/0004-6361/202039448 (open access). Figure 11: the
calibrated along-scan LSF of the AF2-9 CCDs, window class 1, at revolution 3343, for five effective wavenumbers
ν_eff = 1.24, 1.36, 1.48, 1.60, 1.72 µm⁻¹, for each telescope (FOV1, FOV2).

**Digitizing.** The figure is a raster embedded in the published PDF (page 14). Each curve was read off it by hue,
column by column, by the script `digitize_fig11.py` of this folder (the file's header names it by its path in the
development repository, `docs/provenance/rowell2021_fig11_digitize_20261101.py`); the axes were calibrated on the major
ticks and the caption's vertical offsets (0.02 per curve) removed. One image pixel is 0.0017 per pixel in the LSF
(0.4 % of the peak) and 0.038 AL pixels (2.3 mas). The SHA-256 of the embedded image is in the file's header.

**File.** `lsf_af29_wc1.csv` — columns `fov`, `nu_eff` [µm⁻¹], `al_pix` (along-scan offset in pixels of 58.9 mas),
`lsf` (per pixel); 468 columns per curve over ±9 pixels, 4680 rows. Raw: not smoothed, not renormalized. The plotted
range holds 94–95 % of the flux; the rest is in the wings beyond.

**Use.** `notebooks/paper/auxil/kernel_rowell_lsf.ipynb` builds the kernel from it (FOV1, smoothed, Lorentzian wings
beyond ±9 pixels, linear in ν_eff) as a non-Gaussian `Instrument`. A Gaia product: it stays out of the package, which
takes the kernel as plain callables; the notebook reads the file from this folder of the repository.
