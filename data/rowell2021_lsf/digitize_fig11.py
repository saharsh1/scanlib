"""Digitize Fig. 11 of Rowell et al. 2021 (A&A 649, A11): the calibrated AL LSF of AF2-9, window class 1, for five
effective wavenumbers (1.24, 1.36, 1.48, 1.60, 1.72 per micron), both telescopes, at revolution 3343.

The figure is a raster (1280 x 480 RGB) embedded in the published PDF (page 14, the first image there). Each curve is
found by its hue (red, orange, gold, blue, purple; saturation above 0.6, the light pink grid excluded), column by column
inside each panel's axes, the row taken as the saturation-weighted mean of the curve's pixels. The axes are calibrated
on the major ticks read off the image: AL -10 and +10 pixels at columns 92 and 613 (FOV1) and 732 and 1253 (FOV2), LSF
0.0 and 0.5 per pixel at rows 394 and 101. The caption's vertical offsets (0.02 per curve, the 1.24 curve unshifted) are
removed. One image pixel is 0.0017 in LSF (0.4 % of the peak) and 0.038 AL pixels (2.3 mas). The raw columns are written,
nothing smoothed or renormalized: ``notebooks/paper/auxil/kernel_rowell_lsf.ipynb`` builds the kernel from them.

    python data/rowell2021_lsf/digitize_fig11.py <the Rowell 2021 PDF>

Needs PyMuPDF (``pymupdf``) to read the embedded image; it is not a dependency of the package. Writes
``lsf_af29_wc1.csv`` beside this file.
"""
import colorsys
import hashlib
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "lsf_af29_wc1.csv"
PAGE, IMAGE_INDEX = 14, 0                               # the page of Fig. 11 and its first embedded image
NU = (1.24, 1.36, 1.48, 1.60, 1.72)                     # per micron, as in the figure's legend
HUE_DEG = {1.24: 0.0, 1.36: 30.0, 1.48: 48.0, 1.60: 240.0, 1.72: 300.0}
VALUE_RANGE = {1.24: (0.8, 1.01), 1.36: (0.8, 1.01), 1.48: (0.8, 1.01), 1.60: (0.6, 1.01), 1.72: (0.3, 0.7)}
PANELS = {"FOV1": (92, 613), "FOV2": (732, 1253)}       # image columns of AL = -10 and +10 pixels
ROW_LSF0, ROW_LSF05 = 394.0, 101.0                      # image rows of LSF = 0.0 and 0.5 per pixel
OFFSET = 0.02                                           # the caption's vertical offset between neighbouring curves
LEGEND = (slice(60, 290), 125)                          # rows of the legend box, and its width from the panel's left axis
AL_MAX = 9.0                                            # the plotted range [pixels]


def embedded_image(pdf: Path) -> tuple[np.ndarray, str]:
    import pymupdf
    doc = pymupdf.open(pdf)
    xref = doc[PAGE - 1].get_images()[IMAGE_INDEX][0]
    raw = doc.extract_image(xref)["image"]
    pix = pymupdf.Pixmap(raw)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3] / 255.0
    return rgb, hashlib.sha256(raw).hexdigest()


def digitize(rgb: np.ndarray) -> list[tuple[str, float, float, float]]:
    h, w, _ = rgb.shape
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in rgb.reshape(-1, 3)]).reshape(h, w, 3)
    hue, sat, val = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    rows = []
    for fov, (x0, x1) in PANELS.items():
        cols = np.arange(x0 + 1, x1)
        al = -10.0 + 20.0 * (cols - x0) / (x1 - x0)
        for k, nu in enumerate(NU):
            d = np.abs(hue - HUE_DEG[nu]) % 360.0
            m = (sat > 0.6) & (np.minimum(d, 360.0 - d) < 10.0) & (val >= VALUE_RANGE[nu][0]) & (val < VALUE_RANGE[nu][1])
            m[:85] = False
            m[LEGEND[0], x0:x0 + LEGEND[1]] = False
            for c, a in zip(cols, al):
                r = np.where(m[:, c])[0]
                if abs(a) > AL_MAX or r.size == 0:
                    continue
                y = np.average(r, weights=sat[r, c])
                rows.append((fov, nu, float(a), float(0.5 * (ROW_LSF0 - y) / (ROW_LSF0 - ROW_LSF05) - OFFSET * k)))
    return rows


def main():
    rgb, sha = embedded_image(Path(sys.argv[1]))
    rows = digitize(rgb)
    with OUT.open("w") as f:
        f.write(f"# Rowell et al. 2021, A&A 649, A11, Fig. 11 (AF2-9, window class 1, revolution 3343), digitized by\n"
                f"# data/rowell2021_lsf/digitize_fig11.py from the embedded image (sha256 {sha}).\n"
                f"# al_pix: along-scan offset [pixels of 58.9 mas]; lsf: per pixel, the caption's offsets removed.\n"
                "fov,nu_eff,al_pix,lsf\n")
        for fov, nu, a, l in rows:
            f.write(f"{fov},{nu:.2f},{a:.4f},{l:.5f}\n")
    print(f"{len(rows)} points from image {sha[:12]} -> {OUT}")


if __name__ == "__main__":
    main()
