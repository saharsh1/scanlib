# The notebooks

Three folders, for three questions. All data are synthetic. Every scanlib call carries the name of its
module (`dyn.wls`, `joint.fit_joint`), so each line can be found in the package, and each block names the
place in the paper it comes from, or says that it is not in the paper.

| folder | the question it answers | read it when |
|---|---|---|
| [`quickstart/`](quickstart/) | How do I go from windows to a measurement with this package? | first |
| [`scene/`](scene/) | How does each image and the map work, and where does each step stop working? | you want to understand one step |
| [`paper/`](paper/) | Where does this figure or this number of the paper come from? | you check or reproduce the paper |

## Where to start

1. [`quickstart/1_sinogram_fit.ipynb`](quickstart/1_sinogram_fit.ipynb): one companion; the windows, the
   detections, the linear fit of a sinusoid to them.
2. [`quickstart/2_binary_backprojected_seed.ipynb`](quickstart/2_binary_backprojected_seed.ipynb): a moving
   companion; the triangulation map, the seed, the joint fit of all windows.
3. [`quickstart/3_triple_ladder.ipynb`](quickstart/3_triple_ladder.ipynb): two companions; the search, one
   source at a time, and the threshold.

The quickstart's [README](quickstart/README.md) lists the modules and what the calls need from your own
windows.

## One step at a time: `scene/`

| notebook | subject | paper |
|---|---|---|
| [`00_three_images_and_fit`](scene/00_three_images_and_fit.ipynb) | one scene, forty windows: the sinogram, the three images, the joint fit | Sects. 2.1, 3.3, 4.3; App. C |
| [`01_inverse_radon`](scene/01_inverse_radon.ipynb) | the inverse Radon transform, step by step | App. C.2 |
| [`02_faststack`](scene/02_faststack.ipynb) | faststack, the robust stack of the samples | App. C.2 |
| [`03_triangulation_map`](scene/03_triangulation_map.ipynb) | the triangulation map: a detection as a line, the map, its peak | Sect. 3.3; App. C.1 |

Where each step's premise breaks, [`scene/limitations/`](scene/limitations/):

| notebook | the premise that fails |
|---|---|
| [`01_finite_window`](scene/limitations/01_finite_window.ipynb) | the source is well inside the window |
| [`02_scan_law`](scene/limitations/02_scan_law.ipynb) | the scan directions cover the half-turn evenly |
| [`03_few_lines`](scene/limitations/03_few_lines.ipynb) | there are enough lines to disagree |
| [`04_motion`](scene/limitations/04_motion.ipynb) | the scene is the same in every window |
| [`05_blend`](scene/limitations/05_blend.ipynb) | the two profiles of a window are separated |
| [`06_model_errors`](scene/limitations/06_model_errors.ipynb) | the zero points, the LSF, the noise and the list of sources are known |

## The paper: `paper/`

One notebook per figure (Figures 1 to 4, C.1 and C.2), and in `paper/auxil/` the kernel and the simulations
behind the figures' numbers. Each prints and asserts the numbers the paper quotes. The folder's
[README](paper/README.md) has the table: notebook, place in the paper, what is asserted, run time, store.

## Running them

```bash
git clone https://github.com/saharsh1/scanlib.git
pip install -e "./scanlib[notebooks]"      # matplotlib and Jupyter are needed only for the notebooks
jupyter notebook scanlib/notebooks/quickstart/1_sinogram_fit.ipynb
```

The quickstart and `scene/` notebooks are committed with their outputs, so they can be read without running.
The `paper/` notebooks are committed clean and run in seconds to a minute each: their slow parts are stored in
`paper/stored/` and recomputed only with `FULL = True`. The symbols and units are in
[`docs/conventions.md`](../docs/conventions.md).
