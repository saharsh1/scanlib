"""The notebooks that reproduce the paper (``notebooks/paper/``: the figures, and in ``auxil/`` the kernel and the
simulations behind them).

They depend on the package alone. Covered:
  1. each notebook is a valid version-4 notebook, committed clean (no outputs, no execution
     counts), with ``SAVE_PDF = False`` and ``FULL = False`` where it has them;
  2. each imports only numpy, matplotlib, pathlib, dataclasses, a few scipy modules and scanlib, and
     uses only the scanlib names the folder's README lists under "The surface" (no ``sys.path``,
     no ``importlib``, no private name). A name is used either by ``from scanlib.x import name``
     or, with the module imported (``from scanlib import x as alias``, ``import scanlib.x as
     alias``), as ``alias.name``;
  3. every README surface name is used by some notebook, so the list is the contract;
  4. the stores are the ones the README names, and the kernel's data is beside the repository's
     other data;
  5. (slow) each notebook runs top to bottom at its default settings; its inline asserts are
     the paper's numbers.
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FOLDER = REPO / "notebooks" / "paper"
FIGURES = sorted(p.name for p in FOLDER.glob("*.ipynb"))
AUXILIARY = sorted(f"auxil/{p.name}" for p in (FOLDER / "auxil").glob("*.ipynb"))
NOTEBOOKS = FIGURES + AUXILIARY
OTHER_MODULES = {"numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.patheffects", "pathlib",
                 "dataclasses", "scipy.stats", "scipy.ndimage", "scipy.interpolate", "scipy.optimize"}
STORES = {"fig3_coverage.npz", "fig4_coverage.npz", "lambda_c_null.npz", "lambda_c_validation.npz",
          "margin_nulls.npz"}


def _surface():
    """``{module: names}`` from the README's "The surface" list."""
    surface = {}
    for line in (FOLDER / "README.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^- `(scanlib[\w.]*)`: (.+)$", line)
        if m:
            surface[m.group(1)] = set(re.findall(r"`([^`]+)`", m.group(2)))
    return surface


def _notebook(name):
    return json.loads((FOLDER / name).read_text(encoding="utf-8"))


def _code(name):
    return "\n".join("".join(c["source"]) for c in _notebook(name)["cells"] if c["cell_type"] == "code")


def _uses(code):
    """``(modules, names)`` of a notebook's code: every module it imports that is not a scanlib
    module, and every ``(scanlib module, name)`` it uses. A scanlib name is used by
    ``from scanlib.x import name`` or, when the module itself is imported (``from scanlib import
    x [as alias]``, ``import scanlib.x as alias``), as the attribute ``alias.name``."""
    tree = ast.parse(code)
    modules, names, aliases = set(), set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "scanlib":
                for a in node.names:
                    aliases[a.asname or a.name] = f"scanlib.{a.name}"
            elif node.module.startswith("scanlib."):
                names.update((node.module, a.name) for a in node.names)
            else:
                modules.add(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("scanlib.") and a.asname:
                    aliases[a.asname] = a.name
                else:
                    modules.add(a.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            names.add((aliases[node.value.id], node.attr))
    for module in aliases.values():                       # a module imported and never used still counts
        names.add((module, None))
    return modules, names


def test_a_scanlib_name_is_found_through_its_module():
    code = ("import numpy as np\n"
            "from scanlib import dynamics as dyn, joint\n"
            "import scanlib.search as srch\n"
            "from scanlib.instrument import Instrument\n"
            "x = dyn.wls(1, 2, 3); y = joint.fit_joint; z = srch._unassigned; np.zeros(3)\n")
    modules, names = _uses(code)
    assert modules == {"numpy"}
    assert {(m, n) for m, n in names if n} == {("scanlib.dynamics", "wls"), ("scanlib.joint", "fit_joint"),
                                               ("scanlib.search", "_unassigned"),
                                               ("scanlib.instrument", "Instrument")}


def test_the_folder_has_its_twelve_notebooks_a_surface_and_its_stores():
    assert len(FIGURES) == 6 and len(AUXILIARY) == 6      # the figures; the kernel and the simulations in auxil/
    assert _surface(), "the README lists no surface"
    assert {p.name for p in (FOLDER / "stored").glob("*.npz")} == STORES
    assert (REPO / "data" / "rowell2021_lsf" / "lsf_af29_wc1.csv").is_file()
    assert (REPO / "data" / "rowell2021_lsf" / "README.md").is_file()


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_notebook_is_valid_and_clean(name):
    nb = _notebook(name)
    assert nb["nbformat"] == 4 and isinstance(nb["cells"], list)
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown") and isinstance(c["source"], (str, list))
    dirty = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
             and (c.get("outputs") or c.get("execution_count") is not None)]
    assert not dirty, f"code cells with outputs: {dirty}"
    code = _code(name)
    assert not re.search(r"^\s*(SAVE_PDF|FULL)\s*=\s*True\b", code, re.MULTILINE)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_notebook_depends_on_the_package_alone(name):
    surface = _surface()
    code = _code(name)
    modules, names = _uses(code)
    assert modules <= OTHER_MODULES, f"{name}: imports {sorted(modules - OTHER_MODULES)}"
    for module, used in names:
        assert module in surface, f"{name}: {module} is not in the README surface"
        if used is not None:
            assert not used.startswith("_"), f"{name}: a private name of {module}: {used}"
            assert used in surface[module], f"{name}: {module}.{used} is not in the README surface"
    assert "sys.path" not in code and "importlib" not in code


def test_every_surface_name_is_used():
    used = set().union(*(_uses(_code(name))[1] for name in NOTEBOOKS))
    unused = [(m, n) for m, names in _surface().items() for n in names if (m, n) not in used]
    assert not unused, f"README surface names no notebook uses: {unused}"


@pytest.mark.slow
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_notebook_runs(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace = {"__name__": f"<{name}>"}
    previous = os.getcwd()
    os.chdir((FOLDER / name).parent)                      # relative paths resolve from the notebook's own folder
    try:
        exec(compile(_code(name), name, "exec"), namespace)
    finally:
        os.chdir(previous)
        plt.close("all")


#: Functions that are the same thing in more than one paper notebook: each must be one text in all of them.
SHARED_FUNCTIONS = {
    "closed_form": ("auxil/sim_lambda_c.ipynb", "auxil/sim_lambda_c_validation.ipynb"),
    "rate": ("auxil/sim_lambda_c.ipynb", "auxil/sim_lambda_c_validation.ipynb"),
    "polar": ("fig4_contaminated_binary.ipynb", "auxil/sim_coverage_fig4.ipynb"),
}


def _function_text(name, function):
    """The source text of a top-level function of a notebook, from its ``def`` line to its last line."""
    for cell in _notebook(name)["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function:
                return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} does not define {function}")


@pytest.mark.parametrize("function", sorted(SHARED_FUNCTIONS))
def test_a_function_shared_by_two_notebooks_is_one_text(function):
    """A function that two notebooks both write out is the same text in both, so the copies cannot diverge."""
    first, second = SHARED_FUNCTIONS[function]
    assert _function_text(first, function) == _function_text(second, function)


def _notebook_code(path):
    """The code of a notebook given by its path, its code cells joined."""
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_the_settings_of_figure_4_are_the_same_wherever_they_are_written():
    """Figure 4's noise draw, zero-point stream, magnitude and background are written in four notebooks (the
    figure, its coverage simulation, the Lambda_c simulation, and quickstart 3): they hold the same values."""
    figure = _code("fig4_contaminated_binary.ipynb")
    coverage = _code("auxil/sim_coverage_fig4.ipynb")
    lambda_c = _code("auxil/sim_lambda_c.ipynb")
    quickstart = _notebook_code(REPO / "notebooks" / "quickstart" / "3_triple_ladder.ipynb")
    # the noise draw, and the zero points' stream [seed, 3]
    seed = re.search(r"^SEED = (\d+)", figure, re.M).group(1)
    assert "ZERO_POINT_SEED = [SEED, 3]" in figure
    assert f"np.random.default_rng([{seed}, 3])" in lambda_c
    assert re.search(r"^SEED = (\d+)", quickstart, re.M).group(1) == seed
    assert "np.random.default_rng([SEED, 3])" in quickstart
    # the primary's magnitude and the background [electrons per sample per CCD crossing]
    for name in ("PRIMARY_G", "BACKGROUND_E_PER_SAMPLE_PER_CCD"):
        values = {re.search(rf"^{name} = ([\d.]+)", code, re.M).group(1) for code in (figure, coverage, lambda_c)}
        assert len(values) == 1, (name, values)
    magnitude = re.search(r"^PRIMARY_G = ([\d.]+)", figure, re.M).group(1)
    assert f"noise.gaia_like({magnitude}, sim.PRIMARY_FLUX)" in quickstart
