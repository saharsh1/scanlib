"""The law that keeps scanlib standalone: numpy, scipy and the standard
library at module scope, nothing else anywhere but matplotlib inside the
one function that draws (the ``plots`` extra).

Both halves are checked: what the import statements say (an AST walk over
every module, so a new dependency shows up here before it shows up at
install time) and what happens in a fresh interpreter with matplotlib
hidden (every module must import, and the four algorithms must run).

The second law is the tier of each module in ``docs/open_api.md``
(``atom``, ``data``, ``pipeline``, ``simulation``): every module has one
row, every listed name imports, every public name is listed and has one
home, no atom imports a pipeline module, and no atom or pipeline module
imports the simulation: the method never sees the truth.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "src" / "scanlib"

#: Installed with scanlib: fine at module scope.
DEPENDENCIES = frozenset({"numpy", "scipy"})
#: The ``plots`` extra: reachable only from inside a function.
OPTIONAL = frozenset({"matplotlib"})


def _imports(path: Path, module_scope_only: bool) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if module_scope_only else list(ast.walk(tree))
    tops: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            tops.add(node.module.split(".")[0])
    return tops


def _undeclared(allowed: set[str], module_scope_only: bool) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        for top in _imports(path, module_scope_only):
            if top in allowed or top in sys.stdlib_module_names:
                continue
            found.setdefault(top, set()).add(path.name)
    return found


def test_module_scope_imports_are_numpy_scipy_and_the_standard_library():
    bad = _undeclared(DEPENDENCIES | {"scanlib", "__future__"}, module_scope_only=True)
    assert not bad, (
        "imported at module scope and neither numpy, scipy, the standard "
        "library nor scanlib: " + "; ".join(f"{k} ({', '.join(sorted(v))})" for k, v in bad.items())
        + ". Move the import inside the function that needs it, or drop it; do not widen the list.")


def test_every_import_anywhere_is_declared():
    bad = _undeclared(DEPENDENCIES | OPTIONAL | {"scanlib", "__future__"},
                      module_scope_only=False)
    assert not bad, "imported but not declared in pyproject: " + ", ".join(bad)


_PROBE = r'''
import json, sys
class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("No module named %r" % fullname)
for name in list(sys.modules):
    if name.split(".")[0] in BLOCKED:
        del sys.modules[name]
sys.meta_path.insert(0, Blocker())
out = {"failures": [], "modules": []}
import importlib, pathlib
root = pathlib.Path(MODULE_ROOT)
for p in sorted(root.rglob("*.py")):
    parts = list(p.relative_to(root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    name = ".".join(parts)
    out["modules"].append(name)
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        out["failures"].append("%s: %s" % (name, exc))
import numpy as np
import scanlib
inst = scanlib.Instrument.gaussian(fwhm_mas=100.0, pitch_mas=59.0, n_samples=18)
psi = np.linspace(0.0, np.pi, 12, endpoint=False)
rng = np.random.default_rng(0)
eta = scanlib.scene_positions_al([100.0, 50.0], psi)
windows = np.array([scanlib.simulate_window(e, [1000.0, 300.0], inst, rng, 10.0) for e in eta])
scanlib.backprojection_map(psi, eta[:, 1], np.full(psi.size, 3.0))
scanlib.filtered_backprojection(psi, windows, inst)
scanlib.faststack(psi, windows, inst)
out["theta"] = scanlib.fit_joint(windows, psi, [95.0, 55.0], inst, 10.0)["theta"].tolist()
print(json.dumps(out))
'''


@pytest.fixture(scope="module")
def probe() -> dict:
    code = f"MODULE_ROOT = {str(PACKAGE)!r}\nBLOCKED = {sorted(OPTIONAL)!r}\n" + _PROBE
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(REPO))
    assert proc.returncode == 0, f"the probe failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_every_module_imports_without_matplotlib(probe):
    assert len(probe["modules"]) >= 5
    assert not probe["failures"], "\n".join(probe["failures"])


def test_the_four_algorithms_run_on_numpy_and_scipy_alone(probe):
    theta = probe["theta"]
    assert abs(theta[0] - 100.0) < 5.0 and abs(theta[1] - 50.0) < 5.0


# --- the tiers of docs/open_api.md (decision D1 of the port plan) -----------

TIERS = frozenset({"atom", "data", "pipeline", "simulation"})
OPEN_API = REPO / "docs" / "open_api.md"


def _surface_rows() -> dict[str, tuple[str, list[str]]]:
    """``{module: (tier, names)}`` from the Modules table of docs/open_api.md."""
    text = OPEN_API.read_text(encoding="utf-8")
    table = text.split("## Modules", 1)[1].split("\n## ", 1)[0]
    rows: dict[str, tuple[str, list[str]]] = {}
    for line in table.splitlines():
        if not line.startswith("| `scanlib."):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        module = cells[0].strip("`").split(".", 1)[1]
        assert module not in rows, f"{module} has two rows"
        rows[module] = (cells[1], [n.strip().strip("`") for n in cells[2].split(",")])
    return rows


def _package_imports(source: str) -> set[str]:
    """The scanlib modules a module's source imports, by their short names."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:
                found.update([node.module.split(".")[0]] if node.module
                             else [alias.name for alias in node.names])
            elif node.level == 0 and node.module and node.module.split(".")[0] == "scanlib":
                parts = node.module.split(".")
                found.update(parts[1:2] or [alias.name for alias in node.names])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "scanlib" and len(parts) > 1:
                    found.add(parts[1])
    return found


def _importing(tiers: dict[str, str], sources: dict[str, str], importers, imported) -> list[str]:
    """``"m imports other"`` for every module of a tier in ``importers`` that
    imports a module of a tier in ``imported``."""
    return sorted(f"{m} imports {other}" for m, src in sources.items() if tiers[m] in importers
                  for other in _package_imports(src) if tiers.get(other) in imported)


def _atoms_importing_pipeline(tiers: dict[str, str], sources: dict[str, str]) -> list[str]:
    return _importing(tiers, sources, {"atom"}, {"pipeline"})


def _sources() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8")
            for p in PACKAGE.glob("*.py") if p.stem != "__init__"}


def test_every_module_has_one_row_with_a_tier():
    rows = _surface_rows()
    modules = {p.stem for p in PACKAGE.glob("*.py") if p.stem != "__init__"}
    assert set(rows) == modules, (
        f"modules without a row: {sorted(modules - set(rows))}; "
        f"rows without a module: {sorted(set(rows) - modules)}")
    bad = {m: tier for m, (tier, _) in rows.items() if tier not in TIERS}
    assert not bad, f"unknown tiers: {bad}"


def test_every_listed_name_imports_and_every_public_name_is_listed():
    import importlib

    import scanlib
    listed = set()
    for module, (_, names) in _surface_rows().items():
        mod = importlib.import_module(f"scanlib.{module}")
        for name in names:
            assert hasattr(mod, name), f"{name} is listed under scanlib.{module} but is not there"
            listed.add(name)
    missing = sorted(set(scanlib.__all__) - listed)
    assert not missing, f"in scanlib.__all__ without a row in docs/open_api.md: {missing}"


def test_no_atom_imports_a_pipeline_module():
    tiers = {m: tier for m, (tier, _) in _surface_rows().items()}
    bad = _atoms_importing_pipeline(tiers, _sources())
    assert not bad, "an atom imports a pipeline module: " + "; ".join(bad)


def test_the_method_never_imports_the_simulation():
    tiers = {m: tier for m, (tier, _) in _surface_rows().items()}
    assert "simulation" in tiers.values()
    bad = _importing(tiers, _sources(), {"atom", "data", "pipeline"}, {"simulation"})
    assert not bad, "a method module imports the simulation: " + "; ".join(bad)


def test_every_listed_name_has_one_home():
    """A name is defined in the module of its row only: any other module that
    holds it holds the same object (it imported it), so a moved name is gone
    from its old module."""
    import importlib
    rows = _surface_rows()
    mods = {m: importlib.import_module(f"scanlib.{m}") for m in rows}
    seen: dict[str, str] = {}
    for module, (_, names) in rows.items():
        for name in names:
            assert name not in seen, f"{name} is listed under {seen.get(name)} and {module}"
            seen[name] = module
            obj = getattr(mods[module], name)
            for other, mod in mods.items():
                if other != module and hasattr(mod, name):
                    assert getattr(mod, name) is obj, f"{name} is defined in {module} and again in {other}"


def test_the_tier_check_catches_an_atom_that_imports_a_pipeline_module():
    tiers = {"joint": "atom", "reduction": "pipeline", "instrument": "atom"}
    for src in ("from .reduction import ladder_window\n",
                "from . import reduction\n",
                "from scanlib.reduction import ladder_window\n",
                "import scanlib.reduction\n",
                "def f():\n    from .reduction import ladder_window\n"):
        assert _atoms_importing_pipeline(tiers, {"joint": src}) == ["joint imports reduction"], src
    assert _atoms_importing_pipeline(tiers, {"joint": "from .instrument import Instrument\n"}) == []
    assert _atoms_importing_pipeline(tiers, {"reduction": "from .joint import fit_joint\n"}) == []
    tiers["simulation"] = "simulation"
    sim = {"reduction": "from .simulation import simulate_window\n",
           "simulation": "from .reduction import ladder_window\n"}
    assert _importing(tiers, sim, {"atom", "pipeline"}, {"simulation"}) == ["reduction imports simulation"]
