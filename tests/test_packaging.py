"""The packaging contract: installed with a version that matches the
source, a front door whose every name is the object it claims to be, and
a source tree within its budget (under 2200 lines outside the docstrings, no
module above 400).
"""
from __future__ import annotations

import ast
import importlib
import importlib.metadata
from pathlib import Path

import scanlib

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "src" / "scanlib"


def test_version_matches_the_installed_metadata():
    assert scanlib.__version__ == importlib.metadata.version("scanlib")
    assert f"## {scanlib.__version__}" in (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"version: {scanlib.__version__}" in (REPO / "CITATION.cff").read_text(encoding="utf-8")


def test_every_front_door_name_is_the_module_object():
    homes = {"instrument", "noise", "scanlaw", "simulation", "backprojection", "radon", "joint", "optimize"}
    for name in scanlib.__all__:
        obj = getattr(scanlib, name)
        module = obj.__module__.split(".")[-1]
        assert module in homes, f"{name} lives in {obj.__module__}"
        assert getattr(importlib.import_module(obj.__module__), name) is obj
        assert obj.__doc__ and obj.__doc__.strip(), f"{name} has no docstring"


def _code_lines(path):
    """Lines of a module outside its docstrings (module, class and function)."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            doc_lines.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    return sum(1 for i in range(1, len(src.splitlines()) + 1) if i not in doc_lines)


def test_the_source_tree_is_within_its_budget():
    counts = {p.name: _code_lines(p) for p in PACKAGE.glob("*.py")}
    assert sum(counts.values()) < 2200, counts
    assert max(counts.values()) <= 400, counts
