"""Run the notebooks' code cells top to bottom, without writing them.

Each tracked notebook under ``notebooks/`` is executed in-process (its code
cells ``exec``-ed into a fresh namespace, matplotlib on the Agg backend,
a wall-clock cap per notebook) and reported as PASS, FAIL or TIMEOUT. The
``.ipynb`` files are never written: no output lands in a tracked notebook.

Usage::

    conda run -n scanlib python tools/notebook_smoke.py [pattern ...]

A pattern selects the notebooks whose path contains it.
"""
from __future__ import annotations

import io
import json
import os
import re
import signal
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PER_NOTEBOOK_SECONDS = 300


class _Timeout(Exception):
    pass


def _alarm(_sig, _frm):
    raise _Timeout()


def run_notebook(path: Path) -> tuple[str, str, float]:
    """Execute the code cells; return (status, detail, seconds)."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    prev_cwd = Path.cwd()
    os.chdir(path.parent)                      # relative paths resolve from the notebook
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(PER_NOTEBOOK_SECONDS)
    buf = io.StringIO()
    idx = -1
    try:
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            # strip IPython magics and shell escapes only
            src = "\n".join(line for line in src.splitlines()
                            if not re.match(r"\s*(?:%{1,2}[A-Za-z]|![A-Za-z])", line))
            with redirect_stdout(buf), redirect_stderr(buf):
                exec(compile(src, f"{path.name}[cell {idx}]", "exec"), ns)
        return "PASS", "", time.time() - t0
    except _Timeout:
        return "TIMEOUT", f"exceeded {PER_NOTEBOOK_SECONDS}s", time.time() - t0
    except BaseException:  # noqa: BLE001 - report, do not crash the runner
        tb = traceback.format_exc(limit=3).strip().splitlines()[-1][:200]
        return "FAIL", f"cell {idx}: {tb}", time.time() - t0
    finally:
        signal.alarm(0)
        os.chdir(prev_cwd)
        plt.close("all")


def main(patterns: list[str]) -> int:
    results = []
    for path in sorted((REPO / "notebooks").rglob("*.ipynb")):
        rel = str(path.relative_to(REPO))
        if "_local" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue
        if patterns and not any(p in rel for p in patterns):
            continue
        status, detail, secs = run_notebook(path)
        results.append(status)
        print(f"{status:8s} {secs:6.1f}s  {rel}" + (f"  [{detail}]" if detail else ""), flush=True)
    bad = sum(r != "PASS" for r in results)
    print(f"SUMMARY: {len(results)} run, {bad} not passing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
