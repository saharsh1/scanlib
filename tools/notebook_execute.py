"""Execute notebooks in place through a Jupyter kernel and store their outputs.

The ``scanlib`` environment has ``ipykernel`` and ``jupyter_client`` but no
``nbconvert``, so this is the executor: each code cell of each notebook
named on the command line runs in a fresh kernel, in order, with the
notebook's folder as the working directory; the stream, display and error
outputs come back and are written into the ``.ipynb`` with the execution
counts, as ``jupyter nbconvert --execute --inplace`` would write them.

Usage::

    conda run -n scanlib python tools/notebook_execute.py notebooks/quickstart/1_sinogram_fit.ipynb [...]

A cell that raises stops that notebook; the error is written into the cell
and reported, and the exit status is 1.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from queue import Empty

from jupyter_client import KernelManager

CELL_SECONDS = 600


def _run_cell(kc, source: str, count: int) -> tuple[list, bool]:
    msg_id = kc.execute(source)
    outputs: list = []
    failed = False
    deadline = time.time() + CELL_SECONDS
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=max(0.1, deadline - time.time()))
        except Empty:
            raise TimeoutError(f"cell {count} exceeded {CELL_SECONDS}s")
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        typ, content = msg["msg_type"], msg["content"]
        if typ == "status" and content["execution_state"] == "idle":
            break
        if typ == "stream":
            if outputs and outputs[-1]["output_type"] == "stream" \
                    and outputs[-1]["name"] == content["name"]:
                outputs[-1]["text"] += content["text"]
            else:
                outputs.append({"output_type": "stream", "name": content["name"],
                                "text": content["text"]})
        elif typ in ("display_data", "execute_result"):
            out = {"output_type": typ, "data": content["data"],
                   "metadata": content.get("metadata", {})}
            if typ == "execute_result":
                out["execution_count"] = count
            outputs.append(out)
        elif typ == "error":
            outputs.append({"output_type": "error", "ename": content["ename"],
                            "evalue": content["evalue"], "traceback": content["traceback"]})
            failed = True
    # drain the shell reply
    while True:
        try:
            reply = kc.get_shell_msg(timeout=30)
        except Empty:
            break
        if reply["parent_header"].get("msg_id") == msg_id:
            break
    return outputs, failed


def execute(path: Path) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    km = KernelManager(kernel_name="python3")
    km.start_kernel(cwd=str(path.parent))
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=60)
    ok = True
    count = 0
    t0 = time.time()
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            if not ok:
                cell["outputs"], cell["execution_count"] = [], None
                continue
            count += 1
            outputs, failed = _run_cell(kc, "".join(cell["source"]), count)
            cell["outputs"], cell["execution_count"] = outputs, count
            if failed:
                ok = False
                err = next(o for o in outputs if o["output_type"] == "error")
                print(f"  cell {count} failed: {err['ename']}: {err['evalue']}", flush=True)
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    nb.setdefault("metadata", {})
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python",
                                       "version": ".".join(map(str, sys.version_info[:3]))}
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{'PASS' if ok else 'FAIL':5s} {time.time() - t0:6.1f}s  {path}", flush=True)
    return ok


def main(argv: list[str]) -> int:
    paths = [Path(a).resolve() for a in argv]
    if not paths:
        print(__doc__)
        return 2
    bad = sum(not execute(p) for p in paths)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
