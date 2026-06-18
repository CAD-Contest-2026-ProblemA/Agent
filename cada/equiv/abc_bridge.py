"""ABC subprocess bridge: combinational equivalence checking and AIG-based
cost-ranked optimisation.

We feed ABC BLIF (never its Verilog frontend), always ``source`` abc.rc so the
``resyn2``/``dc2`` aliases resolve, and detect equivalence by substring match
("Networks are equivalent") per the empirically-confirmed ABC output.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

_ABC_CANDIDATES = [
    "/home/as6325400/abc/abc",
    os.path.expanduser("~/abc/abc"),
    "/usr/local/bin/abc",
    "/usr/bin/abc",
]

EQUIV_OK = "Networks are equivalent"
EQUIV_NO = "NOT EQUIVALENT"


def find_abc() -> Optional[str]:
    from ..toolpaths import resolve
    return resolve("abc", env_var="ABC_BIN", candidates=_ABC_CANDIDATES)


def _abc_rc(abc_bin: str) -> Optional[str]:
    rc = os.path.join(os.path.dirname(abc_bin), "abc.rc")
    return rc if os.path.exists(rc) else None


def run_abc(commands: List[str], timeout: int = 280) -> Tuple[bool, str]:
    abc = find_abc()
    if abc is None:
        return False, "ABC not found"
    rc = _abc_rc(abc)
    script = []
    if rc:
        script.append(f'source "{rc}"')
    script.extend(commands)
    cmd_str = "; ".join(script)
    try:
        p = subprocess.run([abc, "-q", cmd_str], capture_output=True,
                           text=True, timeout=timeout)
        return True, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return False, "ABC timeout"
    except Exception as exc:
        return False, f"ABC error: {exc}"


def cec_blif(blif_a: str, blif_b: str, timeout: int = 280) -> Optional[bool]:
    """Return True/False for equivalence, or None if the check could not run."""
    d = tempfile.mkdtemp(prefix="cada_cec_")
    pa = os.path.join(d, "a.blif")
    pb = os.path.join(d, "b.blif")
    try:
        with open(pa, "w") as f:
            f.write(blif_a)
        with open(pb, "w") as f:
            f.write(blif_b)
        ok, out = run_abc([f'cec "{pa}" "{pb}"'], timeout=timeout)
        if not ok:
            return None
        if EQUIV_NO in out:
            return False
        if EQUIV_OK in out:
            return True
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def optimize_blif(blif_in: str, recipe: List[str],
                  timeout: int = 280) -> Optional[Tuple[str, dict]]:
    """Run an ABC optimisation recipe on a combinational BLIF and return
    (optimised_blif_text, stats).  The output is an AIG written back as BLIF
    (2-input ANDs + inverters)."""
    d = tempfile.mkdtemp(prefix="cada_opt_")
    pin = os.path.join(d, "in.blif")
    pout = os.path.join(d, "out.blif")
    try:
        with open(pin, "w") as f:
            f.write(blif_in)
        cmds = [f'read_blif "{pin}"', "strash"] + recipe + \
               ["print_stats", f'write_blif "{pout}"']
        ok, out = run_abc(cmds, timeout=timeout)
        if not ok or not os.path.exists(pout):
            return None
        with open(pout) as f:
            text = f.read()
        stats = _parse_stats(out)
        return text, stats
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _parse_stats(out: str) -> dict:
    stats = {}
    import re
    m = re.search(r"and =\s*(\d+)", out)
    if m:
        stats["and"] = int(m.group(1))
    m = re.search(r"lev =\s*(\d+)", out)
    if m:
        stats["lev"] = int(m.group(1))
    m = re.search(r"nd =\s*(\d+)", out)
    if m:
        stats["nd"] = int(m.group(1))
    return stats
