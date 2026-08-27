"""The equivalence gate.

Single source of truth for "are these two netlists functionally equivalent?".
Because every transform in this system preserves the register boundary exactly
(we only ever restructure combinational logic between fixed flip-flops), a
*register-cut combinational* equivalence check via ABC ``cec`` is a sound proof
of full sequential equivalence.  yosys ``equiv_induct`` is available as a
fallback when the register sets differ or ABC is unavailable.
"""

from __future__ import annotations

import time
from typing import Optional

from ..netlist.ir import Netlist
from ..netlist.blif_export import (
    shared_synthetic_labels,
    signal_blif,
    to_blif,
)
from . import abc_bridge


def _same_registers(a: Netlist, b: Netlist) -> bool:
    """Do both designs cut at the same registers?

    Compared by INSTANCE name, not by the Q net.  The cut identifies a
    register, and a transform may legitimately rename the net it drives --
    collapsing a double inverter that feeds a primary output gives the register
    the port's name.  Keying on the net rejected that as "different registers"
    and failed designs that were in fact equivalent.  D is deliberately not
    compared: differing next-state logic is the whole point of the check.

    The BLIF projection additionally exposes D/CK/RN/SN for every matched
    instance, so equal instance sets cannot hide a changed control-pin cone.
    """
    return sorted(ff.name for ff in a.dffs) == sorted(ff.name for ff in b.dffs)


def equivalent(before: Netlist, after: Netlist,
               timeout: int = 280) -> Optional[bool]:
    """True/False, or None if undecidable by the available tools."""
    if not _same_registers(before, after):
        return False
    try:
        labels = shared_synthetic_labels(before, after)
        res = abc_bridge.cec_blif(
            to_blif(before, synthetic_labels=labels),
            to_blif(after, synthetic_labels=labels), timeout=timeout)
    except Exception:
        res = None
    if res is not None:
        return res
    # fallback: yosys sequential equivalence
    try:
        from . import yosys_bridge
        return yosys_bridge.equivalent(before, after, timeout=timeout)
    except Exception:
        return None


def signals_equivalent(nl: Netlist, sig_a: str, sig_b: str,
                       timeout: int = 120) -> Optional[bool]:
    """Are two internal signals functionally identical for all inputs?

    Build a miter that XORs the two cones and SAT-check for a difference.
    ``timeout`` is a TOTAL budget for the query: the cec fallback runs on
    whatever remains after the miter attempt, not on a fresh allowance --
    the analysis caller's 60-second limit (A77/A89) covers both stages.
    """
    deadline = time.time() + timeout
    blif = signal_blif(nl, sig_a, sig_b)
    import os, shutil, tempfile
    d = tempfile.mkdtemp(prefix="cada_sig_")
    p = os.path.join(d, "m.blif")

    def left():
        return max(5, int(deadline - time.time()))

    try:
        with open(p, "w") as f:
            f.write(blif)
        # miter of the two outputs OA, OB; sat returns UNSAT if always equal
        ok, out = abc_bridge.run_abc(
            [f'read_blif "{p}"', "strash", "miter -o", "sat"], timeout=timeout)
        if not ok:
            # fall back: cec-style by building two single-output designs
            return _signals_via_cec(nl, sig_a, sig_b, left())
        low = out.lower()
        if "unsat" in low or "miter is constant 0" in low or \
           "networks are equivalent" in low:
            return True
        if "was asserted" in low or "satisfiable" in low or "is sat" in low:
            return False
        return _signals_via_cec(nl, sig_a, sig_b, left())
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _signals_via_cec(nl: Netlist, sig_a: str, sig_b: str, timeout: int):
    """Compare two single-output projections (OA vs OB) via cec."""
    from ..netlist.blif_export import _TT  # noqa
    from ..analysis import graph
    cone = graph.fanin_cone_nets(nl, [sig_a, sig_b])
    inputs = sorted(i for i in cone
                    if nl.driver(i)[0] in ("pi", "dff", "undriven")
                    and i not in ("1'b0", "1'b1"))

    def proj(sig):
        lines = [".model p", ".inputs " + " ".join(inputs), ".outputs O",
                 ".names __const0", ".names __const1", "1"]
        def ref(n):
            return {"1'b0": "__const0", "1'b1": "__const1"}.get(n, n)
        emitted = set()
        for g in nl.gates:
            if g.out in cone and g.out not in emitted:
                tt = _TT.get(g.type)
                if tt:
                    lines.append(".names " + " ".join([ref(i) for i in g.ins] + [g.out]))
                    lines.extend(tt)
                    emitted.add(g.out)
        lines.append(f".names {ref(sig)} O")
        lines.append("1 1")
        lines.append(".end")
        return "\n".join(lines) + "\n"

    return abc_bridge.cec_blif(proj(sig_a), proj(sig_b), timeout=timeout)
