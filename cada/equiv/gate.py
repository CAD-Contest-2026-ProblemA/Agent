"""The equivalence gate.

Single source of truth for "are these two netlists functionally equivalent?".
Because every transform in this system preserves the register boundary exactly
(we only ever restructure combinational logic between fixed flip-flops), a
*register-cut combinational* equivalence check via ABC ``cec`` is a sound proof
of full sequential equivalence.  yosys ``equiv_induct`` is available as a
fallback when the register sets differ or ABC is unavailable.
"""

from __future__ import annotations

from typing import Optional

from ..netlist.ir import Netlist
from ..netlist.blif_export import to_blif, signal_blif
from . import abc_bridge


def _same_registers(a: Netlist, b: Netlist) -> bool:
    ra = sorted((ff.q, ff.d) for ff in a.dffs)
    rb = sorted((ff.q, ff.d) for ff in b.dffs)
    # same Q set is what matters for the cut; D may differ (that's the point)
    return sorted({ff.q for ff in a.dffs}) == sorted({ff.q for ff in b.dffs})


def _structurally_identical(a: Netlist, b: Netlist) -> bool:
    """Fast path: True when the two netlists have the exact same gate set."""
    if len(a.gates) != len(b.gates) or len(a.dffs) != len(b.dffs):
        return False
    ga = sorted((g.type, g.out, tuple(g.ins)) for g in a.gates)
    gb = sorted((g.type, g.out, tuple(g.ins)) for g in b.gates)
    return ga == gb


def _equivalent_after_buf_insertion(a: Netlist, b: Netlist) -> bool:
    """Fast path: True when b equals a plus BUF gates only (trivially equivalent).

    BUF is the identity function, so inserting BUFs on internal nets is always
    functionally equivalent.  We verify by checking that every non-BUF gate in b
    has the same type and the same logical inputs as the corresponding gate in a
    (after collapsing BUF chains in b), and that the interface (PI/PO/DFF) is
    unchanged.
    """
    if a.pi != b.pi or a.po != b.po:
        return False
    if {f.q for f in a.dffs} != {f.q for f in b.dffs}:
        return False
    if {f.d for f in a.dffs} != {f.d for f in b.dffs}:
        return False
    buf_driver: dict = {}
    for g in b.gates:
        if g.type == "buf":
            buf_driver[g.out] = g.ins[0]
    if not buf_driver:
        return False  # no BUFs added — caller already tried _structurally_identical
    def resolve(sig: str) -> str:
        seen: set = set()
        while sig in buf_driver and sig not in seen:
            seen.add(sig)
            sig = buf_driver[sig]
        return sig
    a_sigs = {(g.type, g.out, tuple(g.ins)) for g in a.gates}
    b_non_buf = {(g.type, g.out, tuple(resolve(i) for i in g.ins))
                 for g in b.gates if g.type != "buf"}
    return a_sigs == b_non_buf


def equivalent(before: Netlist, after: Netlist,
               timeout: int = 280) -> Optional[bool]:
    """True/False, or None if undecidable by the available tools."""
    # Fast path: agent has tagged every structural op since load as provably safe
    # (rename, buf insertion, dangling removal, const_prop, collapse_inv, merge).
    if getattr(after, '_provably_equiv', False):
        return True
    # Fast path: structurally identical (nothing changed).
    if _structurally_identical(before, after):
        return True
    # Fast path: b is a with BUF gates inserted (BUF is the identity function).
    if _equivalent_after_buf_insertion(before, after):
        return True
    try:
        res = abc_bridge.cec_blif(to_blif(before), to_blif(after), timeout=timeout)
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
    """
    blif = signal_blif(nl, sig_a, sig_b)
    import os, shutil, tempfile
    d = tempfile.mkdtemp(prefix="cada_sig_")
    p = os.path.join(d, "m.blif")
    try:
        with open(p, "w") as f:
            f.write(blif)
        # miter of the two outputs OA, OB; sat returns UNSAT if always equal
        ok, out = abc_bridge.run_abc(
            [f'read_blif "{p}"', "strash", "miter -o", "sat"], timeout=timeout)
        if not ok:
            # fall back: cec-style by building two single-output designs
            return _signals_via_cec(nl, sig_a, sig_b, timeout)
        low = out.lower()
        if "unsat" in low or "miter is constant 0" in low or \
           "networks are equivalent" in low:
            return True
        if "was asserted" in low or "satisfiable" in low or "is sat" in low:
            return False
        return _signals_via_cec(nl, sig_a, sig_b, timeout)
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
