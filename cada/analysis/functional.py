"""Functional analysis: signal equivalence, constant outputs, dependence,
Boolean equations, symmetry, and the NAND-pair existence query.

These use ABC (SAT / cec / collapse) on small cones; structure-only questions
fall back to graph reachability.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import List, Optional, Tuple

from ..netlist.ir import Netlist, is_const
from ..netlist.blif_export import _TT
from . import graph, cones
from ..equiv import abc_bridge, gate as equiv_gate


def signals_equivalent(nl: Netlist, a: str, b: str) -> Optional[bool]:
    if a == b:
        return True
    return equiv_gate.signals_equivalent(nl, a, b)


def _splitmix(i: int) -> int:
    """Deterministic 64-bit pseudo-random (splitmix64) — no Math.random, so the
    constant-net detection is reproducible."""
    mask = (1 << 64) - 1
    z = (i + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def constant_nets(nl: Netlist, vectors: int = 256,
                  max_confirm: int = 64) -> dict:
    """Find FUNCTIONALLY-constant nets (provably 0/1 for all inputs), per Q&A
    A21.1.  Random-vector simulation cheaply nominates candidates (signature all
    0s or all 1s); each candidate is then confirmed with a SAT/cec check.
    Returns {net: "1'b0"|"1'b1"} (literal constants are not included).
    Fully guarded — returns {} on any problem."""
    try:
        from . import graph
        mask = (1 << vectors) - 1
        srcs = sorted(set(nl.pi) | {ff.q for ff in nl.dffs})
        val = {name: _splitmix(j) & mask for j, name in enumerate(srcs)}
        val["1'b0"] = 0
        val["1'b1"] = mask
        for net in graph.topo_nets(nl):
            if net in val:
                continue
            drv = nl.driver(net)
            if drv[0] != "gate":
                continue
            g = drv[1]
            iv = [val.get(i, 0) for i in g.ins]
            t = g.type
            if t == "and":
                v = iv[0] & iv[1]
            elif t == "or":
                v = iv[0] | iv[1]
            elif t == "nand":
                v = mask ^ (iv[0] & iv[1])
            elif t == "nor":
                v = mask ^ (iv[0] | iv[1])
            elif t == "xor":
                v = iv[0] ^ iv[1]
            elif t == "xnor":
                v = mask ^ (iv[0] ^ iv[1])
            elif t == "not":
                v = mask ^ iv[0]
            elif t == "buf":
                v = iv[0]
            else:
                continue
            val[net] = v
        cand = [(n, 0) for n, v in val.items() if v == 0 and n != "1'b0"]
        cand += [(n, 1) for n, v in val.items() if v == mask and n != "1'b1"]
        result = {}
        for net, _guess in cand[:max_confirm]:
            c = output_always_constant(nl, net)
            if c is not None:
                result[net] = "1'b%d" % c
        return result
    except Exception:
        return {}


def _cone_inputs(nl: Netlist, sinks) -> List[str]:
    cone = graph.fanin_cone_nets(nl, sinks)
    return sorted(i for i in cone
                  if nl.driver(i)[0] in ("pi", "dff", "undriven") and not is_const(i))


def _single_out_blif(nl: Netlist, sig: str, inputs: List[str],
                     cone: set, model: str = "m") -> str:
    lines = [f".model {model}", ".inputs " + " ".join(inputs), ".outputs O",
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


def output_always_constant(nl: Netlist, out: str) -> Optional[int]:
    """Return 0 or 1 if ``out`` is constant for all inputs, else None."""
    sinks = cones.effective_sinks(nl, out)
    cone = graph.fanin_cone_nets(nl, sinks)
    inputs = _cone_inputs(nl, sinks)
    sig = sinks[0] if sinks else out
    blif = _single_out_blif(nl, sig, inputs, cone)
    # compare to const-0
    const0 = (".model m\n.inputs " + " ".join(inputs) +
              "\n.outputs O\n.names O\n.end\n")
    r0 = abc_bridge.cec_blif(blif, const0)
    if r0 is True:
        return 0
    const1 = (".model m\n.inputs " + " ".join(inputs) +
              "\n.outputs O\n.names O\n1\n.end\n")
    r1 = abc_bridge.cec_blif(blif, const1)
    if r1 is True:
        return 1
    return None


def depends_on(nl: Netlist, out: str, inp: str) -> bool:
    """Structural dependence: is ``inp`` in the fan-in cone of ``out``?"""
    cone = cones.fanin_cone_nets(nl, out)
    if inp in cone:
        return True
    # also consider bus base name
    return any(c == inp or c.split("[")[0] == inp for c in cone)


def is_symmetric(nl: Netlist, out: str, a: str, b: str) -> Optional[bool]:
    """Is the function at ``out`` symmetric in inputs a and b?  (Swap a<->b in
    its cone and check equivalence.)"""
    sinks = cones.effective_sinks(nl, out)
    cone = graph.fanin_cone_nets(nl, sinks)
    if a not in cone and b not in cone:
        return True  # neither in support -> vacuously symmetric
    inputs = _cone_inputs(nl, sinks)
    sig = sinks[0] if sinks else out
    orig = _single_out_blif(nl, sig, inputs, cone, model="o")
    # swapped: rename a<->b in a copied netlist's cone
    swapped = _swapped_blif(nl, sig, inputs, cone, a, b)
    return abc_bridge.cec_blif(orig, swapped)


def _swapped_blif(nl, sig, inputs, cone, a, b):
    lines = [".model s", ".inputs " + " ".join(inputs), ".outputs O",
             ".names __const0", ".names __const1", "1"]

    def sw(n):
        if n == a:
            return b
        if n == b:
            return a
        return n

    def ref(n):
        return {"1'b0": "__const0", "1'b1": "__const1"}.get(n, n)

    emitted = set()
    for g in nl.gates:
        if g.out in cone and g.out not in emitted:
            tt = _TT.get(g.type)
            if tt:
                lines.append(".names " + " ".join([ref(sw(i)) for i in g.ins] + [g.out]))
                lines.extend(tt)
                emitted.add(g.out)
    lines.append(f".names {ref(sig)} O")
    lines.append("1 1")
    lines.append(".end")
    return "\n".join(lines) + "\n"


def boolean_equation(nl: Netlist, out: str, timeout: int = 10) -> Optional[str]:
    """Derive a Boolean equation for ``out`` in terms of its cone leaves
    (primary inputs and register state)."""
    sinks = cones.effective_sinks(nl, out)
    cone = graph.fanin_cone_nets(nl, sinks)
    inputs = _cone_inputs(nl, sinks)
    if len(inputs) > 12:
        return None  # too large to express compactly
    sig = sinks[0] if sinks else out
    blif = _single_out_blif(nl, sig, inputs, cone)
    d = tempfile.mkdtemp(prefix="cada_eqn_")
    p = os.path.join(d, "m.blif")
    pe = os.path.join(d, "m.eqn")
    try:
        with open(p, "w") as f:
            f.write(blif)
        ok, _ = abc_bridge.run_abc(
            [f'read_blif "{p}"', "strash", "collapse", f'write_eqn "{pe}"'],
            timeout=timeout)
        if not ok or not os.path.exists(pe):
            return None
        with open(pe) as f:
            txt = f.read()
        m = re.search(r"O\s*=\s*(.+?);", txt, re.DOTALL)
        if m:
            return m.group(1).strip().replace("\n", " ")
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def exists_nand_pair(nl: Netlist, target: str,
                     max_support: int = 16) -> Optional[Tuple[str, str]]:
    """Find internal signals (a, b) with NAND(a, b) functionally == target,
    restricted to candidates whose support is within the target's support."""
    sinks = cones.effective_sinks(nl, target)
    cone = graph.fanin_cone_nets(nl, sinks)
    inputs = _cone_inputs(nl, sinks)
    if not inputs or len(inputs) > max_support:
        return None
    idx = {n: i for i, n in enumerate(inputs)}
    n = len(inputs)

    # truth-table simulate every net in the cone over all 2^n input patterns
    import itertools
    sigs = [g.out for g in nl.gates if g.out in cone]
    tt = {}
    target_net = sinks[0] if sinks else target

    def sim_all():
        # vectorised over integer bitmasks: each net -> int of length 2^n
        full = (1 << (1 << n)) - 1
        val = {}
        for j, name in enumerate(inputs):
            col = 0
            for p in range(1 << n):
                if (p >> j) & 1:
                    col |= (1 << p)
            val[name] = col
        val["1'b0"] = 0
        val["1'b1"] = full
        order = graph.topo_nets(nl)
        for net in order:
            if net in val:
                continue
            drv = nl.driver(net)
            if drv[0] != "gate":
                continue
            g = drv[1]
            iv = [val.get(i, 0) for i in g.ins]
            t = g.type
            if t == "and":
                val[net] = iv[0] & iv[1]
            elif t == "or":
                val[net] = iv[0] | iv[1]
            elif t == "nand":
                val[net] = full ^ (iv[0] & iv[1])
            elif t == "nor":
                val[net] = full ^ (iv[0] | iv[1])
            elif t == "xor":
                val[net] = iv[0] ^ iv[1]
            elif t == "xnor":
                val[net] = full ^ (iv[0] ^ iv[1])
            elif t == "not":
                val[net] = full ^ iv[0]
            elif t == "buf":
                val[net] = iv[0]
        return val, full

    val, full = sim_all()
    if target_net not in val:
        return None
    want_and = full ^ val[target_net]   # need a&b == ~target
    # index signals by their value
    cand = [s for s in sigs if s in val] + [i for i in inputs]
    by_val = {}
    for s in cand:
        by_val.setdefault(val[s], s)
    for i, s1 in enumerate(cand):
        v1 = val[s1]
        # find s2 with v1 & v2 == want_and ... search is O(n^2) but cone small
        for s2 in cand[i:]:
            if (v1 & val[s2]) == want_and:
                return (s1, s2)
    # Extension: signals just OUTSIDE the cone whose function is still fully
    # determined by it — an external NOT/BUF of a cone net (for example the
    # netlist's own inverter of the target).  sim_all() already computed their
    # truth tables correctly because all of their inputs lie inside the cone.
    base = set(cone) | set(inputs)
    ext = sorted(
        g.out for g in nl.gates
        if g.type in ("not", "buf") and g.out not in base and g.out in val
        and all(i in base or is_const(i) for i in g.ins))
    ordered = sorted(inputs) + [s for s in sigs if s in val]
    for e in ext:
        ve = val[e]
        for s1 in ordered:
            if (val[s1] & ve) == want_and:
                return (s1, e)
    return None
