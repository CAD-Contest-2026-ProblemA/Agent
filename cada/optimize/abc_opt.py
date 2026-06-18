"""Cost-ranked optimisation via ABC.

Project the combinational logic (register cut) to BLIF with controlled,
reversible output labels, optimise the AIG (``resyn2``) then technology-map to a
unit-delay library of our own primitive gates (``map``), so ABC minimises the
*gate-level* depth that the contest cost function actually measures (every gate,
inverters included, costs one level — unlike ABC's inverter-free AIG ``lev``).

The mapped result is read back from the clean ``.gate`` BLIF (cells map 1:1 to
our gate types; ``inv`` -> ``not``; ``zero``/``one`` -> constant nets, so no
out-of-basis ``zero``/``one`` cell ever reaches the writer).  Results are kept
only if cec-equivalent and cost-improving; otherwise we report the original.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

from ..netlist.ir import Gate, Netlist, is_const
from ..netlist.blif_export import _TT, _live_registers
from ..analysis import depth as depth_mod
from ..transform import rewrite
from ..transform.base import Emitter
from ..equiv import abc_bridge, gate as equiv_gate

DEPTH_RECIPE = ["resyn2", "resyn2"]
AREA_RECIPE = ["resyn2", "dc2", "resyn2"]

_CONST_BUF_INV = ("GATE zero 0 O=CONST0;\n"
                  "GATE one 0 O=CONST1;\n"
                  "GATE buf 1 O=a; PIN * NONINV 1 999 1 0 1 0\n"
                  "GATE inv 1 O=!a; PIN * INV 1 999 1 0 1 0\n")

_GATE_DEFS = {
    "and": "GATE and 2 O=a*b; PIN * NONINV 1 999 1 0 1 0\n",
    "or": "GATE or 2 O=a+b; PIN * NONINV 1 999 1 0 1 0\n",
    "nand": "GATE nand 2 O=!(a*b); PIN * INV 1 999 1 0 1 0\n",
    "nor": "GATE nor 2 O=!(a+b); PIN * INV 1 999 1 0 1 0\n",
    "xor": "GATE xor 3 O=(a*!b)+(!a*b); PIN * UNKNOWN 1 999 1 0 1 0\n",
    "xnor": "GATE xnor 3 O=(a*b)+(!a*!b); PIN * UNKNOWN 1 999 1 0 1 0\n",
}

# basis -> which 2-input cells ABC may use during mapping
_BASIS_CELLS = {
    None: ["and", "or", "nand", "nor", "xor", "xnor"],
    "AND_NOT": ["and"],
    "NAND_NOT": ["nand"],
    "NOR_NOT": ["nor"],
    "AND_OR_NOT": ["and", "or"],
}


def _genlib(basis) -> str:
    lib = _CONST_BUF_INV
    for c in _BASIS_CELLS.get(basis, _BASIS_CELLS[None]):
        lib += _GATE_DEFS[c]
    return lib

_GENLIB = _genlib(None)


def _san(net: str) -> str:
    return net.replace("[", "__").replace("]", "")


def _opt_blif(nl: Netlist):
    """Return (blif_text, {po_label: po_net}, {d_label: q_net})."""
    inputs = sorted(nl.pi)
    live = _live_registers(nl)
    q_nets = sorted(live)
    all_inputs = inputs + [q for q in q_nets if q not in set(inputs)]
    in_set = set(all_inputs)

    nl.driver("__force_build__")
    driver = nl._driver

    po_out: Dict[str, str] = {}
    d_out: Dict[str, str] = {}
    rep_d: Dict[str, str] = {}
    for ff in nl.dffs:
        if ff.q in live and ff.q not in rep_d:
            rep_d[ff.q] = ff.d

    out_labels: List[str] = []
    body: List[str] = []

    def ref(net: str) -> str:
        if net == "1'b0":
            return "__const0"
        if net == "1'b1":
            return "__const1"
        return net

    for g in nl.gates:
        tt = _TT.get(g.type)
        if tt is None:
            continue
        body.append(".names " + " ".join([ref(i) for i in g.ins] + [g.out]))
        body.extend(tt)

    for p in sorted(nl.po):
        drv = driver.get(p)
        # Only combinationally-driven POs belong to the comb block.  A PO that
        # is a register Q is driven by its flip-flop (preserved separately); a
        # PO that is a PI is a direct pass-through left untouched.  Adding a
        # buffer for those would create a second driver.
        if drv is not None and drv[0] == "gate":
            label = "PO_" + _san(p)
            po_out[label] = p
            out_labels.append(label)
            body.append(f".names {ref(p)} {label}")
            body.append("1 1")
    for q in sorted(rep_d):
        label = "D_" + _san(q)
        d_out[label] = q
        out_labels.append(label)
        body.append(f".names {ref(rep_d[q])} {label}")
        body.append("1 1")

    head = [".model opt",
            ".inputs " + " ".join(all_inputs),
            ".outputs " + " ".join(out_labels),
            ".names __const0",
            ".names __const1", "1"]
    return "\n".join(head + body) + "\n.end\n", po_out, d_out


def _parse_gate_blif(text: str):
    """Parse a mapped (.gate) BLIF into (gates, const_subst).

    gates: list of (type, out, [ins]); const_subst: net -> "1'b0"/"1'b1".
    """
    gates: List[Tuple[str, str, List[str]]] = []
    const: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(".gate"):
            continue
        toks = line.split()[1:]
        ctype = toks[0]
        pins = {}
        for t in toks[1:]:
            if "=" in t:
                k, v = t.split("=", 1)
                pins[k] = v
        out = pins.get("O")
        if out is None:
            continue
        if ctype == "zero":
            const[out] = "1'b0"
            continue
        if ctype == "one":
            const[out] = "1'b1"
            continue
        mytype = "not" if ctype == "inv" else ctype
        ins = []
        for pn in ("a", "b"):
            if pn in pins:
                ins.append(pins[pn])
        gates.append((mytype, out, ins))
    return gates, const


def optimize_comb(nl: Netlist, recipe: List[str], basis=None,
                  timeout: int = 280) -> Optional[Netlist]:
    blif, po_out, d_out = _opt_blif(nl)
    d = tempfile.mkdtemp(prefix="cada_opt_")
    pin = os.path.join(d, "in.blif")
    pout = os.path.join(d, "out.blif")
    plib = os.path.join(d, "unit.genlib")
    try:
        with open(pin, "w") as f:
            f.write(blif)
        with open(plib, "w") as f:
            f.write(_genlib(basis))
        cmds = ([f'read_blif "{pin}"', "strash"] + recipe +
                [f'read_library "{plib}"', "map", f'write_blif "{pout}"'])
        ok, out = abc_bridge.run_abc(cmds, timeout=timeout)
        if not ok or not os.path.exists(pout):
            return None
        with open(pout) as f:
            text = f.read()
    finally:
        shutil.rmtree(d, ignore_errors=True)

    gates, const = _parse_gate_blif(text)
    if not gates and not const:
        return None

    def resolve(n: str) -> str:
        seen = 0
        while n in const and seen < 1_000_000:
            n = const[n]
            seen += 1
        return n

    # rename comb-output labels to their real targets
    rename: Dict[str, str] = {}
    for label, po in po_out.items():
        rename[label] = po
    qd_net: Dict[str, str] = {}
    new = nl.snapshot()
    for label, q in d_out.items():
        nd = new.fresh_name("nd")
        rename[label] = nd
        qd_net[q] = nd

    def remap(net: str) -> str:
        # apply constant substitution, then output-label rename (a label may be
        # both a primary output AND an internal fan-in to other gates)
        return rename.get(resolve(net), resolve(net))

    new.gates = []
    em = Emitter(new)
    for (gtype, out, ins) in gates:
        real_out = rename.get(out, out)
        real_ins = [remap(i) for i in ins]
        em.emit(gtype, real_out, real_ins)
    new.gates.extend(em.out)

    # any primary output ABC tied directly to a constant/input (no .gate) must
    # still be driven in the rebuilt netlist
    produced = {g.out for g in new.gates}
    for label, po in po_out.items():
        if po not in produced:
            src = resolve(label)
            new.gates.append(Gate("buf", new.fresh_name("cg"),
                                  po, [src if src != label else "1'b0"]))
    for ff in new.dffs:
        if ff.q in qd_net:
            ff.d = qd_net[ff.q]
    new.touch()
    return new


def _finalize(nl, cand, basis, timeout):
    """Ensure basis purity (clean up any buf cells), then cec-verify."""
    if cand is None:
        return None
    if basis:
        want = rewrite.BASES[basis]
        if any(g.type not in want for g in cand.gates):
            rewrite.to_basis(cand, basis)
    if equiv_gate.equivalent(nl, cand, timeout=timeout) is not True:
        return None
    return cand


def minimize_depth(nl: Netlist, basis: Optional[str] = None,
                   timeout: int = 280) -> Tuple[Netlist, bool]:
    before = depth_mod.global_max_depth(nl)
    cand = _finalize(nl, optimize_comb(nl, DEPTH_RECIPE, basis=basis,
                                       timeout=timeout), basis, timeout)
    if cand is not None and depth_mod.global_max_depth(cand) < before:
        return cand, True
    return nl, False


def minimize_area(nl: Netlist, basis: Optional[str] = None,
                  timeout: int = 280) -> Tuple[Netlist, bool]:
    before = len(nl.gates)
    cand = _finalize(nl, optimize_comb(nl, AREA_RECIPE, basis=basis,
                                       timeout=timeout), basis, timeout)
    if cand is not None and len(cand.gates) < before:
        return cand, True
    return nl, False


def optimize_cone_depth(nl: Netlist, output: str, basis: Optional[str] = None,
                        timeout: int = 280) -> Tuple[Netlist, bool]:
    """Depth-optimise the comb block and keep it only if the cone of ``output``
    got shallower (and the design stays equivalent / in basis)."""
    before = depth_mod.depth_of_cone(nl, output)
    cand = _finalize(nl, optimize_comb(nl, DEPTH_RECIPE, basis=basis,
                                       timeout=timeout), basis, timeout)
    if cand is not None and depth_mod.depth_of_cone(cand, output) < before:
        return cand, True
    return nl, False
