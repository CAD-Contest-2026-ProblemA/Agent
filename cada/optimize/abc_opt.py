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
from ..netlist.blif_export import _TT
from ..transform import rewrite
from ..transform.base import Emitter
from ..equiv import abc_bridge, gate as equiv_gate

# The adaptive transform-family scheduler lives in .resynth; minimize_* below
# delegate there and this module provides the ABC plumbing.

_CONST_BUF_INV = ("GATE zero 0 O=CONST0;\n"
                  "GATE one 0 O=CONST1;\n"
                  "GATE buf 1 O=a; PIN * NONINV 1 999 1 0 1 0\n"
                  "GATE inv 1 O=!a; PIN * INV 1 999 1 0 1 0\n")

_GATE_DEFS = {
    # The contest's area metric is primitive gate count, not transistor area:
    # every legal primitive therefore has area 1.  The old 2/3 weights made
    # ABC optimize a different objective from the judge (especially XOR/XNOR).
    "and": "GATE and 1 O=a*b; PIN * NONINV 1 999 1 0 1 0\n",
    "or": "GATE or 1 O=a+b; PIN * NONINV 1 999 1 0 1 0\n",
    "nand": "GATE nand 1 O=!(a*b); PIN * INV 1 999 1 0 1 0\n",
    "nor": "GATE nor 1 O=!(a+b); PIN * INV 1 999 1 0 1 0\n",
    "xor": "GATE xor 1 O=(a*!b)+(!a*b); PIN * UNKNOWN 1 999 1 0 1 0\n",
    "xnor": "GATE xnor 1 O=(a*b)+(!a*!b); PIN * UNKNOWN 1 999 1 0 1 0\n",
}

# Delay mapping still needs a useful area tie-break between equal-depth cuts.
# The historic 2/3 weights empirically preserve shallower XOR-rich structures;
# gate-count objectives use the exact unit weights above instead.
_DEPTH_GATE_DEFS = {
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


def _genlib(basis, unit_area: bool = False) -> str:
    lib = _CONST_BUF_INV
    defs = _GATE_DEFS if unit_area else _DEPTH_GATE_DEFS
    for c in _BASIS_CELLS.get(basis, _BASIS_CELLS[None]):
        lib += defs[c]
    return lib

_GENLIB = _genlib(None)


def _opt_blif(nl: Netlist):
    """Return ``(blif, po-label map, register-pin-label map)`` for the cut.

    Pin labels map to flip-flop *instance names* and pin attributes, because the
    boundary is per register and D/CK/RN/SN must all survive resynthesis.
    """
    nl.driver("__force_build__")
    driver = nl._driver
    inputs = sorted(nl.pi)
    q_nets = sorted({ff.q for ff in nl.dffs})
    # Undriven nets are free combinational sources in the reference harness,
    # not implicit zeroes.  Declare them as BLIF inputs so optimization cannot
    # silently specialize their logic (including dead-register next state).
    floating = sorted(n for n in nl.all_nets()
                      if not is_const(n) and nl.driver(n)[0] == "undriven")
    all_inputs = list(dict.fromkeys(inputs + q_nets + floating))

    po_out: Dict[str, str] = {}
    pin_out: Dict[str, Tuple[str, str]] = {}
    out_labels: List[str] = []
    body: List[str] = []
    used_labels = set(nl.all_nets())

    def fresh_label(kind: str, index: int) -> str:
        label = f"__cada_{kind}_{index}"
        while label in used_labels:
            label += "_"
        used_labels.add(label)
        return label

    const0 = fresh_label("const", 0)
    const1 = fresh_label("const", 1)

    def ref(net: str) -> str:
        if net == "1'b0":
            return const0
        if net == "1'b1":
            return const1
        return net

    for g in nl.gates:
        tt = _TT.get(g.type)
        if tt is None:
            continue
        body.append(".names " + " ".join([ref(i) for i in g.ins] + [g.out]))
        body.extend(tt)

    for po_idx, p in enumerate(sorted(nl.po)):
        drv = driver.get(p)
        # Only combinationally-driven POs belong to the comb block.  A PO that
        # is a register Q is driven by its flip-flop (preserved separately); a
        # PO that is a PI is a direct pass-through left untouched.  Adding a
        # buffer for those would create a second driver.
        if drv is not None and drv[0] == "gate":
            label = fresh_label("po", po_idx)
            po_out[label] = p
            out_labels.append(label)
            body.append(f".names {ref(p)} {label}")
            body.append("1 1")
    for ff_idx, ff in enumerate(sorted(nl.dffs, key=lambda f: f.name)):
        for attr in ("d", "clk", "rn", "sn"):
            label = fresh_label(attr, ff_idx)
            pin_out[label] = (ff.name, attr)
            out_labels.append(label)
            body.append(f".names {ref(getattr(ff, attr))} {label}")
            body.append("1 1")

    head = [".model opt",
            ".inputs " + " ".join(all_inputs),
            ".outputs " + " ".join(out_labels),
            f".names {const0}",
            f".names {const1}", "1"]
    return "\n".join(head + body) + "\n.end\n", po_out, pin_out


def _parse_gate_blif(text: str):
    """Parse a mapped (.gate) BLIF into (gates, const_subst).

    gates: list of (type, out, [ins]); const_subst: net -> "1'b0"/"1'b1".
    """
    gates: List[Tuple[str, str, List[str]]] = []
    const: Dict[str, str] = {}
    lines = text.splitlines()
    for pos, raw in enumerate(lines):
        line = raw.strip()
        # ABC normally emits mapped cells as .gate, but direct output aliases
        # and constants can remain as simple .names nodes.  Preserve the
        # unambiguous 0/1-input forms instead of silently rebuilding them as
        # undriven wires (or zero-valued fallbacks).
        if line.startswith(".names"):
            nets = line.split()[1:]
            table = []
            scan = pos + 1
            while scan < len(lines):
                row = lines[scan].strip()
                if row.startswith("."):
                    break
                if row and not row.startswith("#"):
                    table.append(row)
                scan += 1
            if len(nets) == 1:
                const[nets[0]] = "1'b1" if table == ["1"] else "1'b0"
            elif len(nets) == 2 and table == ["1 1"]:
                gates.append(("buf", nets[1], [nets[0]]))
            elif len(nets) == 2 and table == ["0 1"]:
                gates.append(("not", nets[1], [nets[0]]))
            continue
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
                  timeout: int = 280,
                  area_mode: bool = False,
                  unit_area: Optional[bool] = None,
                  prepared: Optional[Tuple[
                      str, Dict[str, str], Dict[str, Tuple[str, str]]]] = None,
                  ) -> Optional[Netlist]:
    """Optimise the comb core of ``nl`` with an ABC ``recipe`` and map it onto
    the unit-delay library of ``basis``.

    ``prepared`` may carry a pre-computed ``(blif_text, po_out, pin_out)``
    export — e.g. reused across the recipe portfolio, or a yosys-resynthesised
    variant of the same export (the labels must be those of ``_opt_blif(nl)``).
    """
    if prepared is not None:
        blif, po_out, pin_out = prepared
    else:
        blif, po_out, pin_out = _opt_blif(nl)
    d = tempfile.mkdtemp(prefix="cada_opt_")
    pin = os.path.join(d, "in.blif")
    pout = os.path.join(d, "out.blif")
    plib = os.path.join(d, "unit.genlib")
    try:
        with open(pin, "w") as f:
            f.write(blif)
        with open(plib, "w") as f:
            f.write(_genlib(basis, unit_area=(area_mode if unit_area is None
                                              else unit_area)))
        mapper = "map -a -s" if area_mode else "map"
        cmds = ([f'read_blif "{pin}"', "strash"] + recipe +
                [f'read_library "{plib}"', mapper,
                 f'write_blif "{pout}"'])
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
    new = nl.snapshot()
    gate_by_out = {out: (gtype, ins) for gtype, out, ins in gates}
    skip_outputs = set()
    mapped_pins: Dict[Tuple[str, str], str] = {}
    for label, (ff_name, attr) in pin_out.items():
        # ABC can legally collapse a register-pin output straight to a
        # constant.  In that case there is no mapped gate to drive a fresh
        # wire; attach the pin to the constant itself.  Treating the label like
        # an ordinary mapped output left an undriven pin which a
        # constant-zero CEC happened to mask.
        resolved = resolve(label)
        if is_const(resolved):
            mapped_pins[(ff_name, attr)] = resolved
        elif (resolved in gate_by_out
              and gate_by_out[resolved][0] == "buf"
              and len(gate_by_out[resolved][1]) == 1):
            # The pseudo-output alias itself is not real circuit logic.  ABC
            # often maps a direct PI/constant/internal-net -> pin connection as
            # one BUF per register pin; reconnect the preserved D/CK/RN/SN pin
            # to that source and do not charge/materialize the interface BUF.
            # Only the gate whose output is the private pseudo label is skipped;
            # an ordinary internal BUF feeding it remains part of the design.
            skip_outputs.add(resolved)
            mapped_pins[(ff_name, attr)] = resolve(
                gate_by_out[resolved][1][0])
        else:
            mapped = new.fresh_name("np")
            rename[label] = mapped
            mapped_pins[(ff_name, attr)] = mapped

    def remap(net: str) -> str:
        # apply constant substitution, then output-label rename (a label may be
        # both a primary output AND an internal fan-in to other gates)
        return rename.get(resolve(net), resolve(net))

    new.gates = []
    em = Emitter(new)
    for (gtype, out, ins) in gates:
        if out in skip_outputs:
            continue
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
        for attr in ("d", "clk", "rn", "sn"):
            key = (ff.name, attr)
            if key in mapped_pins:
                setattr(ff, attr, remap(mapped_pins[key]))
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


def optimize_nand_super(nl: Netlist, timeout: float = 120.0,
                        dsd: bool = False,
                        verify: bool = True) -> Optional[Netlist]:
    """Map any register-cut netlist with the generic NAND/NOT super library.

    The expensive ``super -I 5 -L 4 -T 20`` enumeration is generated lazily
    once per process and shared by later calls.  ``dsd=True`` enables the
    generic DSD pre-transform; no testcase-specific data or prepared solution
    is consulted.  See :mod:`cada.optimize.nand_super` for fail-closed details.

    This is a separate opt-in arm, so the established :func:`optimize_comb`
    recipes and mapping behaviour are unchanged.
    """
    from . import nand_super
    return nand_super.map_netlist(
        nl, timeout=timeout, dsd=dsd, verify=verify)


def minimize_depth(nl: Netlist, basis: Optional[str] = None,
                   timeout: int = 290,
                   basis_output: Optional[str] = None) -> Tuple[Netlist, bool]:
    from . import resynth
    res, improved, _info = resynth.resynthesize(
        nl, objective="depth", basis=basis, basis_output=basis_output,
        timeout=timeout)
    return res, improved


def minimize_area(nl: Netlist, basis: Optional[str] = None,
                  timeout: int = 290,
                  basis_output: Optional[str] = None) -> Tuple[Netlist, bool]:
    from . import resynth
    res, improved, _info = resynth.resynthesize(
        nl, objective="area", basis=basis, basis_output=basis_output,
        timeout=timeout)
    return res, improved


def minimize_buffered_area(nl: Netlist, fanout_limit: int,
                           include_pi: bool = False, timeout: int = 290,
                           ) -> Tuple[Netlist, int, bool]:
    """Minimize the *post-buffer* primitive count, then insert the provably
    minimal fanout trees required by ``fanout_limit``.

    Ranking plain area first can choose a highly shared netlist that needs more
    buffers than it saved.  The resynthesis objective therefore includes the
    exact buffer lower bound/construction cost for every candidate.
    """
    if fanout_limit < 2:
        raise ValueError("buffered-area optimization requires fanout_limit >= 2")
    from . import resynth
    from ..transform import buffering
    res, improved, _info = resynth.resynthesize(
        nl, objective="buffered_area", fanout_limit=fanout_limit,
        fanout_include_pi=include_pi, timeout=timeout)
    added = buffering.limit_fanout(
        res, fanout_limit, include_pi=include_pi)
    return res, added, improved


def optimize_cone_depth(nl: Netlist, output: str, basis: Optional[str] = None,
                        timeout: int = 290,
                        basis_output: Optional[str] = None,
                        ) -> Tuple[Netlist, bool]:
    """Run the generic runtime cone flow for ``output``.

    The scheduler extracts the complete register-cut cone, preserves shared
    side exits, optimises that small window without prepared-registry lookup,
    safely splices it back, and accepts it only after whole-design CEC.
    """
    from . import resynth
    res, improved, _info = resynth.resynthesize(
        nl, objective="cone_depth", output=output, basis=basis,
        basis_output=basis_output, timeout=timeout)
    return res, improved
