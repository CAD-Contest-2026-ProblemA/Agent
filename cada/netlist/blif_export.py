"""Export the IR to BLIF for ABC.

BLIF (not ABC's Verilog frontend, which cannot parse the named-port ``dff``) is
our bridge to ABC.  ABC's ``cec`` matches primary inputs/outputs *by name*, so
two BLIFs produced from the same net-naming scheme can be compared directly.

Sequential designs are handled by a *register cut*: each flip-flop's Q net
becomes a pseudo-primary-input and its D net a pseudo-primary-output, yielding a
purely combinational projection.  Two designs with identical register
boundaries are functionally equivalent iff their cut projections are
combinationally equivalent (this is exactly what we need to validate structural
combinational transforms).  Aliasing cases are handled:

* a Q net that is also a module output -> emit a buffered copy as the output;
* a D net that is also a declared output -> emit once;
* multiple flip-flops sharing a Q net -> single pseudo-input, separate D outputs.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .ir import Netlist, is_const

_TT = {
    "buf": ("1 1",),
    "not": ("0 1",),
    "and": ("11 1",),
    "or": ("1- 1", "-1 1"),
    "nand": ("0- 1", "-0 1"),
    "nor": ("00 1",),
    "xor": ("10 1", "01 1"),
    "xnor": ("11 1", "00 1"),
}


def _live_registers(nl: Netlist):
    """Q nets of flip-flops whose state can affect a primary output.

    A dangling register (Q reaches no PO) is non-live in *every* design that
    contains it, so excluding non-live registers from the cut keeps the
    before/after comparison interface consistent even when a transform removed a
    dangling register.
    """
    nl.driver("__force_build__")
    driver = nl._driver
    needed = set(nl.po)
    stack = list(nl.po)
    live_q = set()
    while stack:
        n = stack.pop()
        drv = driver.get(n)
        if drv is None:
            continue
        if drv[0] == "gate":
            for i in drv[1].ins:
                if i not in needed:
                    needed.add(i)
                    stack.append(i)
        elif drv[0] == "dff":
            ff = drv[1]
            if ff.q not in live_q:
                live_q.add(ff.q)
            for i in (ff.d,):
                if i not in needed:
                    needed.add(i)
                    stack.append(i)
    return live_q


def to_blif(nl: Netlist, model: str = "top", register_cut: bool = True) -> str:
    lines: List[str] = [f".model {model}"]

    inputs = sorted(nl.pi)
    outputs = list(sorted(nl.po))

    q_nets: List[str] = []
    d_outputs: List[Tuple[str, str]] = []  # (d_net, output_label)
    out_set = set(outputs)

    if register_cut and nl.dffs:
        live = _live_registers(nl)
        # one D pseudo-output per live register, labelled by its Q (register
        # identity), so the interface matches across designs.
        seen_q = set()
        rep_d = {}
        for ff in nl.dffs:
            if ff.q in live and ff.q not in rep_d:
                rep_d[ff.q] = ff.d
        for q in sorted(rep_d):
            seen_q.add(q)
            q_nets.append(q)
            d_outputs.append((rep_d[q], q + "$D"))

    # assemble input list (PI + Q pseudo-inputs)
    all_inputs = inputs + [q for q in q_nets if q not in set(inputs)]

    # assemble output list
    all_outputs = list(outputs)
    # Q that is also a module output: buffer it so an input can be an output.
    buffered: List[Tuple[str, str]] = []
    q_in = set(all_inputs)
    final_outputs = []
    for o in all_outputs:
        if o in q_in:
            lbl = o + "$PO"
            buffered.append((o, lbl))
            final_outputs.append(lbl)
        else:
            final_outputs.append(o)
    for d, lbl in d_outputs:
        final_outputs.append(lbl)

    lines.append(".inputs " + " ".join(all_inputs) if all_inputs else ".inputs")
    lines.append(".outputs " + " ".join(final_outputs) if final_outputs else ".outputs")

    # constants
    lines.append(".names __const0")
    lines.append(".names __const1")
    lines.append("1")

    def ref(net: str) -> str:
        if net == "1'b0":
            return "__const0"
        if net == "1'b1":
            return "__const1"
        return net

    in_set = set(all_inputs)
    driven = set()
    for g in nl.gates:
        tt = _TT.get(g.type)
        if tt is None:
            continue
        ins = [ref(i) for i in g.ins]
        lines.append(".names " + " ".join(ins + [g.out]))
        lines.extend(tt)
        driven.add(g.out)

    # buffered Q->PO copies
    for src, lbl in buffered:
        lines.append(f".names {ref(src)} {lbl}")
        lines.append("1 1")
        driven.add(lbl)
    # D pseudo-outputs are aliases of the D net
    for d, lbl in d_outputs:
        lines.append(f".names {ref(d)} {lbl}")
        lines.append("1 1")
        driven.add(lbl)

    # Any output net that is undriven and not a PI -> tie to const0 so BLIF is
    # well-formed (matches an undriven dangling output).
    for o in final_outputs:
        if o not in driven and o not in in_set:
            lines.append(f".names {o}")  # constant 0

    lines.append(".end")
    return "\n".join(lines) + "\n"


def signal_blif(nl: Netlist, sig_a: str, sig_b: str,
                model: str = "miter_pair") -> str:
    """A BLIF exposing two internal signals as the only two outputs (for
    functional-equivalence checks of internal signals).  Inputs are the union
    fan-in (PIs + register Qs feeding either signal)."""
    from ..analysis import graph

    cone = graph.fanin_cone_nets(nl, [sig_a, sig_b])
    inputs = sorted((cone & (set(nl.pi) | {ff.q for ff in nl.dffs}))
                    | {n for n in cone if nl.driver(n)[0] in ("pi", "dff", "undriven")
                       and not is_const(n)})
    inputs = [i for i in inputs if not is_const(i)]

    lines = [f".model {model}", ".inputs " + " ".join(inputs),
             ".outputs OA OB", ".names __const0", ".names __const1", "1"]

    def ref(net):
        if net == "1'b0":
            return "__const0"
        if net == "1'b1":
            return "__const1"
        return net

    emitted = set()
    for g in nl.gates:
        if g.out in cone and g.out not in emitted:
            tt = _TT.get(g.type)
            if tt is None:
                continue
            lines.append(".names " + " ".join([ref(i) for i in g.ins] + [g.out]))
            lines.extend(tt)
            emitted.add(g.out)
    for sig, lbl in ((sig_a, "OA"), (sig_b, "OB")):
        lines.append(f".names {ref(sig)} {lbl}")
        lines.append("1 1")
    lines.append(".end")
    return "\n".join(lines) + "\n"
