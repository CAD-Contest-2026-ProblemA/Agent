"""Constant propagation / constant-input simplification.

Original benchmark designs have no constant-driven combinational inputs (only
flip-flop .RN/.SN pins carry constants), so on a freshly loaded design these
queries correctly report 0.  After earlier transforms a gate may acquire a
constant input, so everything is computed on the *current* state.

Simplification rewrites each gate through a substitution map (net -> constant
or net), to a fixpoint, then drops gates whose output became a constant or a
direct wire and rewires every consumer (gates, flip-flops, primary outputs).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from ..netlist.ir import Gate, Netlist, is_const

C0, C1 = "1'b0", "1'b1"


def gates_with_const_input(nl: Netlist, gtype: Optional[str] = None,
                           const_val: Optional[str] = None) -> List[Gate]:
    res = []
    for g in nl.gates:
        if gtype is not None and g.type != gtype:
            continue
        consts = [i for i in g.ins if is_const(i)]
        if not consts:
            continue
        if const_val is not None and const_val not in consts:
            continue
        res.append(g)
    return res


def _simplify_one(gtype: str, ins: List[str]) -> Optional[str]:
    """Return a replacement expression for a gate with (possibly) const inputs:
    a constant net, a single net (wire/inverter handled by caller), or one of
    the markers ('NOT', x).  Returns None if no simplification applies.
    """
    if gtype in ("buf",):
        return ins[0]
    if gtype == "not":
        if ins[0] == C0:
            return C1
        if ins[0] == C1:
            return C0
        return None
    a, b = ins[0], ins[1]
    ca, cb = is_const(a), is_const(b)
    if not ca and not cb:
        return None
    if gtype == "and":
        if a == C0 or b == C0:
            return C0
        if a == C1:
            return b
        if b == C1:
            return a
    elif gtype == "or":
        if a == C1 or b == C1:
            return C1
        if a == C0:
            return b
        if b == C0:
            return a
    elif gtype == "nand":
        if a == C0 or b == C0:
            return C1
        if a == C1:
            return ("NOT", b)
        if b == C1:
            return ("NOT", a)
    elif gtype == "nor":
        if a == C1 or b == C1:
            return C0
        if a == C0:
            return ("NOT", b)
        if b == C0:
            return ("NOT", a)
    elif gtype == "xor":
        if a == C0:
            return b
        if b == C0:
            return a
        if a == C1:
            return ("NOT", b)
        if b == C1:
            return ("NOT", a)
    elif gtype == "xnor":
        if a == C1:
            return b
        if b == C1:
            return a
        if a == C0:
            return ("NOT", b)
        if b == C0:
            return ("NOT", a)
    return None


def const_propagate(nl: Netlist, restrict_type: Optional[str] = None) -> int:
    """Propagate constants to a fixpoint.  Returns the number of gates
    eliminated (output became a constant or a direct wire)."""
    subst: Dict[str, str] = {}

    def resolve(net: str) -> str:
        seen = 0
        while net in subst and seen < 1_000_000:
            net = subst[net]
            seen += 1
        return net

    changed = True
    eliminated = 0
    eliminated_by_type: Dict[str, int] = {}
    # We iterate to fixpoint; gates that turn into inverters stay (as NOT).
    inverter_outs: Dict[str, str] = {}   # out -> input (for NOT replacement)
    removed: Set[int] = set()

    while changed:
        changed = False
        for idx, g in enumerate(nl.gates):
            if idx in removed:
                continue
            rins = [resolve(i) for i in g.ins]
            res = _simplify_one(g.type, rins)
            if res is None:
                # update gate inputs in case substitution changed them
                if rins != g.ins:
                    g.ins = rins
                continue
            if isinstance(res, tuple):  # ('NOT', x) -> becomes inverter
                if g.type == "not":
                    continue
                orig_type = g.type
                g.type = "not"
                g.ins = [resolve(res[1])]
                eliminated += 1
                eliminated_by_type[orig_type] = eliminated_by_type.get(orig_type, 0) + 1
                changed = True
            else:  # constant or wire
                subst[g.out] = res
                removed.add(idx)
                eliminated += 1
                eliminated_by_type[g.type] = eliminated_by_type.get(g.type, 0) + 1
                changed = True

    # rebuild gate list, applying substitutions to surviving gates
    new_gates = []
    for idx, g in enumerate(nl.gates):
        if idx in removed:
            continue
        g.ins = [resolve(i) for i in g.ins]
        new_gates.append(g)
    nl.gates = new_gates

    # rewire flip-flops
    for ff in nl.dffs:
        ff.d = resolve(ff.d)
        ff.clk = resolve(ff.clk)
        ff.rn = resolve(ff.rn)
        ff.sn = resolve(ff.sn)

    nl.touch()
    nl._consts_eliminated_by_type = eliminated_by_type  # type: ignore
    if restrict_type is not None:
        return eliminated_by_type.get(restrict_type, 0)
    return eliminated
