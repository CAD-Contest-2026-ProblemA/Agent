"""Shared helpers for structural transforms.

A :class:`Emitter` accumulates freshly-built gates with deterministic names so
that rebuilding ``nl.gates`` is a single pass and never mutates the list while
iterating.
"""

from __future__ import annotations

from typing import List

from ..netlist.ir import Gate, Netlist


class Emitter:
    def __init__(self, nl: Netlist):
        self.nl = nl
        self.out: List[Gate] = []
        self.n_gates = 0

    def fresh_wire(self) -> str:
        return self.nl.fresh_name("cw")        # "cada wire"

    def fresh_gate_name(self) -> str:
        return self.nl.fresh_name("cg")        # "cada gate"

    def emit(self, gtype: str, out: str, ins: List[str]) -> str:
        self.out.append(Gate(gtype, self.fresh_gate_name(), out, list(ins)))
        self.n_gates += 1
        return out

    # convenience
    def not_(self, out, a):
        return self.emit("not", out, [a])

    def and_(self, out, a, b):
        return self.emit("and", out, [a, b])

    def or_(self, out, a, b):
        return self.emit("or", out, [a, b])

    def nand_(self, out, a, b):
        return self.emit("nand", out, [a, b])

    def nor_(self, out, a, b):
        return self.emit("nor", out, [a, b])


def rebuild_gates(nl: Netlist, keep_pred, replace_fn):
    """Rebuild ``nl.gates``: for each gate, if ``keep_pred(g)`` keep it,
    else replace it with the gates returned by ``replace_fn(g, emitter)``.

    Returns the number of original gates that were replaced.
    """
    em = Emitter(nl)
    new_gates: List[Gate] = []
    replaced = 0
    for g in nl.gates:
        if keep_pred(g):
            new_gates.append(g)
        else:
            replaced += 1
            em.out = []
            replace_fn(g, em)
            new_gates.extend(em.out)
    nl.gates = new_gates
    nl.touch()
    return replaced
