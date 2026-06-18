"""Gate counting / reporting queries."""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

from ..netlist.ir import Netlist
from . import cones

GATE_ORDER = ["and", "or", "not", "nand", "nor", "xor", "xnor", "buf", "dff"]
LABEL = {"and": "AND", "or": "OR", "not": "NOT", "nand": "NAND", "nor": "NOR",
         "xor": "XOR", "xnor": "XNOR", "buf": "BUF", "dff": "DFF"}


def type_counts(nl: Netlist) -> dict:
    return nl.type_counts()


def total_gate_count(nl: Netlist, include_dff: bool = True) -> int:
    n = len(nl.gates)
    if include_dff:
        n += len(nl.dffs)
    return n


def count_breakdown_text(nl: Netlist) -> str:
    c = type_counts(nl)
    total = sum(c.get(t, 0) for t in GATE_ORDER)
    parts = [f"{LABEL[t]}: {c.get(t, 0)}" for t in GATE_ORDER]
    return ("Total gate count: %d\n" % total) + "\n".join(parts)


def count_of_type(nl: Netlist, gtype: str) -> int:
    gtype = gtype.lower()
    if gtype == "dff":
        return len(nl.dffs)
    return sum(1 for g in nl.gates if g.type == gtype)


def cone_type_counts(nl: Netlist, output: str) -> Counter:
    c = Counter()
    for g in cones.fanin_cone_gates(nl, output):
        c[g.type] += 1
    return c


def cone_type_counts_text(nl: Netlist, output: str) -> str:
    c = cone_type_counts(nl, output)
    total = sum(c.values())
    lines = [f"Gate-type counts in the cone of {output} (total {total}):"]
    for t in GATE_ORDER:
        if t == "dff":
            continue
        lines.append(f"  {LABEL[t]}: {c.get(t, 0)}")
    return "\n".join(lines)


def gates_in_fanin_cone(nl: Netlist, output: str) -> int:
    return len(cones.fanin_cone_gates(nl, output))


def list_gates_of_type(nl: Netlist, gtype: str) -> List:
    gtype = gtype.lower()
    return [g for g in nl.gates if g.type == gtype]
