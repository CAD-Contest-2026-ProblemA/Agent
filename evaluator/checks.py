"""Independent verification primitives used by the evaluator.

These re-derive the truth a *different* way from the agent where possible:
ABC for equivalence (external oracle), a raw-text regex for gate counts, and
recomputation from the resulting netlist for structural bounds / basis purity.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Set

from cada.netlist.ir import Netlist
from cada.netlist import reader, writer
from cada.equiv import gate as equiv_gate
from cada.transform.rewrite import BASES
from cada.analysis import cones

_GATE_RE = re.compile(r"^\s*(and|or|nand|nor|not|buf|xor|xnor|dff)\b", re.IGNORECASE)


def independent_type_counts(vfile: str) -> Dict[str, int]:
    """Count gate types by scanning the raw Verilog text (independent of the
    agent's parser)."""
    counts = {t: 0 for t in
              ("and", "or", "not", "nand", "nor", "xor", "xnor", "buf", "dff")}
    with open(vfile) as fh:
        for line in fh:
            m = _GATE_RE.match(line)
            if m:
                counts[m.group(1).lower()] += 1
    return counts


def response_counts(text: str) -> Dict[str, int]:
    """Extract 'AND: 12' style numbers from a response."""
    out = {}
    for m in re.finditer(r"\b(AND|OR|NOT|NAND|NOR|XOR|XNOR|BUF|DFF)\s*[:=]\s*(\d+)",
                         text, re.IGNORECASE):
        out[m.group(1).lower()] = int(m.group(2))
    return out


def independent_port_counts(vfile: str):
    """(num_input_ports, num_output_ports) from the declarations."""
    ni = no = 0
    with open(vfile) as fh:
        text = fh.read()
    for stmt in text.split(";"):
        s = stmt.strip()
        head = s.split(None, 1)[0] if s else ""
        if head == "input":
            ni += len(_decl_names(s))
        elif head == "output":
            no += len(_decl_names(s))
    return ni, no


def _decl_names(stmt: str):
    rest = re.sub(r"^\w+\s*", "", stmt)
    rest = re.sub(r"\[\s*\d+\s*:\s*\d+\s*\]", "", rest)
    return [n.strip() for n in rest.split(",") if n.strip()]


def equiv(a: Netlist, b: Netlist, timeout: int = 200) -> Optional[bool]:
    return equiv_gate.equivalent(a, b, timeout=timeout)


def max_fanout(nl: Netlist, include_pi: bool = False) -> int:
    mx = 0
    for g in nl.gates:
        mx = max(mx, len(nl.loads(g.out)))
    if include_pi:
        for p in nl.pi:
            mx = max(mx, len(nl.loads(p)))
    return mx


def fanout_of(nl: Netlist, net: str) -> int:
    return len(nl.loads(net))


def basis_types(nl: Netlist, scope_output: Optional[str] = None) -> Set[str]:
    if scope_output:
        gs = cones.fanin_cone_gates(nl, scope_output)
    else:
        gs = nl.gates
    return {g.type for g in gs}


def basis_pure(nl: Netlist, basis: str, scope_output: Optional[str] = None) -> bool:
    want = BASES.get(basis, set())
    return basis_types(nl, scope_output) <= want


def type_count_in(nl: Netlist, gtype: str, scope_output: Optional[str] = None) -> int:
    gs = cones.fanin_cone_gates(nl, scope_output) if scope_output else nl.gates
    return sum(1 for g in gs if g.type == gtype)


def valid_netlist(nl: Netlist) -> bool:
    """The design re-parses from our own writer with the same gate/dff sets."""
    try:
        txt = writer.to_string(nl)
        nl2 = reader.parse_text(txt)
    except Exception:
        return False
    a = sorted((g.type, g.out, tuple(g.ins)) for g in nl.gates)
    b = sorted((g.type, g.out, tuple(g.ins)) for g in nl2.gates)
    return a == b and len(nl.dffs) == len(nl2.dffs)
