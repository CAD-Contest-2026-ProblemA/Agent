"""Fan-out / connectivity queries."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..netlist.ir import Netlist
from . import graph


def _consumer_gate_names(nl: Netlist, net: str) -> List[str]:
    names = []
    seen = set()
    for c in nl.loads(net):
        inst = c[1]
        nm = inst.name
        if nm not in seen:
            seen.add(nm)
            names.append(nm)
    return names


def fanout_count(nl: Netlist, net: str, include_po: bool = True) -> int:
    """Number of load connections on this net.

    Counts gate input pins and flip-flop pins (D/CK/RN/SN); a gate using a net
    twice counts twice.  Per official Q&A A29 a primary-output connection is
    also a fan-out load, so ``include_po`` (default True) adds one when the net
    is a primary output.  The buffer-insertion guard passes include_po=False
    (it bounds re-routable gate fan-out only).
    """
    n = len(nl.loads(net))
    if include_po and net in nl.po:
        n += 1
    return n


def fanout_load_instances(nl: Netlist, net: str) -> List[str]:
    """Distinct instance names that ``net`` drives directly."""
    return _consumer_gate_names(nl, net)


def gates_driven_by_gate(nl: Netlist, gate_name: str) -> Optional[List[str]]:
    g = nl.gate_by_name(gate_name)
    if g is None:
        return None
    return _consumer_gate_names(nl, g.out)


def immediate_successors(nl: Netlist, inst_name: str) -> Optional[List[str]]:
    inst = nl.instance_by_name(inst_name)
    if inst is None:
        return None
    out = inst[1].out if inst[0] == "gate" else inst[1].q
    return _consumer_gate_names(nl, out)


def highest_fanout_pi(nl: Netlist) -> Tuple[Optional[str], int]:
    best = None
    for p in sorted(nl.pi):
        f = fanout_count(nl, p)
        if best is None or f > best[1]:
            best = (p, f)
    return best if best else (None, 0)


def max_fanout_of(nl: Netlist, base_or_net: str) -> int:
    """Max fanout across a net or, if a bus base name, across all its bits."""
    p = nl.ports.get(base_or_net)
    if p is not None and p.is_bus:
        return max((fanout_count(nl, b) for b in p.bits()), default=0)
    return fanout_count(nl, base_or_net)


def reachable_gates_from(nl: Netlist, net: str) -> List[str]:
    """Instances reachable downstream of ``net``.

    Flip-flops count: a DFF is one of the nine primitive gate types (Q&A A2)
    and its D/CK/RN/SN pins are ordinary loads (Q&A A29), so a flip-flop is
    reached as soon as the traversal lands on any of its input pins.  The
    search itself still stops there -- it does not continue out of Q, which
    would cross the combinational boundary (Q&A A21.2).
    """
    nets = graph.reachable_forward(nl, [net])
    out = []
    for g in nl.gates:
        if g.out in nets:
            out.append(g.name)
    for ff in nl.dffs:
        if ff.d in nets or ff.clk in nets or ff.rn in nets or ff.sn in nets:
            out.append(ff.name)
    return out
