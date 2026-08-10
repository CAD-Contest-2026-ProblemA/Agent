"""Fan-out / connectivity queries."""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

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
    """Highest-fanout *primary input* -- for "which PI has the highest fanout"."""
    best = None
    for p in sorted(nl.pi):
        f = fanout_count(nl, p)
        if best is None or f > best[1]:
            best = (p, f)
    return best if best else (None, 0)


def highest_fanout_net(nl: Netlist) -> Tuple[Optional[str], int]:
    """Highest-fanout net anywhere in the design.

    "Which signal drives the largest number of loads" ranges over every driven
    net -- gate outputs and DFF.Q as well as primary inputs -- and the winner is
    usually an internal net, not a PI.
    """
    nets = {g.out for g in nl.gates} | set(nl.pi) | {ff.q for ff in nl.dffs}
    best = None
    for n in sorted(nets):
        f = fanout_count(nl, n)
        if best is None or f > best[1]:
            best = (n, f)
    return best if best else (None, 0)


def max_fanout_of(nl: Netlist, base_or_net: str) -> int:
    """Max fanout across a net or, if a bus base name, across all its bits."""
    p = nl.ports.get(base_or_net)
    if p is not None and p.is_bus:
        return max((fanout_count(nl, b) for b in p.bits()), default=0)
    return fanout_count(nl, base_or_net)


def gates_within_hops(nl: Netlist, net: str, hops: int) -> List[str]:
    """Instances at most ``hops`` gate levels downstream of ``net``.

    ``hops=1`` is the immediate fanout.  Works from any net, primary inputs
    included -- the question is asked about a signal, not about a gate
    instance.  A flip-flop reached on the way is reported, but the walk does
    not continue out of its Q (Q&A A21.2 keeps the traversal combinational).
    """
    frontier = {net}
    names: List[str] = []
    seen: Set[str] = set()
    for _ in range(max(0, hops)):
        nxt: Set[str] = set()
        for n in frontier:
            for kind, inst, _pin in nl.loads(n):
                if inst.name not in seen:
                    seen.add(inst.name)
                    names.append(inst.name)
                if kind == "gate":
                    nxt.add(inst.out)
        frontier = nxt
    return names


def downstream_instances(nl: Netlist, net: str):
    """Instances downstream of ``net`` — THE definition of "what do I hit
    walking forward from this net".  Every downstream-walk answer
    (reachable_gates_from, cones.fanout_cone_gates) must go through here so
    the two phrasings can never diverge again.

    - A flip-flop IS collected once the walk lands on any of its input pins
      (D/CK/RN/SN).  It is deliberately NOT the mirror image of
      cones.fanin_cone_gates(), because the two walks meet a register at
      opposite ends and the Q&A treats those ends differently: backward the
      walk arrives at DFF.Q, which A21.2 and A50 make a pseudo-primary input
      -- a source, dropped for the same reason a PI is; forward it arrives at
      D/CK/RN/SN, which A29 calls ordinary fan-out loads, and a load this net
      drives is downstream by any reading.  Nothing in the Q&A extends the
      source rule to the load side; symmetry here would be an inference, and
      one that contradicts A29.
    - The walk stops at the register: it does not continue out of Q, which
      would cross the combinational boundary (A21.2).
    - The driver of ``net`` itself is NOT downstream: a gate cannot lie in
      the fan-out of its own output, so the seed net is excluded before
      matching gate outputs.  The seed stays in the set for the flip-flop
      check -- a DFF whose input pin IS ``net`` is a direct load.
    """
    nets = graph.reachable_forward(nl, [net])
    return ([g for g in nl.gates if g.out in nets - {net}]
            + [ff for ff in nl.dffs
               if ff.d in nets or ff.clk in nets or ff.rn in nets or ff.sn in nets])


def reachable_gates_from(nl: Netlist, net: str) -> List[str]:
    """Names of the instances reachable downstream of ``net``."""
    return [i.name for i in downstream_instances(nl, net)]
