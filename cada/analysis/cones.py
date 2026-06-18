"""Logic-cone queries (fan-in / fan-out cones, shared cones).

DFF-Q convention: when a queried "output" net is actually a flip-flop's Q
(register output), its *combinational* fan-in cone is empty.  Because the
benchmark questions about "the cone of output X" are clearly about the logic
feeding that output, we redirect a Q-net query to the D-side next-state
cone(s).  This is controlled by ``redirect_q`` (default True).
"""

from __future__ import annotations

from typing import List, Set

from ..netlist.ir import Netlist
from . import graph


def effective_sinks(nl: Netlist, net: str, redirect_q: bool = True) -> List[str]:
    drv = nl.driver(net)
    if drv[0] == "gate":
        return [net]
    if drv[0] == "dff" and redirect_q:
        # gather every D feeding a DFF whose Q is this net (handles multi-driver)
        ds = [ff.d for ff in nl.dffs if ff.q == net]
        return ds or [net]
    return [net]


def fanin_cone_nets(nl: Netlist, net: str, redirect_q: bool = True) -> Set[str]:
    return graph.fanin_cone_nets(nl, effective_sinks(nl, net, redirect_q))


def fanin_cone_gates(nl: Netlist, net: str, redirect_q: bool = True):
    nets = fanin_cone_nets(nl, net, redirect_q)
    return [g for g in nl.gates if g.out in nets]


def transitive_fanin(nl: Netlist, net: str) -> Set[str]:
    return graph.fanin_cone_nets(nl, [net])


def transitive_fanout(nl: Netlist, net: str) -> Set[str]:
    return graph.reachable_forward(nl, [net])


def fanout_cone_gates(nl: Netlist, net: str):
    nets = transitive_fanout(nl, net)
    return [g for g in nl.gates if g.out in nets or any(i == net for i in g.ins)]


def shared_fanin_gates(nl: Netlist, a: str, b: str):
    ga = {g.name for g in fanin_cone_gates(nl, a)}
    gb = {g.name for g in fanin_cone_gates(nl, b)}
    shared = ga & gb
    return [g for g in nl.gates if g.name in shared]


def largest_fanin_output(nl: Netlist, redirect_q: bool = True):
    """Return (output_net, gate_count) for the PO with the biggest fan-in cone."""
    best = None
    for po in sorted(nl.po):
        n = len(fanin_cone_gates(nl, po, redirect_q))
        if best is None or n > best[1]:
            best = (po, n)
    return best if best else (None, 0)
