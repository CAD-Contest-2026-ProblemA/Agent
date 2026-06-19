"""Logic-cone queries (fan-in / fan-out cones, shared cones).

DFF-Q convention (per official Q&A A21.2): fan-in cones are *combinational
only* — a flip-flop's Q output is treated as a primary input, so the cone of a
net terminates at DFF.Q.  If a queried output net is itself a DFF.Q, its
combinational fan-in cone is therefore empty.  ``redirect_q`` (default False)
can optionally redirect a Q-net query to the D-side next-state cone, but the
default matches the official "treat DFF.Q as a primary input" rule.
"""

from __future__ import annotations

from typing import List, Set

from ..netlist.ir import Netlist
from . import graph


def effective_sinks(nl: Netlist, net: str, redirect_q: bool = False) -> List[str]:
    drv = nl.driver(net)
    if drv[0] == "gate":
        return [net]
    if drv[0] == "dff" and redirect_q:
        # gather every D feeding a DFF whose Q is this net (handles multi-driver)
        ds = [ff.d for ff in nl.dffs if ff.q == net]
        return ds or [net]
    return [net]


def fanin_cone_nets(nl: Netlist, net: str, redirect_q: bool = False) -> Set[str]:
    return graph.fanin_cone_nets(nl, effective_sinks(nl, net, redirect_q))


def fanin_cone_gates(nl: Netlist, net: str, redirect_q: bool = False):
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


def largest_fanin_output(nl: Netlist, redirect_q: bool = False):
    """Return (output_net, gate_count) for the PO with the biggest fan-in cone."""
    best = None
    for po in sorted(nl.po):
        n = len(fanin_cone_gates(nl, po, redirect_q))
        if best is None or n > best[1]:
            best = (po, n)
    return best if best else (None, 0)
