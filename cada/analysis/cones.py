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
    """Combinational gates in the cone.

    Deliberately excludes flip-flops: callers that scope a *rewrite* to a cone
    use this, and a transform must never restructure a DFF.  Use
    :func:`fanin_cone_instances` for questions that count what is in the cone.
    """
    nets = fanin_cone_nets(nl, net, redirect_q)
    return [g for g in nl.gates if g.out in nets]


def fanin_cone_dffs(nl: Netlist, net: str, redirect_q: bool = False):
    """Flip-flops bounding the cone -- those whose Q feeds it.

    A flip-flop whose Q *is* the queried net is excluded: per Q&A A65 the cone
    of a register output is empty, that register being the boundary itself
    rather than something inside the cone.
    """
    nets = fanin_cone_nets(nl, net, redirect_q)
    sinks = set(effective_sinks(nl, net, redirect_q))
    return [ff for ff in nl.dffs if ff.q in nets and ff.q not in sinks]


def fanin_cone_instances(nl: Netlist, net: str, redirect_q: bool = False):
    """Every instance in the cone: combinational gates plus the boundary DFFs.

    This is what "how many gates are in the fan-in cone of X" asks for -- a
    flip-flop is one of the nine primitive gate types (Q&A A2), so it counts.
    """
    return (fanin_cone_gates(nl, net, redirect_q)
            + fanin_cone_dffs(nl, net, redirect_q))


def transitive_fanin(nl: Netlist, net: str) -> Set[str]:
    return graph.fanin_cone_nets(nl, [net])


def transitive_fanout(nl: Netlist, net: str) -> Set[str]:
    return graph.reachable_forward(nl, [net])


def fanout_cone_gates(nl: Netlist, net: str):
    nets = transitive_fanout(nl, net)
    return [g for g in nl.gates if g.out in nets or any(i == net for i in g.ins)]


def shared_fanin_gates(nl: Netlist, a: str, b: str):
    ia = {inst.name: inst for inst in fanin_cone_instances(nl, a)}
    ib = {inst.name for inst in fanin_cone_instances(nl, b)}
    return [inst for name, inst in ia.items() if name in ib]


def largest_fanin_output(nl: Netlist, redirect_q: bool = False):
    """Return (output_net, gate_count) for the PO with the biggest fan-in cone."""
    best = None
    for po in sorted(nl.po):
        n = len(fanin_cone_instances(nl, po, redirect_q))
        if best is None or n > best[1]:
            best = (po, n)
    return best if best else (None, 0)
