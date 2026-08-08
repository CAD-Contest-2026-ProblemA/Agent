"""Logic-cone queries (fan-in / fan-out cones, shared cones).

DFF-Q convention.  Official Q&A A21.2 says "combinational depth only; treat
DFF.Q outputs as primary inputs", and the organisers' 2026-07-20 reply extends
that from depth to membership: DFF.Q counts as a primary input for cone
*contents* too, so a PO driven straight off a Q has an empty cone (0 gates).

The consequence, and the thing to keep straight when editing this module: a
flip-flop is a *boundary* of the combinational subgraph, not a member of it.
A cone stops at DFF.Q on the way back and at a DFF input pin on the way
forward, and in neither direction is the flip-flop itself inside.  That is the
same reason a primary input's driver is not in the cone.

This is about MEMBERSHIP only.  It does not touch load counting: per Q&A A29 a
DFF's D/CK/RN/SN pins are ordinary fan-out loads, so fanout_count() and the
max-fanout constraint still count them.  "What does this net drive?" and "what
is inside this cone?" are different questions with different answers.

``redirect_q`` (default False) can optionally redirect a Q-net query to the
D-side next-state cone, but the default matches the official rule.
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
    """The gates in the cone — every instance in it, and the only ones.

    Flip-flops whose Q feeds the cone bound it rather than belong to it, so
    they are not here; see the module docstring for the ruling.  This is both
    "how many gates are in the fan-in cone of X" and the set a cone-scoped
    rewrite may touch, which is the same set precisely because a DFF is
    outside the combinational subgraph and a transform must never restructure
    one.
    """
    nets = fanin_cone_nets(nl, net, redirect_q)
    return [g for g in nl.gates if g.out in nets]


def transitive_fanin(nl: Netlist, net: str) -> Set[str]:
    return graph.fanin_cone_nets(nl, [net])


def fanout_cone_gates(nl: Netlist, net: str):
    from . import connectivity
    return connectivity.downstream_instances(nl, net)


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
