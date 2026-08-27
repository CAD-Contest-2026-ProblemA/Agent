"""Logic-cone queries (fan-in / fan-out cones, shared cones).

DFF-Q convention.  Official Q&A A21.2 says "combinational depth only; treat
DFF.Q outputs as primary inputs" and A50 restates it as a general rule ("DFF.Q
signals are treated as pseudo-PIs"), so a flip-flop whose Q feeds a fan-in cone
is that cone's *source*, dropped for exactly the reason a primary input is.
A65 applies the same rule to the queried net itself: a PO driven straight off a
Q has an empty cone (0 gates).

The rule is NOT symmetric, and that is the thing to keep straight when editing
this module.  Walking backward the cone meets a register at Q -- a source, so
the register is outside.  Walking forward it meets D/CK/RN/SN, which A29 calls
ordinary fan-out loads -- something this net drives, so the register is
downstream and IS collected; see connectivity.downstream_instances().  No Q&A
item extends the source rule to the load side (searching all of Q1-Q70 for
"reachable", "transitive fanout" and "downstream" returns nothing), and doing
so would contradict A29.  Both walks still stop AT the register: neither
crosses from D to Q or from Q to D.

Load counting is a third question again, untouched by either: per A29
fanout_count() and the max-fanout constraint count D/CK/RN/SN pins through
nl.loads().  "What does this net drive?", "what does it reach?" and "what is
inside this cone?" have three different answers.

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
    """The COMBINATIONAL gates in the cone — the set a cone-scoped rewrite may
    touch, which excludes flip-flops precisely because a transform must never
    restructure one.

    For COUNTING and LISTING answers use :func:`fanin_cone_instances` instead:
    A94 rules that the flip-flops bounding the cone are counted as members.
    """
    nets = fanin_cone_nets(nl, net, redirect_q)
    return [g for g in nl.gates if g.out in nets]


def fanin_cone_instances(nl: Netlist, net: str):
    """(combinational gates, bounding flip-flops) of the cone, for answers.

    A94: the walk stops at every DFF.Q it meets, and each of those registers
    is counted inside the cone — including the driver when the queried net is
    itself a Q (A74/A94.3, where the cone is exactly that one DFF).  The
    combinational half is byte-identical to :func:`fanin_cone_gates`, so
    rewrite scoping and this counting view cannot drift apart.
    """
    nets = fanin_cone_nets(nl, net)
    gates = [g for g in nl.gates if g.out in nets]
    dffs = [ff for ff in sorted(nl.dffs, key=lambda f: f.name) if ff.q in nets]
    return gates, dffs


def transitive_fanin(nl: Netlist, net: str) -> Set[str]:
    return graph.fanin_cone_nets(nl, [net])


def fanout_cone_gates(nl: Netlist, net: str):
    from . import connectivity
    return connectivity.downstream_instances(nl, net)


def shared_fanin_gates(nl: Netlist, a: str, b: str):
    ga_g, ga_d = fanin_cone_instances(nl, a)
    gb_g, gb_d = fanin_cone_instances(nl, b)
    ga = {g.name for g in ga_g} | {ff.name for ff in ga_d}
    gb = {g.name for g in gb_g} | {ff.name for ff in gb_d}
    shared = ga & gb
    return ([g for g in nl.gates if g.name in shared]
            + [ff for ff in nl.dffs if ff.name in shared])


def largest_fanin_outputs(nl: Netlist, redirect_q: bool = False):
    """Return (gate_count, [every PO reaching it]) for the biggest fan-in cone.

    All of them, because ties are ordinary rather than exceptional: outputs of
    one bus are usually built from the same logic, so several share the maximum
    and naming one of them answers "which output" only by accident.  The list
    is in sorted PO order, so the answer does not depend on iteration order.
    """
    best, winners = None, []
    for po in sorted(nl.po):
        gates, dffs = fanin_cone_instances(nl, po)
        n = len(gates) + len(dffs)
        if best is None or n > best:
            best, winners = n, [po]
        elif n == best:
            winners.append(po)
    return (best or 0), winners


def largest_fanin_output(nl: Netlist, redirect_q: bool = False):
    """First tied winner only.  Prefer :func:`largest_fanin_outputs`."""
    n, w = largest_fanin_outputs(nl, redirect_q)
    return (w[0] if w else None), n
