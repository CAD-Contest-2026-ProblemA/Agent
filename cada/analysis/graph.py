"""Shared combinational-graph utilities.

The *combinational* graph treats a flip-flop as a boundary: a DFF's Q net is a
combinational source (depth 0, like a primary input) and its D net is a
combinational sink (like a primary output).  This keeps every traversal acyclic
even for sequential designs.

Depth convention: one gate == one level (BUF and NOT included); a net that is
itself a source has depth 0, so a primary input wired straight to a primary
output is depth 0.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set

from ..netlist.ir import Netlist


def comb_sources(nl: Netlist) -> Set[str]:
    """PIs + DFF Q nets + constants (everything that has no comb fan-in)."""
    s = set(nl.pi)
    for ff in nl.dffs:
        s.add(ff.q)
    s.add("1'b0")
    s.add("1'b1")
    return s


def comb_sinks(nl: Netlist) -> Set[str]:
    """POs + DFF D nets."""
    s = set(nl.po)
    for ff in nl.dffs:
        s.add(ff.d)
    return s


def gate_inputs(nl: Netlist, net: str) -> List[str]:
    """Combinational predecessors of ``net`` (empty if it is a source)."""
    drv = nl.driver(net)
    if drv[0] == "gate":
        return drv[1].ins
    return []


def topo_nets(nl: Netlist) -> List[str]:
    """All nets in combinational topological order (sources first).

    Uses Kahn's algorithm on the gate fan-in DAG.  Nets driven by a DFF or that
    are PIs/constants are sources.
    """
    nl.driver("__force_build__")  # ensure conn indices exist
    driver = nl._driver
    loads = nl._loads

    # in-degree = number of distinct combinational predecessors
    indeg: Dict[str, int] = {}
    nets: Set[str] = set()
    for g in nl.gates:
        nets.add(g.out)
        preds = set(g.ins)
        indeg[g.out] = len(preds)
        for i in preds:
            nets.add(i)
    # sources / nets that are never a gate output have indeg 0
    for n in nets:
        indeg.setdefault(n, 0)
    # DFF Q nets and PIs are sources even if listed: force indeg 0
    for ff in nl.dffs:
        indeg[ff.q] = 0
        nets.add(ff.q)
    for p in nl.pi:
        indeg[p] = 0
        nets.add(p)

    q = deque(n for n in nets if indeg[n] == 0)
    order: List[str] = []
    # track remaining preds per net
    remaining = dict(indeg)
    while q:
        n = q.popleft()
        order.append(n)
        for consumer in loads.get(n, []):
            if consumer[0] != "gate":
                continue
            out = consumer[1].out
            # decrement once per predecessor edge; but a gate may use n twice
            remaining[out] -= 1
            if remaining[out] == 0:
                q.append(out)
    # any nets left (shouldn't happen in a DAG) appended at end
    if len(order) < len(nets):
        for n in nets:
            if n not in order:
                order.append(n)
    return order


def forward_levels(nl: Netlist) -> Dict[str, int]:
    """Global longest-path depth of every net (max over all sources)."""
    nl.driver("__force_build__")
    driver = nl._driver
    memo: Dict[str, int] = {}
    for start in list(_all_nets_for_depth(nl)):
        if start in memo:
            continue
        stack = [(start, False)]
        while stack:
            x, done = stack.pop()
            if done:
                drv = driver.get(x)
                if drv is not None and drv[0] == "gate":
                    memo[x] = 1 + max((memo[i] for i in drv[1].ins), default=0)
                else:
                    memo[x] = 0
                continue
            if x in memo:
                continue
            drv = driver.get(x)
            if drv is not None and drv[0] == "gate":
                stack.append((x, True))
                for i in drv[1].ins:
                    if i not in memo:
                        stack.append((i, False))
            else:
                memo[x] = 0
    return memo


def _all_nets_for_depth(nl: Netlist):
    for g in nl.gates:
        yield g.out
    for ff in nl.dffs:
        yield ff.d
    for p in nl.po:
        yield p


def longest_from(nl: Netlist, src: str,
                 topo: Optional[List[str]] = None) -> Dict[str, int]:
    """Longest path length (number of gates) from ``src`` to every reachable net."""
    if topo is None:
        topo = topo_nets(nl)
    nl.driver("__force_build__")
    driver = nl._driver
    dist: Dict[str, int] = {src: 0}
    for net in topo:
        drv = driver.get(net)
        if drv is None or drv[0] != "gate":
            continue
        best = None
        for i in drv[1].ins:
            if i in dist:
                cand = dist[i] + 1
                if best is None or cand > best:
                    best = cand
        if best is not None and (net not in dist or best > dist[net]):
            dist[net] = best
    return dist


def reachable_forward(nl: Netlist, srcs: Iterable[str],
                      avoid: Optional[Set[str]] = None) -> Set[str]:
    """Nets reachable downstream from ``srcs`` (through gate fan-out)."""
    avoid = avoid or set()
    seen: Set[str] = set()
    stack = [s for s in srcs if s not in avoid]
    nl.driver("__force_build__")
    loads = nl._loads
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        for consumer in loads.get(n, []):
            if consumer[0] == "gate":
                out = consumer[1].out
                if out not in seen and out not in avoid:
                    stack.append(out)
    return seen


def fanin_cone_nets(nl: Netlist, sinks: Iterable[str],
                    avoid: Optional[Set[str]] = None) -> Set[str]:
    """All nets in the transitive fan-in (combinational) of ``sinks``."""
    avoid = avoid or set()
    seen: Set[str] = set()
    stack = [s for s in sinks if s not in avoid]
    while stack:
        n = stack.pop()
        if n in seen or n in avoid:
            continue
        seen.add(n)
        for i in gate_inputs(nl, n):
            if i not in seen:
                stack.append(i)
    return seen


def fanin_cone_gates(nl: Netlist, sinks: Iterable[str]):
    """Gates whose output lies in the fan-in cone of ``sinks``."""
    nets = fanin_cone_nets(nl, sinks)
    return [g for g in nl.gates if g.out in nets]


def fanout_cone_gates(nl: Netlist, srcs: Iterable[str]):
    nets = reachable_forward(nl, srcs)
    return [g for g in nl.gates if g.out in nets]
