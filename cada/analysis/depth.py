"""Combinational depth queries (1 gate == 1 level)."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from ..netlist.ir import Netlist
from . import graph, cones


def longest_from_set(nl: Netlist, srcs: Set[str],
                     topo: Optional[List[str]] = None) -> Dict[str, int]:
    """Longest path length from the *set* of sources to every reachable net."""
    if topo is None:
        topo = graph.topo_nets(nl)
    nl.driver("__force_build__")
    driver = nl._driver
    dist: Dict[str, int] = {s: 0 for s in srcs}
    for net in topo:
        drv = driver.get(net)
        if drv is None or drv[0] != "gate":
            continue
        best = None
        for i in drv[1].ins:
            d = dist.get(i)
            if d is not None:
                c = d + 1
                if best is None or c > best:
                    best = c
        if best is not None:
            prev = dist.get(net)
            if prev is None or best > prev:
                dist[net] = best
    return dist


def global_max_depth(nl: Netlist) -> int:
    """Max combinational depth in the whole design (any source -> any sink)."""
    lv = graph.forward_levels(nl)
    sinks = graph.comb_sinks(nl)
    return max((lv.get(s, 0) for s in sinks), default=0)


def depth_of_cone(nl: Netlist, output: str, redirect_q: bool = False) -> int:
    lv = graph.forward_levels(nl)
    sinks = cones.effective_sinks(nl, output, redirect_q)
    return max((lv.get(s, 0) for s in sinks), default=0)


def max_depth_from_to(nl: Netlist, a: str, b: str,
                      redirect_q: bool = False) -> Optional[int]:
    dist = longest_from_set(nl, {a})
    sinks = cones.effective_sinks(nl, b, redirect_q)
    vals = [dist[s] for s in sinks if s in dist]
    return max(vals) if vals else None


def pi_to_po_max_depth(nl: Netlist) -> int:
    dist = longest_from_set(nl, set(nl.pi))
    return max((dist.get(p, -1) for p in nl.po), default=-1)


def pi_to_dff_d_max_depth(nl: Netlist) -> int:
    # "From any primary input" is strict: sources are the primary-input bits
    # only.  DFF.Q-sourced paths belong to the separate register-to-register
    # query (reg_to_reg_max_depth) — answering this question with Q sources
    # would just repeat that number.  Every gate on the path (including BUF)
    # counts one level.
    dist = longest_from_set(nl, set(nl.pi))
    ds = {ff.d for ff in nl.dffs}
    return max((dist.get(d, -1) for d in ds), default=-1)


def reg_to_reg_max_depth(nl: Netlist) -> int:
    """Longest combinational path from a register output (Q) to a register
    input (D)."""
    qs = {ff.q for ff in nl.dffs}
    ds = {ff.d for ff in nl.dffs}
    if not qs or not ds:
        return -1
    dist = longest_from_set(nl, qs)
    return max((dist.get(d, -1) for d in ds), default=-1)


def outputs_depth_greater_than(nl: Netlist, k: int) -> List[str]:
    lv = graph.forward_levels(nl)
    out = []
    for p in sorted(nl.po):
        sinks = cones.effective_sinks(nl, p)
        d = max((lv.get(s, 0) for s in sinks), default=0)
        if d > k:
            out.append(p)
    return out


def deepest_output(nl: Netlist, redirect_q: bool = False):
    lv = graph.forward_levels(nl)
    best = None
    for p in sorted(nl.po):
        sinks = cones.effective_sinks(nl, p, redirect_q)
        d = max((lv.get(s, 0) for s in sinks), default=0)
        if best is None or d > best[1]:
            best = (p, d)
    return best if best else (None, 0)


def gate_on_max_depth_path(nl: Netlist, gate_name: str) -> Optional[bool]:
    """Is the given gate on some globally-maximum-depth path?"""
    g = nl.gate_by_name(gate_name)
    if g is None:
        return None
    fwd = graph.forward_levels(nl)              # longest path ending at net
    gmax = global_max_depth(nl)
    # backward longest path (to a sink) from each net
    back = _backward_levels(nl)
    # the gate contributes level 1 (itself). depth ending at g.out is fwd[g.out];
    # remaining to a sink from g.out is back[g.out]. On a max path iff their sum
    # equals gmax (back includes the gate's own out as 0 distance to itself, fwd
    # includes it). fwd[out] + back[out] - 0 ... fwd counts gate, back counts
    # downstream gates after out. total path length through this gate:
    return fwd.get(g.out, 0) + back.get(g.out, 0) == gmax


def _backward_levels(nl: Netlist) -> Dict[str, int]:
    """Longest path length from each net forward to any comb sink."""
    topo = graph.topo_nets(nl)
    nl.driver("__force_build__")
    loads = nl._loads
    sinks = graph.comb_sinks(nl)
    back: Dict[str, int] = {}
    for net in reversed(topo):
        best = 0
        for consumer in loads.get(net, []):
            if consumer[0] == "gate":
                out = consumer[1].out
                best = max(best, 1 + back.get(out, 0))
        # if it's also a sink, distance-to-sink is 0 (already covered by best=0)
        back[net] = best
    return back
