"""Path queries — counted/analysed, never blindly enumerated.

Combinational paths only (flip-flops are boundaries).  A path from net A to
net B is a chain of gates A -> ... -> B.  Because the comb graph is a DAG we
can:

* test existence (optionally avoiding a node) with a BFS,
* count paths exactly with a topological DP (counts can be astronomically
  large; Python big ints stay exact),
* find mandatory vertices (dominators / articulation points) using the
  identity  #paths(A->B through X) = #paths(A->X) * #paths(X->B), so X is on
  every A->B path iff that product equals the total.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from ..netlist.ir import Netlist
from . import graph

ENUM_LIMIT = 2000  # materialise paths only when there are at most this many


def path_exists(nl: Netlist, a: str, b: str,
                avoid: Optional[Set[str]] = None) -> bool:
    if a in (avoid or set()) or b in (avoid or set()):
        return False
    reach = graph.reachable_forward(nl, [a], avoid=avoid)
    return b in reach


def _forward_counts(nl: Netlist, a: str, topo: List[str]) -> Dict[str, int]:
    """#paths from A to each net."""
    nl.driver("__force_build__")
    driver = nl._driver
    cnt: Dict[str, int] = {a: 1}
    for net in topo:
        if net == a:
            continue
        drv = driver.get(net)
        if drv is None or drv[0] != "gate":
            continue
        total = 0
        for i in drv[1].ins:
            c = cnt.get(i)
            if c:
                total += c
        if total:
            cnt[net] = total
    return cnt


def _backward_counts(nl: Netlist, b: str, topo: List[str]) -> Dict[str, int]:
    """#paths from each net to B."""
    nl.driver("__force_build__")
    loads = nl._loads
    cnt: Dict[str, int] = {b: 1}
    for net in reversed(topo):
        if net == b:
            continue
        total = 0
        for consumer in loads.get(net, []):
            if consumer[0] == "gate":
                c = cnt.get(consumer[1].out)
                if c:
                    total += c
        if total:
            cnt[net] = total
    return cnt


def count_paths(nl: Netlist, a: str, b: str) -> int:
    topo = graph.topo_nets(nl)
    cnt = _forward_counts(nl, a, topo)
    return cnt.get(b, 0)


def mandatory_gates(nl: Netlist, a: str, b: str) -> List[str]:
    """Gate instances whose output is on *every* A->B path (dominators).

    Same set as :func:`articulation_points`; kept as a name for the "does every
    path go through ...?" phrasing.
    """
    return articulation_points(nl, a, b)


def every_path_passes_through(nl: Netlist, a: str, b: str,
                              gate_name: str) -> Optional[bool]:
    g = nl.gate_by_name(gate_name)
    if g is None:
        return None
    if not path_exists(nl, a, b):
        return None  # caller reports "no path exists"
    return not path_exists(nl, a, b, avoid={g.out})


def articulation_points(nl: Netlist, a: str, b: str) -> List[str]:
    """Gate instances whose removal disconnects A from B, in topological order.

    Q&A A51 fixes the directed, pair-local reading: a cut is one whose removal
    breaks *this* A->B pair, not all PI-PO connectivity in the design.  A52
    excludes the endpoints themselves and A53 makes an unconnected pair report
    nothing.  The prompts ask which *gates* are articulation points, so the
    answer is gate instances -- reporting the nets instead names the wrong kind
    of object and misses one gate (see below).

    A gate's output net has a single driver, so exactly
    ``fwd[g.out] * bwd[g.out]`` of the A->B paths run through the gate, and it
    is an articulation point iff that equals the total path count.  Endpoint
    handling falls out of this: the gate driving B is traversed by every path
    and counts, while the gate driving A sits upstream of A and is traversed by
    none, so it is skipped explicitly.
    """
    topo = graph.topo_nets(nl)
    fwd = _forward_counts(nl, a, topo)
    bwd = _backward_counts(nl, b, topo)
    total = fwd.get(b, 0)
    if total == 0:
        return []                      # A53: no path -> no articulation points
    driver = nl._driver
    pts = []
    for net in topo:
        if net == a:                   # paths start at A; its driver is not on one
            continue
        drv = driver.get(net)
        if drv is None or drv[0] != "gate":
            continue
        f = fwd.get(net, 0)
        if f and f * bwd.get(net, 0) == total:
            pts.append(drv[1].name)
    return pts


def length_zero_paths(nl: Netlist) -> List[str]:
    """Direct PI->PO connections: a net that is both a primary input and a
    primary output (zero gates between them)."""
    return sorted(nl.pi & nl.po)


def is_cut_pi_po(nl: Netlist, wire: str) -> bool:
    """True iff removing ``wire`` disconnects at least one previously
    connected PI-to-PO pair (Q&A A51, option A — the pair-local reading).

    A pair (pi, po) is broken exactly when every pi->po path crosses the
    wire, so only PIs that reach the wire and POs the wire reaches can be
    part of a broken pair; other pairs keep their connectivity untouched.
    The earlier reading here — some PO losing ALL of its PI connectivity —
    was stricter than A51: a pair can be severed while its PO stays
    reachable from another input, and A51 counts that as a cut."""
    upstream = graph.fanin_cone_nets(nl, [wire])
    pis = [p for p in nl.pi if p in upstream or p == wire]
    if not pis:
        return False
    downstream_pos = graph.reachable_forward(nl, [wire]) & nl.po
    if not downstream_pos:
        return False
    for pi in pis:
        before = graph.reachable_forward(nl, [pi]) & downstream_pos
        if not before:
            continue
        after = graph.reachable_forward(nl, [pi], avoid={wire})
        if before - after:
            return True
    return False


def enumerate_paths(nl: Netlist, a: str, b: str,
                    limit: int = ENUM_LIMIT):
    """Return (count, paths_or_None).  Paths materialised only if count<=limit.

    Each path is a list of gate instance names from the gate just after A to the
    gate driving B.
    """
    count = count_paths(nl, a, b)
    if count == 0:
        return 0, []
    if count > limit:
        return count, None

    # Restrict the DFS to nets that can actually reach B (its fan-in cone),
    # otherwise the forward search is exponential on large designs even when
    # only a few paths reach B.
    reach_b = graph.fanin_cone_nets(nl, [b])
    paths: List[List[str]] = []
    nl.driver("__force_build__")
    loads = nl._loads
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 200000))

    def dfs(net: str, acc: List[str]):
        if net == b:
            paths.append(list(acc))
            return
        for consumer in loads.get(net, []):
            if consumer[0] != "gate":
                continue
            g = consumer[1]
            if g.out not in reach_b:
                continue
            acc.append(g.name)
            dfs(g.out, acc)
            acc.pop()

    dfs(a, [])
    return count, paths


def stream_paths(nl: Netlist, a: str, b: str, out_fh, cap: int) -> int:
    """Write up to ``cap`` A->B paths (one per line, gate sequence) to a file
    handle.  Pruned to nets that can reach B so the DFS stays productive.
    Returns the number of paths written."""
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 200000))
    reach_b = graph.fanin_cone_nets(nl, [b])
    nl.driver("__force_build__")
    loads = nl._loads
    written = [0]

    def dfs(net: str, acc: List[str]):
        if written[0] >= cap:
            return
        if net == b:
            out_fh.write((a + " -> " + " -> ".join(acc) + " -> " + b) if acc
                         else (a + " -> " + b))
            out_fh.write("\n")
            written[0] += 1
            return
        for consumer in loads.get(net, []):
            if consumer[0] != "gate":
                continue
            g = consumer[1]
            if g.out not in reach_b:
                continue
            acc.append(g.name)
            dfs(g.out, acc)
            acc.pop()
            if written[0] >= cap:
                return

    dfs(a, [])
    return written[0]
