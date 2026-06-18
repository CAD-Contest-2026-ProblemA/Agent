"""Sequential analysis: clock domains, register-to-register paths, and
enable/hold structure detection on flip-flop D inputs."""

from __future__ import annotations

from typing import List, Tuple

from ..netlist.ir import Netlist
from . import graph, cones


def ffs_on_clock(nl: Netlist, clk: str) -> List:
    return [ff for ff in nl.dffs if ff.clk == clk]


def same_clock_domain(nl: Netlist, a_name: str, b_name: str):
    fa = nl.dff_by_name(a_name)
    fb = nl.dff_by_name(b_name)
    if fa is None or fb is None:
        return None
    return fa.clk == fb.clk


def reg_to_reg_pairs(nl: Netlist, limit: int = 200) -> Tuple[int, List[Tuple[str, str]]]:
    """Pairs (src_ff, dst_ff) where src.Q reaches dst.D through comb logic."""
    q_to_ffs = {}
    for ff in nl.dffs:
        q_to_ffs.setdefault(ff.q, []).append(ff)
    pairs = []
    count = 0
    for dst in nl.dffs:
        cone = cones.transitive_fanin(nl, dst.d)
        # iterate in a deterministic order (set iteration order depends on
        # PYTHONHASHSEED, which would make the reported examples non-reproducible)
        for q in sorted(cone):
            if q in q_to_ffs:
                for src in q_to_ffs[q]:
                    count += 1
                    if len(pairs) < limit:
                        pairs.append((src.name, dst.name))
    return count, pairs


def enable_hold_ffs(nl: Netlist) -> List:
    """Flip-flops whose next-state logic depends on their own current state
    (a hold/enable structure: the Q feeds back into its own D cone)."""
    res = []
    for ff in nl.dffs:
        cone = cones.transitive_fanin(nl, ff.d)
        if ff.q in cone:
            res.append(ff)
    return res
