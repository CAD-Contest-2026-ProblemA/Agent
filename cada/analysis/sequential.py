"""Sequential analysis: clock domains, register-to-register paths, and
enable/hold structure detection on flip-flop D inputs."""

from __future__ import annotations

from typing import List, Tuple

import time

from ..netlist.ir import Netlist
from . import graph, cones, functional
from .functional import _splitmix


def ffs_on_clock(nl: Netlist, clk: str) -> List:
    return [ff for ff in nl.dffs if ff.clk == clk]


def same_clock_domain(nl: Netlist, a_name: str, b_name: str):
    fa = nl.dff_by_name(a_name)
    fb = nl.dff_by_name(b_name)
    if fa is None or fb is None:
        return None
    return fa.clk == fb.clk


def reg_to_reg_path_count(nl: Netlist) -> int:
    """How many distinct combinational PATHS run from a DFF.Q to a DFF.D.

    Not the same question as :func:`reg_to_reg_pairs`, and the difference is
    large: one (src, dst) pair can be joined by many paths, so on real designs
    the two numbers differ by orders of magnitude.  "List all
    register-to-register paths" asks for this one.

    Counted by dynamic programming over the combinational fan-in of each D pin
    -- paths(net) = 1 at a DFF.Q, otherwise the sum over the driving gate's
    inputs -- so the count is exact and cheap even when the paths themselves
    are far too numerous to write down.  A reconvergent net is visited once and
    its subtotal reused, which is what keeps this linear where enumeration is
    exponential.
    """
    qs = {ff.q for ff in nl.dffs}
    nl.driver("__force_build__")
    driver = nl._driver
    memo = {}

    def count(net: str) -> int:
        if net in qs:
            return 1
        if net in memo:
            return memo[net]
        drv = driver.get(net)
        if drv is None or drv[0] != "gate":
            return 0
        memo[net] = 0                      # cycle guard; combinational loops
        memo[net] = sum(count(i) for i in drv[1].ins)
        return memo[net]

    return sum(count(ff.d) for ff in nl.dffs)


def stream_reg_to_reg_paths(nl: Netlist, out_fh, cap: int) -> int:
    """Write up to ``cap`` register-to-register paths, one per line.

    Each line is ``srcFF -> gate -> ... -> dstFF``, walking forward from every
    DFF.Q and stopping at the first DFF input pin reached -- the walk never
    crosses a register, matching the combinational-only reading of Q&A A21.2.

    Pruned to nets that can still reach some D pin, so the DFS does not
    wander into logic that only feeds primary outputs.  Returns the number of
    lines written.
    """
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 200000))
    nl.driver("__force_build__")
    loads = nl._loads

    d_pins = {}
    for ff in nl.dffs:
        d_pins.setdefault(ff.d, []).append(ff.name)
    productive = graph.fanin_cone_nets(nl, list(d_pins))
    written = [0]

    def dfs(net: str, src: str, acc: List[str]):
        if written[0] >= cap:
            return
        for dst in d_pins.get(net, ()):
            out_fh.write(src + " -> " + " -> ".join(acc + [dst]) if acc
                         else src + " -> " + dst)
            out_fh.write("\n")
            written[0] += 1
            if written[0] >= cap:
                return
        for consumer in loads.get(net, []):
            if consumer[0] != "gate":
                continue
            g = consumer[1]
            if g.out not in productive:
                continue
            acc.append(g.name)
            dfs(g.out, src, acc)
            acc.pop()
            if written[0] >= cap:
                return

    for ff in sorted(nl.dffs, key=lambda f: f.name):
        dfs(ff.q, ff.name, [])
        if written[0] >= cap:
            break
    return written[0]


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


def _sim_disproves_unate(nl: Netlist, d: str, q: str, words: int = 4) -> bool:
    """Did simulation *witness* D falling as Q rises?  A hit is a proof.

    Bit-parallel evaluation with Q forced low and then high over the same input
    words.  Finding a pattern where D is high at Q=0 and low at Q=1 settles the
    question -- that flip-flop is not a hold structure and needs no SAT call.
    Finding none proves nothing, so the caller must still confirm.

    Vectors come from splitmix64 rather than `random`, so repeated runs give
    the same answer instead of drifting.
    """
    mask = (1 << 64) - 1
    free: dict = {}
    drv = {g.out: g for g in nl.gates}

    def ev(net, qval, cache):
        if net == q:
            return qval
        hit = cache.get(net)
        if hit is not None:
            return hit
        g = drv.get(net)
        if g is None:                       # PI, another DFF's Q, or a constant
            if net == "1'b1":
                return mask
            if net == "1'b0":
                return 0
            if net not in free:
                free[net] = _splitmix(len(free) * words + seed) & mask
            return free[net]
        a = ev(g.ins[0], qval, cache)
        b = ev(g.ins[1], qval, cache) if len(g.ins) > 1 else 0
        t = g.type
        if t == "and":
            v = a & b
        elif t == "or":
            v = a | b
        elif t == "nand":
            v = mask ^ (a & b)
        elif t == "nor":
            v = mask ^ (a | b)
        elif t == "xor":
            v = a ^ b
        elif t == "xnor":
            v = mask ^ (a ^ b)
        elif t == "not":
            v = mask ^ a
        else:                                # buf and anything unmodelled
            v = a
        cache[net] = v
        return v

    for seed in range(words):
        free.clear()
        try:
            lo = ev(d, 0, {})                # the cache must not be shared:
            hi = ev(d, mask, {})             # every net is re-evaluated at Q=1
        except RecursionError:
            return False                     # can't judge here; leave it to SAT
        if lo & ~hi & mask:                  # D fell when Q rose
            return True
    return False


def enable_hold_ffs(nl: Netlist, budget: float = 240.0) -> List:
    """Flip-flops with an enable/hold structure on their D input.

    Q&A A46 settles that the structure is recognised by the *function* of D,
    not by matching a gate pattern, and A69.1 gives the criterion: D must be
    positive-unate in Q with true Q-dependence.  Any Q-free decomposition into
    EN/DATA counts, whether or not EN and DATA exist as nets in the netlist --
    under D = (EN & DATA) | (!EN & Q), raising Q can only raise D.

    Three stages, cheapest first:

      1. structural -- Q must reach its own D at all;
      2. simulation -- a witnessed violation is a proof of non-unateness and
         retires that candidate without a solver call;
      3. SAT       -- the exact test, on whatever survives.

    Simulation alone is not enough to *accept*: on test40 it settles at 1949
    against the true 1796, and the figure moves with the vector set (1878-1968
    across five of them) because more vectors both find more violations and
    detect more dependence, so the count does not converge.

    ``budget`` caps the solver stage.  If it runs out, the remaining candidates
    fall back to the simulation verdict rather than the whole query failing.
    """
    t0 = time.time()
    cand = [ff for ff in nl.dffs
            if ff.q in cones.transitive_fanin(nl, ff.d)
            and not _sim_disproves_unate(nl, ff.d, ff.q)]
    if not cand:
        return []

    unate = functional.cofactors_empty_batch(
        nl, [(ff.d, ff.q, "viol") for ff in cand])
    keep = [ff for ff, u in zip(cand, unate) if u is not False]

    if time.time() - t0 > budget or not keep:
        return keep                          # degrade to the simulation verdict

    dep = functional.cofactors_empty_batch(
        nl, [(ff.d, ff.q, "diff") for ff in keep])
    return [ff for ff, empty in zip(keep, dep) if empty is not True]
