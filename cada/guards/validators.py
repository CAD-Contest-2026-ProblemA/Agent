"""Hard-requirement validators run before committing a structural transform.

A transform is committed only if it (a) stays functionally equivalent, (b)
respects the requested gate basis, and (c) respects any structural bound
(max-fanout / max-depth).  Equivalence is decisive: an explicit ``False`` from
the equivalence gate triggers a rollback; an undecidable ``None`` trusts the
by-construction correctness of the template transforms (the gate only returns
None when neither ABC nor yosys could run).
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..netlist.ir import Netlist
from ..transform.rewrite import BASES
from ..equiv import gate as equiv_gate
from ..analysis import connectivity, depth as depth_mod


def check(before: Netlist, after: Netlist, *,
          basis: Optional[str] = None,
          max_fanout: Optional[int] = None,
          max_fanout_pi: bool = True,
          max_depth: Optional[int] = None,
          verify_equiv: bool = True,
          timeout: int = 280) -> Tuple[bool, str]:
    if basis:
        want = BASES.get(basis, set())
        bad = {g.type for g in after.gates if g.type not in want}
        if bad:
            return False, f"basis violation: found {sorted(bad)} outside {basis}"

    if max_fanout is not None:
        # "no gate drives more than K" bounds gate outputs AND DFF Q outputs --
        # a flip-flop is one of the nine primitive gate types (Q&A A2).  Only
        # the broader "no signal/net..." adds primary inputs, which are nets and
        # not gates.  This must match buffering.limit_fanout: when the guard was
        # the looser of the two it silently passed a netlist the transform had
        # left over the bound.
        mx = 0
        for g in after.gates:
            mx = max(mx, connectivity.fanout_count(after, g.out, include_po=False))
        for ff in after.dffs:
            mx = max(mx, connectivity.fanout_count(after, ff.q, include_po=False))
        if max_fanout_pi:
            for p in after.pi:
                mx = max(mx, connectivity.fanout_count(after, p, include_po=False))
        if mx > max_fanout:
            return False, f"max-fanout {mx} exceeds {max_fanout}"

    if max_depth is not None:
        if depth_mod.global_max_depth(after) > max_depth:
            return False, "max-depth bound exceeded"

    if verify_equiv:
        eq = equiv_gate.equivalent(before, after, timeout=timeout)
        if eq is False:
            return False, "functional equivalence violated"

    return True, "ok"
