"""Buffer insertion for fan-out limiting.

* ``limit_fanout`` builds a *minimal* balanced buffer tree (combine K sinks
  under one buffer at a time, which is the minimum number of buffers needed to
  bound every driver to K loads) — good for the gate-count-ranked cases.
* ``dedicated_buffer_per_load`` inserts exactly one buffer per load.

Both preserve functionality (a buffer is the identity).
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional, Set

from ..netlist.ir import Gate, Netlist, is_const
from .base import Emitter


def _pin_setter(inst_tuple) -> Callable[[str], None]:
    kind, inst, pin = inst_tuple

    def setter(net: str):
        if kind == "gate":
            inst.ins[pin] = net
        else:  # dff
            if pin == 0:
                inst.d = net
            elif pin == 1:
                inst.clk = net
            elif pin == 2:
                inst.rn = net
            elif pin == 3:
                inst.sn = net
    return setter


def _buffer_one_net(nl: Netlist, em: Emitter, source: str, k: int) -> int:
    loads = nl.loads(source)
    if len(loads) <= k:
        return 0
    queue: deque = deque(_pin_setter(l) for l in loads)
    buffers = 0
    while len(queue) > k:
        chunk = [queue.popleft() for _ in range(k)]
        bnet = em.fresh_wire()
        bgate = Gate("buf", em.fresh_gate_name(), bnet, ["__tbd__"])
        em.out.append(bgate)
        buffers += 1
        for setter in chunk:
            setter(bnet)
        # the buffer itself becomes a sink needing a parent
        queue.append(lambda net, g=bgate: g.ins.__setitem__(0, net))
    for setter in queue:
        setter(source)
    return buffers


def limit_fanout(nl: Netlist, k: int, include_pi: bool = False,
                 only_nets: Optional[Set[str]] = None) -> int:
    """Ensure no driver fans out to more than ``k`` loads.  Returns #buffers.

    Gate outputs and DFF Q outputs are always bounded: a flip-flop is one of
    the nine primitive gate types (Q&A A2), so "no gate drives more than K
    loads" covers its Q just as it covers an AND's output.

    ``include_pi`` additionally bounds primary inputs, which is the broader
    "no signal/net drives more than K" phrasing -- a PI is a net but not a
    gate, so it is in scope only there.  Constants are exempt.
    """
    em = Emitter(nl)

    # snapshot the set of driver nets to process (new buffer nets are <=k by
    # construction, so we don't need to revisit them)
    targets: List[str] = []
    if only_nets is not None:
        targets = [n for n in only_nets if not is_const(n)]
    else:
        seen: Set[str] = set()

        def add(net: str) -> None:
            if net not in seen and not is_const(net):
                seen.add(net)
                targets.append(net)

        for g in nl.gates:
            add(g.out)
        for q in sorted({ff.q for ff in nl.dffs}):
            add(q)
        if include_pi:
            for p in sorted(nl.pi):
                add(p)
    total = 0
    for net in targets:
        total += _buffer_one_net(nl, em, net, k)
    nl.gates.extend(em.out)
    nl.touch()
    return total


def dedicated_buffer_per_load(nl: Netlist, source: str) -> int:
    """Insert one dedicated buffer for every load of ``source``."""
    em = Emitter(nl)
    loads = nl.loads(source)
    count = 0
    for l in loads:
        bnet = em.fresh_wire()
        em.out.append(Gate("buf", em.fresh_gate_name(), bnet, [source]))
        _pin_setter(l)(bnet)
        count += 1
    nl.gates.extend(em.out)
    nl.touch()
    return count
