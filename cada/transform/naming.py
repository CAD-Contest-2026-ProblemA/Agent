"""Rename gates, wires, and signals, updating every reference.

Renaming is purely cosmetic and always functionality-preserving.  A *signal*
(net) rename rewrites the net everywhere it is used or driven; a *gate* rename
changes only the instance name.
"""

from __future__ import annotations

from typing import List

from ..netlist.ir import Netlist


def rename_gate(nl: Netlist, old: str, new: str) -> bool:
    g = nl.gate_by_name(old)
    if g is not None:
        g.name = new
        nl.touch()
        return True
    ff = nl.dff_by_name(old)
    if ff is not None:
        ff.name = new
        nl.touch()
        return True
    return False


def rename_net(nl: Netlist, old: str, new: str) -> bool:
    """Rename a wire/signal net everywhere (drivers, loads, ports)."""
    touched = False
    for g in nl.gates:
        if g.out == old:
            g.out = new
            touched = True
        for i, inet in enumerate(g.ins):
            if inet == old:
                g.ins[i] = new
                touched = True
    for ff in nl.dffs:
        if ff.q == old:
            ff.q = new; touched = True
        if ff.d == old:
            ff.d = new; touched = True
        if ff.clk == old:
            ff.clk = new; touched = True
        if ff.rn == old:
            ff.rn = new; touched = True
        if ff.sn == old:
            ff.sn = new; touched = True
    # ports (rename base name or a bit) — only if exact port net match
    if old in nl.ports:
        p = nl.ports.pop(old)
        p.name = new
        nl.ports[new] = p
        nl.port_order = [new if x == old else x for x in nl.port_order]
        touched = True
    if touched:
        nl.touch()
    return touched


def gates_connected_to_net(nl: Netlist, net: str) -> List[str]:
    """Instances that read or drive ``net`` (used by the companion 'list gates
    now connected to the renamed signal' queries)."""
    names = []
    seen = set()
    drv = nl.driver(net)
    if drv[0] in ("gate", "dff"):
        nm = drv[1].name
        if nm not in seen:
            seen.add(nm); names.append(nm)
    for c in nl.loads(net):
        nm = c[1].name
        if nm not in seen:
            seen.add(nm); names.append(nm)
    return names
