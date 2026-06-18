"""Canonical structural Verilog writer.

We never ship ABC's ``write_verilog`` output: it is behavioural
(``assign x = ~(a & b)``) and shatters buses into escaped scalars
(``\\n0[0]``), which breaks the expected ``input [7:0] n0;`` netlist format.

This writer emits a clean, positional, gate-level netlist that:
* preserves the module port order and bus port declarations,
* declares every internal net as a wire (buses reconstructed from index use),
* writes gates output-pin-first, and
* restores flip-flops to the named-port ``dff`` form.

The result round-trips exactly through :mod:`cada.netlist.reader`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .ir import Netlist, is_const

_BIT = re.compile(r"^(.*)\[(\d+)\]$")


def _base_index(net: str) -> Tuple[str, Optional[int]]:
    m = _BIT.match(net)
    if m:
        return m.group(1), int(m.group(2))
    return net, None


def write_file(nl: Netlist, path: str) -> None:
    with open(path, "w") as fh:
        fh.write(to_string(nl))


def to_string(nl: Netlist) -> str:
    lines: List[str] = []
    lines.append("")
    lines.append(f"module {nl.module}({', '.join(nl.port_order)});")

    # ---- port declarations (preserve bus ranges) -----------------------
    port_bits = set()
    for name in nl.port_order:
        p = nl.ports.get(name)
        if p is None:
            continue
        port_bits.update(p.bits())
        port_bits.add(name)

    for direction in ("input", "output"):
        # group by identical [msb:lsb] for compact, readable output
        groups: Dict[Tuple[Optional[int], Optional[int]], List[str]] = defaultdict(list)
        for name in nl.port_order:
            p = nl.ports.get(name)
            if p is None or p.direction != direction:
                continue
            groups[(p.msb, p.lsb)].append(name)
        for (msb, lsb), names in groups.items():
            rng = "" if msb is None else f"[{msb}:{lsb}] "
            lines.append(f"  {direction} {rng}{', '.join(names)};")

    # ---- internal wires -------------------------------------------------
    bus_max: Dict[str, int] = {}
    bus_min: Dict[str, int] = {}
    scalars: List[str] = []
    seen = set()
    for net in _iter_nets(nl):
        if is_const(net) or net in port_bits:
            continue
        base, idx = _base_index(net)
        if base in port_bits:  # a bit of a declared port bus
            continue
        if idx is None:
            if net not in seen:
                seen.add(net)
                scalars.append(net)
        else:
            bus_max[base] = max(bus_max.get(base, idx), idx)
            bus_min[base] = min(bus_min.get(base, idx), idx)

    for base in bus_max:
        lines.append(f"  wire [{bus_max[base]}:{bus_min[base]}] {base};")
    # chunk scalar wire decls 8 per line for readability
    for i in range(0, len(scalars), 8):
        lines.append(f"  wire {', '.join(scalars[i:i + 8])};")

    lines.append("")

    # ---- gates (output-pin-first) --------------------------------------
    for g in nl.gates:
        args = ", ".join([g.out] + list(g.ins))
        lines.append(f"  {g.type} {g.name}({args});")

    # ---- flip-flops (named-port) ---------------------------------------
    for ff in nl.dffs:
        lines.append(
            f"  dff {ff.name}(.RN({ff.rn}), .SN({ff.sn}), "
            f".CK({ff.clk}), .D({ff.d}), .Q({ff.q}));"
        )

    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)


def _iter_nets(nl: Netlist):
    for g in nl.gates:
        yield g.out
        for i in g.ins:
            yield i
    for ff in nl.dffs:
        yield ff.q
        yield ff.d
        yield ff.clk
        yield ff.rn
        yield ff.sn
