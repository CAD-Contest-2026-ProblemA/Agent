"""Netlist intermediate representation (the single source of truth).

A flat, gate-level netlist with one top module.  Everything in the system
(analysis / transform / optimize / equivalence) operates on this IR; ABC and
yosys are only ever used as *equivalence / cost-ranked-synthesis* oracles and
never as a naming source of truth.

Net names are plain strings exactly as they appear in the Verilog, e.g.
``"n5"`` (scalar) or ``"n0[3]"`` (one bit of a bus).  Constants are the
strings ``"1'b0"`` / ``"1'b1"``.

Gate convention is *output-pin-first* positional, matching the input format::

    not  g3(out, in)
    nand g12(out, a, b)

Flip-flops use the named-port model from the problem statement (generalised so
that *both* the async active-low reset ``RN`` and async active-low set ``SN``
can each be either a constant or a signal)::

    dff g906(.RN(n1), .SN(1'b1), .CK(n0), .D(n3238), .Q(n14[0]))
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

# Gate primitive types and their input arity.
TWO_INPUT = {"and", "or", "nand", "nor", "xor", "xnor"}
ONE_INPUT = {"not", "buf"}
GATE_TYPES = TWO_INPUT | ONE_INPUT
CONSTS = {"1'b0", "1'b1"}


def is_const(net: str) -> bool:
    return net in CONSTS


class Gate:
    """A combinational primitive gate instance."""

    __slots__ = ("type", "name", "out", "ins")

    def __init__(self, gtype: str, name: str, out: str, ins: List[str]):
        self.type = gtype
        self.name = name
        self.out = out
        self.ins = list(ins)

    def __repr__(self):
        return f"Gate({self.type} {self.name} {self.out} <- {self.ins})"


class Dff:
    """A flip-flop instance (named-port model)."""

    __slots__ = ("name", "clk", "d", "q", "rn", "sn")

    def __init__(self, name: str, clk: str, d: str, q: str, rn: str, sn: str):
        self.name = name
        self.clk = clk
        self.d = d
        self.q = q
        self.rn = rn  # async active-low reset (constant or signal)
        self.sn = sn  # async active-low set   (constant or signal)

    def __repr__(self):
        return f"Dff({self.name} Q={self.q} D={self.d} CK={self.clk})"


class Port:
    """A module port declaration (scalar or bus)."""

    __slots__ = ("name", "direction", "msb", "lsb")

    def __init__(self, name: str, direction: str,
                 msb: Optional[int] = None, lsb: Optional[int] = None):
        self.name = name
        self.direction = direction  # "input" | "output"
        self.msb = msb
        self.lsb = lsb

    @property
    def is_bus(self) -> bool:
        return self.msb is not None

    def bits(self) -> List[str]:
        """Enumerate the concrete net names for this port, LSB..MSB order."""
        if self.msb is None:
            return [self.name]
        lo, hi = sorted((self.msb, self.lsb))
        return [f"{self.name}[{i}]" for i in range(lo, hi + 1)]

    @property
    def width(self) -> int:
        if self.msb is None:
            return 1
        return abs(self.msb - self.lsb) + 1


class Netlist:
    """The whole design."""

    def __init__(self, module: str = "top"):
        self.module = module
        self.port_order: List[str] = []          # base names, header order
        self.ports: Dict[str, Port] = {}         # base name -> Port
        # Declared wire base-names (for round-trip fidelity); not semantically
        # important but kept so the writer can reproduce a clean netlist.
        self.wire_decls: List[Tuple[str, Optional[int], Optional[int]]] = []
        # Nets that must be DECLARED even when nothing uses them: an identifier
        # an earlier request created (a rename) whose logic was later optimized
        # away.  Kept apart from wire_decls, which mirrors the input file --
        # emitting those unconditionally would resurrect wires the source
        # happened to declare and the design no longer needs.
        self.forced_wires: List[str] = []
        self.gates: List[Gate] = []
        self.dffs: List[Dff] = []

        # Lazily-built indices (invalidated by .touch()).
        self._driver: Optional[Dict[str, tuple]] = None
        self._loads: Optional[Dict[str, list]] = None
        self._pi: Optional[set] = None
        self._po: Optional[set] = None
        self._gate_by_name: Optional[Dict[str, Gate]] = None
        # Monotonic counter for deterministically-named inserted gates/wires.
        self._fresh = 0

    # ----- maintenance ---------------------------------------------------
    def touch(self):
        """Invalidate cached indices after a structural mutation."""
        self._driver = None
        self._loads = None
        self._gate_by_name = None
        # PI/PO only change when ports change; keep unless explicitly reset.

    def snapshot(self) -> "Netlist":
        """A deep, index-free copy (cheap to take, safe to mutate)."""
        d = self._driver, self._loads, self._pi, self._po, self._gate_by_name
        self._driver = self._loads = self._gate_by_name = None
        self._pi = self._po = None
        clone = copy.deepcopy(self)
        (self._driver, self._loads, self._pi, self._po,
         self._gate_by_name) = d
        return clone

    def fresh_name(self, prefix: str) -> str:
        self._fresh += 1
        return f"{prefix}{self._fresh}"

    # ----- primary inputs / outputs -------------------------------------
    @property
    def pi(self) -> set:
        if self._pi is None:
            self._build_po_pi()
        return self._pi

    @property
    def po(self) -> set:
        if self._po is None:
            self._build_po_pi()
        return self._po

    def _build_po_pi(self):
        pi, po = set(), set()
        for name in self.port_order:
            p = self.ports[name]
            if p.direction == "input":
                pi.update(p.bits())
            elif p.direction == "output":
                po.update(p.bits())
        self._pi, self._po = pi, po

    def reset_ports(self):
        self._pi = self._po = None

    # ----- driver / loads ------------------------------------------------
    def driver(self, net: str):
        """Return how ``net`` is produced.

        ('gate', Gate) | ('dff', Dff) | ('pi',) | ('const', "1'b0"/"1'b1")
        | ('undriven',)
        """
        if self._driver is None:
            self._build_conn()
        if is_const(net):
            return ("const", net)
        return self._driver.get(net, ("pi",) if net in self.pi else ("undriven",))

    def loads(self, net: str) -> list:
        """Consumers of ``net``: list of ('gate', Gate, pin) | ('dff', Dff, pin)."""
        if self._loads is None:
            self._build_conn()
        return self._loads.get(net, [])

    def _build_conn(self):
        driver: Dict[str, tuple] = {}
        loads: Dict[str, list] = {}
        for g in self.gates:
            driver[g.out] = ("gate", g)
            for pin, inet in enumerate(g.ins):
                loads.setdefault(inet, []).append(("gate", g, pin))
        for ff in self.dffs:
            # A flip-flop *drives* its Q net.
            driver[ff.q] = ("dff", ff)
            for pin, inet in ((0, ff.d), (1, ff.clk), (2, ff.rn), (3, ff.sn)):
                loads.setdefault(inet, []).append(("dff", ff, pin))
        self._driver, self._loads = driver, loads

    def gate_by_name(self, name: str) -> Optional[Gate]:
        if self._gate_by_name is None:
            self._gate_by_name = {g.name: g for g in self.gates}
        return self._gate_by_name.get(name)

    def dff_by_name(self, name: str) -> Optional[Dff]:
        for ff in self.dffs:
            if ff.name == name:
                return ff
        return None

    def instance_by_name(self, name: str):
        g = self.gate_by_name(name)
        if g is not None:
            return ("gate", g)
        ff = self.dff_by_name(name)
        if ff is not None:
            return ("dff", ff)
        return None

    # ----- convenience ---------------------------------------------------
    def port_of_net(self, net: str) -> Optional[Port]:
        """Return the Port whose bit-set contains ``net`` (or whose base name is net)."""
        base = net.split("[")[0]
        p = self.ports.get(base)
        if p is None:
            return None
        if p.is_bus:
            return p if net in p.bits() or net == base else None
        return p if net == base else None

    def type_counts(self) -> Dict[str, int]:
        counts = {t: 0 for t in
                  ("and", "or", "not", "nand", "nor", "xor", "xnor", "buf")}
        for g in self.gates:
            counts[g.type] = counts.get(g.type, 0) + 1
        counts["dff"] = len(self.dffs)
        return counts

    def all_nets(self) -> set:
        nets = set()
        for g in self.gates:
            nets.add(g.out)
            nets.update(g.ins)
        for ff in self.dffs:
            nets.update((ff.q, ff.d, ff.clk, ff.rn, ff.sn))
        nets.update(self.pi)
        nets.update(self.po)
        nets.discard("1'b0")
        nets.discard("1'b1")
        return nets

    def __repr__(self):
        return (f"Netlist({self.module}: {len(self.gates)} gates, "
                f"{len(self.dffs)} dffs, {len(self.pi)} PI, {len(self.po)} PO)")
