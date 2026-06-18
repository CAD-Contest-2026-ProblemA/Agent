"""Self-written gate-level Verilog parser.

Why self-written: ABC's ``read_verilog`` cannot parse the named-port ``dff``
model used by the benchmarks (it errors with "Cannot parse a standard gate"),
so we never rely on it as a front end.  This parser accepts exactly the subset
the benchmarks use:

* ``module top(a, b, ...);``
* ``input [7:0] a, b;`` / ``output c;`` / ``wire d, e;`` (scalar or bus,
  comma-separated, possibly spanning several physical lines)
* positional gate instances ``nand g(out, a, b);`` / ``not g(out, in);``
* named-port flip-flops ``dff g(.RN(x), .SN(y), .CK(c), .D(d), .Q(q));``
* ``endmodule``

It is tolerant of ``//`` and ``/* */`` comments and arbitrary whitespace.
"""

from __future__ import annotations

import re
from typing import List

from .ir import Dff, Gate, Netlist, Port, GATE_TYPES

_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE = re.compile(r"//[^\n]*")
_BUS = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]")
_INST = re.compile(r"^([A-Za-z_][\w$]*)\s+([\\]?[\w$]+)\s*\((.*)\)$", re.DOTALL)
_DFF_PORT = re.compile(r"\.\s*([A-Za-z_]\w*)\s*\(\s*([^)]*?)\s*\)")


class VerilogParseError(Exception):
    pass


def _strip_comments(text: str) -> str:
    text = _COMMENT_BLOCK.sub(" ", text)
    text = _COMMENT_LINE.sub(" ", text)
    return text


def _split_statements(text: str) -> List[str]:
    """Split into statements on ';'.  ``endmodule`` has no ';' so it is dropped."""
    out = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(chunk)
    return out


def _split_top_commas(s: str) -> List[str]:
    """Split on commas that are not inside brackets/parens."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
            cur.append(ch)
        elif ch in ")]":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def parse_file(path: str) -> Netlist:
    with open(path, "r") as fh:
        return parse_text(fh.read())


def parse_text(text: str) -> Netlist:
    text = _strip_comments(text)
    nl = Netlist()
    saw_module = False

    for stmt in _split_statements(text):
        if stmt == "endmodule" or stmt.startswith("endmodule"):
            continue

        head = stmt.split(None, 1)[0]

        if head == "module":
            _parse_module_header(stmt, nl)
            saw_module = True
            continue

        if head in ("input", "output", "wire"):
            _parse_decl(stmt, nl)
            continue

        if head == "dff":
            _parse_dff(stmt, nl)
            continue

        if head in GATE_TYPES:
            _parse_gate(stmt, nl)
            continue

        # Unknown leading token: could be a multi-line wire decl continuation
        # already handled, or something unexpected — be strict to catch bugs.
        if saw_module:
            raise VerilogParseError(f"Unrecognised statement: {stmt[:80]!r}")

    nl.reset_ports()
    return nl


def _parse_module_header(stmt: str, nl: Netlist):
    # module <name> ( p0, p1, ... )
    m = re.match(r"module\s+([\\]?[\w$]+)\s*\((.*)\)\s*$", stmt, re.DOTALL)
    if not m:
        # Some headers may omit a port list.
        m2 = re.match(r"module\s+([\\]?[\w$]+)", stmt)
        if not m2:
            raise VerilogParseError(f"Bad module header: {stmt[:80]!r}")
        nl.module = m2.group(1)
        return
    nl.module = m.group(1)
    for p in _split_top_commas(m.group(2)):
        # Port list entries are bare names here; direction comes from decls.
        nl.port_order.append(p.strip())


def _parse_decl(stmt: str, nl: Netlist):
    head = stmt.split(None, 1)[0]
    rest = stmt[len(head):].strip()
    msb = lsb = None
    mb = _BUS.search(rest)
    if mb and rest[:mb.start()].strip() == "":
        msb, lsb = int(mb.group(1)), int(mb.group(2))
        rest = rest[mb.end():]
    names = _split_top_commas(rest)
    for nm in names:
        nm = nm.strip()
        if not nm:
            continue
        if head in ("input", "output"):
            if nm not in nl.ports:
                nl.ports[nm] = Port(nm, head, msb, lsb)
            else:
                # Re-declared with direction (port list had it as bare name).
                p = nl.ports[nm]
                p.direction = head
                if msb is not None:
                    p.msb, p.lsb = msb, lsb
            if nm not in nl.port_order:
                nl.port_order.append(nm)
        else:  # wire
            nl.wire_decls.append((nm, msb, lsb))


def _parse_gate(stmt: str, nl: Netlist):
    m = _INST.match(stmt)
    if not m:
        raise VerilogParseError(f"Bad gate instance: {stmt[:80]!r}")
    gtype, name, arglist = m.group(1), m.group(2), m.group(3)
    args = [a.strip() for a in _split_top_commas(arglist)]
    out, ins = args[0], args[1:]
    nl.gates.append(Gate(gtype, name, out, ins))


def _parse_dff(stmt: str, nl: Netlist):
    m = _INST.match(stmt)
    if not m:
        raise VerilogParseError(f"Bad dff instance: {stmt[:80]!r}")
    name, arglist = m.group(2), m.group(3)
    pins = {k.upper(): v.strip() for k, v in _DFF_PORT.findall(arglist)}
    if "Q" not in pins or "D" not in pins:
        raise VerilogParseError(f"dff missing D/Q: {stmt[:80]!r}")
    nl.dffs.append(Dff(
        name=name,
        clk=pins.get("CK", pins.get("CLK", "")),
        d=pins["D"],
        q=pins["Q"],
        rn=pins.get("RN", "1'b1"),
        sn=pins.get("SN", "1'b1"),
    ))
