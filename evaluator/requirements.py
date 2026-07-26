"""Derive each request's checkable requirements from its natural-language text.

This is deliberately conservative: it only emits a requirement when the wording
clearly implies one, so a passing check is meaningful and a failing one is real.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

NET = r"[A-Za-z_]\w*(?:\[\d+\])?"

# phrases that promise functional equivalence is preserved
_EQUIV_PHRASES = (
    "nothing changes functionally", "functional equivalence is preserved",
    "without changing functionality", "functionality does not change",
    "preserve functional equivalence", "preserving functional equivalence",
    "does not change the design", "ensure the design functionality",
)


@dataclass
class Requirement:
    kind: str                       # equiv | basis | gate_absent | max_fanout
    k: Optional[int] = None         #                 | max_fanout_net | valid
    include_pi: bool = False
    basis: Optional[str] = None
    scope_output: Optional[str] = None
    net: Optional[str] = None       # for max_fanout_net; or gate type for gate_absent
    gate_type: Optional[str] = None


@dataclass
class OptimizeInfo:
    metric: str                     # depth | cone_depth | gate_count
    output: Optional[str] = None


@dataclass
class LineSpec:
    requirements: List[Requirement] = field(default_factory=list)
    optimize: Optional[OptimizeInfo] = None
    is_transform: bool = False


def _basis_phrase(line: str) -> Optional[str]:
    """Map a basis-purity instruction to a basis name.

    Gate names are written in UPPERCASE in the prompts (NAND, NOT, ...), while
    the conjunction 'and' is lowercase — so we extract only uppercase gate
    tokens.  That avoids mistaking the 'and' in "NAND and NOT" for the AND gate
    (which would otherwise yield {nand, and, not} and match no basis), and
    avoids unrelated words (e.g. "does not change") fabricating a basis.
    A basis is returned only on an exact match of the recognised gate set.
    """
    toks = set(t.lower() for t in
               re.findall(r"\b(NAND|NOR|XNOR|XOR|AND|OR|NOT|BUF)\b", line))
    toks.discard("buf")
    if toks == {"nand", "not"}:
        return "NAND_NOT"
    if toks == {"nor", "not"}:
        return "NOR_NOT"
    if toks == {"and", "or", "not"}:
        return "AND_OR_NOT"
    if toks == {"and", "not"}:
        return "AND_NOT"
    return None


def _scope_output(line: str) -> Optional[str]:
    m = re.search(r"cone of (?:output )?(%s)" % NET, line, re.IGNORECASE)
    return m.group(1) if m else None


def derive(line: str) -> LineSpec:
    spec = LineSpec()
    low = line.lower()

    # ---- functional equivalence ----
    if any(p in low for p in _EQUIV_PHRASES):
        spec.requirements.append(Requirement(kind="equiv"))
        spec.is_transform = True

    # ---- gate-basis: distinguish targeted type removal vs whole/cone purity ----
    basis = _basis_phrase(line)
    scope = _scope_output(line)
    # (C) targeted: "convert/replace/decompose all/every <TYPE> gates ..." removes
    # exactly that gate type (the rest of the design keeps its gates); the basis
    # mention describes the replacement, not whole-design purity.  Excludes the
    # conditional "NAND gates that have one input tied to constant 1" case.
    m = re.search(r"(?:convert|replace|decompose)\s+(?:all|every)\s+(?:2-input\s+)?"
                  r"(xnor|xor|nand|nor|and|or)\b", low)
    targeted = bool(m) and not any(q in low for q in
                                   ("tied to", "constant", "that have"))
    if targeted:
        spec.requirements.append(
            Requirement(kind="gate_absent", gate_type=m.group(1), scope_output=scope))
        spec.is_transform = True
    elif basis:
        # (A) whole design / netlist purity
        if re.search(r"(entire|whole)\s+(design|netlist)", low) or \
                re.search(r"netlist remains\b", low):
            spec.requirements.append(Requirement(kind="basis", basis=basis))
        # (B) a specific cone must be in the basis
        elif scope and re.search(r"(to use only|using only|use only|maintains only|"
                                 r"contains only|continues to use only)", low):
            spec.requirements.append(
                Requirement(kind="basis", basis=basis, scope_output=scope))

    # ---- fanout bounds ----
    m = re.search(r"no (gate|signal) (?:drives|has fanout)[^0-9]*(\d+)", low)
    if m:
        spec.requirements.append(
            Requirement(kind="max_fanout", k=int(m.group(2)),
                        include_pi=(m.group(1) == "signal")))
        spec.is_transform = True
    m = re.search(r"(?:reduce|fanout).{0,40}?signal (%s).{0,40}?(\d+) loads" % NET, low)
    if m:
        spec.requirements.append(
            Requirement(kind="max_fanout_net", net=m.group(1), k=int(m.group(2))))
        spec.is_transform = True

    # ---- transforms in general (for equiv even if phrase missing) ----
    if re.search(r"\b(remap|reconstruct|convert|decompose|replace|remove|delete|"
                 r"trim|sweep|prune|eliminate|merge|collapse|insert|rename|"
                 r"simplify|optimize|reduce|minimize|restructure)\b", low):
        spec.is_transform = True

    # ---- optimize cost metric ----
    if "cost function" in low or re.search(r"\b(optimi[sz]e|minimi[sz]e|reduce)\b", low):
        # Read the metric off the clause that *states* the cost, not the whole
        # line.  These prompts routinely also name a cone as a structural
        # constraint ("... ensuring the cone of n15 contains only AND, OR and
        # NOT gates") while costing the whole design, and matching "cone"
        # anywhere would scope the cost to that cone instead.
        mc = re.search(r"cost(?: function)?\s+is\s+(.*?)(?:;|\.\s|$)", low)
        clause = mc.group(1) if mc else low
        if "gate count" in clause:
            spec.optimize = OptimizeInfo(metric="gate_count")
        elif "cone" in clause:
            spec.optimize = OptimizeInfo(
                metric="cone_depth",
                output=_scope_output(clause) or _scope_output(line))
        elif "depth" in clause:
            spec.optimize = OptimizeInfo(metric="depth")

    return spec
