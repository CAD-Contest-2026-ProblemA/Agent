"""Deterministic bit-parallel register-cut simulation.

This is deliberately a *testing* oracle, not a formal equivalence proof.  It
is used only to validate a lazily loaded, offline-CEC'd prepared artifact.  All
ordinary online resynthesis candidates continue to use :mod:`cada.equiv.gate`.

Patterns are shared by source-net name and observations cover every primary
output plus every DFF D/CK/RN/SN pin.  The first block contains deterministic
corner cases (all-zero/all-one, checkerboards, walking one/zero, prefix/suffix,
and correlated same-bit bus patterns); the remaining vectors come from
reproducible independent PRNG streams.  Simulation is batched and releases
dead net values, keeping the large prepared circuits within a bounded memory
footprint.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import random
import re
import time
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..netlist.ir import Gate, Netlist, is_const


DEFAULT_BATCH_PATTERNS = 8192
DEFAULT_SEED = 0xCADA2026


@dataclass(frozen=True)
class PatternResult:
    """Result of a bounded simulation comparison.

    ``matched`` is ``True`` when every requested vector agreed, ``False``
    on a witnessed counterexample, and ``None`` when simulation could not be
    completed (deadline, incompatible cut boundary, cycle, or unsupported
    netlist).  Only ``False`` is a proof; ``True`` remains probabilistic.
    """

    matched: Optional[bool]
    patterns_checked: int
    requested_patterns: int
    directed_patterns: int
    seed: int
    batch_patterns: int
    seconds: float
    reason: str
    mismatch_observation: Optional[str] = None
    mismatch_pattern: Optional[int] = None


@dataclass(frozen=True)
class _Plan:
    gates: Tuple[Gate, ...]
    sources: FrozenSet[str]
    observations: Tuple[Tuple[str, str], ...]
    last_use: Dict[str, int]
    keep: FrozenSet[str]
    release_after: Tuple[Tuple[str, ...], ...]


class _UnsupportedSimulation(RuntimeError):
    pass


def _natural_key(value: str):
    return tuple(
        int(piece) if piece.isdigit() else piece
        for piece in re.split(r"(\d+)", value))


def _observations(nl: Netlist) -> Tuple[Tuple[str, str], ...]:
    result: List[Tuple[str, str]] = [
        (f"PO:{net}", net) for net in sorted(nl.po, key=_natural_key)]
    seen_q = set()
    for ff in sorted(nl.dffs, key=lambda item: _natural_key(item.q)):
        if ff.q in seen_q:
            raise _UnsupportedSimulation(f"duplicate DFF Q boundary {ff.q}")
        seen_q.add(ff.q)
        for attr in ("d", "clk", "rn", "sn"):
            result.append((f"FFQ:{ff.q}:{attr}", getattr(ff, attr)))
    return tuple(result)


def _build_plan(nl: Netlist) -> _Plan:
    """Build a topological plan for only the observable combinational cone."""
    observations = _observations(nl)
    relevant: Dict[str, Gate] = {}
    sources = set()
    stack = [net for _label, net in observations]
    visited = set()
    while stack:
        net = stack.pop()
        if net in visited or is_const(net):
            continue
        visited.add(net)
        drv = nl.driver(net)
        if drv[0] == "gate":
            gate = drv[1]
            previous = relevant.get(gate.out)
            if previous is not None and previous is not gate:
                raise _UnsupportedSimulation(
                    f"multiple relevant drivers for {gate.out}")
            relevant[gate.out] = gate
            stack.extend(gate.ins)
        elif drv[0] in ("pi", "dff", "undriven"):
            sources.add(net)
        else:
            raise _UnsupportedSimulation(
                f"unsupported driver {drv[0]} for {net}")

    position = {gate.out: index for index, gate in enumerate(nl.gates)}
    if len(position) != len(nl.gates):
        raise _UnsupportedSimulation("duplicate gate output net")
    indegree: Dict[str, int] = {}
    consumers: Dict[str, List[str]] = {}
    for out, gate in relevant.items():
        dependencies = {
            inp for inp in gate.ins
            if inp in relevant and nl.driver(inp)[0] == "gate"}
        indegree[out] = len(dependencies)
        for dependency in dependencies:
            consumers.setdefault(dependency, []).append(out)
    for outputs in consumers.values():
        outputs.sort(key=lambda net: position.get(net, 0))

    ready = deque(sorted(
        (out for out, count in indegree.items() if count == 0),
        key=lambda net: position.get(net, 0)))
    ordered: List[Gate] = []
    while ready:
        out = ready.popleft()
        ordered.append(relevant[out])
        for consumer in consumers.get(out, ()):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                ready.append(consumer)
    if len(ordered) != len(relevant):
        raise _UnsupportedSimulation("combinational cycle or invalid topology")

    last_use: Dict[str, int] = {}
    for index, gate in enumerate(ordered):
        for net in gate.ins:
            last_use[net] = index
    keep = frozenset(net for _label, net in observations)
    release_after = tuple(
        tuple(dict.fromkeys(
            net for net in gate.ins
            if last_use.get(net) == index and net not in keep))
        for index, gate in enumerate(ordered))
    return _Plan(tuple(ordered), frozenset(sources), observations,
                 last_use, keep, release_after)


def _boundary_sources(nl: Netlist) -> FrozenSet[str]:
    return frozenset(set(nl.pi) | {ff.q for ff in nl.dffs})


def _source_order(nl: Netlist, extras: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for name in nl.port_order:
        port = nl.ports.get(name)
        if port is None or port.direction != "input":
            continue
        for bit in port.bits():
            if bit not in seen:
                seen.add(bit)
                ordered.append(bit)
    for net in sorted((ff.q for ff in nl.dffs), key=_natural_key):
        if net not in seen:
            seen.add(net)
            ordered.append(net)
    for net in sorted(extras, key=_natural_key):
        if net not in seen:
            seen.add(net)
            ordered.append(net)
    return ordered


def _range_mask(global_lo: int, global_hi: int,
                batch_start: int, batch_count: int) -> int:
    lo = max(global_lo, batch_start)
    hi = min(global_hi, batch_start + batch_count)
    if lo >= hi:
        return 0
    return ((1 << (hi - lo)) - 1) << (lo - batch_start)


def _directed_bits(source_index: int, source_count: int,
                   batch_start: int, batch_count: int,
                   directed_limit: int) -> Tuple[int, int]:
    """Return ``(directed_mask, ones)`` for one source in this batch."""
    mask = _range_mask(0, directed_limit, batch_start, batch_count)
    ones = 0

    def bit(position: int) -> int:
        if (position < directed_limit
                and batch_start <= position < batch_start + batch_count):
            return 1 << (position - batch_start)
        return 0

    # 0: all zero; 1: all one; 2/3: complementary checkerboards.
    ones |= bit(1)
    ones |= bit(2 if source_index % 2 == 0 else 3)

    one_hot = 4
    one_cold = one_hot + source_count
    prefix = one_cold + source_count
    suffix = prefix + source_count

    ones |= bit(one_hot + source_index)
    cold = _range_mask(
        one_cold, min(one_cold + source_count, directed_limit),
        batch_start, batch_count)
    cold &= ~bit(one_cold + source_index)
    ones |= cold
    # prefix[i] sets sources 0..i; suffix[i] sets sources i..N-1.
    ones |= _range_mask(
        prefix + source_index,
        min(prefix + source_count, directed_limit),
        batch_start, batch_count)
    ones |= _range_mask(
        suffix,
        min(suffix + source_index + 1, directed_limit),
        batch_start, batch_count)
    return mask, ones


def _rng_for(name: str, seed: int) -> random.Random:
    digest = hashlib.blake2b(
        f"cada-prepared-pattern-v1\0{seed}\0{name}".encode(),
        digest_size=16).digest()
    return random.Random(int.from_bytes(digest, "little"))


def _word_bit_index(name: str) -> Optional[int]:
    match = re.search(r"\[(\d+)\]$", name)
    return int(match.group(1)) if match else None


def _evaluate(plan: _Plan, source_values: Dict[str, int], mask: int,
              deadline: Optional[float]) -> Tuple[int, ...]:
    values: Dict[str, int] = {"1'b0": 0, "1'b1": mask}
    for source in plan.sources:
        try:
            values[source] = source_values[source]
        except KeyError as exc:
            raise _UnsupportedSimulation(
                f"no shared stimulus for source {source}") from exc

    for index, gate in enumerate(plan.gates):
        if ((index & 1023) == 0 and deadline is not None
                and time.monotonic() >= deadline):
            raise TimeoutError("pattern simulation deadline exhausted")
        try:
            a = values[gate.ins[0]]
        except (IndexError, KeyError) as exc:
            raise _UnsupportedSimulation(
                f"unavailable input while simulating {gate.name}") from exc
        if gate.type in {"and", "nand"}:
            value = a
            for pin in gate.ins[1:]:
                value &= values[pin]
            if gate.type == "nand":
                value = mask ^ value
        elif gate.type in {"or", "nor"}:
            value = a
            for pin in gate.ins[1:]:
                value |= values[pin]
            if gate.type == "nor":
                value = mask ^ value
        elif gate.type in {"xor", "xnor"}:
            value = a
            for pin in gate.ins[1:]:
                value ^= values[pin]
            if gate.type == "xnor":
                value = mask ^ value
        elif gate.type == "not" and len(gate.ins) == 1:
            value = mask ^ a
        elif gate.type == "buf" and len(gate.ins) == 1:
            value = a
        else:
            raise _UnsupportedSimulation(
                f"unsupported gate {gate.type}/{len(gate.ins)}")

        if gate.out in plan.keep or gate.out in plan.last_use:
            values[gate.out] = value
        for net in plan.release_after[index]:
            values.pop(net, None)

    try:
        return tuple(values[net] if not is_const(net)
                     else (mask if net == "1'b1" else 0)
                     for _label, net in plan.observations)
    except KeyError as exc:
        raise _UnsupportedSimulation(
            f"unavailable observable {exc.args[0]}") from exc


def equivalent(before: Netlist, after: Netlist, *, patterns: int = 100_000,
               seed: int = DEFAULT_SEED,
               batch_patterns: int = DEFAULT_BATCH_PATTERNS,
               deadline: Optional[float] = None) -> PatternResult:
    """Compare two register-cut netlists on deterministic packed patterns."""
    started = time.monotonic()
    directed_patterns = 0

    def result(value: Optional[bool], checked: int, reason: str,
               observation: Optional[str] = None,
               pattern: Optional[int] = None) -> PatternResult:
        return PatternResult(
            matched=value,
            patterns_checked=checked,
            requested_patterns=patterns,
            directed_patterns=directed_patterns,
            seed=seed,
            batch_patterns=batch_patterns,
            seconds=time.monotonic() - started,
            reason=reason,
            mismatch_observation=observation,
            mismatch_pattern=pattern)

    if patterns < 1 or batch_patterns < 1:
        return result(None, 0, "patterns and batch_patterns must be positive")
    checked = 0
    try:
        before_plan = _build_plan(before)
        after_plan = _build_plan(after)
        before_labels = tuple(label for label, _net in before_plan.observations)
        after_labels = tuple(label for label, _net in after_plan.observations)
        if before_labels != after_labels:
            return result(None, 0, "observable register-cut boundaries differ")

        before_boundary = _boundary_sources(before)
        after_boundary = _boundary_sources(after)
        if before_boundary != after_boundary:
            return result(None, 0, "PI/DFF-Q source boundaries differ")
        before_extra = before_plan.sources - before_boundary
        after_extra = after_plan.sources - after_boundary
        if before_extra != after_extra:
            return result(None, 0, "undriven free-source boundaries differ")

        sources = _source_order(before, tuple(before_extra))
        source_count = len(sources)
        # Always reserve at least half the requested vectors for independent
        # pseudo-random testing, even on designs with enormous boundaries.
        echo_start = 4 + 4 * source_count
        echo_patterns = 256
        directed_limit = min(
            echo_start + echo_patterns, max(2, patterns // 2))
        directed_patterns = directed_limit
        rngs = {name: _rng_for(name, seed) for name in sources}
        bit_indices = {
            name: index for name in sources
            if (index := _word_bit_index(name)) is not None}
        echo_rngs = {
            index: _rng_for(f"__word_echo_bit_{index}", seed)
            for index in set(bit_indices.values())}

        while checked < patterns:
            if deadline is not None and time.monotonic() >= deadline:
                return result(None, checked, "pattern simulation deadline exhausted")
            count = min(batch_patterns, patterns - checked)
            mask = (1 << count) - 1
            echo_mask = _range_mask(
                echo_start, min(echo_start + echo_patterns, directed_limit),
                checked, count)
            echo_values = {
                index: rng.getrandbits(count) & echo_mask
                for index, rng in echo_rngs.items()}
            source_values: Dict[str, int] = {}
            for index, name in enumerate(sources):
                value = rngs[name].getrandbits(count)
                directed_mask, directed_ones = _directed_bits(
                    index, source_count, checked, count, directed_limit)
                echo_index = bit_indices.get(name)
                echo_value = (
                    echo_values.get(echo_index, 0)
                    if echo_index is not None else 0)
                source_values[name] = (
                    (value & (mask ^ directed_mask)) | directed_ones
                    | echo_value)

            lhs = _evaluate(before_plan, source_values, mask, deadline)
            rhs = _evaluate(after_plan, source_values, mask, deadline)
            mismatch_bit = None
            mismatch_label = None
            for index, (left, right) in enumerate(zip(lhs, rhs)):
                difference = left ^ right
                if not difference:
                    continue
                position = (difference & -difference).bit_length() - 1
                if mismatch_bit is None or position < mismatch_bit:
                    mismatch_bit = position
                    mismatch_label = before_labels[index]
            if mismatch_bit is not None:
                absolute = checked + mismatch_bit
                return result(
                    False, absolute + 1, "counterexample pattern witnessed",
                    mismatch_label, absolute)
            checked += count
        return result(True, checked, "all requested patterns agreed")
    except TimeoutError as exc:
        return result(None, checked, str(exc))
    except (MemoryError, _UnsupportedSimulation) as exc:
        return result(None, checked, str(exc))
    except Exception as exc:
        return result(None, checked, f"pattern simulation failed: {exc}")
