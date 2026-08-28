"""Safe extraction and reinsertion of one combinational logic cone.

This module is deliberately independent of the optimisation scheduler.  It
turns a named net's complete register-cut fan-in cone into a small
standalone :class:`~cada.netlist.ir.Netlist`, and can splice an optimised form
of that standalone netlist back into the full design.

The standalone netlist may have more than one output.  A gate in the fan-in
cone may also feed unrelated logic, another primary output, or a DFF pin.  Its
net is exposed as a *side-exit output* in addition to the queried root.  This
lets synthesis optimise the complete cone (and retains the real arrival depth
from PIs/registers), while requiring every value observed outside the cone to
remain equivalent.  Reinsertion maps every side exit back to its original net.

Extraction uses private scalar aliases for every cutpoint and for the root.
Consequently Verilog bus-bit names such as ``n12[0]`` never become malformed
standalone port declarations.  Reinsertion maps aliases back to the original
names, while all optimiser-created internal nets and instances receive fresh
names that cannot collide with the full design.

Structural validation is intentionally strict.  Unsupported primitives,
multiple drivers, combinational cycles, stale extraction metadata, unexpected
standalone ports, floating optimiser nets, dead optimiser gates, and namespace
collisions are rejected with :class:`ConeTransformError`.  These checks make
the edit structurally safe; they do *not* prove that the optimiser preserved
the Boolean function.  Call :func:`splice_cone_verified`, or pass the result of
:func:`splice_cone` through the scheduler's normal whole-design CEC gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..netlist.ir import GATE_TYPES, ONE_INPUT, Gate, Netlist, Port


_CONE_MODULE = "__cada_runtime_cone"
_INPUT_PREFIX = "__cada_ci_"
_ROOT_ALIAS = "__cada_co_root"
_SIDE_OUTPUT_PREFIX = "__cada_co_"
_NET_PREFIX = "__cada_cone_net_"
_GATE_PREFIX = "__cada_cone_gate_"


class ConeTransformError(ValueError):
    """The requested cone transformation cannot be performed safely."""


@dataclass(frozen=True)
class ConeExtraction:
    """A standalone cone plus the information required to reinsert it.

    ``boundary`` is an ordered tuple of input ``(standalone_alias,
    original_net)`` pairs.  ``outputs`` has the same form for the root and all
    side exits; its first entry is always ``(root_alias, root)``.  Callers may
    freely optimise a snapshot of ``cone`` but must pass this unchanged
    extraction record to :func:`splice_cone`; a digest prevents using it after
    the source design has changed.
    """

    root: str
    root_alias: str
    boundary: Tuple[Tuple[str, str], ...]
    outputs: Tuple[Tuple[str, str], ...]
    preserved_nets: Tuple[str, ...]
    removed_gate_outputs: Tuple[str, ...]
    source_digest: str
    cone: Netlist

    @property
    def boundary_aliases(self) -> Tuple[str, ...]:
        return tuple(alias for alias, _net in self.boundary)

    @property
    def boundary_nets(self) -> Tuple[str, ...]:
        return tuple(net for _alias, net in self.boundary)

    @property
    def output_aliases(self) -> Tuple[str, ...]:
        return tuple(alias for alias, _net in self.outputs)

    @property
    def side_exits(self) -> Tuple[Tuple[str, str], ...]:
        return self.outputs[1:]


def extract_cone(nl: Netlist, root: str,
                 preserve_nets: Optional[Sequence[str]] = None,
                 *, preserve_outputs: Optional[Sequence[str]] = None,
                 ) -> ConeExtraction:
    """Extract the replaceable combinational fan-in region of ``root``.

    PIs, DFF-Q nets, and undriven/floating nets become standalone inputs.
    Constants remain constants.  Any intermediate cone net also consumed
    outside the cone becomes an additional standalone output.  The returned
    cone has no DFFs and uses safe scalar aliases for its complete interface.

    The input netlist is never mutated.  ``root`` must be the exact concrete
    net name (for example ``n12[0]``, not the base name of a multi-bit bus).

    ``preserve_nets`` forces named gate nets in this cone to become side-exit
    outputs even if they currently have no outside load.  This is useful for a
    scoped basis contract whose ``basis_output`` name must still exist after
    reinsertion.  ``preserve_outputs`` is an equivalent keyword alias; values
    supplied through both arguments are combined.
    """

    if not isinstance(root, str) or not root or root in ("1'b0", "1'b1"):
        raise ConeTransformError("cone root must be a non-constant net name")

    gate_by_out = _validate_full_netlist(nl)
    if root not in _all_named_nets(nl):
        raise ConeTransformError(f"unknown cone root {root!r}")
    if root not in gate_by_out:
        kind = _source_kind(nl, root, gate_by_out)
        raise ConeTransformError(
            f"cone root {root!r} is {kind}, not a combinational gate output")

    full_outputs = _fanin_gate_outputs(root, gate_by_out)
    _require_acyclic(full_outputs, gate_by_out, "source cone")

    requested_preserve = _normalise_preserve_nets(
        preserve_nets, preserve_outputs)
    for net in requested_preserve:
        if net not in full_outputs:
            raise ConeTransformError(
                f"preserved net {net!r} is not a gate net in root cone")

    # Every gate in the complete root cone is replaceable as long as all values
    # visible outside it are preserved.  Expose those side exits as additional
    # outputs.  This is preferable to cutting shared gates into inputs: doing
    # the latter resets their arrival depths to zero and can rank a locally
    # shallower cone that is actually deeper in the full design.
    loads = _loads_by_net(nl)
    side_exit_nets: Set[str] = set()
    for out in full_outputs:
        if out == root:
            continue
        if out in nl.po:
            side_exit_nets.add(out)
            continue
        for kind, consumer_out in loads.get(out, ()):
            if kind != "gate" or consumer_out not in full_outputs:
                side_exit_nets.add(out)
                break
    side_exit_nets.update(net for net in requested_preserve if net != root)

    boundary_nets: Set[str] = set()
    for out in full_outputs:
        for net in gate_by_out[out].ins:
            if net in ("1'b0", "1'b1"):
                continue
            if net not in full_outputs:
                boundary_nets.add(net)

    # Stable aliases make extraction reproducible and keep arbitrary original
    # identifiers (especially bus bits) out of standalone port declarations.
    boundary = tuple(
        (f"{_INPUT_PREFIX}{idx}", net)
        for idx, net in enumerate(sorted(boundary_nets))
    )
    original_to_alias = {net: alias for alias, net in boundary}

    outputs = ((_ROOT_ALIAS, root),) + tuple(
        (f"{_SIDE_OUTPUT_PREFIX}{idx}", net)
        for idx, net in enumerate(sorted(side_exit_nets))
    )
    original_output_to_alias = {net: alias for alias, net in outputs}

    ordered_outputs = _topological_outputs(full_outputs, gate_by_out)
    internal_outputs = [
        out for out in ordered_outputs if out not in original_output_to_alias
    ]
    original_to_local = dict(original_to_alias)
    original_to_local.update(
        (out, f"__cada_cn_{idx}")
        for idx, out in enumerate(internal_outputs)
    )
    original_to_local.update(original_output_to_alias)

    cone = Netlist(_CONE_MODULE)
    for alias, _net in boundary:
        cone.port_order.append(alias)
        cone.ports[alias] = Port(alias, "input")
    for alias, _net in outputs:
        cone.port_order.append(alias)
        cone.ports[alias] = Port(alias, "output")
    for idx, out in enumerate(ordered_outputs):
        gate = gate_by_out[out]
        cone.gates.append(Gate(
            gate.type,
            f"__cada_cg_{idx}",
            original_to_local[out],
            [original_to_local.get(net, net) for net in gate.ins],
        ))
    cone.reset_ports()
    _validate_optimised_cone(
        cone,
        tuple(alias for alias, _net in boundary),
        tuple(alias for alias, _net in outputs),
    )

    # Preserve original gate-list order in the removal record.  It makes the
    # splice insertion point deterministic even if graph traversal order changes.
    removed = tuple(g.out for g in nl.gates if g.out in full_outputs)
    return ConeExtraction(
        root=root,
        root_alias=_ROOT_ALIAS,
        boundary=boundary,
        outputs=outputs,
        preserved_nets=requested_preserve,
        removed_gate_outputs=removed,
        source_digest=_netlist_digest(nl),
        cone=cone,
    )


def splice_cone(nl: Netlist, extraction: ConeExtraction,
                optimised: Netlist) -> Netlist:
    """Return a full-design snapshot with ``optimised`` spliced at the root.

    This function performs structural validation only.  It preserves every
    interface declaration, DFF, and gate outside the extracted fan-in region,
    as well as the original root net and all of its fanout connections.
    Optimiser net and instance names are not trusted or reused.
    """

    if not isinstance(extraction, ConeExtraction):
        raise ConeTransformError("invalid cone extraction record")
    if _netlist_digest(nl) != extraction.source_digest:
        raise ConeTransformError(
            "source netlist changed after cone extraction; extract it again")

    # Recompute the cut rather than trusting caller-constructible metadata.
    expected = extract_cone(
        nl, extraction.root, preserve_nets=extraction.preserved_nets)
    if not _same_extraction_contract(expected, extraction):
        raise ConeTransformError("cone extraction metadata is inconsistent")

    aliases = extraction.boundary_aliases
    _validate_optimised_cone(
        optimised, aliases, extraction.output_aliases)

    candidate = nl.snapshot()
    removed = set(extraction.removed_gate_outputs)
    if not removed:
        raise ConeTransformError("cone extraction contains no gates")

    alias_to_original = dict(extraction.boundary)
    output_to_original = dict(extraction.outputs)
    used_nets = _reserved_net_names(candidate) | {"1'b0", "1'b1"}
    used_instances = {g.name for g in candidate.gates}
    used_instances.update(ff.name for ff in candidate.dffs)

    output_map: Dict[str, str] = dict(output_to_original)
    fresh_net_index = 0
    for gate in optimised.gates:
        if gate.out in output_to_original:
            continue
        name, fresh_net_index = _fresh_identifier(
            _NET_PREFIX, fresh_net_index, used_nets)
        output_map[gate.out] = name
        used_nets.add(name)

    replacement: List[Gate] = []
    fresh_gate_index = 0
    for gate in optimised.gates:
        gate_name, fresh_gate_index = _fresh_identifier(
            _GATE_PREFIX, fresh_gate_index, used_instances)
        used_instances.add(gate_name)
        ins = []
        for net in gate.ins:
            if net in ("1'b0", "1'b1"):
                ins.append(net)
            elif net in alias_to_original:
                ins.append(alias_to_original[net])
            else:
                # Validation guarantees that every other input is driven by
                # an optimiser gate, so it has an entry in output_map.
                try:
                    ins.append(output_map[net])
                except KeyError as exc:  # defensive against future validators
                    raise ConeTransformError(
                        f"unmapped optimiser net {net!r}") from exc
        replacement.append(Gate(
            gate.type, gate_name, output_map[gate.out], ins))

    new_gates: List[Gate] = []
    inserted = False
    for gate in candidate.gates:
        if gate.out in removed:
            if not inserted:
                new_gates.extend(replacement)
                inserted = True
            continue
        new_gates.append(gate)
    if not inserted:
        raise ConeTransformError("source cone gates disappeared before splice")
    candidate.gates = new_gates
    candidate.touch()

    # Validate the assembled netlist as a final invariant check.  The full
    # function is intentionally not checked here; use whole-design CEC below.
    _validate_full_netlist(candidate)
    if candidate.driver(extraction.root)[0] != "gate":
        raise ConeTransformError("spliced root is not gate-driven")
    for _alias, original_net in extraction.outputs:
        if candidate.driver(original_net)[0] != "gate":
            raise ConeTransformError(
                f"spliced side exit {original_net!r} is not gate-driven")
    return candidate


def splice_cone_verified(nl: Netlist, extraction: ConeExtraction,
                         optimised: Netlist, timeout: int = 280) -> Netlist:
    """Splice a cone and require a successful whole-design CEC proof.

    ``False`` and an inconclusive/timeout result are both safe rejection.  This
    wrapper is convenient for direct users; schedulers that already CEC every
    candidate should call :func:`splice_cone` and keep their existing proof
    gate instead of proving twice.
    """

    candidate = splice_cone(nl, extraction, optimised)
    from ..equiv import gate as equiv_gate

    if equiv_gate.equivalent(nl, candidate, timeout=timeout) is not True:
        raise ConeTransformError("whole-design CEC did not prove equivalence")
    return candidate


def _same_extraction_contract(a: ConeExtraction, b: ConeExtraction) -> bool:
    return (
        a.root == b.root
        and a.root_alias == b.root_alias
        and a.boundary == b.boundary
        and a.outputs == b.outputs
        and a.preserved_nets == b.preserved_nets
        and a.removed_gate_outputs == b.removed_gate_outputs
        and a.source_digest == b.source_digest
    )


def _validate_full_netlist(nl: Netlist) -> Dict[str, Gate]:
    """Validate invariants relied on by extraction and return gate drivers."""

    if not isinstance(nl, Netlist):
        raise ConeTransformError("expected a Netlist")

    instance_names: Set[str] = set()
    gate_by_out: Dict[str, Gate] = {}
    for gate in nl.gates:
        _validate_gate(gate)
        if gate.name in instance_names:
            raise ConeTransformError(f"duplicate instance name {gate.name!r}")
        instance_names.add(gate.name)
        if gate.out in ("1'b0", "1'b1"):
            raise ConeTransformError("a gate cannot drive a constant literal")
        if gate.out in gate_by_out:
            raise ConeTransformError(f"multiple gate drivers for {gate.out!r}")
        gate_by_out[gate.out] = gate

    for ff in nl.dffs:
        if not ff.name or ff.name in instance_names:
            raise ConeTransformError(f"duplicate/empty instance name {ff.name!r}")
        instance_names.add(ff.name)
        if not ff.q or ff.q in ("1'b0", "1'b1"):
            raise ConeTransformError(f"invalid DFF Q net {ff.q!r}")
        # Multiple DFF instances sharing one Q are supported by the existing
        # register-cut BLIF/CEC convention.  A combinational gate driving the
        # same net would be a genuinely ambiguous cross-kind driver.
        if ff.q in gate_by_out:
            raise ConeTransformError(
                f"gate and DFF both drive Q net {ff.q!r}")
        if ff.q in nl.pi:
            raise ConeTransformError(f"DFF Q {ff.q!r} is also a primary input")
        if any(not isinstance(pin, str) or not pin
               for pin in (ff.d, ff.clk, ff.rn, ff.sn)):
            raise ConeTransformError(f"DFF {ff.name!r} has an empty pin")
    for out in gate_by_out:
        if out in nl.pi:
            raise ConeTransformError(f"gate output {out!r} is also a primary input")
    return gate_by_out


def _validate_gate(gate: Gate) -> None:
    if not isinstance(gate, Gate):
        raise ConeTransformError("gate list contains a non-Gate object")
    if gate.type not in GATE_TYPES:
        raise ConeTransformError(f"unsupported gate primitive {gate.type!r}")
    if not isinstance(gate.name, str) or not gate.name:
        raise ConeTransformError("gate has an empty instance name")
    if not isinstance(gate.out, str) or not gate.out:
        raise ConeTransformError(f"gate {gate.name!r} has an empty output")
    if any(not isinstance(net, str) or not net for net in gate.ins):
        raise ConeTransformError(f"gate {gate.name!r} has an empty input")
    want = 1 if gate.type in ONE_INPUT else 2
    if len(gate.ins) != want:
        raise ConeTransformError(
            f"gate {gate.name!r} ({gate.type}) has arity {len(gate.ins)}, "
            f"expected {want}")


def _validate_optimised_cone(nl: Netlist, aliases: Sequence[str],
                             output_aliases: Sequence[str]) -> None:
    """Require a closed, acyclic standalone combinational netlist."""

    gate_by_out = _validate_full_netlist(nl)
    if nl.dffs:
        raise ConeTransformError("optimised cone must not contain DFFs")

    expected_pi = set(aliases)
    expected_po = set(output_aliases)
    if not expected_po:
        raise ConeTransformError("optimised cone has no output interface")
    if set(nl.pi) != expected_pi or set(nl.po) != expected_po:
        raise ConeTransformError(
            "optimised cone interface differs from the extracted interface")
    expected_ports = expected_pi | expected_po
    if (set(nl.ports) != expected_ports
            or len(nl.port_order) != len(expected_ports)
            or set(nl.port_order) != expected_ports):
        raise ConeTransformError(
            "optimised cone has missing, duplicate, or unexpected ports")
    for name, port in nl.ports.items():
        direction = "input" if name in expected_pi else "output"
        if (port.name != name or port.direction != direction
                or port.msb is not None or port.lsb is not None):
            raise ConeTransformError(
                f"optimised cone port {name!r} has an invalid declaration")
    if expected_pi & expected_po:
        raise ConeTransformError("cone input and output aliases overlap")
    missing_drivers = expected_po - set(gate_by_out)
    if missing_drivers:
        raise ConeTransformError(
            f"optimised cone output {sorted(missing_drivers)[0]!r} "
            "is not gate-driven")

    allowed_sources = expected_pi | {"1'b0", "1'b1"}
    for gate in nl.gates:
        for net in gate.ins:
            if net not in allowed_sources and net not in gate_by_out:
                raise ConeTransformError(
                    f"optimised cone contains floating/cross-cone net {net!r}")

    productive = _fanin_gate_outputs_many(expected_po, gate_by_out)
    if productive != set(gate_by_out):
        dead = sorted(set(gate_by_out) - productive)
        raise ConeTransformError(
            f"optimised cone contains dead gates, e.g. {dead[0]!r}")
    _require_acyclic(productive, gate_by_out, "optimised cone")


def _fanin_gate_outputs(root: str, gate_by_out: Dict[str, Gate]) -> Set[str]:
    return _fanin_gate_outputs_many((root,), gate_by_out)


def _fanin_gate_outputs_many(roots: Iterable[str],
                             gate_by_out: Dict[str, Gate]) -> Set[str]:
    seen: Set[str] = set()
    stack = list(roots)
    while stack:
        net = stack.pop()
        if net in seen or net not in gate_by_out:
            continue
        seen.add(net)
        stack.extend(gate_by_out[net].ins)
    return seen


def _require_acyclic(outputs: Set[str], gate_by_out: Dict[str, Gate],
                     context: str) -> None:
    _topological_outputs(outputs, gate_by_out, context=context)


def _topological_outputs(outputs: Set[str], gate_by_out: Dict[str, Gate],
                         context: str = "cone") -> List[str]:
    # Stable Kahn traversal.  Repeated pins of one gate represent one graph
    # predecessor for cycle purposes.
    preds: Dict[str, Set[str]] = {
        out: {net for net in gate_by_out[out].ins if net in outputs}
        for out in outputs
    }
    succ: Dict[str, Set[str]] = {out: set() for out in outputs}
    for out, incoming in preds.items():
        for net in incoming:
            succ[net].add(out)
    ready = [out for out, incoming in preds.items() if not incoming]
    heapq.heapify(ready)
    order: List[str] = []
    while ready:
        out = heapq.heappop(ready)
        order.append(out)
        for nxt in sorted(succ[out]):
            preds[nxt].discard(out)
            if not preds[nxt]:
                heapq.heappush(ready, nxt)
    if len(order) != len(outputs):
        raise ConeTransformError(f"combinational cycle in {context}")
    return order


def _loads_by_net(nl: Netlist) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    loads: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    for gate in nl.gates:
        for net in gate.ins:
            loads.setdefault(net, []).append(("gate", gate.out))
    for ff in nl.dffs:
        for net in (ff.d, ff.clk, ff.rn, ff.sn):
            loads.setdefault(net, []).append(("dff", None))
    return loads


def _all_named_nets(nl: Netlist) -> Set[str]:
    nets = set(nl.all_nets())
    nets.update(nl.pi)
    nets.update(nl.po)
    return nets


def _reserved_net_names(nl: Netlist) -> Set[str]:
    """Used and explicitly declared names that fresh splice nets must avoid."""

    nets = _all_named_nets(nl)
    nets.update(getattr(nl, "forced_wires", ()))
    for name, port in nl.ports.items():
        nets.add(name)
        nets.update(port.bits())
    for base, msb, lsb in nl.wire_decls:
        nets.add(base)
        if msb is not None and lsb is not None:
            lo, hi = sorted((msb, lsb))
            nets.update(f"{base}[{idx}]" for idx in range(lo, hi + 1))
    return nets


def _source_kind(nl: Netlist, net: str,
                 gate_by_out: Dict[str, Gate]) -> str:
    if net in nl.pi:
        return "a primary input"
    if any(ff.q == net for ff in nl.dffs):
        return "a DFF-Q source"
    if net in gate_by_out:
        return "a gate output"
    return "undriven"


def _fresh_identifier(prefix: str, start: int,
                      occupied: Set[str]) -> Tuple[str, int]:
    index = start
    while True:
        candidate = f"{prefix}{index}"
        index += 1
        if candidate not in occupied:
            return candidate, index


def _normalise_preserve_nets(
        preserve_nets: Optional[Sequence[str]],
        preserve_outputs: Optional[Sequence[str]]) -> Tuple[str, ...]:
    values: List[str] = []
    for supplied in (preserve_nets, preserve_outputs):
        if supplied is None:
            continue
        if isinstance(supplied, str):
            supplied = (supplied,)
        for net in supplied:
            if not isinstance(net, str) or not net:
                raise ConeTransformError("preserved cone net must be a name")
            if net in ("1'b0", "1'b1"):
                raise ConeTransformError("cannot preserve a constant as a net")
            values.append(net)
    return tuple(sorted(set(values)))


def _netlist_digest(nl: Netlist) -> str:
    """Digest every structural field that a stale splice could invalidate."""

    ports = []
    for name in sorted(nl.ports):
        port = nl.ports[name]
        ports.append((name, port.direction, port.msb, port.lsb))
    payload = {
        "module": nl.module,
        "port_order": list(nl.port_order),
        "ports": ports,
        "wire_decls": list(nl.wire_decls),
        "forced_wires": list(getattr(nl, "forced_wires", ())),
        "gates": [
            (gate.type, gate.name, gate.out, list(gate.ins))
            for gate in nl.gates
        ],
        "dffs": [
            (ff.name, ff.clk, ff.d, ff.q, ff.rn, ff.sn)
            for ff in nl.dffs
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = [
    "ConeExtraction",
    "ConeTransformError",
    "extract_cone",
    "splice_cone",
    "splice_cone_verified",
]
