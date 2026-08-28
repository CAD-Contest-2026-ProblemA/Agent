"""Runtime registry for offline-prepared optimisation candidates.

This module deliberately performs *matching and loading only*.  A returned
candidate is a hint, not a proof: the caller applies the prepared-bank trust
policy (offline CEC identity for exact hits, bounded register-cut simulation
for runtime validation) before accepting it.

The default manifest is searched in locations that work both from source and
from a PyInstaller extraction directory.  Artifacts are parsed lazily and may
be plain Verilog or gzip-compressed Verilog.  A minimal manifest is::

    {"version": 1, "fingerprint_version": 2, "candidates": [{
      "id": "case44-depth56", "artifact": "case44-depth56.v.gz",
      "sha256": "...", "objective": "depth", "basis": "AND_NOT",
      "scope": "global", "fanout_model": "unit",
      "match": {"topology_key": "...", "register_identity_key": "...",
                "interface": {...}}
    }]}

Unknown manifest fields are retained in :attr:`PreparedCandidate.metadata` so
the format can grow without changing the runtime API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from ..analysis import graph
from ..analysis.depth import global_max_depth
from ..netlist.ir import Netlist
from ..netlist.reader import parse_text
from ..toolpaths import _bundledir, _exedir


MANIFEST_VERSION = 1
FINGERPRINT_VERSION = 2
_DEFAULT_RELATIVE_PATHS = (
    "cada/optimize/prepared_assets/manifest.json",
    "prepared/manifest.json",
)


class PreparedRegistryError(RuntimeError):
    """Raised for a malformed manifest or corrupt prepared artifact."""


def _norm_token(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip().upper().replace("-", "_")


def interface_signature(nl: Netlist) -> Dict[str, Any]:
    """Return a name-independent, conservative interface summary.

    Width multisets permit coarse matching across renamed ports.  Register
    count avoids obvious false matches while leaving structural control-pin
    changes, exact correspondence, and equivalence to later stages.
    """
    inputs = sorted(p.width for p in nl.ports.values()
                    if p.direction == "input")
    outputs = sorted(p.width for p in nl.ports.values()
                     if p.direction == "output")

    return {
        "input_widths": inputs,
        "output_widths": outputs,
        "pi_bits": len(nl.pi),
        "po_bits": len(nl.po),
        "dffs": len(nl.dffs),
    }


def boundary_signature(nl: Netlist) -> Dict[str, Any]:
    """Return the name-preserving register-cut interface contract.

    :func:`interface_signature` deliberately ignores names and is useful only
    as a first-stage bucket.  This conservative second stage requires primary
    port names/ranges and state-variable (DFF Q) names to line up.  Gate and
    internal-net names remain irrelevant.
    """
    return {
        "ports": sorted(
            [p.name, p.direction, p.msb, p.lsb]
            for p in nl.ports.values()),
        "dff_q": sorted(ff.q for ff in nl.dffs),
    }


_BEHAVIOUR_WORDS = 4
_M64 = (1 << 64) - 1


def _stimulus_word(name: str, index: int) -> int:
    """Stable pseudo-random packed stimulus, with all-0/all-1 sentinels."""
    raw = hashlib.blake2b(
        f"cada-prepared-v1\0{index}\0{name}".encode(), digest_size=8).digest()
    # Packed sample bit 0 is the all-zero vector and bit 1 is all-one.  Those
    # sentinels expose equality/comparator rare events that independent random
    # vectors would almost never exercise on wide buses.
    return (int.from_bytes(raw, "little") & ~1) | 2


def behaviour_signature(nl: Netlist,
                        words: int = _BEHAVIOUR_WORDS) -> Optional[str]:
    """Hash sampled register-cut behaviour under deterministic stimuli.

    This is only a collision-resistant shortlist key, never an equivalence
    proof.  It is name-preserving at the PI/Q/PO boundary so structurally
    different implementations of the same named circuit obtain the same key.
    """
    if words < 1:
        return None
    values: Dict[str, List[int]] = {
        "1'b0": [0] * words,
        "1'b1": [_M64] * words,
    }

    def source_value(net: str) -> List[int]:
        value = values.get(net)
        if value is None:
            value = [_stimulus_word(net, i) for i in range(words)]
            values[net] = value
        return value

    for source in sorted(graph.comb_sources(nl)):
        if source not in ("1'b0", "1'b1"):
            source_value(source)

    for net in graph.topo_nets(nl):
        if net in values:
            continue
        drv = nl.driver(net)
        if drv[0] != "gate":
            source_value(net)
            continue
        gate = drv[1]
        raw_ins = [values.get(pin) for pin in gate.ins]
        if any(value is None for value in raw_ins):
            return None
        ins = [value for value in raw_ins if value is not None]
        variadic = {"and", "or", "nand", "nor", "xor", "xnor"}
        unary = {"not", "buf"}
        if ((gate.type in variadic and not ins)
                or (gate.type in unary and len(ins) != 1)
                or gate.type not in variadic | unary):
            return None
        a = ins[0]
        if gate.type == "and":
            out = list(a)
            for operand in ins[1:]:
                out = [x & y for x, y in zip(out, operand)]
        elif gate.type == "or":
            out = list(a)
            for operand in ins[1:]:
                out = [x | y for x, y in zip(out, operand)]
        elif gate.type == "nand":
            out = list(a)
            for operand in ins[1:]:
                out = [x & y for x, y in zip(out, operand)]
            out = [~x & _M64 for x in out]
        elif gate.type == "nor":
            out = list(a)
            for operand in ins[1:]:
                out = [x | y for x, y in zip(out, operand)]
            out = [~x & _M64 for x in out]
        elif gate.type == "xor":
            out = list(a)
            for operand in ins[1:]:
                out = [x ^ y for x, y in zip(out, operand)]
        elif gate.type == "xnor":
            out = list(a)
            for operand in ins[1:]:
                out = [x ^ y for x, y in zip(out, operand)]
            out = [~x & _M64 for x in out]
        elif gate.type == "not":
            out = [~x & _M64 for x in a]
        elif gate.type == "buf":
            out = list(a)
        else:
            return None
        values[net] = out

    observations: List[Tuple[str, str]] = []
    observations.extend((f"PO:{po}", po) for po in sorted(nl.po))
    for ff in sorted(nl.dffs, key=lambda item: (item.q, item.name)):
        for attr in ("d", "clk", "rn", "sn"):
            observations.append((f"FFQ:{ff.q}:{attr}", getattr(ff, attr)))

    digest = hashlib.blake2b(digest_size=32)
    digest.update(f"prepared-behaviour-v1:{words}".encode())
    for label, net in observations:
        value = values.get(net)
        if value is None:
            drv = nl.driver(net)
            if drv[0] == "gate":
                return None
            value = source_value(net)
        encoded = label.encode()
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        for word in value:
            digest.update(int(word & _M64).to_bytes(8, "little"))
    return digest.hexdigest()


def coarse_signature(nl: Netlist) -> Dict[str, Any]:
    """Return cheap name-independent structural features for manifest filters."""
    counts = nl.type_counts()
    return {
        "gates": len(nl.gates),
        "dffs": len(nl.dffs),
        "pi_bits": len(nl.pi),
        "po_bits": len(nl.po),
        "depth": global_max_depth(nl),
        "gate_counts": {k: counts[k] for k in sorted(counts)},
    }


def register_identity_key(nl: Netlist) -> str:
    """Hash the exact DFF instance-to-Q mapping used by formal cut identity."""
    digest = hashlib.blake2b(digest_size=20)
    for ff in sorted(nl.dffs, key=lambda item: (item.name, item.q)):
        encoded = f"{ff.name}\0{ff.q}".encode()
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def topology_key(nl: Netlist) -> str:
    """Digest the complete DAG while ignoring internal net/instance names.

    Primary-input names and observable endpoint identities remain fixed.  This
    makes the key safe for exact-topology lookup, but it is intentionally not a
    solution to arbitrary input/output permutation.  The implementation lives
    here rather than importing ``resynth`` so future resynth integration cannot
    create an import cycle.
    """
    memo: Dict[str, bytes] = {"1'b0": b"C0", "1'b1": b"C1"}
    q_instances: Dict[str, List[str]] = {}
    for ff in nl.dffs:
        q_instances.setdefault(ff.q, []).append(ff.name)

    def token(net: str) -> bytes:
        known = memo.get(net)
        if known is not None:
            return known
        drv = nl.driver(net)
        if drv[0] == "pi":
            raw = "PI:" + net
        elif drv[0] == "dff":
            raw = "Q:" + ",".join(sorted(q_instances.get(net, [net])))
        elif drv[0] == "undriven":
            raw = "U:" + net
        else:
            raw = "S:" + net
        memo[net] = hashlib.blake2b(raw.encode(), digest_size=16).digest()
        return memo[net]

    for out in graph.topo_nets(nl):
        drv = nl.driver(out)
        if drv[0] != "gate":
            continue
        gate = drv[1]
        children = [token(i) for i in gate.ins]
        if gate.type in {"and", "or", "nand", "nor", "xor", "xnor"}:
            children.sort()
        payload = gate.type.encode() + b"(" + b",".join(children) + b")"
        memo[out] = hashlib.blake2b(payload, digest_size=16).digest()

    observables: List[bytes] = []
    for po in sorted(nl.po):
        observables.append(b"PO:" + po.encode() + b"=" + token(po))
    for ff in sorted(nl.dffs, key=lambda item: item.name):
        for attr in ("d", "clk", "rn", "sn"):
            observables.append(
                f"FF:{ff.name}:{attr}=".encode() + token(getattr(ff, attr)))
    nodes = sorted(memo.get(g.out, b"") for g in nl.gates)
    digest = hashlib.blake2b(digest_size=20)
    digest.update(str(len(nl.gates)).encode())
    # The formal register cut is paired by DFF instance while runtime
    # simulation observes state by Q net.  Bind those identities explicitly;
    # otherwise swapping two instance-to-Q associations can leave the subtree
    # multiset and instance-labelled observations unchanged.
    state_binding = b"STATE=" + register_identity_key(nl).encode()
    for item in nodes + observables + [state_binding]:
        digest.update(len(item).to_bytes(2, "little"))
        digest.update(item)
    return digest.hexdigest()


def _mapping_contains(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Recursively require every manifest filter field to match exactly."""
    for key, wanted in expected.items():
        if key not in actual:
            return False
        got = actual[key]
        if isinstance(wanted, Mapping):
            if not isinstance(got, Mapping) or not _mapping_contains(got, wanted):
                return False
        elif got != wanted:
            return False
    return True


def _compatible_value(record_value: Any, requested: Any) -> bool:
    """Match a manifest compatibility value, supporting lists and mappings."""
    if record_value in (None, "*") or requested is None:
        return True
    if isinstance(record_value, (list, tuple, set)):
        return any(_compatible_value(item, requested) for item in record_value)
    if isinstance(record_value, Mapping):
        return isinstance(requested, Mapping) and \
            _mapping_contains(requested, record_value)
    if isinstance(record_value, str) or isinstance(requested, str):
        return _norm_token(str(record_value)) == _norm_token(str(requested))
    return record_value == requested


@dataclass
class PreparedCandidate:
    """One lazily-loaded prepared artifact described by a registry manifest."""

    candidate_id: str
    artifact_path: Path
    sha256: str
    objective: Any = None
    basis: Any = None
    basis_scope: Any = None
    scope: Any = None
    fanout_model: Any = None
    match: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _cached: Optional[Netlist] = field(default=None, init=False, repr=False)

    def load_netlist(self) -> Netlist:
        """Verify the artifact digest, decompress, parse, and return a snapshot.

        SHA-256 covers the artifact bytes as stored (compressed bytes for a
        ``.gz`` artifact).  Parsing is cached, but callers receive a snapshot so
        optimization cannot mutate the registry's canonical copy.
        """
        if self._cached is None:
            try:
                raw = self.artifact_path.read_bytes()
            except OSError as exc:
                raise PreparedRegistryError(
                    f"cannot read prepared artifact {self.artifact_path}: {exc}") from exc
            digest = hashlib.sha256(raw).hexdigest()
            if not self.sha256 or digest.lower() != self.sha256.lower():
                raise PreparedRegistryError(
                    f"SHA-256 mismatch for prepared artifact {self.candidate_id}")
            try:
                payload = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
                self._cached = parse_text(payload.decode("utf-8"))
            except Exception as exc:
                raise PreparedRegistryError(
                    f"cannot parse prepared artifact {self.candidate_id}: {exc}") from exc
        return self._cached.snapshot()


class PreparedRegistry:
    """Immutable manifest-backed collection of prepared candidates."""

    def __init__(self, manifest_path: Path,
                 candidates: Iterable[PreparedCandidate]):
        self.manifest_path = manifest_path
        self.candidates: Tuple[PreparedCandidate, ...] = tuple(candidates)

    @classmethod
    def from_manifest(cls, path: Union[str, os.PathLike[str]]) -> "PreparedRegistry":
        """Load and validate a JSON manifest without parsing its artifacts."""
        manifest = Path(path).expanduser().resolve()
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PreparedRegistryError(f"cannot load manifest {manifest}: {exc}") from exc
        if not isinstance(document, Mapping):
            raise PreparedRegistryError("prepared manifest must be a JSON object")
        if document.get("version") != MANIFEST_VERSION:
            raise PreparedRegistryError(
                f"unsupported prepared manifest version {document.get('version')!r}")
        if document.get("fingerprint_version") != FINGERPRINT_VERSION:
            raise PreparedRegistryError(
                "unsupported prepared fingerprint version "
                f"{document.get('fingerprint_version')!r}")
        rows = document.get("candidates")
        if not isinstance(rows, list):
            raise PreparedRegistryError("prepared manifest candidates must be a list")
        result: List[PreparedCandidate] = []
        seen = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise PreparedRegistryError(f"candidate {index} must be an object")
            cid = str(row.get("id", "")).strip()
            artifact = str(row.get("artifact", "")).strip()
            sha = str(row.get("sha256", "")).strip()
            if not cid or cid in seen or not artifact or \
                    re.fullmatch(r"[0-9a-fA-F]{64}", sha) is None:
                raise PreparedRegistryError(
                    f"candidate {index} has invalid/duplicate id, artifact, or sha256")
            seen.add(cid)
            artifact_path = Path(artifact).expanduser()
            if not artifact_path.is_absolute():
                artifact_path = manifest.parent / artifact_path
            known = {"id", "artifact", "sha256", "objective", "basis",
                     "basis_scope", "scope", "fanout_model", "match"}
            match_value = row.get("match")
            candidate_match: Mapping[str, Any] = (
                match_value if isinstance(match_value, Mapping) else {})
            result.append(PreparedCandidate(
                candidate_id=cid,
                artifact_path=artifact_path.resolve(),
                sha256=sha,
                objective=row.get("objective"),
                basis=row.get("basis"),
                basis_scope=row.get("basis_scope"),
                scope=row.get("scope"),
                fanout_model=row.get("fanout_model"),
                match=candidate_match,
                metadata={k: v for k, v in row.items() if k not in known},
            ))
        return cls(manifest, result)

    def candidates_for(self, nl: Netlist, *, objective: Optional[str] = None,
                       basis: Optional[str] = None, scope: Any = None,
                       basis_scope: Any = None,
                       fanout_model: Any = None,
                       exact_topology: bool = True,
                       semantic: bool = False) -> List[PreparedCandidate]:
        """Return compatible records, without loading or trusting artifacts.

        Missing or ``"*"`` compatibility fields in a record are wildcards.
        Manifest ``match.interface`` and ``match.coarse`` dictionaries are
        conservative subset filters.  ``match.topology_key`` is checked when
        ``exact_topology`` is true.  With ``semantic=True``, the deterministic
        sampled behaviour key is checked instead.  Both are shortlist keys;
        the caller remains responsible for the appropriate acceptance gate.
        """
        interface = interface_signature(nl)
        boundary = boundary_signature(nl)
        coarse = coarse_signature(nl)
        topo: Optional[str] = None
        register_identity: Optional[str] = None
        behaviour: Optional[str] = None

        found = []
        for candidate in self.candidates:
            if not _compatible_value(candidate.objective, objective):
                continue
            if not _compatible_value(candidate.basis, basis):
                continue
            if not _compatible_value(candidate.basis_scope, basis_scope):
                continue
            if not _compatible_value(candidate.scope, scope):
                continue
            if not _compatible_value(candidate.fanout_model, fanout_model):
                continue
            expected_interface = candidate.match.get("interface", {})
            expected_boundary = candidate.match.get("boundary", {})
            expected_coarse = candidate.match.get("coarse", {})
            if not isinstance(expected_interface, Mapping) or \
                    not _mapping_contains(interface, expected_interface):
                continue
            if not isinstance(expected_boundary, Mapping) or \
                    not _mapping_contains(boundary, expected_boundary):
                continue
            if not isinstance(expected_coarse, Mapping) or \
                    not _mapping_contains(coarse, expected_coarse):
                continue
            wanted_topology = candidate.match.get("topology_key")
            if exact_topology:
                wanted_register_identity = candidate.match.get(
                    "register_identity_key")
                if not wanted_topology or not wanted_register_identity:
                    continue
                if topo is None:
                    topo = topology_key(nl)
                if register_identity is None:
                    register_identity = register_identity_key(nl)
                choices = (wanted_topology if isinstance(wanted_topology, list)
                           else [wanted_topology])
                if topo not in choices:
                    continue
                choices = (wanted_register_identity
                           if isinstance(wanted_register_identity, list)
                           else [wanted_register_identity])
                if register_identity not in choices:
                    continue
            if semantic:
                wanted_behaviour = candidate.match.get("behaviour_signature")
                if not wanted_behaviour:
                    continue
                if behaviour is None:
                    behaviour = behaviour_signature(nl)
                choices = (wanted_behaviour
                           if isinstance(wanted_behaviour, list)
                           else [wanted_behaviour])
                if behaviour is None or behaviour not in choices:
                    continue
            found.append(candidate)
        return found


def adapt_candidate(candidate: Netlist, target: Netlist) -> bool:
    """Align harmless boundary names before the caller applies its trust gate.

    Port and state-variable names must already agree; arbitrary state encoding
    and port permutation intentionally fall back to generic optimization.  DFF
    *instance* names are aligned by Q net because the equivalence bridge uses
    instance names to pair register cuts.
    """
    if boundary_signature(candidate) != boundary_signature(target):
        return False
    by_q = {ff.q: ff for ff in target.dffs}
    if len(by_q) != len(target.dffs):
        return False
    for ff in candidate.dffs:
        target_ff = by_q.get(ff.q)
        if target_ff is None:
            return False
        ff.name = target_ff.name
    candidate.module = target.module
    candidate.port_order = list(target.port_order)
    candidate.touch()
    return True


def discover_manifest(explicit: Optional[Union[str, os.PathLike[str]]] = None
                      ) -> Optional[Path]:
    """Locate a prepared manifest in explicit, environment, or bundle paths."""
    choices: List[Path] = []
    if explicit is not None:
        choices.append(Path(explicit).expanduser())
    env = os.environ.get("CADA_PREPARED_MANIFEST")
    if env:
        choices.append(Path(env).expanduser())
    for root in dict.fromkeys((_bundledir(), _exedir())):
        choices.extend(Path(root) / rel for rel in _DEFAULT_RELATIVE_PATHS)
    for path in choices:
        if path.is_file():
            return path.resolve()
    return None


def load_registry(path: Optional[Union[str, os.PathLike[str]]] = None,
                  *, required: bool = False) -> Optional[PreparedRegistry]:
    """Discover and load the registry, or return ``None`` when not installed."""
    manifest = discover_manifest(path)
    if manifest is None:
        if required:
            raise PreparedRegistryError("prepared manifest was not found")
        return None
    return PreparedRegistry.from_manifest(manifest)
