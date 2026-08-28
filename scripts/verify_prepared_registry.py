#!/usr/bin/env python3
"""Validate the packaged prepared-solution registry.

The default fast mode is suitable as a build gate: it checks every compressed
artifact, parses it, and recomputes its contract and metrics without needing
the research tree.  ``--full`` additionally reloads each recorded source and
runs source/candidate plus writer-roundtrip CEC.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cada.analysis import cones, depth  # noqa: E402
from cada.equiv import gate as equiv_gate  # noqa: E402
from cada.netlist import reader, writer  # noqa: E402
from cada.optimize import prepared  # noqa: E402
from cada.transform import rewrite  # noqa: E402


DEFAULT_MANIFEST = (
    ROOT / "cada" / "optimize" / "prepared_assets" / "manifest.json")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _max_fanout(nl, include_pi: bool) -> int:
    nets = {gate.out for gate in nl.gates} | {ff.q for ff in nl.dffs}
    if include_pi:
        nets.update(nl.pi)
    return max((len(nl.loads(net)) for net in nets), default=0)


def _cost(nl, row: Dict[str, Any]) -> int:
    if row["objective"] == "depth":
        return depth.global_max_depth(nl)
    if row["objective"] == "buffered_area":
        return len(nl.gates)
    raise ValueError(f"unsupported objective {row['objective']}")


def _check_basis(nl, row: Dict[str, Any]) -> None:
    basis = row.get("basis")
    if not basis:
        return
    wanted = rewrite.BASES[basis]
    scope = row.get("basis_scope")
    if scope == "global":
        gates = nl.gates
    elif scope in nl.all_nets():
        gates = cones.fanin_cone_gates(nl, scope)
    elif row.get("provenance", {}).get("allow_missing_basis_scope"):
        gates = []
    else:
        raise ValueError(f"missing basis scope {scope}")
    bad = sorted({gate.type for gate in gates if gate.type not in wanted})
    if bad:
        raise ValueError(f"basis {basis} violated by {bad}")


def verify(path: Path, *, full: bool, timeout: int) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("fingerprint_version") != prepared.FINGERPRINT_VERSION:
        raise ValueError("unsupported/missing prepared fingerprint_version")
    registry = prepared.PreparedRegistry.from_manifest(path)
    rows = document["candidates"]
    if any(row.get("objective") == "cone_depth" for row in rows):
        raise ValueError(
            "cone_depth entries are not allowed in the whole-design "
            "prepared registry")
    if len(rows) != len(registry.candidates):
        raise ValueError("manifest/runtime candidate counts differ")
    referenced = set()

    for row, record in zip(rows, registry.candidates):
        referenced.add(record.artifact_path.resolve())
        nl = record.load_netlist()
        metrics = row.get("metrics") or {}
        actual = _cost(nl, row)
        if actual != metrics.get("cost"):
            raise ValueError(
                f"{record.candidate_id}: cost {actual} != {metrics.get('cost')}")
        expected_metrics = {
            "global_depth": depth.global_max_depth(nl),
            "gates": len(nl.gates),
            "dffs": len(nl.dffs),
        }
        for key, value in expected_metrics.items():
            if metrics.get(key) != value:
                raise ValueError(
                    f"{record.candidate_id}: {key} {value} != {metrics.get(key)}")
        _check_basis(nl, row)
        fanout = row.get("fanout_model")
        if fanout:
            maximum = _max_fanout(nl, bool(fanout.get("include_pi")))
            if maximum > int(fanout["limit"]):
                raise ValueError(
                    f"{record.candidate_id}: max fanout {maximum} exceeds "
                    f"{fanout['limit']}")
            if metrics.get("max_fanout") != maximum:
                raise ValueError(
                    f"{record.candidate_id}: recorded max fanout is stale")

        raw = record.artifact_path.read_bytes()
        uncompressed = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
        provenance = row.get("provenance") or {}
        if provenance.get("offline_cec_rechecked") is not True:
            raise ValueError(
                f"{record.candidate_id}: packaged artifact lacks offline CEC")
        if _sha(uncompressed) != provenance.get("candidate_sha256"):
            raise ValueError(
                f"{record.candidate_id}: uncompressed candidate hash mismatch")
        if prepared.boundary_signature(nl) != row["match"].get("boundary"):
            raise ValueError(f"{record.candidate_id}: boundary signature mismatch")
        if prepared.register_identity_key(nl) != \
                row["match"].get("register_identity_key"):
            raise ValueError(
                f"{record.candidate_id}: register identity mismatch")
        if prepared.behaviour_signature(nl) != \
                row["match"].get("behaviour_signature"):
            raise ValueError(f"{record.candidate_id}: behaviour signature mismatch")

        if full:
            source_path = ROOT / provenance["source"]
            source_bytes = source_path.read_bytes()
            if _sha(source_bytes) != provenance.get("source_sha256"):
                raise ValueError(f"{record.candidate_id}: source hash mismatch")
            source = reader.parse_text(source_bytes.decode("utf-8"))
            if prepared.topology_key(source) != row["match"].get("topology_key"):
                raise ValueError(f"{record.candidate_id}: source topology mismatch")
            if prepared.interface_signature(source) != \
                    row["match"].get("interface"):
                raise ValueError(f"{record.candidate_id}: source interface mismatch")
            if prepared.boundary_signature(source) != row["match"].get("boundary"):
                raise ValueError(f"{record.candidate_id}: source boundary mismatch")
            if prepared.register_identity_key(source) != \
                    row["match"].get("register_identity_key"):
                raise ValueError(
                    f"{record.candidate_id}: source register identity mismatch")
            if prepared.behaviour_signature(source) != \
                    row["match"].get("behaviour_signature"):
                raise ValueError(f"{record.candidate_id}: source behaviour mismatch")
            if equiv_gate.equivalent(source, nl, timeout=timeout) is not True:
                raise ValueError(f"{record.candidate_id}: source/candidate CEC failed")
            roundtrip = reader.parse_text(writer.to_string(nl))
            if equiv_gate.equivalent(nl, roundtrip, timeout=timeout) is not True:
                raise ValueError(f"{record.candidate_id}: round-trip CEC failed")
        print(f"PASS {provenance.get('origin', record.candidate_id)} cost={actual}")

    artifact_dir = path.parent / "artifacts"
    installed = {item.resolve() for item in artifact_dir.glob("*.v.gz")}
    if installed != referenced:
        missing = sorted(str(item) for item in referenced - installed)
        stale = sorted(str(item) for item in installed - referenced)
        raise ValueError(f"artifact set mismatch; missing={missing}, stale={stale}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full", action="store_true",
                        help="also run source/candidate and round-trip CEC")
    parser.add_argument("--timeout", type=int, default=280,
                        help="per-tool CEC timeout in full mode")
    args = parser.parse_args(argv)
    try:
        verify(args.manifest.resolve(), full=args.full, timeout=args.timeout)
    except Exception as exc:
        print(f"prepared registry verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"verified {args.manifest} ({'full' if args.full else 'fast'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
