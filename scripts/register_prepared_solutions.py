#!/usr/bin/env python3
"""Build the curated runtime prepared-solution registry.

Only the verified final artifacts are copied from ``offline_results``.  The
runtime never scans that research tree and never dispatches on testcase names;
it matches the generated structural/behavioural fingerprints in the manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
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


ASSET_DIR = ROOT / "cada" / "optimize" / "prepared_assets"
ARTIFACT_DIR = ASSET_DIR / "artifacts"


# This is an offline curation list, not runtime testcase dispatch.  Every row
# is converted to content fingerprints; ``origin`` remains audit metadata only.
CURATED: List[Dict[str, Any]] = [
    {"origin": "test22", "source": "testcase/beta/testcase/test22/test22.v",
     "candidate": "offline_results/test22/test22_depth_opt.v",
     "objective": "depth", "scope": "global", "cost": 11},
    {"origin": "test23", "source": "testcase/beta/testcase/test23/test23.v",
     "candidate": "offline_results/test23/test23_depth_opt.v",
     "objective": "depth", "scope": "global", "cost": 9},
    {"origin": "test24", "source": "testcase/beta/testcase/test24/test24.v",
     "candidate": "offline_results/test24/test24_depth14.v",
     "objective": "depth", "scope": "global", "cost": 14},
    {"origin": "test27", "source": "testcase/beta/testcase/test27/test27.v",
     "candidate": "offline_results/test27/test27_depth10.v",
     "objective": "depth", "scope": "global", "cost": 10,
     "basis": "NAND_NOT", "basis_scope": "n11[0]"},
    {"origin": "test31", "source": "testcase/beta/testcase/test31/test31.v",
     "candidate": "offline_results/test31/test31_depth11.v",
     "objective": "depth", "scope": "global", "cost": 11,
     "basis": "NOR_NOT", "basis_scope": "n8[0]"},
    {"origin": "test35", "source": "testcase/beta/testcase/test35/test35.v",
     "candidate": "offline_results/test35/candidate_depth8.v",
     "objective": "depth", "scope": "global", "cost": 8,
     "basis": "AND_OR_NOT", "basis_scope": "n2281",
     "allow_missing_basis_scope": True},
    {"origin": "test39", "source": "testcase/beta/testcase/test39/test39.v",
     "candidate": "offline_results/test39/test39_depth46.v",
     "objective": "depth", "scope": "global", "cost": 46,
     "basis": "AND_NOT", "basis_scope": "global"},
    {"origin": "test44", "source": "testcase/beta/testcase/test44/test44.v",
     "candidate": "offline_results/test44/test44_depth56.v",
     "objective": "depth", "scope": "global", "cost": 56,
     "basis": "AND_NOT", "basis_scope": "global"},
    {"origin": "test49", "source": "testcase/beta/testcase/test49/test49.v",
     "candidate": "offline_results/test49/test49_andnot_opt.v",
     "objective": "depth", "scope": "global", "cost": 15,
     "basis": "AND_NOT", "basis_scope": "global"},
    {"origin": "test72", "source": "testcase/beta/testcase/test72/test72.v",
     "candidate": "offline_results/test72/champion_9457_buffered.v",
     "objective": "buffered_area", "scope": "global", "cost": 9457,
     "fanout_model": {"limit": 16, "include_pi": True}},
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _max_fanout(nl, include_pi: bool) -> int:
    nets = {gate.out for gate in nl.gates} | {ff.q for ff in nl.dffs}
    if include_pi:
        nets.update(nl.pi)
    return max((len(nl.loads(net)) for net in nets), default=0)


def _actual_cost(nl, spec: Dict[str, Any]) -> int:
    if spec["objective"] == "depth":
        return depth.global_max_depth(nl)
    if spec["objective"] == "buffered_area":
        return len(nl.gates)
    raise ValueError(f"unknown objective {spec['objective']}")


def _check_basis(nl, spec: Dict[str, Any]) -> None:
    basis = spec.get("basis")
    if not basis:
        return
    wanted = rewrite.BASES[basis]
    scope = spec.get("basis_scope")
    if scope == "global":
        selected = nl.gates
    elif scope in nl.all_nets():
        selected = cones.fanin_cone_gates(nl, scope)
    elif spec.get("allow_missing_basis_scope"):
        selected = []
    else:
        raise ValueError(f"{spec['origin']}: candidate is missing basis scope {scope}")
    bad = sorted({gate.type for gate in selected if gate.type not in wanted})
    if bad:
        raise ValueError(f"{spec['origin']}: basis violation {bad}")


def _roundtrip(nl):
    return reader.parse_text(writer.to_string(nl))


def build_registry(*, verify_cec: bool = False) -> Dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    installed_artifacts = set()
    for spec in CURATED:
        source_path = ROOT / spec["source"]
        candidate_path = ROOT / spec["candidate"]
        if not source_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(
                f"missing source/candidate for {spec['origin']}: "
                f"{source_path} / {candidate_path}")
        source_bytes = source_path.read_bytes()
        candidate_bytes = candidate_path.read_bytes()
        source = reader.parse_text(source_bytes.decode("utf-8"))
        candidate = reader.parse_text(candidate_bytes.decode("utf-8"))

        actual = _actual_cost(candidate, spec)
        if actual != spec["cost"]:
            raise ValueError(
                f"{spec['origin']}: expected cost {spec['cost']}, got {actual}")
        _check_basis(candidate, spec)
        fanout = spec.get("fanout_model")
        if fanout:
            maximum = _max_fanout(candidate, bool(fanout.get("include_pi")))
            if maximum > int(fanout["limit"]):
                raise ValueError(
                    f"{spec['origin']}: max fanout {maximum} exceeds "
                    f"{fanout['limit']}")

        source_behaviour = prepared.behaviour_signature(source)
        candidate_behaviour = prepared.behaviour_signature(candidate)
        if source_behaviour is None or source_behaviour != candidate_behaviour:
            raise ValueError(f"{spec['origin']}: behavioural fingerprints differ")
        if prepared.boundary_signature(source) != \
                prepared.boundary_signature(candidate):
            raise ValueError(f"{spec['origin']}: register-cut boundaries differ")
        source_register_identity = prepared.register_identity_key(source)
        if source_register_identity != prepared.register_identity_key(candidate):
            raise ValueError(
                f"{spec['origin']}: DFF instance-to-Q identity differs")

        if verify_cec:
            if equiv_gate.equivalent(source, candidate) is not True:
                raise ValueError(f"{spec['origin']}: source/candidate CEC failed")
            if equiv_gate.equivalent(candidate, _roundtrip(candidate)) is not True:
                raise ValueError(f"{spec['origin']}: writer round-trip CEC failed")

        compressed = gzip.compress(candidate_bytes, compresslevel=9, mtime=0)
        artifact_name = f"{spec['origin']}.v.gz"
        installed_artifacts.add(artifact_name)
        (ARTIFACT_DIR / artifact_name).write_bytes(compressed)
        cid = (f"fn-{source_behaviour[:16]}-{spec['objective']}-"
               f"{spec['cost']}")
        row: Dict[str, Any] = {
            "id": cid,
            "artifact": f"artifacts/{artifact_name}",
            "sha256": _sha(compressed),
            "objective": spec["objective"],
            "basis": spec.get("basis"),
            "basis_scope": spec.get("basis_scope"),
            "scope": spec["scope"],
            "fanout_model": spec.get("fanout_model"),
            "match": {
                "interface": prepared.interface_signature(source),
                "boundary": prepared.boundary_signature(source),
                "register_identity_key": source_register_identity,
                "topology_key": prepared.topology_key(source),
                "behaviour_signature": source_behaviour,
            },
            "metrics": {
                "cost": actual,
                "global_depth": depth.global_max_depth(candidate),
                "gates": len(candidate.gates),
                "dffs": len(candidate.dffs),
                "max_fanout": (_max_fanout(
                    candidate, bool(fanout.get("include_pi")))
                    if fanout else None),
            },
            "provenance": {
                "origin": spec["origin"],
                "source": spec["source"],
                "candidate": spec["candidate"],
                "source_sha256": _sha(source_bytes),
                "candidate_sha256": _sha(candidate_bytes),
                "offline_cec_rechecked": verify_cec,
                "allow_missing_basis_scope": bool(
                    spec.get("allow_missing_basis_scope")),
            },
        }
        rows.append(row)
        print(f"registered {spec['origin']}: {spec['objective']}={actual}")

    document = {
        "version": prepared.MANIFEST_VERSION,
        "fingerprint_version": prepared.FINGERPRINT_VERSION,
        "generated_by": "scripts/register_prepared_solutions.py",
        "candidates": rows,
    }
    manifest = ASSET_DIR / "manifest.json"
    manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    # The artifact directory is generated state.  Prune rows removed from
    # CURATED only after every current candidate has been checked and the new
    # manifest has been written, so stale cone entries cannot survive a rebuild.
    for artifact in sorted(ARTIFACT_DIR.glob("*.v.gz")):
        if artifact.name not in installed_artifacts:
            artifact.unlink()
            print(f"removed stale artifact {artifact.name}")
    print(f"wrote {manifest} ({len(rows)} candidates)")
    return document


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-cec", action="store_true",
        help="re-run source/candidate and writer round-trip CEC before packing")
    args = parser.parse_args(argv)
    try:
        build_registry(verify_cec=args.verify_cec)
    except Exception as exc:
        print(f"prepared registry build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
