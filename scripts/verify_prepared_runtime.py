#!/usr/bin/env python3
"""Small end-to-end checks for prepared lookup and resynth integration."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cada.analysis import depth  # noqa: E402
from cada.equiv import gate as equiv_gate  # noqa: E402
from cada.equiv import pattern as pattern_equiv  # noqa: E402
from cada.netlist.reader import parse_text  # noqa: E402
from cada.optimize import prepared, resynth  # noqa: E402


SOURCE = """
module top(a, b, c, d, o);
  input a, b, c, d;
  output o;
  wire t0, t1;
  and g0(t0, a, b);
  and g1(t1, t0, c);
  and g2(o, t1, d);
endmodule
"""

GOOD = """
module cached(a, b, c, d, o);
  input a, b, c, d;
  output o;
  wire t0, t1;
  and x0(t0, a, b);
  and x1(t1, c, d);
  and x2(o, t0, t1);
endmodule
"""

SEMANTIC_VARIANT = """
module another(a, b, c, d, o);
  input a, b, c, d;
  output o;
  wire inv, ab, abc;
  nand h0(inv, a, b);
  not h1(ab, inv);
  and h2(abc, ab, c);
  and h3(o, abc, d);
endmodule
"""

WRONG = """
module cached(a, b, c, d, o);
  input a, b, c, d;
  output o;
  wire t0, t1;
  or x0(t0, a, b);
  or x1(t1, c, d);
  or x2(o, t0, t1);
endmodule
"""

NAND_GOOD = """
module cached(a, b, c, d, o);
  input a, b, c, d;
  output o;
  wire nab, ab, ncd, cd, no;
  nand x0(nab, a, b);
  not x1(ab, nab);
  nand x2(ncd, c, d);
  not x3(cd, ncd);
  nand x4(no, ab, cd);
  not x5(o, no);
endmodule
"""

DFF_A = """
module top(clk, a, q);
  input clk, a;
  output q;
  dff ff0(.RN(1'b1), .SN(1'b1), .CK(clk), .D(a), .Q(q));
endmodule
"""

DFF_WRONG_D = """
module top(clk, a, q);
  input clk, a;
  output q;
  wire na;
  not g0(na, a);
  dff ff0(.RN(1'b1), .SN(1'b1), .CK(clk), .D(na), .Q(q));
endmodule
"""

DFF_IDENTITY_A = """
module top(i, o);
  input i;
  output o;
  wire q0, q1, n0;
  not g0(n0, q0);
  buf go(o, q0);
  dff A(.RN(1'b1), .SN(1'b1), .CK(i), .D(n0), .Q(q0));
  dff B(.RN(1'b1), .SN(1'b1), .CK(i), .D(q1), .Q(q1));
endmodule
"""

DFF_IDENTITY_SWAPPED = """
module top(i, o);
  input i;
  output o;
  wire q0, q1, n0;
  not g0(n0, q1);
  buf go(o, q1);
  dff A(.RN(1'b1), .SN(1'b1), .CK(i), .D(n0), .Q(q1));
  dff B(.RN(1'b1), .SN(1'b1), .CK(i), .D(q0), .Q(q0));
endmodule
"""

FLOAT_A = """
module top(o);
  output o;
  wire floating;
  buf g0(o, floating);
endmodule
"""

FLOAT_WRONG = """
module top(o);
  output o;
  wire floating;
  not g0(o, floating);
endmodule
"""

NARY = """
module top(a, b, c, o);
  input a, b, c;
  output o;
  and g0(o, a, b, c);
endmodule
"""

NARY_BINARY = """
module top(a, b, c, o);
  input a, b, c;
  output o;
  wire ab;
  and g0(ab, a, b);
  and g1(o, ab, c);
endmodule
"""


def _write_registry(root: Path, candidate_text: str, *, basis=None,
                    basis_scope=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = parse_text(SOURCE)
    compressed = gzip.compress(candidate_text.encode(), mtime=0)
    artifact = root / "candidate.v.gz"
    artifact.write_bytes(compressed)
    row = {
        "id": "tiny-depth2",
        "artifact": artifact.name,
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "objective": "depth",
        "basis": basis,
        "basis_scope": basis_scope,
        "scope": "global",
        "fanout_model": None,
        "match": {
            "interface": prepared.interface_signature(source),
            "boundary": prepared.boundary_signature(source),
            "register_identity_key": prepared.register_identity_key(source),
            "topology_key": prepared.topology_key(source),
            "behaviour_signature": prepared.behaviour_signature(source),
        },
        "provenance": {
            "offline_cec_rechecked": True,
            "allow_missing_basis_scope": False,
        },
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "version": 1,
        "fingerprint_version": prepared.FINGERPRINT_VERSION,
        "candidates": [row],
    }))
    return manifest


def main() -> int:
    old = os.environ.get("CADA_PREPARED_MANIFEST")
    try:
        with tempfile.TemporaryDirectory(prefix="cada_prepared_test_") as tmp:
            root = Path(tmp)
            manifest = _write_registry(root, GOOD)
            os.environ["CADA_PREPARED_MANIFEST"] = str(manifest)
            source = parse_text(SOURCE)

            registry = prepared.load_registry(required=True)
            assert registry is not None
            query: Dict[str, Any] = dict(
                objective="depth", basis=None, scope="global",
                basis_scope=None, fanout_model=None)
            exact = registry.candidates_for(
                source, exact_topology=True, **query)
            assert len(exact) == 1
            semantic = registry.candidates_for(
                parse_text(SEMANTIC_VARIANT), exact_topology=False,
                semantic=True, **query)
            assert len(semantic) == 1
            assert not registry.candidates_for(
                source, objective="area", basis=None, scope="global",
                basis_scope=None, fanout_model=None, exact_topology=True)

            identity_a = parse_text(DFF_IDENTITY_A)
            identity_swapped = parse_text(DFF_IDENTITY_SWAPPED)
            assert (prepared.boundary_signature(identity_a)
                    == prepared.boundary_signature(identity_swapped))
            assert (prepared.register_identity_key(identity_a)
                    != prepared.register_identity_key(identity_swapped))
            assert (prepared.topology_key(identity_a)
                    != prepared.topology_key(identity_swapped))

            # Prepared acceptance itself must not call the formal CEC gate.
            # A two-second request is too short to launch an online arm, so
            # any call here would necessarily be the removed prepared CEC.
            with patch.object(
                    resynth.equiv_gate, "equivalent",
                    side_effect=AssertionError("prepared stage called CEC")):
                result, improved, info = resynth.resynthesize(
                    source, objective="depth", timeout=2,
                    use_templates=False, use_yosys=False)
            assert improved
            assert depth.global_max_depth(result) == 2
            assert info["prepared"]["status"] == "accepted"
            assert info["prepared"]["mode"] == "exact"
            assert info["prepared"]["validation"]["patterns"] == 4096
            assert equiv_gate.equivalent(source, result) is True

            # A combined basis+opt request with a prepared hit must not spend
            # time proving the generic basis conversion before lookup.
            basis_manifest = _write_registry(
                root / "basis", NAND_GOOD, basis="NAND_NOT",
                basis_scope="global")
            os.environ["CADA_PREPARED_MANIFEST"] = str(basis_manifest)
            with patch.object(
                    resynth.equiv_gate, "equivalent",
                    side_effect=AssertionError("basis hit called CEC")):
                basis_result, basis_improved, basis_info = resynth.resynthesize(
                    parse_text(SOURCE), objective="depth", basis="NAND_NOT",
                    timeout=2, use_templates=False, use_yosys=False)
            assert basis_improved
            assert resynth._basis_compliant(
                basis_result, "NAND_NOT", None)
            assert basis_info["basis_fallback"] == "deferred-for-prepared"
            assert basis_info["prepared"]["status"] == "accepted"
            assert basis_info["prepared"]["validation"]["patterns"] == 4096
            os.environ["CADA_PREPARED_MANIFEST"] = str(manifest)

            result, improved, info = resynth.resynthesize(
                parse_text(SEMANTIC_VARIANT), objective="depth", timeout=2,
                use_templates=False, use_yosys=False)
            assert improved
            assert info["prepared"]["status"] == "accepted"
            assert info["prepared"]["mode"] == "semantic"
            assert info["prepared"]["validation"]["patterns"] == 100_000
            assert equiv_gate.equivalent(
                parse_text(SEMANTIC_VARIANT), result) is True

            # Conversely, a later online candidate that reproduces the
            # simulation-accepted prepared topology still receives CEC.
            cec_calls = []

            def online_cec(before, after, timeout=280):
                cec_calls.append((before, after, timeout))
                return True

            with patch.object(resynth.abc_opt, "_opt_blif",
                              return_value=("dummy", {}, {})), \
                    patch.object(resynth.abc_opt, "optimize_comb",
                                 return_value=parse_text(GOOD)), \
                    patch.object(resynth.equiv_gate, "equivalent",
                                 side_effect=online_cec):
                resynth.resynthesize(
                    parse_text(SOURCE), objective="depth", timeout=5,
                    use_templates=False, use_yosys=False)
            assert cec_calls, "online candidate skipped the ordinary CEC gate"

            # A non-matching objective leaves the ordinary optimizer in
            # charge; registry absence is not an error path.
            result, _improved, info = resynth.resynthesize(
                parse_text(SOURCE), objective="area", timeout=2,
                use_templates=False, use_yosys=False)
            assert info["prepared"]["status"] == "miss"
            assert equiv_gate.equivalent(parse_text(SOURCE), result) is True

            # A fingerprint hit is still only a hint: a wrong artifact must be
            # rejected by the prepared simulation gate and never escape.
            wrong_manifest = _write_registry(root / "wrong", WRONG)
            os.environ["CADA_PREPARED_MANIFEST"] = str(wrong_manifest)
            result, _improved, info = resynth.resynthesize(
                source, objective="depth", timeout=2,
                use_templates=False, use_yosys=False)
            assert info["prepared"]["status"] == "rejected"
            assert equiv_gate.equivalent(source, result) is True

            # Corruption is detected before parsing and degrades to a normal
            # registry error instead of trusting the modified bytes.
            os.environ["CADA_PREPARED_MANIFEST"] = str(manifest)
            corrupt_registry = prepared.load_registry(required=True)
            assert corrupt_registry is not None
            record = corrupt_registry.candidates[0]
            raw = record.artifact_path.read_bytes()
            record.artifact_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            try:
                record.load_netlist()
            except prepared.PreparedRegistryError:
                pass
            else:
                raise AssertionError("corrupt artifact was accepted")

            result, _improved, info = resynth.resynthesize(
                parse_text(SOURCE), objective="depth", timeout=2,
                use_templates=False, use_yosys=False)
            assert info["prepared"]["status"] == "rejected"
            assert equiv_gate.equivalent(parse_text(SOURCE), result) is True

            # Missing scoped nets are rejected generally; only a curated
            # manifest waiver may represent the legacy eliminated-target
            # contract used by one benchmark.
            assert not resynth._basis_compliant(
                parse_text(GOOD), "AND_OR_NOT", "missing_scope")
            assert resynth._basis_compliant(
                parse_text(GOOD), "AND_OR_NOT", "missing_scope",
                allow_missing_scope=True)

            dff_result = pattern_equiv.equivalent(
                parse_text(DFF_A), parse_text(DFF_WRONG_D), patterns=100_000)
            assert dff_result.matched is False
            assert dff_result.mismatch_observation == "FFQ:q:d"
            floating_result = pattern_equiv.equivalent(
                parse_text(FLOAT_A), parse_text(FLOAT_WRONG), patterns=100_000)
            assert floating_result.matched is False
            nary = parse_text(NARY)
            nary_binary = parse_text(NARY_BINARY)
            assert (prepared.behaviour_signature(nary)
                    == prepared.behaviour_signature(nary_binary))
            nary_result = pattern_equiv.equivalent(
                nary, nary_binary, patterns=100_000)
            assert nary_result.matched is True
    finally:
        if old is None:
            os.environ.pop("CADA_PREPARED_MANIFEST", None)
        else:
            os.environ["CADA_PREPARED_MANIFEST"] = old
    print("prepared runtime checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
