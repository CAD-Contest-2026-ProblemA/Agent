#!/usr/bin/env python3
"""Regression checks for the generic runtime NAND-super cone seed."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cada.analysis import cones, depth  # noqa: E402
from cada.equiv import gate as equiv_gate  # noqa: E402
from cada.netlist import reader, writer  # noqa: E402
from cada.optimize import abc_opt, cone_runtime, resynth  # noqa: E402
from cada.transform import rewrite  # noqa: E402


FAIL_CLOSED_SOURCE = r"""
module top(a, b, c, y);
  input a, b, c;
  output y;
  wire t0, t1;
  nand g0(t0, a, b);
  nand g1(t1, t0, c);
  not g2(y, t1);
endmodule
"""


def _beta(case: str):
    return reader.parse_file(str(
        ROOT / "testcase" / "beta" / "testcase" / case / f"{case}.v"))


def verify_test63_runtime() -> None:
    raw = _beta("test63")
    current = raw.snapshot()
    target = "n12[0]"
    scope = {
        gate.name for gate in cones.fanin_cone_gates(current, target)
    }
    rewrite.to_basis(current, "NAND_NOT", scope_gates=scope)
    assert depth.depth_of_cone(current, target) == 26
    extraction = cone_runtime.extract_cone(current, target)
    assert len(extraction.boundary) == 42
    assert len(extraction.side_exits) == 42

    with patch(
            "cada.optimize.prepared.load_registry",
            side_effect=AssertionError("cone_depth consulted prepared registry")):
        result, changed, info = resynth.resynthesize(
            current,
            objective="cone_depth",
            output=target,
            basis="NAND_NOT",
            basis_output=target,
            timeout=45,
            use_templates=False,
            use_yosys=False,
        )
    assert changed is True
    assert depth.depth_of_cone(result, target) == 11
    assert all(gate.type in rewrite.BASES["NAND_NOT"]
               for gate in cones.fanin_cone_gates(result, target))
    assert info["prepared"]["status"] == "skipped"
    assert info["runtime_cone"]["side_outputs"] == 42
    assert info["runtime_cone"]["nand_super"]["status"] == "accepted"
    assert info["runtime_cone"]["nand_super"]["stage1_depth"] == 12
    assert info["runtime_cone"]["nand_super"]["final_depth"] == 11
    assert info["runtime_cone"]["nand_super"]["cec"] == "passed"
    assert info["whole_design_cec"] == "passed"
    assert equiv_gate.equivalent(raw, result, timeout=30) is True

    roundtrip = reader.parse_text(writer.to_string(result))
    assert depth.depth_of_cone(roundtrip, target) == 11
    assert equiv_gate.equivalent(result, roundtrip, timeout=30) is True


def verify_test90_lower_bound_bypass() -> None:
    raw = _beta("test90")
    current = raw.snapshot()
    rewrite.to_basis(current, "NAND_NOT")
    with patch(
            "cada.optimize.abc_opt.optimize_nand_super",
            side_effect=AssertionError("depth-two cone loaded super library")):
        result, changed, info = resynth.resynthesize(
            current,
            objective="cone_depth",
            output="n14",
            basis="NAND_NOT",
            basis_output=None,
            timeout=5,
        )
    assert changed is False
    assert depth.depth_of_cone(result, "n14") == 2
    assert info["runtime_cone"]["nand_super"]["status"] == "skipped"
    assert all(gate.type in rewrite.BASES["NAND_NOT"]
               for gate in result.gates)
    assert equiv_gate.equivalent(raw, result, timeout=30) is True


def verify_fail_closed_seed() -> None:
    source = reader.parse_text(FAIL_CLOSED_SOURCE)
    with patch.object(abc_opt, "optimize_nand_super", return_value=None):
        result, changed, info = resynth._runtime_nand_super_seed(
            source, "y", 2.0)
    assert result is source and changed is False
    assert info["status"] == "failed"

    wrong = source.snapshot()
    wrong.gates = [
        gate for gate in wrong.gates if gate.out == "y"
    ]
    wrong.gates[0].ins = ["a"]
    wrong.touch()
    with patch.object(abc_opt, "optimize_nand_super", return_value=wrong):
        result, changed, info = resynth._runtime_nand_super_seed(
            source, "y", 5.0)
    assert result is source and changed is False
    assert info["cec"] == "failed"

    with patch.object(abc_opt, "optimize_nand_super", return_value=wrong), \
            patch("cada.optimize.resynth.equiv_gate.equivalent",
                  return_value=None):
        result, changed, info = resynth._runtime_nand_super_seed(
            source, "y", 5.0)
    assert result is source and changed is False
    assert info["cec"] == "inconclusive"


def main() -> int:
    verify_test63_runtime()
    verify_test90_lower_bound_bypass()
    verify_fail_closed_seed()
    print("runtime NAND-super cone checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
