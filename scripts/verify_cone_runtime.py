#!/usr/bin/env python3
"""Fast regression checks for the generic runtime cone-depth flow."""

from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cada.analysis import depth  # noqa: E402
from cada.equiv import gate as equiv_gate  # noqa: E402
from cada.netlist import reader, writer  # noqa: E402
from cada.optimize import cone_runtime, resynth  # noqa: E402
from cada.transform import rewrite  # noqa: E402


SHARED_SOURCE = r"""
module top(a, b, c, clk, y, z, q);
  input a, b, c, clk;
  output y, z, q;
  wire shared, mid;
  and g0(shared, a, b);
  xor g1(mid, shared, c);
  not g2(y, mid);
  or g3(z, shared, c);
  dff ff0(.RN(1'b1), .SN(1'b1), .CK(clk), .D(shared), .Q(q));
endmodule
"""


GLOBAL_BASIS_SOURCE = r"""
module top(a, b, y, z);
  input a, b;
  output y, z;
  wire t;
  nand g0(t, a, b);
  not g1(y, t);
  or g2(z, a, b);
endmodule
"""


def verify_extract_and_splice() -> None:
    source = reader.parse_text(SHARED_SOURCE)
    extraction = cone_runtime.extract_cone(
        source, "y", preserve_nets=("mid",))
    assert len(extraction.removed_gate_outputs) == 3
    assert extraction.boundary_nets == ("a", "b", "c")
    assert {net for _alias, net in extraction.side_exits} == {"mid", "shared"}
    assert depth.depth_of_cone(extraction.cone,
                               extraction.root_alias) == 3

    # The writer/reader boundary is part of the real runtime path.
    roundtrip = reader.parse_text(writer.to_string(extraction.cone))
    rebuilt = cone_runtime.splice_cone(source, extraction, roundtrip)
    assert depth.depth_of_cone(rebuilt, "y") == 3
    assert len(rebuilt.gates) == len(source.gates)
    assert equiv_gate.equivalent(source, rebuilt, timeout=10) is True

    # Structural splice accepts a well-formed cone, but the verified wrapper
    # must reject a Boolean change at the whole-design CEC gate.
    wrong = extraction.cone.snapshot()
    root_gate = next(
        gate for gate in wrong.gates if gate.out == extraction.root_alias)
    assert root_gate.type == "not"
    root_gate.type = "buf"
    wrong.touch()
    try:
        cone_runtime.splice_cone_verified(
            source, extraction, wrong, timeout=10)
    except cone_runtime.ConeTransformError:
        pass
    else:
        raise AssertionError("wrong cone escaped whole-design CEC")


def verify_runtime_routing() -> None:
    source = reader.parse_text(GLOBAL_BASIS_SOURCE)
    with patch(
            "cada.optimize.prepared.load_registry",
            side_effect=AssertionError("cone_depth consulted prepared registry")):
        result, changed, info = resynth.resynthesize(
            source,
            objective="cone_depth",
            output="y",
            basis="NAND_NOT",
            basis_output=None,
            timeout=5,
            use_templates=False,
            use_yosys=False,
        )
    assert changed is True
    assert depth.depth_of_cone(result, "y") == 2
    assert all(gate.type in rewrite.BASES["NAND_NOT"]
               for gate in result.gates)
    assert info["prepared"]["status"] == "skipped"
    assert info["runtime_cone"]["gates"] == 2
    assert info["whole_design_cec"] == "passed"
    result_roundtrip = reader.parse_text(writer.to_string(result))
    assert equiv_gate.equivalent(result, result_roundtrip, timeout=10) is True

    # A legal register-cut source is a depth-zero cone and a safe no-op.
    sequential = reader.parse_text(SHARED_SOURCE)
    result, changed, info = resynth.resynthesize(
        sequential, objective="cone_depth", output="q", timeout=1)
    assert result is sequential
    assert changed is False
    assert info["runtime_cone"]["status"] == "source-boundary"

    for missing in (None, "not_a_net"):
        try:
            resynth.resynthesize(
                sequential, objective="cone_depth", output=missing, timeout=1)
        except ValueError:
            pass
        else:
            raise AssertionError("missing cone-depth target was accepted")


def main() -> int:
    verify_extract_and_splice()
    verify_runtime_routing()
    print("runtime cone-depth checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
