"""Allowed LLM fallback intents and lightweight validation.

Place this file next to fallback.py, for example:
    cada/llm/allowed_intents.py

The important detail is that validation is intent-aware.  For example,
load_design.dir may be "testcase/test41/", while list_ports.dir must be
"input" or "output".  Therefore we must not globally validate every key named
"dir" as a port direction.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Set, Tuple


INTENT_CATALOG = """
You translate a single natural-language EDA (gate-level netlist) request into ONE JSON object.
Output ONLY the JSON — no prose, no markdown fences. Schema:
  {"intent": "<name>", "params": { ... }}

Rules:
• Never reason about or simulate the circuit. Only classify the request type and extract names/numbers.
• If the request does not match any available operation, output: {"intent": "noop", "params": {}}
• Gate/wire/signal names follow the pattern: letter/underscore then alphanumerics, optionally with [index].
  Examples: n14, n440, cg288, renamed_wire, clk, reset_n, q[0].
• "gate", "cell", "primitive", "element", "logic block" are all synonyms — treat identically.

━━━ EDA VOCABULARY REFERENCE ━━━

GATE TYPE SYNONYMS (use the canonical type name in params):
  AND  : conjunction gate, product gate, AND cell
  OR   : disjunction gate, sum gate, OR cell
  NOT  : inverter, INV, complementor, negation gate, NOT cell
  NAND : negative-AND, Sheffer stroke gate, NAND cell, anti-conjunction
  NOR  : negative-OR, Pierce arrow gate, NOR cell, anti-disjunction
  XOR  : exclusive-OR, parity gate, XOR cell, EX-OR, difference gate
  XNOR : exclusive-NOR, equivalence gate, XNOR cell, EX-NOR
  BUF  : buffer, repeater, driver cell, BUF gate, signal repeater
  DFF  : D flip-flop, register, sequential element, memory element, storage cell

NETLIST CONCEPTS:
  Primary Input (PI) : design input, top-level input, circuit input, "PI", port (input)
  Primary Output (PO): design output, top-level output, circuit output, "PO", port (output)
  Fanout            : load count, drive strength, branching factor, output connections, loads driven
  Fanin             : predecessors, input connections, antecedents, drivers
  Cone / Fanin cone : support set, input dependencies, transitive fanin, supply cone, logic cone
  Dangling gate     : dead logic, unobservable cell, logically inert gate, unused logic
  Constant input    : hardwired logic (0/1), static high/low, tied-off input, fixed logic level

GRAPH THEORY (applied to netlist):
  Articulation point: cut vertex, cut point, cut vertice, separator vertex, bridge node, critical node
  Critical path     : longest path, timing critical path, worst-case depth, maximum delay path
  DAG               : directed acyclic graph, combinational graph, logic graph, dependency graph
  PI-to-PO path     : zero-hop path, direct connection, pass-through, combinational path of length 0

OPERATION SYNONYMS:
  begin_case      : initialize testcase, commence session, start benchmark
  load_design     : import netlist, parse circuit, ingest schematic, read design
  write_design    : export netlist, persist circuit, emit design, save to file
  count_gates     : audit gate inventory, cell histogram, primitive count by type, enumerate cell types
  collapse_inv.   : eliminate double inverters, remove tandem NOT chains, short-circuit NOT-NOT pairs,
                    dissolve back-to-back inverters, replace double inversions with wire
  remove_dangling : prune dead logic, excise unobservable cells, eliminate unused gates
  insert_buffers  : add repeaters, apply fanout buffering, buffer signal, replicate driver
  verify_equiv.   : certify equivalence, confirm functionally identical, assert SAT equivalence,
                    prove combinational equivalence, validate against reference netlist
  rename          : relabel, alias, reassign identifier, replace all occurrences of name
  const_propagate : fold constant inputs, eliminate constant-driven gates, constant folding, simplify
  report_const_g. : find/locate/scan gates with static inputs, identify constant-driven cells

━━━ DISAMBIGUATION ━━━

▸ fanout vs max_fanout_of
  Use "fanout {net}" when the request asks WHAT a net drives (the gate list matters).
  Use "max_fanout_of {net}" when the request asks only for a PEAK NUMBER after buffering,
  e.g. "peak load count on n1", "maximum fanout value of n1 now", "highest load on n1".

▸ count_gates vs delta_count
  "count_gates" → current gate inventory broken down by type.
    Triggers: "count all gates", "primitive count per logic family", "audit gate types",
              "how many gates in total", "breakdown by gate type".
  "delta_count {kind}" → change from the PREVIOUS operation (added/removed count).
    Triggers: "how many were excised/absorbed/swept/collapsed/eliminated/deleted/removed",
              "tally of X gates absorbed", "gates removed in last step".
  KEY RULE: if the request contains a past-tense verb referring to the last action
  (excised, absorbed, swept, folded, eliminated, inserted), use delta_count, not count_gates.

▸ enumerate_paths vs articulation
  "enumerate_paths {a, b}" → list all combinational signal paths between two net names.
  "articulation {a, b}"    → find structural graph cut points (articulation points,
    cut vertices, bridge nodes) between two net names. No path listing involved.
  KEY RULE: if the request mentions "bridge node", "cut vertex", "cut vertice",
  "articulation point", use articulation, NOT enumerate_paths.

▸ cone_gate_count vs enumerate_paths
  "cone_gate_count {output}" → list every gate that feeds (contributes to) ONE output net.
    Triggers: "fanin cone of X", "supply cone of X", "logic cone of X",
              "trace back from X", "gates contributing to X", "transitive fanin of X".
  "enumerate_paths {a, b}" → paths between TWO DISTINCT nodes.
  KEY RULE: if only ONE net is named and the request is about its cone/contributors,
  use cone_gate_count, not enumerate_paths.

▸ report_const_gates vs const_propagate
  "report_const_gates {type}" → QUERY only: find/locate/list/identify gates whose inputs
    are hardwired constants. Does not modify the design.
    Triggers: "locate NAND cells with constant inputs", "find AND gates driven by logic 0/1",
              "which NAND gates have a static/hardwired/constant input".
  "const_propagate {type}"   → TRANSFORM: simplify/fold/reduce those gates. Modifies design.
    Triggers: "fold constant inputs", "propagate constants", "simplify constant-input gates",
              "apply constant propagation".

▸ verify_equivalence vs noop
  "verify_equivalence" covers: verify, prove, assert, confirm, check, validate that the
  current design is (combinationally/functionally) equivalent to the original or a snapshot.

▸ DEPTH PATH GROUPS — class endpoints vs concrete nets
  Five class-level depth queries take NO params:
    "from any primary input to any primary output"          → pi_to_po_depth {}
    "from any primary input to any DFF D-pin / register input" → pi_to_dff_depth {}
    "on any register-to-register path"                      → reg_to_reg_depth {}
    "from any register/DFF/flip-flop output to any primary output" → reg_to_po_depth {}
    "maximum combinational logic depth in the design (now)" → global_max_depth {}
  Use "max_depth_between {a, b}" ONLY when BOTH endpoints are concrete net names
  that literally appear in the request (n24[0], n26, ...).
  KEY RULE: "any primary input", "any DFF", "any output" are CLASSES, never param
  values — never emit params like {"a":"primary_input"} or {"b":"DFF"}.
  KEY RULE: bracket indices are part of the name: n31[1] and n31 are different nets.
  Copy the index if the request has one.

▸ targeted gate-type conversion vs convert_basis
  If the request names a SPECIFIC gate type to replace (XOR, XNOR, NAND-with-constant),
  use the targeted intent — xor_to_nand, xnor_to_nor, xor_to_aoi, nand_const1_to_inv —
  NOT convert_basis.  convert_basis rewrites EVERY gate and is only correct when the
  request says to rebuild the whole netlist (or a cone) using only the basis gates.

▸ minimize_depth vs optimize_cone
  "optimize_cone {output}" only when the CONE ITSELF is the thing being optimized
  ("optimize/re-synthesize the cone of n9").
  If the cost function is the DESIGN-level maximum logic depth and a cone is mentioned
  only as a side constraint ("...while ensuring the cone of n11[0] continues to use only
  NAND and NOT gates"), use minimize_depth with that basis: {"basis":"NAND_NOT"}.

▸ fanout vs transitive_fanout vs connected_to_net
  "fanout {net}"            → what the net drives DIRECTLY (list of driven gates/loads).
    Also correct for a clock/reset net: its loads are the flip-flops it clocks.
  "transitive_fanout {net}" → everything downstream through multiple gate levels.
  "connected_to_net {net}"  → every gate touching the net (its driver AND its loads),
    e.g. "list all gates that connect to the renamed signal X".

▸ insert_buffers scope
  "no GATE drives more than k loads"        → {"k":k, "scope":"gate"}
  "no SIGNAL/NET drives more than k loads"  → {"k":k, "scope":"signal"}  (also bounds PIs)

▸ enable/hold structures (flip-flop D-input logic)
  "enable or hold structures", "D input logic ... enable or hold" are NOT net names.
  Report/list request → enable_hold_report {} ; "how many flip-flops ..." → enable_hold_count {}

▸ count_type vs delta_count after a transform
  "How many X gates are NOW in the design (after the conversion)?" → count_type {type}
  (present-tense inventory).  delta_count only for "how many were added/removed/...".
  For gates ADDED by a conversion, kind is "<added-type>_added":
  "How many NOR gates were added by replacing the XNOR gates?" → {"kind":"nor_added"}

▸ reachable_from vs transitive_fanout
  "reachable_from {net}" for "(determine/list/how many) gates reachable from X" —
  reachability CROSSES flip-flops (Q continues the trace).
  "transitive_fanout {net}" only when the request literally says "transitive fanout".

▸ rename kind
  kind echoes the noun used in the request: "rename wire X" → "wire",
  "rename (internal) signal X" → "signal", "rename gate X" → "gate".

━━━ FEW-SHOT EXAMPLES ━━━

"Establish a new test environment. Benchmark name: test38."
→ {"intent":"begin_case","params":{"name":"test38"}}

"Ingest the circuit schematic from testcase/test38/test38.v."
→ {"intent":"load_design","params":{"file":"test38.v","dir":"testcase/test38"}}

"Audit the primitive count per logic family: AND, OR, NOT, NAND, NOR, XOR, XNOR, BUF, DFF."
→ {"intent":"count_gates","params":{}}

"Apply fanout-reduction buffers to net n1 until no driver fanout exceeds 4."
→ {"intent":"insert_buffers","params":{"k":4,"net":"n1","mode":"fanout"}}

"What is the current peak load count on signal n1?"
→ {"intent":"max_fanout_of","params":{"net":"n1"}}

"Which primary input node exhibits the maximum branching factor in this circuit?"
→ {"intent":"highest_fanout_pi","params":{}}

"Dissolve all tandem NOT chains by substituting a plain wire for each pair."
→ {"intent":"collapse_inverters","params":{}}

"Quantify the critical-path gate depth of the design."
→ {"intent":"global_max_depth","params":{}}

"Shorten the worst-case combinational path by restructuring the logic. The cost function is the maximum logic depth of the final design."
→ {"intent":"minimize_depth","params":{}}

"Minimize the total number of gates in the design. The cost function is the total gate count of the final design; smaller is better."
→ {"intent":"minimize_area","params":{}}

"Resynthesize the design to reduce the logic depth as much as possible."
→ {"intent":"minimize_depth","params":{}}

"Re-synthesize the cone of n9 for minimum depth while keeping it NAND and NOT only."
→ {"intent":"optimize_cone","params":{"output":"n9","basis":"NAND_NOT"}}

"Rebuild the entire netlist using only AND and NOT primitives while preserving functional equivalence."
→ {"intent":"convert_basis","params":{"basis":"AND_NOT"}}

"Excise all logically inert gates from the netlist."
→ {"intent":"remove_dangling","params":{}}

"How many gates were excised in the previous step?"
→ {"intent":"delta_count","params":{"kind":"dangling"}}

"Give the tally of NAND gates absorbed by constant folding."
→ {"intent":"delta_count","params":{"kind":"nand"}}

"Identify all zero gate-hop paths from primary inputs to primary outputs."
→ {"intent":"length_zero_paths","params":{}}

"Enumerate the bridge nodes in the combinational DAG spanning from n2 to n14."
→ {"intent":"articulation","params":{"a":"n2","b":"n14"}}

"Alias net n440 as renamed_wire across the entire netlist."
→ {"intent":"rename","params":{"kind":"signal","old":"n440","new":"renamed_wire"}}

"Assert that the current netlist is combinationally equivalent to the loaded original."
→ {"intent":"verify_equivalence","params":{"against":"original"}}

"Locate every NAND cell whose input pins are driven by a static logic constant (0 or 1)."
→ {"intent":"report_const_gates","params":{"type":"nand"}}

"Fold the static input values into the identified NAND gates."
→ {"intent":"const_propagate","params":{"type":"nand"}}

"For output n14, tabulate the primitive-type breakdown within its transitive fanin region."
→ {"intent":"cone_type_counts","params":{"output":"n14"}}

"Trace back from n14 and enumerate every gate in its supply cone."
→ {"intent":"cone_gate_count","params":{"output":"n14"}}

"Emit the modified netlist to the output file test38_out.v."
→ {"intent":"write_design","params":{"file":"test38_out.v"}}

"What is the maximum combinational depth from any primary input to any primary output in the entire design?"
→ {"intent":"pi_to_po_depth","params":{}}

"What is the maximum logic depth from any primary input to any DFF D-pin in this design?"
→ {"intent":"pi_to_dff_depth","params":{}}

"What is the maximum combinational depth from any DFF output to any primary output in this design?"
→ {"intent":"reg_to_po_depth","params":{}}

"Determine the longest combinational path depth from n30 to n31[1]."
→ {"intent":"max_depth_between","params":{"a":"n30","b":"n31[1]"}}

"Determine whether gate g0 lies on any maximum-depth path of the design. Report yes or no."
→ {"intent":"gate_on_max_path","params":{"gate":"g0"}}

"Is output n16 always 0 regardless of all inputs? Report yes or no."
→ {"intent":"output_constant","params":{"output":"n16"}}

"Write the logic expression for n30 using only the primary input names."
→ {"intent":"boolean_equation","params":{"output":"n30"}}

"Report the number of each gate type in the cone of n8."
→ {"intent":"cone_type_counts","params":{"output":"n8"}}

"List all gates with one or more inputs tied to 1'b1."
→ {"intent":"const1_gates","params":{}}

"Check if there are any floating inputs or unconnected output ports in this design."
→ {"intent":"check_floating","params":{}}

"How many floating signals were found?"
→ {"intent":"floating_count","params":{}}

"Report the D input logic of the flip-flops to report any existing enable or hold structures implemented through multiplexers or AND gates."
→ {"intent":"enable_hold_report","params":{}}

"How many flip-flops were found to have enable or hold structures in their D input logic?"
→ {"intent":"enable_hold_count","params":{}}

"Which output has the largest fanin cone?"
→ {"intent":"largest_fanin_cone","params":{}}

"Convert every XNOR gate in this design to an equivalent NOR-only circuit. Ensure the design functionality does not change."
→ {"intent":"xnor_to_nor","params":{}}

"Try to replace all XOR gates in this design with equivalent NAND-only implementations. Each 2-input XOR can be realized with 4 NAND gates."
→ {"intent":"xor_to_nand","params":{}}

"Decompose all XOR gates in the fanin cone of n15 into AND, OR, and NOT gates without changing functionality."
→ {"intent":"xor_to_aoi","params":{"scope":"n15"}}

"Try to replace all 2-input NAND gates that have one input tied to constant 1 with inverters."
→ {"intent":"nand_const1_to_inv","params":{}}

"Perform depth optimization on the combinational logic while ensuring the cone of n11[0] continues to use only NAND and NOT gates. The cost function is the maximum logic depth of the final design; smaller is better."
→ {"intent":"minimize_depth","params":{"basis":"NAND_NOT"}}

"Insert buffers wherever needed so that no gate drives more than 4 loads. Make sure nothing changes functionally."
→ {"intent":"insert_buffers","params":{"k":4,"scope":"gate"}}

"List all gates that now connect to the renamed signal renamed_sig."
→ {"intent":"connected_to_net","params":{"net":"renamed_sig"}}

"What is the transitive fanout of primary input n0? List all gates reachable from n0."
→ {"intent":"fanout","params":{"net":"n0"}}

"How many NAND gates are now in the design after the XOR-to-NAND conversion?"
→ {"intent":"count_type","params":{"type":"nand"}}

"How many NOR gates were added by replacing the XNOR gates?"
→ {"intent":"delta_count","params":{"kind":"nor_added"}}

"Determine all gates reachable from n2."
→ {"intent":"reachable_from","params":{"net":"n2"}}

"Does every path from input n2 to output n12 pass through gate g0? Report yes or no."
→ {"intent":"dominator","params":{"a":"n2","b":"n12","gate":"g0"}}

"Does there exist any pair of internal signals (a, b) already in the netlist such that NAND(a, b) is equivalent to n25?"
→ {"intent":"exists_nand_pair","params":{"target":"n25"}}

"Change the identifier of wire n74 to renamed_wire and update all references."
→ {"intent":"rename","params":{"kind":"wire","old":"n74","new":"renamed_wire"}}

━━━ AVAILABLE INTENTS ━━━
- begin_case {name}
- load_design {file, dir}
- write_design {file}
- count_gates {}
- total_gate_count {}
- count_type {type, scope}      # scope = optional output net: count only inside its fanin cone
- delta_count {kind}
- gate_info {gate}
- list_type {type}
- cone_gate_count {output}
- cone_type_counts {output}
- list_ports {dir}              # dir = input|output
- count_ports {}
- fanout {net}
- max_fanout_of {net}
- connected_to_net {net}        # driver AND loads of a net (e.g. a renamed signal)
- gates_driven_by {gate}
- successors {gate}
- transitive_fanin {net}
- transitive_fanout {net}
- reachable_from {net}
- highest_fanout_pi {}
- shared_cone {a, b}
- connected_to_output {gate}
- path_exists {a, b, avoid}
- enumerate_paths {a, b}
- length_zero_paths {}
- dominator {a, b, gate}
- articulation {a, b}
- is_cut {wire}
- max_depth_between {a, b}      # both endpoints must be concrete net names
- cone_depth {output}
- global_max_depth {}
- pi_to_po_depth {}             # any primary input -> any primary output
- pi_to_dff_depth {}            # any primary input -> any DFF D-pin
- reg_to_reg_depth {}           # any register output -> any register input
- reg_to_po_depth {}            # any register/DFF output -> any primary output
- outputs_depth_gt {k}
- deepest_output {}             # deepest cone (by depth)
- largest_fanin_cone {}         # largest cone (by gate count)
- gate_on_max_path {gate}
- signals_equivalent {a, b}
- output_constant {output}
- depends_on {output, input}
- boolean_equation {output}
- symmetric {output, a, b}
- exists_nand_pair {target}
- ffs_on_clock {clk}
- same_clock {a, b}
- reg_to_reg_paths {}
- enable_hold_report {}
- enable_hold_count {}
- convert_basis {basis, scope}  # basis = NAND_NOT|NOR_NOT|AND_NOT|AND_OR_NOT
- xor_to_nand {scope}
- xnor_to_nor {scope}
- xor_to_aoi {scope}
- nand_const1_to_inv {}
- report_const_gates {type, value}
- const1_gates {}               # list ALL gates (any type, incl. DFF pins) tied to 1'b1
- check_floating {}             # floating inputs / unconnected output ports?
- floating_count {}             # how many floating signals were found
- const_propagate {type}
- collapse_inverters {}
- remove_dangling {}
- merge_duplicates {}
- rename {kind, old, new}       # kind = gate|wire|signal
- insert_buffers {k, net, mode, scope} # mode = fanout|dedicated; scope = gate|signal
- minimize_depth {basis}
- minimize_area {basis}
- optimize_cone {output, basis}
- verify_equivalence {against}  # against = original|pre|last_loaded
- noop {}
"""


ALLOWED_INTENTS: Set[str] = {
    "load_design", "write_design", "count_gates", "total_gate_count",
    "count_type", "delta_count", "gate_info", "list_type",
    "cone_gate_count", "cone_type_counts", "list_ports", "count_ports",
    "fanout", "gates_driven_by", "successors", "transitive_fanin",
    "transitive_fanout", "reachable_from", "highest_fanout_pi",
    "max_fanout_of", "shared_cone", "connected_to_output", "path_exists",
    "enumerate_paths", "length_zero_paths", "dominator", "articulation",
    "is_cut", "max_depth_between", "cone_depth", "global_max_depth",
    "pi_to_dff_depth", "reg_to_reg_depth", "outputs_depth_gt",
    "deepest_output", "largest_fanin_cone", "gate_on_max_path",
    "pi_to_po_depth", "reg_to_po_depth", "check_floating", "floating_count",
    "const1_gates", "connected_to_net", "signals_equivalent",
    "output_constant", "depends_on", "boolean_equation", "symmetric",
    "exists_nand_pair", "ffs_on_clock", "same_clock", "reg_to_reg_paths",
    "enable_hold_report", "enable_hold_count", "convert_basis",
    "xor_to_nand", "xnor_to_nor", "xor_to_aoi", "nand_const1_to_inv",
    "report_const_gates", "const_propagate", "collapse_inverters",
    "remove_dangling", "merge_duplicates", "rename", "insert_buffers",
    "minimize_depth", "minimize_area", "optimize_cone",
    "verify_equivalence", "begin_case", "noop",
}


# Required params only. Optional params are accepted but not required.
REQUIRED_PARAMS: Mapping[str, Set[str]] = {
    "load_design": {"file"},
    "write_design": {"file"},
    "count_type": {"type"},
    "delta_count": {"kind"},
    "gate_info": {"gate"},
    "list_type": {"type"},
    "cone_gate_count": {"output"},
    "cone_type_counts": {"output"},
    "list_ports": {"dir"},
    "fanout": {"net"},
    "gates_driven_by": {"gate"},
    "successors": {"gate"},
    "transitive_fanin": {"net"},
    "transitive_fanout": {"net"},
    "reachable_from": {"net"},
    "max_fanout_of": {"net"},
    "shared_cone": {"a", "b"},
    "connected_to_output": {"gate"},
    "connected_to_net": {"net"},
    "path_exists": {"a", "b"},
    "enumerate_paths": {"a", "b"},
    "dominator": {"a", "b", "gate"},
    "articulation": {"a", "b"},
    "is_cut": {"wire"},
    "max_depth_between": {"a", "b"},
    "cone_depth": {"output"},
    "outputs_depth_gt": {"k"},
    "gate_on_max_path": {"gate"},
    "signals_equivalent": {"a", "b"},
    "output_constant": {"output"},
    "depends_on": {"output", "input"},
    "boolean_equation": {"output"},
    "symmetric": {"output", "a", "b"},
    "exists_nand_pair": {"target"},
    "ffs_on_clock": {"clk"},
    "same_clock": {"a", "b"},
    "rename": {"old", "new"},
    "insert_buffers": {"k"},
    "optimize_cone": {"output"},
    "begin_case": {"name"},
}


OPTIONAL_PARAMS: Mapping[str, Set[str]] = {
    "load_design": {"dir"},
    "count_type": {"scope"},
    "path_exists": {"avoid"},
    "convert_basis": {"basis", "scope"},
    "xor_to_nand": {"scope"},
    "xnor_to_nor": {"scope"},
    "xor_to_aoi": {"scope"},
    "report_const_gates": {"type", "value"},
    "const_propagate": {"type"},
    "rename": {"kind"},
    "insert_buffers": {"net", "mode", "scope"},
    "minimize_depth": {"basis"},
    "minimize_area": {"basis"},
    "optimize_cone": {"basis"},
    "verify_equivalence": {"against"},
}


GATE_TYPES = {"and", "or", "not", "nand", "nor", "xor", "xnor", "buf", "dff"}
BASIS_VALUES = {"NAND_NOT", "NOR_NOT", "AND_NOT", "AND_OR_NOT"}
LIST_PORT_DIRS = {"input", "output"}
RENAME_KINDS = {"gate", "wire", "signal"}
BUFFER_MODES = {"fanout", "dedicated"}
EQUIV_TARGETS = {"original", "pre", "last_loaded"}
CONST_VALUES = {"0", "1", "1'b0", "1'b1"}


def validate_intent_object(obj: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate and normalize one LLM-produced intent object.

    Returns:
        (clean_obj, None) on success
        (None, error_message) on failure
    """
    if not isinstance(obj, dict):
        return None, "LLM output is not a JSON object."

    intent = obj.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        return None, "Missing or invalid intent."
    intent = intent.strip()

    if intent not in ALLOWED_INTENTS:
        return None, f'Unknown intent "{intent}".'

    raw_params = obj.get("params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        return None, "params must be a JSON object."

    required = REQUIRED_PARAMS.get(intent, set())
    optional = OPTIONAL_PARAMS.get(intent, set())
    allowed_keys = required | optional

    missing = [k for k in sorted(required) if _is_missing(raw_params.get(k))]
    if missing:
        return None, f'Missing required param(s) for intent "{intent}": {", ".join(missing)}.'

    clean_params: Dict[str, Any] = {}
    for key in allowed_keys:
        if key in raw_params and not _is_missing(raw_params[key]):
            clean_params[key] = _normalize_param(intent, key, raw_params[key])

    err = _validate_param_values(intent, clean_params)
    if err:
        return None, err

    return {"intent": intent, "params": clean_params}, None


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _normalize_param(intent: str, key: str, value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()

    # Intent-aware normalization.  Do not globally normalize every key named
    # "dir", because load_design.dir is a filesystem path.
    if key == "type" and isinstance(value, str):
        return value.lower()
    if intent == "list_ports" and key == "dir" and isinstance(value, str):
        return value.lower()
    if intent == "rename" and key == "kind" and isinstance(value, str):
        return value.lower()
    if intent == "insert_buffers" and key in ("mode", "scope") and isinstance(value, str):
        return value.lower()
    if intent == "verify_equivalence" and key == "against" and isinstance(value, str):
        return value.lower()
    if key == "basis" and isinstance(value, str):
        return value.upper()
    if key == "k":
        try:
            return int(value)
        except Exception:
            return value
    return value


def _validate_param_values(intent: str, params: Dict[str, Any]) -> Optional[str]:
    if "type" in params:
        t = params["type"]
        if isinstance(t, str) and t.lower() not in GATE_TYPES:
            return f'Invalid gate type "{t}".'

    if intent == "list_ports" and "dir" in params:
        d = params["dir"]
        if d not in LIST_PORT_DIRS:
            return f'Invalid port direction "{d}". Expected input or output.'

    if "basis" in params:
        b = params["basis"]
        if b not in BASIS_VALUES:
            return f'Invalid basis "{b}".'

    if intent == "rename" and "kind" in params:
        kind = params["kind"]
        if kind not in RENAME_KINDS:
            return f'Invalid rename kind "{kind}". Expected gate, wire, or signal.'

    if intent == "insert_buffers" and "mode" in params:
        mode = params["mode"]
        if mode not in BUFFER_MODES:
            return f'Invalid buffer mode "{mode}". Expected fanout or dedicated.'

    if intent == "insert_buffers" and "scope" in params:
        scope = params["scope"]
        if scope not in ("gate", "signal"):
            return f'Invalid buffer scope "{scope}". Expected gate or signal.'

    if intent == "verify_equivalence" and "against" in params:
        against = params["against"]
        if against not in EQUIV_TARGETS:
            return f'Invalid equivalence target "{against}".'

    if intent == "report_const_gates" and "value" in params:
        value = params["value"]
        if isinstance(value, str) and value not in CONST_VALUES:
            return f'Invalid constant value "{value}".'

    if "k" in params:
        k = params["k"]
        if not isinstance(k, int) or k < 1:
            return f'Invalid bound k="{k}". Expected positive integer.'

    return None
