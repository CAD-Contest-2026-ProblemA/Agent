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
                      "What is the fanout of X?" = "How many loads on X?" = fanout {net: X}
                      NOTE: "directly drive/feed" → successors (not fanout); see disambiguation section
  Fanin             : predecessors, input connections, antecedents, drivers
  Cone / Fanin cone : support set, input dependencies, transitive fanin, supply cone, logic cone
                      "belong to the fanin cone of X" = "are in the fanin cone of X" = members of the cone
                      "rooted at output X" = "of output X" (cone depth / cone gate count queries)
  Dangling gate     : dead logic, unobservable cell, logically inert gate, unused logic
  Constant input    : hardwired logic (0/1), static high/low, tied-off input, fixed logic level

PATH AVOIDANCE SYNONYMS ("avoid node Z" in path_exists):
  bypasses node Z, skips node Z, skips Z, circumvents Z, does not pass through Z,
  without going through Z, not traversing Z, excluding Z, steering clear of Z,
  does not traverse Z, avoiding Z, avoids Z, while avoiding Z

DEPTH / PATH-LENGTH SYNONYMS (for max_depth_between {a, b}):
  "Find the longest combinational path from X to Y" = max_depth_between
  "Find the longest path from X to Y" = max_depth_between
  "maximum path depth between X and Y" = max_depth_between
  "what is the maximum depth between X and Y" = max_depth_between
  Compare: "maximum logic depth FROM X TO Y" uses max_depth_between too.
  These are all the SAME intent; only the phrasing differs.

FANOUT / SUCCESSORS SYNONYMS:
  gates_driven_by {gate} : "how many gates does X drive", "how many logic gates does X directly drive",
                           "direct fanout count of gate X", "how many cells are driven by X",
                           "number of gates driven by X"
  successors {gate}      : "direct output gates of X", "direct children of gate X",
                           "immediate successors of X", "gates that X directly feeds",
                           "list the next gates after X", "output neighbors of X"

TRANSITIVE CONE SYNONYMS:
  transitive_fanin {net} : "transitive fanin of X", "all gates in the transitive fanin of X",
                           "find all gates that feed X transitively",
                           "transitive fanin cone of X", "gates in the transitive input cone of X"
  transitive_fanout {net}: "transitive fanout of X", "all gates in the transitive fanout of X",
                           "find all gates reachable from X", "gates reachable from X"

FUNCTIONAL EQUIVALENCE SYNONYMS (signals_equivalent {a, b}):
  "logically equivalent" = "functionally equivalent"
  "always yield the same Boolean output" = "identical logic values for all inputs"
  "logically the same signal" = "functionally equivalent"
  "same output for all input combinations" = signals_equivalent
  "produce identical results" = signals_equivalent

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
                    dissolve back-to-back inverters, dissolve back-to-back inverter chains into a wire,
                    replace double inversions with wire
  remove_dangling : prune dead logic, excise unobservable cells, eliminate unused gates,
                    excise logically inert cells, excise floating logic nodes,
                    remove cells that drive no output, delete gates not connected to any primary output
  insert_buffers  : add buffer gates so no net/signal drives more than X loads,
                    add repeaters, apply fanout buffering, buffer signal, replicate driver
  minimize_depth  : shorten the worst-case combinational path, rebalance logic depth,
                    reduce the longest path, lower the maximum logic depth,
                    reduce the combinational delay, shorten the worst-case path
  convert_basis   : translate logic using only X and Y gates, transform logic to use only X and Y,
                    rebuild netlist using only X and Y primitives, remap to X-only basis
  xor_to_aoi      : break down XOR gates into AND/OR/NOT, break down XOR cells into AND, OR, and NOT,
                    expand XOR into AND/OR/NOT primitives, decompose XOR into AOI gates
  merge_duplicates: identify and combine duplicate gates, consolidate equivalent gate pairs,
                    find and deduplicate equivalent cells, combine functionally identical gates
  verify_equiv.   : certify equivalence, confirm functionally identical, assert SAT equivalence,
                    prove combinational equivalence, validate against reference netlist
  rename          : relabel X as Y, alias, reassign identifier, replace all occurrences of name
  const_propagate : fold constant inputs, eliminate constant-driven gates, constant folding, simplify
  report_const_g. : find/locate/scan gates with static inputs, identify constant-driven cells
  xnor_to_nor     : transform XNOR cells to NOR-only circuit, substitute XNOR with NOR equivalents,
                    replace XNOR gates with NOR-only implementation, convert XNOR to NOR
  xor_to_nand     : replace XOR with 4-NAND circuit, substitute XOR with NAND-only equivalents,
                    convert XOR gates to NAND-only implementation (each 2-input XOR → 4 NANDs)
  nand_const1_inv : substitute NAND-with-constant-1 with inverter, replace NAND+const-1 with NOT gate

━━━ DISAMBIGUATION ━━━

▸ fanout vs max_fanout_of
  Use "fanout {net}" when the request asks WHAT a net drives (the gate list matters).
  Use "max_fanout_of {net}" when the request asks only for a PEAK NUMBER after buffering,
  e.g. "peak load count on n1", "maximum fanout value of n1 now", "highest load on n1".

▸ count_gates vs count_type vs delta_count
  "count_gates" → current gate inventory broken down by ALL types (full histogram).
    Triggers: "count all gates", "primitive count per logic family", "audit gate types",
              "how many gates in total", "breakdown by gate type".
  "count_type {type}" → current count of ONE specific gate type in the ENTIRE design.
    Triggers: "how many NAND gates are now in the design?",
              "how many NOT gates are currently in the design?",
              "how many X gates are now in the [translated/restructured] cone of Y?" ← IMPORTANT:
                even when the prompt says "in the cone of Y", use count_type (total design count),
                because the operation just done applies to the whole design or the count reflects
                the current full-design state. cone_type_counts would give a sub-cone breakdown.
              "how many X gates are now in the design after the Y transformation?" ← IMPORTANT:
                "now in the design" + "after transformation" → count_type (current total count).
                NOT delta_count — delta_count only when "added", "removed", "excised", "inserted".
    Examples:
      "How many NAND gates are now in the restructured cone of output n8?" → count_type {type:"nand"}
      "How many NOT gates are currently in the design?" → count_type {type:"not"}
      "How many NOR gates are now in the design after the XNOR-to-NOR transformation?" → count_type {type:"nor"}
      "How many NAND gates are now in the design after the XOR-to-NAND conversion?" → count_type {type:"nand"}
  "delta_count {kind}" → change from the PREVIOUS operation (added/removed count).
    Triggers: "how many were excised/absorbed/swept/collapsed/eliminated/deleted/removed",
              "how many X gates were ADDED/REMOVED by Y", "tally of X gates absorbed",
              "gates removed in last step", "how many were pruned/merged/inserted/found".
  KEY RULE: if the request contains a past-tense verb referring to the last action
  (excised, absorbed, swept, folded, eliminated, inserted, added, pruned, merged, found),
  use delta_count. If the request says "are now" or "currently", use count_type.

▸ enumerate_paths vs articulation
  "enumerate_paths {a, b}" → list all combinational signal paths between two net names.
  "articulation {a, b}"    → find structural graph cut points (articulation points,
    cut vertices, bridge nodes) between two net names. No path listing involved.
  KEY RULE: if the request mentions "bridge node", "cut vertex", "cut vertice",
  "articulation point", use articulation, NOT enumerate_paths.

▸ cone_gate_count vs enumerate_paths
  "cone_gate_count {output}" → list every gate that feeds (contributes to) ONE output net.
    Triggers: "fanin cone of X", "supply cone of X", "logic cone of X",
              "trace back from X", "gates contributing to X", "transitive fanin of X",
              "belong to the fanin cone of X", "in the fanin cone of X",
              "how many gates are in / belong to the cone of X",
              "enumerate all gates that contribute to the fanin cone of X",
              "list all gates in the fanin logic cone of X".
  "enumerate_paths {a, b}" → paths between TWO DISTINCT nodes.
  CRITICAL RULE: The word "enumerate" alone does NOT imply enumerate_paths.
    "enumerate gates contributing to a cone" = cone_gate_count (only ONE net mentioned).
    "enumerate paths between X and Y" = enumerate_paths (TWO nets mentioned, path traversal).
  KEY RULE: if only ONE net is named and the request is about its cone/contributors,
  ALWAYS use cone_gate_count, not enumerate_paths — even if the word "enumerate" appears.

▸ cone_depth for "cone rooted at / cone depth"
  "cone_depth {output}" → maximum logic depth inside the fanin cone of ONE output.
    Triggers: "max/maximum logic depth of the fanin cone of X",
              "max/maximum logic depth of the cone rooted at X",
              "depth of the fanin cone rooted at output X",
              "how deep is the cone of X", "critical path length inside cone of X".
  KEY RULE: if the query asks for DEPTH (not gate count) of a SINGLE output's cone, use cone_depth.

▸ path_exists parameter extraction
  Intent: path_exists {a, b, avoid}  — answers YES/NO whether a path exists.
  Extract net names carefully: strip the labels "input", "output", "primary input",
  "primary output" that appear BEFORE the actual net name.
    "from input n0[1] to output n6" → a = "n0[1]", b = "n6"
    "from primary input n1 to primary output n117[1]" → a = "n1", b = "n117[1]"
  "avoid" synonyms: bypasses, avoids, does not traverse, without, skips,
    circumvents, does not pass through, excluding, while avoiding.
    "that bypasses node n95" → avoid = "n95"
    "that avoids n94"        → avoid = "n94"
  KEY RULE: "Check if there is a path/combinational path..." = path_exists (yes/no).
  Do NOT use enumerate_paths for a check/verify/does-it-exist question.

▸ enumerate_paths for "list all paths"
  Intent: enumerate_paths {a, b} — lists every combinational path between two nets.
    Triggers: "enumerate all paths", "list all paths between X and Y",
              "list every path from X to Y", "find all combinational paths",
              "all paths between X and Y", "paths connecting X and Y".
  Strip "primary input"/"primary output"/"input"/"output" labels from net names.
    "List all paths between n2 and n117[0]" → a = "n2", b = "n117[0]"

▸ report_const_gates vs const_propagate
  "report_const_gates {type, value}" → QUERY only: find/locate/list/identify gates whose inputs
    are hardwired constants. Does not modify the design.
    DEFAULT: if no gate type is specified in the request, type defaults to "nand".
    Triggers: "locate NAND cells with constant inputs", "find AND gates driven by logic 0/1",
              "which NAND gates have a static/hardwired/constant input",
              "Identify all gates with at least one input pin hardwired to logic-1".
    CRITICAL param rules:
    • If request names a specific gate type (NAND, NOR, etc.) → report_const_gates {type=that_type}.
    • If request says "all gates" or does NOT name a specific gate type → report_const_gates {} (no params; default type=nand).
    • If request says "hardwired to logic-0" / "tied to 1'b0" / "constant-0" → value="1'b0".
    Examples:
      "Locate all NAND gates with a constant-1 input" → {type:"nand", value:"1'b1"}
      "Identify all gates with any hardwired input" → {} (no type, no value)
      "Identify all gates in this design that have at least one input pin hardwired to logic-1" → {} (no type)
  "const_propagate {type}"   → TRANSFORM: simplify/fold/reduce those gates. Modifies design.
    Triggers: "fold constant inputs", "propagate constants", "simplify constant-input gates",
              "apply constant propagation".

▸ max_depth_between vs global_max_depth
  "max_depth_between {a, b}" → depth of the LONGEST path between two SPECIFIC nets.
    Triggers: "longest combinational path from X to Y", "longest path from X to Y",
              "maximum path depth between X and Y", "critical path depth between X and Y",
              "depth from X to Y", any phrasing naming TWO specific nets for depth.
  "global_max_depth {}" → maximum depth anywhere in the ENTIRE design (no specific nets).
  KEY RULE: if TWO nets are named, use max_depth_between {a, b}. If no nets (or "in the design"),
  use global_max_depth.

▸ gates_driven_by vs successors vs fanout
  "gates_driven_by {gate}" → returns the COUNT of gates driven by a gate.
    Triggers: "how many gates does X drive", "how many logic gates does X directly drive",
              "number of gates driven by X", "direct fanout count of X".
  "successors {gate}" → returns the LIST of gates directly driven by a gate (treating X as a gate instance).
    CRITICAL: if X is a primary input (not a gate instance), returns "No instance named X exists."
    Triggers: "which gates does X directly drive", "list every gate X feeds immediately",
              "list the direct output gates of X", "immediate successors of X",
              "direct children of X", "which gates does X feed", "gates that X directly feeds".
    NOTE: use successors when the request says "directly drive" or "feeds immediately" — even if X is
    labeled "primary input". If X is a PI, successors returns "No instance named X exists."
  "fanout {net}" → returns the fanout COUNT and gate list for a NET (not a gate instance).
    Triggers: "what is the fanout of X", "fanout of X", "how many loads on X",
              "list all fanout gates of X", "List all fanout gates of n5".
    NOTE: use fanout when "fanout" is explicitly mentioned anywhere in the request.
  KEY RULE:
    • If "fanout" appears anywhere in the request → fanout (fanout keyword wins over "directly drive").
    • If only "directly drive" or "feeds immediately" (no "fanout") → successors.
    • COUNT question → gates_driven_by.

▸ transitive_fanin vs cone_gate_count
  "transitive_fanin {net}" → all gates in the transitive fanin of a net (set of predecessors).
    Triggers: "transitive fanin of X", "all gates in the transitive fanin of X",
              "gates that feed X transitively", "input cone gates of X".
  "cone_gate_count {output}" → same meaning; use when "fanin cone" appears explicitly.
  In practice: if the request says "transitive fanin" use transitive_fanin.

▸ signals_equivalent: "logically equivalent" always means signals_equivalent {a, b}.
  Do NOT use verify_equivalence (which compares the full design to a saved snapshot).
  signals_equivalent compares two INTERNAL signals within the current design.

▸ CRITICAL: convert_basis vs optimize_cone vs minimize_depth — MUST READ BEFORE ROUTING
  RULE 1: If the request says "to use only X and Y gates/primitives" or "using only X and Y"
    → ALWAYS use convert_basis regardless of whether "cone" appears in the sentence.
    convert_basis is for GATE TYPE TRANSFORMATION (changing gate types to a new basis).
    Example: "Transform the logic cone of n10 to use only NOR and NOT primitives"
    → convert_basis {scope:"n10", basis:"NOR_NOT"}  ← NOT optimize_cone!
  RULE 2: optimize_cone and minimize_depth are for DEPTH REDUCTION only.
    They only apply when the request contains "minimize depth", "optimize depth",
    "reduce depth", "rebalance depth", or similar depth-reduction phrases.
    If NO depth-reduction phrase is present → do NOT use optimize_cone or minimize_depth.

▸ minimize_depth (with basis) vs optimize_cone
  Both reduce logic depth, but target DIFFERENT things:
  "minimize_depth {basis}" → globally reduce the ENTIRE design's maximum depth, while
    maintaining basis purity as an optional constraint.
    Triggers: "minimize/reduce/rebalance depth + maintaining/keeping/ensuring X-basis",
              "rebalance logic depth, keeping the netlist in AND+NOT only",
              "rebalance depth with constraint that cone of X uses only Y-basis",
              "shorten the worst-case path, keeping basis X".
    KEY: when "cone of X" appears alongside a depth-reduction request, the cone name
    describes WHERE the BASIS CONSTRAINT applies, not WHAT is being optimized.
    The optimization is still GLOBAL depth.
    Examples:
      "rebalance depth, keeping cone of n10 in NOR+NOT only" → minimize_depth {basis:"NOR_NOT"}
      "minimize depth, cone of n15 in AND/OR/NOT only"       → minimize_depth {basis:"AND_OR_NOT"}
      "rebalance depth with NAND+NOT constraint"             → minimize_depth {basis:"NAND_NOT"}
      "rebalance depth, keeping netlist in AND+NOT only"     → minimize_depth {basis:"AND_NOT"}
  "optimize_cone {output, basis}" → reduce depth WITHIN ONE SPECIFIC output's cone only.
    Triggers: "optimize the cone of X", "optimize the depth of the cone of X",
              "reduce the depth inside the cone rooted at output X",
              "minimize the depth of the logic cone of output X",
              "minimize the depth of the cone of X while keeping it in Y basis".
    KEY DISTINCTION: if the phrase says "minimize depth OF THE CONE OF X" (i.e., the CONE
    is the SUBJECT being minimized), use optimize_cone {output:"X"}.
    Examples:
      "Minimize the depth of the logic cone of output n8 while keeping it in NAND+NOT"
        → optimize_cone {output:"n8", basis:"NAND_NOT"}
      "Minimize the depth of the cone of n14 while keeping netlist in NAND+NOT only"
        → optimize_cone {output:"n14", basis:"NAND_NOT"}

▸ convert_basis parameter extraction
  "convert_basis {basis, scope}" — remap gate types to a specific basis.
  Basis mapping:
    "NAND and NOT" / "NAND+NOT" / "only NAND and NOT"  → basis = "NAND_NOT"
    "NOR and NOT"  / "NOR+NOT"  / "only NOR and NOT"   → basis = "NOR_NOT"
    "AND and NOT"  / "AND+NOT"  / "only AND and NOT"   → basis = "AND_NOT"
    "AND, OR, and NOT" / "AND/OR/NOT"                  → basis = "AND_OR_NOT"
  Scope: extract the net name after "cone of", "in the cone of", "in the fanin cone of".
    CRITICAL: strip "output", "primary output" labels — extract only the net identifier.
      "in the cone of output n8" → scope = "n8"  (NOT "output n8")
      "in the cone of n11[0]"   → scope = "n11[0]"
    If the conversion applies to the "entire netlist" / "whole design", omit scope.
    "Translate all OR gates in the cone of n11[0] into NAND and NOT"
      → convert_basis {scope:"n11[0]", basis:"NAND_NOT"}
    "Translate the logic in the cone of output n8 to use only NAND and NOT primitives"
      → convert_basis {scope:"n8", basis:"NAND_NOT"}
    "Rebuild the entire netlist using only AND and NOT"
      → convert_basis {basis:"AND_NOT"}  (no scope)
  Synonyms for convert_basis: translate, transform, rebuild, remap, convert, restructure, replace.

▸ rename parameter extraction
  "rename {kind, old, new}" — rename a gate or wire.
  kind rules — follow exactly:
    → "gate" ONLY when request explicitly says "gate", "cell", "element", or "primitive"
    → "wire" ONLY when request explicitly says "wire"
    → "signal" when request says "signal", "net", or does not specify wire/gate
  CRITICAL: "Relabel WIRE X as Y" MUST produce kind="wire", not kind="signal".
  Extract old and new from "X as Y", "X to Y", "from X to Y".
    "Relabel gate g0 as renamed_gate" → {kind:"gate", old:"g0", new:"renamed_gate"}
    "Relabel wire n74 as renamed_wire" → {kind:"wire", old:"n74", new:"renamed_wire"}
    "Rename wire n74 to renamed_wire"  → {kind:"wire", old:"n74", new:"renamed_wire"}
    "Rename signal n440 to renamed_wire" → {kind:"signal", old:"n440", new:"renamed_wire"}
    "Alias net n440 as renamed_wire"   → {kind:"signal", old:"n440", new:"renamed_wire"}

▸ insert_buffers parameter extraction
  "insert_buffers {k, include_pi}" — add buffers globally so no driver exceeds k loads.
  k is the integer fanout limit. mode defaults to "fanout" when not specified.
  When no specific net is named, omit the "net" field.
  include_pi RULE (default false):
    • "no gate/net/cell drives more than X loads"   → include_pi omitted (defaults to false)
    • "no signal/wire/connection drives more than X" → include_pi: true
      (The word "signal" implies primary inputs are also treated as drivers to buffer.)
  CRITICAL: k comes from the number in the request — extract it correctly.
    "more than 4 loads" → k:4    "more than 16 loads" → k:16
  Do NOT use mode:"dedicated" for global fanout-limit requests.
    "Add buffer gates so no net drives more than 4 loads"      → {k:4}
    "Insert buffers so that no gate drives more than 4 loads"  → {k:4}
    "Ensure no signal drives more than 4 loads"                → {k:4, include_pi:true}
    "Insert buffers so that no signal drives more than 16 loads" → {k:16, include_pi:true}
    "Apply fanout-reduction buffers to net n1 until fanout ≤ 4" → {k:4, net:"n1", mode:"fanout"}

▸ count_ports (combined PI/PO count query)
  "count_ports {}" → report how many primary inputs AND primary outputs the design has.
  Triggers: "How many primary inputs and primary outputs does this design have?",
            "determine the number of primary inputs and outputs",
            "report the PI and PO counts", "how many inputs and outputs?".
  KEY RULE: use count_ports (not list_ports) when the request asks HOW MANY inputs and outputs,
  not for a listing of their names. count_ports returns a single combined count line.
  Examples:
    "How many primary inputs and primary outputs does this design have?"
    → {"intent":"count_ports","params":{}}
    "How many primary inputs and outputs are in this circuit?"
    → {"intent":"count_ports","params":{}}

▸ xor_to_aoi parameter extraction
  "xor_to_aoi {scope}" — decompose XOR gates into AND/OR/NOT within a scope.
  scope: extract the net name after "cone of" / "fanin cone of" / "in the cone of".
  If no cone is mentioned, omit scope (applies globally).
    "Break down all XOR cells in the fanin cone of n15 into AND, OR, and NOT primitives"
      → xor_to_aoi {scope:"n15"}

▸ output_constant
  "output_constant {output}" → check if a specific output is always 0 or always 1 (constant function).
  Triggers: "Is output X always 0 regardless of inputs?", "always evaluates to constant-0",
            "constant-0 function for all input assignments", "always 1 regardless of inputs".
  KEY RULE: do NOT use signals_equivalent for output_constant queries.

▸ cone_type_counts vs cone_gate_count
  "cone_type_counts {output}" → BREAKDOWN of gate types inside a cone (how many AND, NOT, NAND, etc.).
  Triggers: "primitive-type distribution within the cone of X", "gate-type breakdown within cone of X",
            "how many of each gate type in the cone of X", "report each gate type in the cone of X",
            "count gates by type in the cone of X".
  "cone_gate_count {output}" → TOTAL gate count in a cone (single integer).
  KEY RULE: breakdown by type → cone_type_counts. Total count → cone_gate_count.

▸ dominator (every path passes through)
  "dominator {a, b, gate}" → whether EVERY combinational path from net A to net B traverses gate G.
  Triggers: "does every path from A to B pass through gate G?",
            "does every combinational path from A to B traverse G?",
            "is G on every path from A to B?".
  KEY RULE: "every path...pass through GATE" → dominator {a, b, gate}. NOT path_exists.

▸ gate_on_max_path
  "gate_on_max_path {gate}" → whether gate G lies on any maximum-depth (critical) path.
  Triggers: "does gate G lie on any maximum-depth path?", "is G on the critical path?",
            "does G reside on a max-depth path?", "determine whether G is on any maximum-depth path".

▸ pi_to_dff_depth
  "pi_to_dff_depth {}" → worst-case combinational depth from any primary input to any DFF D-pin.
  Triggers: "maximum depth from primary input to DFF D-pin", "worst-case combinational depth from any
             primary input to any flip-flop D-pin", "maximum logic depth from PI to DFF data input".

▸ deepest_output
  "deepest_output {}" → which primary output has the deepest (longest) fanin cone.
  Triggers: "which output has the deepest fanin cone?", "which output bit has the largest logic depth?",
            "which output has the largest fanin cone?", "output bit with the deepest fanin logic cone",
            "which output has the largest fan-in cone?", "which primary output has the most logic levels?".
  KEY RULE: use deepest_output when the question asks WHICH output (not how many).

▸ outputs_depth_gt
  "outputs_depth_gt {k}" → count of primary outputs whose fanin depth EXCEEDS k logic levels.
  Triggers: "how many outputs have a logic depth greater than N?",
            "how many output bits have a fan-in depth exceeding N levels?",
            "how many primary outputs have depth > N?".
  Extract k as the integer threshold value.
  Example: "How many outputs have a logic depth greater than 4?" → outputs_depth_gt {k: 4}

▸ exists_nand_pair
  "exists_nand_pair {target}" → does any existing NAND(a,b) pair in the netlist compute the same
  Boolean function as net TARGET?
  Triggers: "does any NAND pair compute same function as N?", "is there a NAND(a,b) = N in the netlist?",
            "does any existing pair (a, b) have NAND(a, b) equivalent to N?".

▸ shared_cone
  "shared_cone {a, b}" → list all gates shared between the fanin cones of two outputs.
  Triggers: "gates shared between the fanin cones of A and B", "gates in both the cone of A and cone of B",
            "identify gates in the intersection of the cones of A and B".

▸ "gates connected to a signal/net" → connected_to_net (driver ∪ loads, NOT fanout)
  "gates connected to signal/wire/net X" means EVERY gate touching net X — the gate
  that DRIVES X plus all gates that read X as a load. That is connected_to_net {net: X}.
  This differs from fanout {net: X}, which lists only the loads X drives (it omits the
  driver). Use connected_to_net whenever the request says "gates connected to" a NET —
  including a renamed net (a rename makes the new name a NET name).
  "List all gates that now connect to the renamed signal X" → connected_to_net {net: X}
  "Which gates are connected to renamed signal X?" → connected_to_net {net: X}
  KEY RULE: "gates connected to signal/wire/net X" (X is a net) → connected_to_net {net: X}.
  Do NOT use fanout here (fanout omits the driver); do NOT use connected_to_output
  (that is for the output of a gate INSTANCE, not a net).
  Example:
    "List all gates that now connect to the renamed signal renamed_sig."
    → {"intent":"connected_to_net","params":{"net":"renamed_sig"}}

▸ connected_to_output vs fanout (output-net queries)
  CRITICAL RULE: when the request says "output net of gate X" (explicit "net" keyword) →
    use fanout {net: X}  (treats X as a net name).
    Example: "List all cells connected to the output net of gate g0"
    → {"intent":"fanout","params":{"net":"g0"}}
  "connected_to_output {gate}" → list all gates driven by the output of a specific gate
    (gate-instance interpretation, NO "net" keyword in phrasing).
    Triggers: "gates connected to the output of X", "which gates are directly driven by gate X",
              "gates that gate X drives".
  KEY RULE:
    • "output NET of gate X" → fanout {net: X}
    • "output of gate X" (no NET) → connected_to_output {gate: X}

▸ reachable_from vs transitive_fanout
  Both find all gates reachable from a net. Use "reachable_from {net}" when the phrasing says
  "reachable from", "all cells reachable from N", "enumerate cells transitively reachable from N".
  Use "transitive_fanout {net}" when phrasing says "transitive fanout of N".

▸ enable_hold_report vs enable_hold_count
  "enable_hold_report {}" → list/report D-input logic of flip-flops (enable/hold structures).
  Triggers: "report D-input logic of flip-flops", "identify enable or hold structures in D-input logic",
            "report any enable or hold implemented in DFF D inputs".
  "enable_hold_count {}" → count how many flip-flops have such structures.
  Triggers: "how many flip-flops have enable or hold structures?", "count DFFs with enable/hold logic".

▸ optimize_cone (extended triggers)
  Additional triggers for optimize_cone {output, basis}:
  "minimize the depth of the logic cone of output X while keeping the cone in Y-basis only",
  "minimize the depth of the cone of X while ensuring Y-basis", "reduce cone depth for X in Y only".

▸ insert_buffers: dedicated mode
  For dedicated buffer insertion on a specific net (mode:"dedicated"), include k=4 as a placeholder;
  the actual insertion gives one buffer per load regardless of k.
  "For signal N, insert a dedicated repeater for each fanout load" → {k:4, net:"N", mode:"dedicated"}
  "Insert one dedicated buffer per load on net N"                 → {k:4, net:"N", mode:"dedicated"}

▸ boolean_equation (derive symbolic logic expression)
  "boolean_equation {output}" → express the Boolean function at output X entirely in terms
  of primary inputs as a symbolic logic expression.
  Triggers: "What Boolean function does output X compute? Express it in terms of the primary inputs.",
            "Express the Boolean function computed at output X entirely in terms of its primary inputs.",
            "Express the Boolean function at output X in terms of its primary inputs.",
            "Derive the Boolean equation for output X in terms of its primary inputs.",
            "What is the Boolean equation for output X?".
  Extract the output net name (e.g., n8, n16, n30, n12) into the "output" param.
  KEY RULES:
    • boolean_equation: asks for a SYMBOLIC EXPRESSION for an output — not a yes/no, not a count.
    • NOT output_constant (which checks if output is always 0 or always 1 for all inputs).
    • NOT depends_on (which checks whether output X uses/depends on input Y at all).
  Example: "What Boolean function does output n8 compute? Express it in terms of the primary inputs."
  → {"intent":"boolean_equation","params":{"output":"n8"}}

▸ list_ports (with bit widths phrasing)
  "list_ports {dir}" → list the primary inputs or outputs of the design, including bit widths.
  dir = "input" for primary inputs, "output" for primary outputs.
  Triggers: "List all primary inputs of this design with their bit widths.",
            "Please list all the primary inputs of this design with their bit widths.",
            "List all primary outputs of this design with their bit widths.",
            "Enumerate all primary inputs/outputs with bus widths.",
            "What are the primary inputs/outputs and their widths?".
  Examples:
    "List all primary inputs of this design with their bit widths."
    → {"intent":"list_ports","params":{"dir":"input"}}
    "List all primary outputs of this design with their bit widths."
    → {"intent":"list_ports","params":{"dir":"output"}}

▸ verify_equivalence vs noop
  "verify_equivalence" covers: verify, prove, assert, confirm, check, validate that the
  current design is (combinationally/functionally) equivalent to the original or a snapshot.

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

"Excise all logically inert gates from the netlist."
→ {"intent":"remove_dangling","params":{}}

"How many gates were excised in the previous step?"
→ {"intent":"delta_count","params":{"kind":"dangling"}}

"How many logic cells were excised in the previous step?"
→ {"intent":"delta_count","params":{"kind":"dangling"}}

"How many cells were pruned in the previous step?"
→ {"intent":"delta_count","params":{"kind":"dangling"}}

"How many floating signals were found?"
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

"Enumerate all gates that contribute to the fanin logic cone of output n14."
→ {"intent":"cone_gate_count","params":{"output":"n14"}}

"List all gates that belong to the fanin cone of output n8."
→ {"intent":"cone_gate_count","params":{"output":"n8"}}

"Emit the modified netlist to the output file test38_out.v."
→ {"intent":"write_design","params":{"file":"test38_out.v"}}

"How many gates belong to the fanin cone of primary output n15?"
→ {"intent":"cone_gate_count","params":{"output":"n15"}}

"What is the maximum logic depth of the fanin cone rooted at output n3?"
→ {"intent":"cone_depth","params":{"output":"n3"}}

"What gates does primary input n5 directly drive? List all fanout gates of n5."
→ {"intent":"fanout","params":{"net":"n5"}}

"Which gates does primary input n0 directly drive? List every gate n0 feeds immediately."
→ {"intent":"successors","params":{"gate":"n0"}}

"Check if there is a combinational path from n0[0] to n4 that bypasses node n719."
→ {"intent":"path_exists","params":{"a":"n0[0]","b":"n4","avoid":"n719"}}

"Check if there is a path from input n0[1] to output n6 that avoids n94."
→ {"intent":"path_exists","params":{"a":"n0[1]","b":"n6","avoid":"n94"}}

"Check if there is a combinational path from n12 to n25[0] that bypasses node n1127."
→ {"intent":"path_exists","params":{"a":"n12","b":"n25[0]","avoid":"n1127"}}

"Check if there is a path from input n1 to output n117[1] that avoids n1646."
→ {"intent":"path_exists","params":{"a":"n1","b":"n117[1]","avoid":"n1646"}}

"List all paths between n2 and n117[0]."
→ {"intent":"enumerate_paths","params":{"a":"n2","b":"n117[0]"}}

"List all paths between n14 and n25[0]."
→ {"intent":"enumerate_paths","params":{"a":"n14","b":"n25[0]"}}

"Is there a combinational path from n29 to n31[0] that skips n1552?"
→ {"intent":"path_exists","params":{"a":"n29","b":"n31[0]","avoid":"n1552"}}

"Find the longest combinational path from n30 to n31[1]."
→ {"intent":"max_depth_between","params":{"a":"n30","b":"n31[1]"}}

"What is the maximum path depth between n4 and n5[0]?"
→ {"intent":"max_depth_between","params":{"a":"n4","b":"n5[0]"}}

"How many logic gates does g0 directly drive?"
→ {"intent":"gates_driven_by","params":{"gate":"g0"}}

"List the direct output gates of g0."
→ {"intent":"successors","params":{"gate":"g0"}}

"Find all gates in the transitive fanin of n3."
→ {"intent":"transitive_fanin","params":{"net":"n3"}}

"Find all gates in the transitive fanin of n33[0]."
→ {"intent":"transitive_fanin","params":{"net":"n33[0]"}}

"Are signals n2122 and n2116 logically equivalent?"
→ {"intent":"signals_equivalent","params":{"a":"n2122","b":"n2116"}}

"Do n1039 and n1046 always yield the same Boolean output?"
→ {"intent":"signals_equivalent","params":{"a":"n1039","b":"n1046"}}

"Confirm that n1035 and n1029 are logically the same signal."
→ {"intent":"signals_equivalent","params":{"a":"n1035","b":"n1029"}}

"How many gates does the design contain in total?"
→ {"intent":"total_gate_count","params":{}}

"Add buffer gates so no net drives more than 4 loads."
→ {"intent":"insert_buffers","params":{"k":4}}

"Buffer all gate outputs so that no gate output drives more than 4 loads."
→ {"intent":"insert_buffers","params":{"k":4}}

"Insert repeaters on gate outputs exceeding 4 loads."
→ {"intent":"insert_buffers","params":{"k":4}}

"Ensure no signal drives more than 4 loads, including primary inputs."
→ {"intent":"insert_buffers","params":{"k":4,"include_pi":true}}

"Apply fanout buffering so no signal or primary input exceeds 4 loads."
→ {"intent":"insert_buffers","params":{"k":4,"include_pi":true}}

"Shorten the worst-case combinational path by restructuring the logic."
→ {"intent":"minimize_depth","params":{}}

"Rebalance the logic depth to minimize the maximum combinational path."
→ {"intent":"minimize_depth","params":{}}

"Excise all cells that are logically inert and drive no output."
→ {"intent":"remove_dangling","params":{}}

"Excise all cells that are logically inert and drive no primary output."
→ {"intent":"remove_dangling","params":{}}

"Excise floating logic nodes that have no connection to primary outputs."
→ {"intent":"remove_dangling","params":{}}

"Relabel gate g0 as renamed_gate throughout the netlist."
→ {"intent":"rename","params":{"kind":"gate","old":"g0","new":"renamed_gate"}}

"Relabel wire n74 as renamed_wire throughout the netlist."
→ {"intent":"rename","params":{"kind":"wire","old":"n74","new":"renamed_wire"}}

"Translate all OR gates in the cone of n11[0] into equivalent circuits using only NAND and NOT gates."
→ {"intent":"convert_basis","params":{"scope":"n11[0]","basis":"NAND_NOT"}}

"Translate the logic in the cone of output n8 to use only NAND and NOT primitives while preserving functional equivalence."
→ {"intent":"convert_basis","params":{"scope":"n8","basis":"NAND_NOT"}}

"Transform the logic cone of n10 to use only NOR and NOT primitives while preserving functional equivalence."
→ {"intent":"convert_basis","params":{"scope":"n10","basis":"NOR_NOT"}}

"Rebuild the entire netlist using only AND and NOT primitives while preserving functional equivalence."
→ {"intent":"convert_basis","params":{"basis":"AND_NOT"}}

"Dissolve all back-to-back inverter chains into a wire."
→ {"intent":"collapse_inverters","params":{}}

"Break down all XOR cells in the fanin cone of n15 into AND, OR, and NOT primitives without changing functionality."
→ {"intent":"xor_to_aoi","params":{"scope":"n15"}}

"Rebalance the logic depth with the constraint that the cone of n11[0] uses only NAND and NOT gates."
→ {"intent":"minimize_depth","params":{"basis":"NAND_NOT"}}

"Rebalance the logic depth, keeping the cone of n10 in NOR and NOT only."
→ {"intent":"minimize_depth","params":{"basis":"NOR_NOT"}}

"Rebalance the logic depth, keeping the cone of n15 in AND, OR, and NOT only."
→ {"intent":"minimize_depth","params":{"basis":"AND_OR_NOT"}}

"Rebalance the logic depth, keeping the netlist in AND and NOT only."
→ {"intent":"minimize_depth","params":{"basis":"AND_NOT"}}

"Identify and combine all structurally or functionally duplicate gate pairs."
→ {"intent":"merge_duplicates","params":{}}

"Transform every XNOR cell into an equivalent circuit using only NOR gates."
→ {"intent":"xnor_to_nor","params":{}}

"Transform every XNOR gate in this design into an equivalent NOR-only sub-circuit."
→ {"intent":"xnor_to_nor","params":{}}

"Replace every XOR gate in this design with an equivalent 4-NAND circuit."
→ {"intent":"xor_to_nand","params":{}}

"Substitute all 2-input NAND gates whose one input is hardwired to logic-1 with inverter gates."
→ {"intent":"nand_const1_to_inv","params":{}}

"Does primary output n16 evaluate to a constant-0 function for all possible input assignments?"
→ {"intent":"output_constant","params":{"output":"n16"}}

"Report the primitive-type distribution within the fanin cone of output n8."
→ {"intent":"cone_type_counts","params":{"output":"n8"}}

"Report the primitive-type distribution within the fanin cone of output n14."
→ {"intent":"cone_type_counts","params":{"output":"n14"}}

"Does every combinational path from input n2 to output n12 pass through gate g0? Report yes or no."
→ {"intent":"dominator","params":{"a":"n2","b":"n12","gate":"g0"}}

"Does gate g0 lie on any maximum-depth path of the design? Report yes or no."
→ {"intent":"gate_on_max_path","params":{"gate":"g0"}}

"What is the worst-case combinational depth from any primary input to any flip-flop D-pin?"
→ {"intent":"pi_to_dff_depth","params":{}}

"Which output bit has the deepest fanin logic cone?"
→ {"intent":"deepest_output","params":{}}

"Does any existing NAND gate pair (a, b) in the netlist compute the same function as primary output n25?"
→ {"intent":"exists_nand_pair","params":{"target":"n25"}}

"Identify all logic gates that appear in both the fanin cone of n16 and the fanin cone of n17."
→ {"intent":"shared_cone","params":{"a":"n16","b":"n17"}}

"Enumerate all logic cells transitively reachable from net n2."
→ {"intent":"reachable_from","params":{"net":"n2"}}

"List all logic cells transitively reachable from primary input n0."
→ {"intent":"transitive_fanout","params":{"net":"n0"}}

"Report the D-input logic of the flip-flops to identify any enable or hold structures."
→ {"intent":"enable_hold_report","params":{}}

"How many flip-flops were found to have enable or hold structures in their D-input logic?"
→ {"intent":"enable_hold_count","params":{}}

"List all register-to-register combinational paths in this design."
→ {"intent":"reg_to_reg_paths","params":{}}

"What is the maximum combinational depth on any register-to-register path in this design?"
→ {"intent":"reg_to_reg_depth","params":{}}

"Is the function at output n8 symmetric with respect to inputs n3 and n4[0]? Report yes or no."
→ {"intent":"symmetric","params":{"output":"n8","a":"n3","b":"n4[0]"}}

"Is the function at output n11 symmetric with respect to inputs n3 and n9[0]? Report yes or no."
→ {"intent":"symmetric","params":{"output":"n11","a":"n3","b":"n9[0]"}}

"List all flip-flops driven by clock signal n0."
→ {"intent":"ffs_on_clock","params":{"clk":"n0"}}

"For signal n2, insert a dedicated repeater for each of its fanout loads so every load gets its own driver."
→ {"intent":"insert_buffers","params":{"k":4,"net":"n2","mode":"dedicated"}}

"Limit the fanout of signal n1 to at most 4 loads per driver by inserting buffer stages."
→ {"intent":"insert_buffers","params":{"k":4,"net":"n1","mode":"fanout"}}

"Apply fanout buffering so that no signal drives more than 16 loads."
→ {"intent":"insert_buffers","params":{"k":16,"include_pi":true}}

"Minimize the depth of the logic cone of output n8 while keeping the cone in NAND and NOT basis only."
→ {"intent":"optimize_cone","params":{"output":"n8","basis":"NAND_NOT"}}

"Minimize the depth of the logic cone of output n14 while keeping the netlist in NAND and NOT basis only."
→ {"intent":"optimize_cone","params":{"output":"n14","basis":"NAND_NOT"}}

"What cell type is gate g0, and which input/output nets does it connect to?"
→ {"intent":"gate_info","params":{"gate":"g0"}}

"Check if there are any floating inputs or unconnected output ports in this design."
→ {"intent":"remove_dangling","params":{}}

"Does this design have any undriven input pins or unloaded output ports?"
→ {"intent":"remove_dangling","params":{}}

"Excise all redundant gates that can be removed without altering functionality."
→ {"intent":"remove_dangling","params":{}}

"Simplify the OR gates identified above by folding their constant-1 inputs."
→ {"intent":"const_propagate","params":{"type":"or"}}

"Simplify the NOR gates identified above by propagating their constant inputs."
→ {"intent":"const_propagate","params":{"type":"nor"}}

"Identify all gates in this design that have at least one input pin hardwired to logic-1."
→ {"intent":"report_const_gates","params":{}}

"List all gates with one or more inputs tied to 1'b1."
→ {"intent":"report_const_gates","params":{}}

"List all cells connected to the output net of gate g0."
→ {"intent":"fanout","params":{"net":"g0"}}

"Which gates are directly driven by the output of gate g0?"
→ {"intent":"connected_to_output","params":{"gate":"g0"}}

"How many NAND gates are now in the restructured cone of output n8?"
→ {"intent":"count_type","params":{"type":"nand"}}

"How many NOT gates are currently in the design?"
→ {"intent":"count_type","params":{"type":"not"}}

"List all gates that now connect to the renamed signal renamed_sig."
→ {"intent":"connected_to_net","params":{"net":"renamed_sig"}}

"Which gates are connected to the renamed signal renamed_wire?"
→ {"intent":"connected_to_net","params":{"net":"renamed_wire"}}

"How many outputs have a logic depth greater than 4?"
→ {"intent":"outputs_depth_gt","params":{"k":4}}

"How many primary output bits have a fan-in depth exceeding 6 levels?"
→ {"intent":"outputs_depth_gt","params":{"k":6}}

"Which output has the largest fanin cone?"
→ {"intent":"deepest_output","params":{}}

"Which output has the largest fan-in cone?"
→ {"intent":"deepest_output","params":{}}

"List every path originating at primary input n24[1] and terminating at primary output n26[1]."
→ {"intent":"enumerate_paths","params":{"a":"n24[1]","b":"n26[1]"}}

"Provide a complete enumeration of paths between n24[2] and n26[0]."
→ {"intent":"enumerate_paths","params":{"a":"n24[2]","b":"n26[0]"}}

"What Boolean function does output n8 compute? Express it in terms of the primary inputs."
→ {"intent":"boolean_equation","params":{"output":"n8"}}

"What Boolean function does output n25 compute? Express it in terms of the primary inputs."
→ {"intent":"boolean_equation","params":{"output":"n25"}}

"Express the Boolean function at output n30 in terms of its primary inputs."
→ {"intent":"boolean_equation","params":{"output":"n30"}}

"Express the Boolean function computed at output n16 entirely in terms of its primary inputs."
→ {"intent":"boolean_equation","params":{"output":"n16"}}

"Express the Boolean function at output n12 in terms of its primary inputs."
→ {"intent":"boolean_equation","params":{"output":"n12"}}

"Derive the Boolean equation for output n12 in terms of its primary inputs."
→ {"intent":"boolean_equation","params":{"output":"n12"}}

"List all primary inputs of this design with their bit widths."
→ {"intent":"list_ports","params":{"dir":"input"}}

"Please list all the primary inputs of this design with their bit widths."
→ {"intent":"list_ports","params":{"dir":"input"}}

"List all primary outputs of this design with their bit widths."
→ {"intent":"list_ports","params":{"dir":"output"}}

"How many NOR gates are now in the design after the XNOR-to-NOR transformation?"
→ {"intent":"count_type","params":{"type":"nor"}}

"How many NAND gates are now in the design after the XOR-to-NAND conversion?"
→ {"intent":"count_type","params":{"type":"nand"}}

"How many primary inputs and primary outputs does this design have?"
→ {"intent":"count_ports","params":{}}

"Insert buffers so that no signal drives more than 16 loads. Ensure functional equivalence is preserved."
→ {"intent":"insert_buffers","params":{"k":16,"include_pi":true}}

━━━ AVAILABLE INTENTS ━━━
- begin_case {name}
- load_design {file, dir}
- write_design {file}
- count_gates {}
- total_gate_count {}
- count_type {type}
- delta_count {kind}
- gate_info {gate}
- list_type {type}
- cone_gate_count {output}
- cone_type_counts {output}
- list_ports {dir}              # dir = input|output
- count_ports {}
- fanout {net}
- max_fanout_of {net}
- gates_driven_by {gate}
- successors {gate}
- transitive_fanin {net}
- transitive_fanout {net}
- reachable_from {net}
- highest_fanout_pi {}
- shared_cone {a, b}
- connected_to_output {gate}
- connected_to_net {net}        # driver ∪ loads of a net (gates connected to signal X)
- path_exists {a, b, avoid}
- enumerate_paths {a, b}
- length_zero_paths {}
- dominator {a, b, gate}
- articulation {a, b}
- is_cut {wire}
- max_depth_between {a, b}
- cone_depth {output}
- global_max_depth {}
- pi_to_dff_depth {}
- reg_to_reg_depth {}
- outputs_depth_gt {k}
- deepest_output {}
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
- report_const_gates {type, value}   # default type=nand when no gate type specified
- const_propagate {type}
- collapse_inverters {}
- remove_dangling {}
- merge_duplicates {}
- rename {kind, old, new}       # kind = gate|wire|signal
- insert_buffers {k, net, mode, include_pi} # mode = fanout|dedicated; include_pi = true|false (default false)
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
    "max_fanout_of", "shared_cone", "connected_to_output", "connected_to_net",
    "path_exists",
    "enumerate_paths", "length_zero_paths", "dominator", "articulation",
    "is_cut", "max_depth_between", "cone_depth", "global_max_depth",
    "pi_to_dff_depth", "reg_to_reg_depth", "outputs_depth_gt",
    "deepest_output", "gate_on_max_path", "signals_equivalent",
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
    "path_exists": {"avoid"},
    "convert_basis": {"basis", "scope"},
    "xor_to_nand": {"scope"},
    "xnor_to_nor": {"scope"},
    "xor_to_aoi": {"scope"},
    "report_const_gates": {"type", "value"},
    "const_propagate": {"type"},
    "rename": {"kind"},
    "insert_buffers": {"net", "mode", "include_pi"},
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
    if intent == "insert_buffers" and key == "mode" and isinstance(value, str):
        return value.lower()
    if intent == "insert_buffers" and key == "include_pi":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
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
