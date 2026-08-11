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
  (excised, absorbed, swept, folded, eliminated, inserted, removed, merged), use
  delta_count, not count_gates.  Adjectives like "redundant", "duplicate",
  "superfluous", "dangling", "floating" describe WHAT was removed, not a new
  analysis — "How many redundant gates were removed?" → delta_count
  {"kind":"removed"}.  A "how many ... were <verb>ed" question after a transform
  is NEVER noop.

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
    "COMBINATIONAL depth from any primary input to any primary output" → pi_to_po_depth {}
    "from any primary input to any DFF D-pin / register input" → pi_to_dff_depth {}
    "on any register-to-register path"                      → reg_to_reg_depth {}
    "from any register/DFF/flip-flop output to any primary output" → reg_to_po_depth {}
    "maximum combinational logic depth in the design (now)" → global_max_depth {}
  KEY RULE: pi_to_po_depth requires BOTH conditions at once — (a) the word
  "combinational" AND (b) explicit primary-input/primary-output endpoints in
  the request.  Everything else is global_max_depth {}:
    - "maximum combinational logic depth in the design (now)" — endpoints NOT
      mentioned → global_max_depth.  The word "combinational" alone does NOT
      select pi_to_po_depth; global_max_depth IS a combinational measure.
    - "maximum logic depth from any PI to any PO" / "between the PIs and POs"
      — endpoints but no "combinational" → global_max_depth (per Q&A A21.2
      the graded value treats DFF.Q as PIs and DFF.D as POs).
  Use "max_depth_between {a, b}" ONLY when BOTH endpoints are concrete net names
  that literally appear in the request (n24[0], n26, ...).
  KEY RULE: "any primary input", "any DFF", "any output" are CLASSES, never param
  values — never emit params like {"a":"primary_input"} or {"b":"DFF"}.
  KEY RULE: bracket indices are part of the name: n31[1] and n31 are different nets.
  Copy the index if the request has one.

▸ choosing the basis value: NAND vs NAND_NOT (and NOR vs NOR_NOT)
  "NAND"      → the request allows ONE gate type: "using only 2-input NAND
    gates", "so that the final design contains no gate type other than NAND",
    "NAND-only".  Inverters are built as NAND(a, a), so the result really does
    contain nothing else.
  "NAND_NOT"  → the request names both: "using only NAND and NOT gates",
    "rebuild it from NANDs and inverters".
  Same split for NOR / NOR_NOT.
  KEY RULE: count the gate types the request permits.  "only NAND" is one type
  and "NAND and NOT" is two; picking NAND_NOT for a one-type request leaves
  inverters in the netlist, and a prompt that says "no gate type other than
  NAND" is checked against the FINAL netlist (Q&A A63/A64), so those inverters
  fail it even though the logic is equivalent.

▸ picking the targeted conversion: the SOURCE type and the TARGET type
  Four conversions exist and each names both ends.  Match BOTH, not just the
  gate being replaced:
    XOR  -> NAND      xor_to_nand {scope}
    XOR  -> AOI       xor_to_aoi {scope}
    XNOR -> NOR       xnor_to_nor {scope}
    XNOR -> NAND      xnor_to_nand {scope}
  KEY RULE: "replace all XNOR gates with NAND-only implementations" is
  xnor_to_nand, not xnor_to_nor.  Both leave zero XNOR gates and both preserve
  equivalence, so the answer looks right either way -- but the prompt named the
  target basis and that is checked against the final netlist (Q&A A63).

▸ targeted gate-type conversion vs convert_basis
  If the request names a SPECIFIC gate type to replace (XOR, XNOR, NAND-with-constant),
  use the targeted intent — xor_to_nand, xnor_to_nor, xnor_to_nand, xor_to_aoi,
  nand_const1_to_inv —
  NOT convert_basis.  convert_basis rewrites EVERY gate and is only correct when the
  request says to rebuild the whole netlist (or a cone) using only the basis gates.

▸ minimize_depth vs optimize_cone
  "optimize_cone {output}" only when the CONE ITSELF is the thing being optimized
  ("optimize/re-synthesize the cone of n9").
  If the cost function is the DESIGN-level maximum logic depth and a cone is mentioned
  only as a side constraint ("...while ensuring the cone of n11[0] continues to use only
  NAND and NOT gates"), use minimize_depth with that basis: {"basis":"NAND_NOT"}.
  KEY RULE: "minimize (maximum) path depth, ensuring the cone of X continues to
  use only <basis>" — the TARGET is the whole design, the cone is only the
  constraint → minimize_depth {"basis":<basis>}, NOT optimize_cone.  Choosing
  optimize_cone here silently skips the requested design-level optimization.

▸ cone CONVERSION vs cone OPTIMIZATION
  KEY RULE: "replace/convert/restructure/rebuild the (logic) cone of X using only
  <basis> gates while preserving functional equivalence" with NO cost-function
  sentence is a CONVERSION, not an optimization:
    → convert_basis {"basis": <basis>, "scope": X}
  Route to optimize_cone ONLY when the request states an optimization goal or a
  cost function ("for minimum depth", "optimize", "the cost function is ...",
  "smaller is better").  "Try to ..." / "while preserving equivalence" alone do
  NOT make it an optimization.

▸ fanout vs transitive_fanout vs connected_to_net
  "fanout {net}"            → what the net drives DIRECTLY (list of driven gates/loads).
    For a clock/reset net this is still fanout ONLY when the request asks for
    its loads generically; if it asks which FLIP-FLOPS the net clocks, that is
    ffs_on_clock (see below).
  "transitive_fanout {net}" → everything downstream through multiple gate levels.
  "connected_to_net {net}"  → every gate touching the net (its driver AND its loads),
    e.g. "list all gates that connect to the renamed signal X".

▸ insert_buffers scope
  "no GATE drives more than k loads"        → {"k":k, "scope":"gate"}
  "no SIGNAL/NET drives more than k loads"  → {"k":k, "scope":"signal"}  (also bounds PIs)
  KEY RULE: a "no ... drives more than k loads" bound ALWAYS carries BOTH k and
  scope — never omit scope, and never emit mode for these.  Omitting
  scope:"signal" silently exempts primary inputs from the bound and fails the
  request.
  "insert a BUF on signal X so each load is driven through a DEDICATED buffer"
  (one buffer per load, no numeric bound stated) → {"net":X, "mode":"dedicated"}.
  mode is ONLY for this dedicated-per-load form.  Either way this is ALWAYS
  insert_buffers, never noop — it is a structural transform.

▸ deepest_output vs largest_fanin_cone
  "DEEPEST fan-in cone / deepest logic cone" (by logic DEPTH) → deepest_output {}
  "LARGEST/BIGGEST fan-in cone" (by GATE COUNT)               → largest_fanin_cone {}
  KEY RULE: "deepest" asks about depth, not size — never answer it with
  largest_fanin_cone.

▸ enable/hold structures (flip-flop D-input logic)
  "enable or hold structures", "D input logic ... enable or hold" are NOT net names.
  Report/list request → enable_hold_report {} ; "how many flip-flops ..." → enable_hold_count {}

▸ count_type vs delta_count after a transform
  "How many X gates are NOW in the design (after the conversion)?" → count_type {type}
  (present-tense inventory).  delta_count only for "how many were added/removed/...".
  For gates ADDED by a conversion, kind is "<added-type>_added":
  "How many NOR gates were added by replacing the XNOR gates?" → {"kind":"nor_added"}

▸ reachable_from vs transitive_fanout
  Both walk downstream from the net and return the SAME instance set: a
  flip-flop counts as soon as the walk lands on any of its input pins
  (D/CK/RN/SN), and the walk STOPS at the register boundary — it does not
  continue out of Q (Q&A A21.2).  The net's own driver is not downstream.
  "reachable_from {net}" for "(determine/list/how many) gates reachable from X"
  (answers with the name list); "transitive_fanout {net}" when the request says
  "transitive fanout (cone)" (answers with the count).

▸ rename kind
  kind echoes the noun used in the request: "rename wire X" → "wire",
  "rename (internal) signal X" → "signal", "rename gate X" → "gate".

▸ cone_depth vs max_depth_between vs cone_gate_count
  "cone_depth {output}"      → the DEPTH of ONE named net's fanin cone.
    Triggers: "logic depth of the fanin cone of n30[0]", "how deep does the
              logic that ends at n31[0] run", "worst-case number of gates
              stacked between the inputs and n31[0]".
  "max_depth_between {a, b}" → depth between TWO concrete net names.
  "cone_gate_count {output}" → how MANY gates the cone holds (size, not depth).
  KEY RULE: count the concrete net names.  ONE net + depth wording →
  cone_depth, never max_depth_between: "between the inputs and n31[0]" names
  only n31[0], and "the inputs" is a class, not an endpoint.  TWO concrete
  names → max_depth_between.  One net but asking "how many gates" →
  cone_gate_count.

▸ gate-named downstream queries: connected_to_output vs successors vs fanout
  Both gate intents return what the gate feeds; the VOCABULARY decides which.
  "successors {gate}"          → GRAPH / ORDERING wording: adjacent, neighbour,
    border, edge, hop, next, follows, immediate, successor, descendant, heir.
    Triggers: "which cells border g12 on its output side", "one edge away from
              g0, downstream", "one hop downstream of g12", "the very next
              gates after g71713", "the first-generation descendants of g1259",
              "which cells directly follow g868 in the graph", "where does the
              signal go the instant it leaves g926".
  "connected_to_output {gate}" → PIN / WIRE / LOAD wording: output pin, output
    signal, output terminal, connected/wired/attached/soldered to, loads,
    consumers, takers, "who reads what X writes".
    Triggers: "what hangs off the output pin of g12", "which loads does the
              output pin of g454 carry", "who consumes what g1259 produces",
              "show every cell that taps the output of g868".
  "fanout {net}"               → only when the named thing is a NET.
  KEY RULE: a gate/instance name (g0, cg288) NEVER routes to fanout.  Between
  the two gate intents choose by vocabulary — adjacency/ordering → successors,
  pin/wire/load → connected_to_output.  Both describe the same walk, so the
  wording is the only signal.

▸ dominator vs path_exists vs depends_on
  dominator and path_exists BOTH name two endpoints plus a third thing to get
  around, and BOTH are phrased with bypass / around / avoid / skip / dodge /
  free-of wording.  That wording therefore decides NOTHING.  What decides is
  whether the third name is a GATE or a NET:
  "dominator {a, b, gate}"    → third name is a GATE instance (g0, g454, g1259).
    Asks whether that gate is unavoidable.  Answered yes/no.
    Triggers: "is there any way around g454 when traveling from n24[2] to
              n25[0]", "does g926 hold a monopoly on the traffic between n0[0]
              and n117[1]", "is g100 a mandatory stop on every route from
              n24[2] to n26[1]", "would blocking g1259 silence every message
              n24[2] sends toward n31[1]", "traveling from n14 to n117[1], is
              skipping g926 ever an option".
  "path_exists {a, b, avoid}" → third name is a NET / node (n141, n719, n208).
    Asks whether a path survives with that net off-limits.
    Triggers: "does n14 have a way to n31[1] that bypasses n719", "steering
              clear of n95, does n0 still connect to n26[1]", "is there a
              n86984-free route from n2 to n13[0]", "with n4156 declared
              off-limits, does n3 still find its way to n31[1]".
  "depends_on {output, input}"→ only TWO names, asking functional influence:
    "does n2 have any influence over n30[0]", "will changing n4 ever change
    n32[0]".
  KEY RULE: gate name in the avoid/through position → dominator; net name there
  → path_exists.  Never choose between them on the phrasing.
  depends_on answers FUNCTIONAL influence by default: can changing the input
  ever change the output.  A net can sit in the fan-in cone and still have no
  influence, so "X lies in the structural fanin cone of Y -- is Y functionally
  sensitive to X?" is depends_on with the default kind, NOT a cone query.  Pass
  kind="structural" only when the request asks about cone membership itself.

▸ is_cut vs articulation vs dominator
  "is_cut {wire}"          → is this ONE wire a cut between ANY primary input and
    ANY primary output? yes/no.  No endpoints are named.
    Triggers: "would the design split apart if we removed n1203", "is n1209
              load-bearing for the input-to-output connectivity", "if n1207
              were severed, would some output lose contact with the inputs".
  "articulation {a, b}"    → LIST the cut vertices between TWO named endpoints.
  "dominator {a, b, gate}" → two endpoints AND a named gate, answered yes/no.
  KEY RULE: if the request names ONE wire and leaves the endpoints as classes
  ("any primary input", "the inputs", "some output"), it is is_cut.
  articulation requires two concrete endpoints and dominator requires two
  endpoints plus a gate — neither can be chosen when only one name appears.

▸ ffs_on_clock vs fanout / reachable_from / gates_driven_by
  "ffs_on_clock {clk}" → which flip-flops are clocked by this net.
    Triggers: "which state elements does n0 drive", "which registers march to
              the beat of n0", "who gets clocked by n0", "round up every DFF
              ticking on n0", "which flip-flops belong to the domain of n0",
              "show the flops whose clock pin is wired to n0".
  KEY RULE: when the request asks which FLIP-FLOPS / registers / state
  elements / DFFs / flops a net drives or clocks, use ffs_on_clock — NOT
  fanout, reachable_from or gates_driven_by.  The deciding clue is the NOUN
  being asked for (a sequential element), not the verb ("drive").  A bare
  "what does n0 drive" with no sequential noun stays fanout.

▸ same_clock vs signals_equivalent
  "same_clock {a, b}"         → are these two FLIP-FLOPS clocked by the same net?
    Triggers: "are g3 and g53 on the same clock domain", "do g9 and g59 tick
              together", "g7 versus g57: common clock or different clocks".
  "signals_equivalent {a, b}" → do these two SIGNALS carry the same logic value
    for every input?  Triggers: "would swapping n3 for n41 change anything
    logically", "check if n5 mirrors n43 exactly".
  KEY RULE: the objects decide, not the word "same".  Two flip-flop/instance
  names + clock wording → same_clock.  Two net names + value/equivalence
  wording → signals_equivalent.

▸ highest_fanout_pi vs highest_fanout_net
  "highest_fanout_pi {}"  → the question names primary inputs: "which primary
    input drives the most loads", "which PI has the highest fanout".
  "highest_fanout_net {}" → every driven net: "which signal drives the largest
    number of loads", "the highest fanout over all nets in this design",
    "which net has the highest fanout".  Also answers "report the driver and
    the fanout count of <net>" for that net.
  KEY RULE: the busiest net is usually INTERNAL, so the two return different
  nets and different numbers -- on one reference design 8 (PI) against 12
  (design).  Choose the PI version only when the request says "primary input"
  or "PI"; "signal", "net" and "over all nets" are the design-wide question.

▸ "does the design contain any X" -- existence questions
  There is no existence intent.  Ask for the COUNT of the thing and the number
  answers the question; zero means no.
    a gate type ("does this design contain any sequential element such as a
      DFF", "are there any XOR gates", "does it use buffers")
        → count_type {type}          type = and|or|not|nand|nor|xor|xnor|buf|dff
    floating signals            → floating_count {}
    constant-input gates        → report_const_gates {type}
    a path between two nets     → path_exists {a, b}
  KEY RULE: "does ... contain" is not a relation between two named objects.
  depends_on takes an OUTPUT and an INPUT that must both be real nets, so
  {"output":"design","input":"DFF"} is not a smaller mistake than picking the
  wrong intent -- it names nothing that exists and the request is answered
  with nothing at all.  A trailing "answer yes or no" does not change which
  intent computes the fact.

▸ "answer yes or no" -- form and threshold
  A request that asks for a verdict is not answered by a measurement alone.
  Two shapes, and the right one depends on whether a number is compared:
    EXISTENCE ("do they share at least one gate", "does this design contain
      any XOR gates", "are there dangling gates") -- the answer is whether the
      set is non-empty, so pass form="yesno":
        → {"intent":"shared_cone","params":{"a":"n1","b":"n2","form":"yesno"}}
        → {"intent":"count_type","params":{"type":"dff","form":"yesno"}}
    THRESHOLD ("is the depth at most 5", "does any net drive more than 8
      loads") -- pass BOTH the number and the direction:
        threshold = the number from the request
        compare   = at_most | at_least | greater | less | equal
        → {"intent":"max_depth_between","params":{"a":"g30","b":"g99",
           "threshold":5,"compare":"at_most"}}
  KEY RULE: the direction is not optional.  The same measurement answers yes
  to "at most 5" and no to "more than 5", so a threshold without a direction
  cannot be turned into a verdict and the reply falls back to the bare number.
  Do not switch to a different intent because the request ends in "answer yes
  or no" -- it describes the reply, not the operation.

▸ threshold questions ("more than k", "exceeds k", "at most k")
  The QUANTITY being compared picks the intent -- the threshold wording is
  common to all of them and decides nothing.
    depth of outputs      → outputs_depth_gt {k}
    depth between two nets→ max_depth_between {a, b}
    fanout of a named net → fanout {net} / max_fanout_of {net}
    fanout of any PI      → highest_fanout_pi {}
  outputs_depth_gt is the only intent whose name contains a threshold, which
  makes it a magnet for every "exceeds k" sentence.  It answers ONE question:
  how many primary outputs have a combinational depth greater than k.  A
  request about FANOUT exceeding k is not that question, however similar the
  sentence looks.
  A yes/no threshold question is still answered by the intent that computes
  the quantity: report the number and it settles the comparison.  Do not
  reach for a different intent because the request ends in "answer yes or no".

▸ ANSWER SHAPE is a parameter, not a different intent
  Some questions select the same set of gates and differ only in what is
  reported.  Do NOT hunt for a second intent for the other shape -- pass
  "form" instead:
    form = "count"  when the request says how many / count / the number of
    form = "list"   when it says list / name / enumerate / which gates /
                    "report only the instance names"
  Omit form when the request asks for neither in particular; the answer then
  carries both.
  Applies to: transitive_fanin, transitive_fanout, successors.
    "How many gates are immediate successors of g36?"
      → {"intent":"successors","params":{"gate":"g36","form":"count"}}
    "List the instance names of all gates in the fanin cone of N426."
      → {"intent":"transitive_fanin","params":{"net":"N426","form":"list"}}
  KEY RULE: pick the intent from WHAT IS BEING ASKED ABOUT (the set), and the
  form from HOW IT SHOULD BE REPORTED.  Choosing a neighbouring intent because
  it happens to print the shape you want gives the wrong set.

▸ transitive_fanin vs reachable_from
  "transitive_fanin {net}" → everything UPSTREAM of the net.
    Triggers: "the full ancestry of n31[0]", "the complete upstream closure of
              n33[0]", "compute the transitive fanin of n30[0]".
  "reachable_from {net}"   → everything DOWNSTREAM of the net.
  KEY RULE: direction decides it.  ancestry / upstream / feeds-into / closure
  of what drives it → transitive_fanin.  reachable / downstream / what it
  drives → reachable_from.

▸ remove_buffers vs remove_dangling vs collapse_inverters
  "remove_buffers {}"      → the request names BUF/buffer gates: "remove every
    BUF gate by connecting each buffered signal directly to its destination",
    "strip the buffers so no BUF remains".
  "remove_dangling {}"     → gates that drive nothing / do not reach a PO.
  "collapse_inverters {}"  → back-to-back NOT pairs.
  KEY RULE: all three delete gates, but each names WHICH gates in the request.
  A buffer is not dangling -- it is on a live path -- so remove_dangling
  removes none of them and reports success while every BUF is still there.

▸ check_dangling vs remove_dangling -- asking is not instructing
  "check_dangling {}"  → the request only ASKS: "does this design contain any
    dangling gates", "are there gates that do not reach a primary output",
    "report any unused logic".  The design is left untouched.
  "remove_dangling {}" → the request INSTRUCTS: "remove them", "delete unused
    gates", "prune the netlist".  Note that "Remove them if found" is an
    instruction, not a question, even though it is conditional.
  KEY RULE: choosing the transform for a question is not a wrong answer, it is
  an unrequested EDIT -- every later request in the case then runs against a
  netlist the caller never asked for, and an equivalence check against the
  original will not flag it because removing dead logic preserves function.
  The same asking/instructing split applies to report_const_gates vs
  const_propagate.

▸ minimize_area vs remove_dangling
  "minimize_area {basis}" → RESYNTHESIZE the logic to reduce total gate count.
    Triggers: "minimize the total gate count", "rework the logic so the design
              ends up with as few cells as possible", "shrink this netlist —
              every gate you can spare, remove".
  "remove_dangling {}"    → delete only gates that are ALREADY unused/dead.
  KEY RULE: a request to make the design smaller by restructuring is
  minimize_area even when it uses the word "remove".  remove_dangling applies
  only when the request names dead / unused / unobservable / inert logic
  specifically.

▸ symmetric vs signals_equivalent
  "symmetric {output, a, b}"  → THREE names: does the OUTPUT keep its function
    when the two inputs trade places?
    Triggers: "would n13[0] even notice if n2 and n1 switched roles", "is
              n31[0] blind to which of n1 and n3 carries which value",
              "interchangeable or not: n24[1] and n0[3], as seen from n63[1]",
              "does the function at n15 stay fixed under swapping n0[1] with
              n9[0]".
  "signals_equivalent {a, b}" → TWO names: do these two signals carry the same
    value for every input?
  KEY RULE: swap/exchange/interchange wording with a THIRD name naming where
  it is observed → symmetric.  Only two names and no observation point →
  signals_equivalent.  Both read as "would anything change"; the name count
  decides.

▸ counting vs listing: how many vs which
  Several op pairs walk the same graph and differ only in whether the request
  wants a NUMBER or the NAMES:
  "gates_driven_by {gate}"     → HOW MANY gates the gate drives.
    Triggers: "how many cells hang off g868 directly", "the output of g926
              lands on how many gates".
  "connected_to_output {gate}" → WHICH gates.  "which cells receive their
    signal straight off the back of g1259".
  "cone_gate_count {output}"   → HOW MANY gates the fanin cone holds.
    Triggers: "what is the size of the machinery upstream of n11[0]", "the
              number of gates ancestral to n12 is what", "how many gates are
              inside the fence around everything n11[0] depends on".
  "transitive_fanin {net}"     → the cone itself ("compute the transitive
    fanin of n30[0]", "the full ancestry of n31[0]", "the complete upstream
    closure of n33[0]").
  KEY RULE: "how many" / "the number of" / "the size of" / "count" asks for a
  number — take the counting op.  "which" / "name them" / "list" / "compute
  the cone" asks for the objects.

▸ connected_to_net vs connected_to_output
  "connected_to_net {net}"     → the named thing is a NET: every gate touching
    it, driver and loads alike.  Triggers: "identify all hardware attached to
    stage2_out", "what plugs into stage2_out? every gate counts".
  "connected_to_output {gate}" → the named thing is a GATE.
  KEY RULE: same discriminator as elsewhere — a gate/instance name (g0, cg288)
  versus a net name (n14, stage2_out) decides it, not the wording.

▸ articulation vs shared_cone
  "articulation {a, b}"  → cut vertices ON THE PATHS between two nets: points
    every route from a to b must pass.  Triggers: "across the web of routes
    linking n1 and n63[0], which points are common to all of them".
  "shared_cone {a, b}"   → gates in BOTH nets' fanin cones — shared ancestry,
    no path between the two required.  Triggers: "which gates would both
    n63[1] and n31[0] lose if they were removed".
  KEY RULE: routes/paths BETWEEN the two → articulation.  What the two have in
  COMMON upstream → shared_cone.

▸ floating_count vs delta_count
  "floating_count {}"  → how many floating/unconnected signals the preceding
    check reported.  Triggers: "how many floating signals were found", "so how
    many loose ends did that check turn up", "what is the count of
    unconnected nets".
  "delta_count {kind}" → how many GATES a preceding TRANSFORM added or removed.
  KEY RULE: "how many ... floating / unconnected / undriven / loose ends" is
  floating_count even though it refers back to an earlier step.  The
  "count_gates vs delta_count" rule above lists "floating" among the adjectives
  describing removed GATES; that does not apply when the question asks how many
  floating SIGNALS a check found.

━━━ FEW-SHOT EXAMPLES ━━━

"Establish a new test environment. Benchmark name: test38."
→ {"intent":"begin_case","params":{"name":"test38"}}

"Ingest the circuit schematic from testcase/test38/test38.v."
→ {"intent":"load_design","params":{"file":"test38.v","dir":"testcase/test38"}}

"Audit the primitive count per logic family: AND, OR, NOT, NAND, NOR, XOR, XNOR, BUF, DFF."
→ {"intent":"count_gates","params":{}}

"Determine the number of primary inputs and primary outputs in this design."
→ {"intent":"count_ports","params":{}}

"Size of the I/O footprint, please."
→ {"intent":"count_ports","params":{}}

"Count the pins on the boundary: how many in, how many out?"
→ {"intent":"count_ports","params":{}}

"Apply fanout-reduction buffers to net n1 until no driver fanout exceeds 4."
→ {"intent":"insert_buffers","params":{"k":4,"net":"n1"}}

"Insert buffers wherever needed so that no signal drives more than 16 loads. Make sure nothing changes functionally. The cost function is the total gate count of the final design; smaller is better."
→ {"intent":"insert_buffers","params":{"k":16,"scope":"signal"}}

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

"Convert the logic cone of n10 to use only NOR and NOT gates while preserving functional equivalence."
→ {"intent":"convert_basis","params":{"basis":"NOR_NOT","scope":"n10"}}

"Try to restructure the logic cone of output n8 using only NAND and NOT gates while preserving functional equivalence."
→ {"intent":"convert_basis","params":{"basis":"NAND_NOT","scope":"n8"}}

"Replace all 2-input OR gates in the cone of n11[0] with equivalent logic built only from NAND and NOT gates. Ensure the design functionality does not change."
→ {"intent":"convert_basis","params":{"basis":"NAND_NOT","scope":"n11[0]"}}

"What is the maximum logic depth from any primary input to any primary output?"
→ {"intent":"global_max_depth","params":{}}

"What is the PI-to-PO depth of this design?"
→ {"intent":"global_max_depth","params":{}}   (no "combinational" → design-wide maximum)

"What is the maximum combinational logic depth from any primary input to any primary output?"
→ {"intent":"pi_to_po_depth","params":{}}

"Excise all logically inert gates from the netlist."
→ {"intent":"remove_dangling","params":{}}

"How many gates were excised in the previous step?"
→ {"intent":"delta_count","params":{"kind":"dangling"}}

"How many redundant gates were removed?"
→ {"intent":"delta_count","params":{"kind":"removed"}}

"Give the tally of NAND gates absorbed by constant folding."
→ {"intent":"delta_count","params":{"kind":"nand"}}

"Identify all zero gate-hop paths from primary inputs to primary outputs."
→ {"intent":"length_zero_paths","params":{}}

"Enumerate the bridge nodes in the combinational DAG spanning from n2 to n14."
→ {"intent":"articulation","params":{"a":"n2","b":"n14"}}

"Determine whether the wire n1200 is a cut between any primary input and any primary output. Report yes or no."
→ {"intent":"is_cut","params":{"wire":"n1200"}}

"Would the design split apart if we removed n1203?"
→ {"intent":"is_cut","params":{"wire":"n1203"}}

"Is n1209 load-bearing for the input-to-output connectivity?"
→ {"intent":"is_cut","params":{"wire":"n1209"}}

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

"How many gates are in the logic cone of output n12?"
→ {"intent":"cone_gate_count","params":{"output":"n12"}}   (plain count; cone_type_counts only when a per-type breakdown is asked)

"Report every gate connected to the output of g0."
→ {"intent":"connected_to_output","params":{"gate":"g0"}}

"Optimize the logic to minimize maximum path depth, ensuring the cone of n15 continues to use only NAND and NOT gates. Preserve functional equivalence."
→ {"intent":"minimize_depth","params":{"basis":"NAND_NOT"}}

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

"Check whether the function at n30[0] is symmetric with respect to inputs n2 and n0[0]."
→ {"intent":"symmetric","params":{"output":"n30[0]","a":"n2","b":"n0[0]"}}

"Could I swap n3 and n0[1] without n31[0] noticing?"
→ {"intent":"symmetric","params":{"output":"n31[0]","a":"n3","b":"n0[1]"}}

"Are n9 and n0[3] equal citizens as far as n37[0] is concerned?"
→ {"intent":"symmetric","params":{"output":"n37[0]","a":"n9","b":"n0[3]"}}

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

"Do flip-flops g0 and g50 use the same clock signal? Report yes or no."
→ {"intent":"same_clock","params":{"a":"g0","b":"g50"}}

"Are g3 and g53 on the same clock domain?"
→ {"intent":"same_clock","params":{"a":"g3","b":"g53"}}

"Do g9 and g59 tick together — same clock source?"
→ {"intent":"same_clock","params":{"a":"g9","b":"g59"}}

"Which output has the largest fanin cone?"
→ {"intent":"largest_fanin_cone","params":{}}

"Which output bit has the deepest fanin logic cone?"
→ {"intent":"deepest_output","params":{}}

"What is the maximum combinational logic depth in the design now?"
→ {"intent":"global_max_depth","params":{}}

"Please insert a BUF gate on signal n2 so that each load of n2 is driven through a dedicated buffer. Ensure the design functionality does not change."
→ {"intent":"insert_buffers","params":{"net":"n2","mode":"dedicated"}}

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
- successors {gate, form}       # form = count|list (see ANSWER SHAPE)
- transitive_fanin {net, form}  # form = count|list
- transitive_fanout {net, form} # form = count|list
- reachable_from {net}
- highest_fanout_pi {}          # PI-scoped ONLY
- highest_fanout_net {threshold, compare}  # busiest net in the design, plus its driver
- shared_cone {a, b, form}
- connected_to_output {gate}
- path_exists {a, b, avoid}
- enumerate_paths {a, b}
- length_zero_paths {}
- dominator {a, b, gate}
- articulation {a, b}
- is_cut {wire}
- max_depth_between {a, b, threshold, compare}   # both endpoints concrete net names
- cone_depth {output}
- global_max_depth {}
- pi_to_po_depth {}             # any primary input -> any primary output
- pi_to_dff_depth {}            # any primary input -> any DFF D-pin
- reg_to_reg_depth {}           # any register output -> any register input
- reg_to_po_depth {}            # any register/DFF output -> any primary output
- outputs_depth_gt {k}          # DEPTH only: how many POs sit deeper than k levels
- deepest_output {}             # deepest cone (by depth)
- largest_fanin_cone {}         # largest cone (by gate count)
- gate_on_max_path {gate}
- signals_equivalent {a, b}
- output_constant {output}
- depends_on {output, input, kind}  # kind = functional (default) | structural
- boolean_equation {output}
- symmetric {output, a, b}
- exists_nand_pair {target}
- ffs_on_clock {clk}
- same_clock {a, b}
- reg_to_reg_paths {}
- enable_hold_report {}
- enable_hold_count {}
- convert_basis {basis, scope}  # basis = NAND|NOR|NAND_NOT|NOR_NOT|AND_NOT|AND_OR_NOT
- xor_to_nand {scope}
- xnor_to_nor {scope}
- xnor_to_nand {scope}
- xor_to_aoi {scope}
- nand_const1_to_inv {}
- report_const_gates {type, value}
- const1_gates {}               # list ALL gates (any type, incl. DFF pins) tied to 1'b1
- check_floating {}             # floating inputs / unconnected output ports?
- floating_count {}             # how many floating signals were found
- const_propagate {type}
- collapse_inverters {}         # NOT(NOT(a)) pairs -> direct wire
- remove_buffers {}             # delete every BUF, rewiring around it
- check_dangling {form}         # REPORT them, design untouched
- remove_dangling {}            # DELETE them
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
    "highest_fanout_net",
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
    "xor_to_nand", "xnor_to_nor", "xnor_to_nand", "xor_to_aoi",
    "nand_const1_to_inv",
    "report_const_gates", "const_propagate", "collapse_inverters",
    "remove_buffers",
    "remove_dangling", "check_dangling", "merge_duplicates", "rename",
    "insert_buffers",
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
    "shared_cone": {"form"},
    "max_depth_between": {"threshold", "compare"},
    "highest_fanout_net": {"threshold", "compare"},
    "check_dangling": {"form"},
    "depends_on": {"kind"},
    "transitive_fanin": {"form"},
    "transitive_fanout": {"form"},
    "successors": {"form"},
    "load_design": {"dir"},
    "count_type": {"scope", "form"},
    "path_exists": {"avoid"},
    "convert_basis": {"basis", "scope"},
    "xor_to_nand": {"scope"},
    "xnor_to_nor": {"scope"},
    "xnor_to_nand": {"scope"},
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
# Must track cada.transform.rewrite.BASES.  A value accepted there and
# rejected here reads as a routing failure: the model picks the right
# intent with the right basis and the request is answered with nothing.
BASIS_VALUES = {"NAND", "NOR",
                "NAND_NOT", "NOR_NOT", "AND_NOT", "AND_OR_NOT"}
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

    # k bounds fanout-mode buffering; dedicated-per-load has no bound to name,
    # but it does need the net it buffers.  Without that net the request is not
    # under-specified in a harmless way: op_insert_buffers falls past its
    # dedicated branch into the design-wide default bound, which on test31
    # turns 2 requested buffers into 649 and reports the wrong count for the
    # rest of the case.
    if intent == "insert_buffers" and raw_params.get("mode") == "dedicated":
        required = (required - {"k"}) | {"net"}

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

    # A fanout bound of 1 is not satisfiable: bounding every driver to a single
    # load turns the buffer tree into a path, which can reach exactly one sink.
    # The request that sounds like it ("one buffer per load") is the dedicated
    # form, so say so rather than letting buffering spin on an unreachable goal.
    if intent == "insert_buffers" and params.get("k") == 1:
        return ('A fanout bound of k=1 cannot be met — one load per driver '
                'admits no tree. For "a dedicated buffer per load of X" use '
                '{"mode":"dedicated","net":X}; for a real bound use k >= 2.')

    # Endpoint pairs must name two different endpoints.  a == b is what the
    # model emits when a single-endpoint question ("how deep is the logic
    # ending at X") gets routed to a two-endpoint intent; rejecting it sends
    # the request back with the reason instead of answering the degenerate
    # question the model actually asked.
    if "a" in params and "b" in params and params["a"] == params["b"]:
        return (f'intent "{intent}" needs two different endpoints, but a and b '
                f'are both "{params["a"]}". If the request names only one '
                "endpoint, it is asking about that endpoint's cone — use a "
                "cone-scoped intent (cone_depth, cone_gate_count, "
                "transitive_fanin) instead.")

    return None
