"""LLM fallback: translate one NL line into a structured intent.

Used only when the deterministic regex router fails to recognise a line.  The
LLM is constrained to emit exactly one JSON object ``{"intent","params"}``
drawn from our published intent catalogue; the result is validated and cached
(so temperature noise never makes behaviour non-deterministic within a run).
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

# A compact description of the EDA interface we expose to the LLM.
INTENT_CATALOG = """
You translate a single natural-language EDA request into ONE JSON object.
Output ONLY the JSON, no prose, no markdown.  Schema:
  {"intent": "<name>", "params": { ... }}
Never reason about the circuit; only classify the request and extract names.

Available intents (params in brackets):
- load_design {file, dir}
- write_design {file}
- count_gates {}                         # full type breakdown
- total_gate_count {}
- count_type {type}                      # how many <type> now
- delta_count {kind}                     # how many added/removed/merged/eliminated
- gate_info {gate}                       # type + pin connections
- list_type {type}                       # list gates of a type
- cone_gate_count {output}
- cone_type_counts {output}
- list_ports {dir}                       # dir = input|output
- count_ports {}
- fanout {net}                           # fanout + loads
- gates_driven_by {gate}
- successors {gate}
- transitive_fanin {net}
- transitive_fanout {net}
- reachable_from {net}
- highest_fanout_pi {}
- max_fanout_of {net}
- shared_cone {a, b}
- connected_to_output {gate}
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
- convert_basis {basis, scope}           # basis = NAND_NOT|NOR_NOT|AND_NOT|AND_OR_NOT
- xor_to_nand {scope}
- xnor_to_nor {scope}
- xor_to_aoi {scope}
- nand_const1_to_inv {}
- report_const_gates {type, value}
- const_propagate {type}
- collapse_inverters {}
- remove_dangling {}
- merge_duplicates {}
- rename {kind, old, new}                # kind = gate|wire|signal
- insert_buffers {k, net, mode}          # mode = fanout|dedicated
- minimize_depth {basis}
- minimize_area {basis}
- optimize_cone {output, basis}
- verify_equivalence {against}           # against = original|pre|last_loaded
- begin_case {name}
- noop {}
"""


def parse_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict) and "intent" in obj:
            obj.setdefault("params", {})
            return obj
    except Exception:
        return None
    return None


class Fallback:
    def __init__(self, client):
        self.client = client
        self.cache: Dict[str, dict] = {}

    def translate(self, line: str) -> Optional[dict]:
        key = line.strip()
        if key in self.cache:
            return self.cache[key]
        if self.client is None or not self.client.available:
            return None
        out = self.client.complete(INTENT_CATALOG, key)
        obj = parse_json_object(out or "")
        if obj is not None:
            self.cache[key] = obj
        return obj
