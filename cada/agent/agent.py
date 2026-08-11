"""The agent: rule-first natural-language router over the deterministic EDA
engine, with an LLM fallback for unrecognised phrasings.

Each request is mapped to one intent + params (by regex first, LLM second),
then dispatched to a handler that runs the engine, applies transforms with
rollback-on-violation, and formats a direct answer.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Tuple

from ..io_.config import Config
from ..netlist import reader, writer
from ..netlist.ir import Netlist
from ..analysis import (counts, cones, depth, connectivity, paths,
                        functional, sequential, graph)
from ..transform import rewrite, constprop, cleanup, buffering, naming
from ..optimize import abc_opt
from ..equiv import gate as equiv_gate
from ..guards import validators
from .state import State
from ..llm.client import LLMClient
from ..llm.fallback import Fallback

NET = r"[A-Za-z_]\w*(?:\[\d+\])?"
BASIS_WORDS = {
    ("nand", "not"): "NAND_NOT", ("nor", "not"): "NOR_NOT",
    ("and", "not"): "AND_NOT",
}


def _basis_from_text(text: str) -> Optional[str]:
    t = text.lower()
    has = lambda *ws: all(w in t for w in ws)
    if has("nand") and "not" in t and "nor" not in t:
        return "NAND_NOT"
    if has("nor") and "not" in t:
        return "NOR_NOT"
    if has("and", "or") and "not" in t:
        return "AND_OR_NOT"
    if has("and") and "not" in t and "nand" not in t and "nor" not in t:
        return "AND_NOT"
    return None


class Agent:
    def __init__(self, config: Config, use_rules: bool = True):
        self.config = config
        self.state = State()
        self.llm = LLMClient(config)
        # "auto" = dense retrieval when the model ships, BM25 otherwise, and
        # None if even the example bank is missing.  Every step degrades to
        # the catalog-only behaviour rather than failing the run.
        from ..llm.retrieval import build_retriever
        self.fallback = Fallback(self.llm, retriever=build_retriever("auto"))
        self.use_rules = use_rules
        self.rules = self._build_rules()
        self.const_nets = {}     # functionally-constant nets (from last report)

    # ===================================================================
    # main entry
    # ===================================================================
    def handle(self, line: str, ident: int) -> str:
        line = line.strip()
        if self.use_rules:
            for rx, fn in self.rules:
                m = rx.search(line)
                if m:
                    try:
                        return fn(m, line)
                    except Exception as exc:
                        return f"Could not complete the request ({exc})."
        # LLM fallback
        obj = self.fallback.translate(line)
        if obj is not None:
            # Semantic gate: params that must name a real net/instance but
            # don't (e.g. {"a":"primary_input"}) are sent back with the error
            # so the model can pick a class-level intent instead.
            bad = self._invalid_name_params(obj)
            if bad:
                obj2 = self.fallback.retranslate(line, bad)
                if obj2 is not None and not self._invalid_name_params(obj2):
                    obj = obj2
            try:
                return self._dispatch_intent(obj.get("intent"), obj.get("params", {}), line)
            except Exception as exc:
                return f"Could not complete the request ({exc})."
        return self._default_ack(line)

    # params that must refer to an existing net or instance name
    _NAME_PARAM_KEYS = ("a", "b", "net", "gate", "output", "wire", "input",
                        "target", "clk", "old")

    def _invalid_name_params(self, obj) -> str:
        """Error text if a name-typed param doesn't exist in the design."""
        nl = self.state.current
        if nl is None:
            return ""
        params = obj.get("params") or {}
        names = self._known_names(nl)
        bad = [f'{k}="{params[k]}"' for k in self._NAME_PARAM_KEYS
               if isinstance(params.get(k), str) and params[k] not in names]
        if not bad:
            return ""
        return (f'intent "{obj.get("intent")}" has param(s) {", ".join(bad)} '
                "that name no net, gate, or flip-flop in the current design.")

    def _known_names(self, nl):
        # Rebuilt per call: transforms and renames mutate the netlist in
        # place, so a cached name set would go stale and misreport fresh
        # identifiers (e.g. a just-renamed signal) as unknown.
        names = {"1'b0", "1'b1"}
        names.update(nl.pi); names.update(nl.po)
        for g in nl.gates:
            names.add(g.name); names.add(g.out); names.update(g.ins)
        for ff in nl.dffs:
            names.update((ff.name, ff.d, ff.q, ff.clk, ff.rn, ff.sn))
        names.discard(None)
        return names

    def _default_ack(self, line: str) -> str:
        return ("Acknowledged. The request was not mapped to a specific EDA "
                "operation; the current design is unchanged.")

    # ===================================================================
    # rule table
    # ===================================================================
    def _build_rules(self) -> List[Tuple[re.Pattern, Callable]]:
        R = lambda p: re.compile(p, re.IGNORECASE)
        rules = [
            (R(r"beginning of (a new )?testcase|case name is"), self.h_begin),

            # buffer-insertion rules must precede the load rule: "each load of
            # n2" + "the design functionality" otherwise false-positives the
            # load-design regex (the historic test31/test39 misroute).
            (R(r"insert a BUF gate on signal (%s).*dedicated buffer" % NET), self.h_buffers_dedicated),
            (R(r"insert.*buffers?.*no (gate|signal|net)\b.*?(drives|drive|has fanout).*?(\d+)"), self.h_buffers_fanout),
            (R(r"insert.*buffers? on (?:the )?(?:reset )?signal (%s).*?(\d+) loads" % NET), self.h_buffers_signal),
            (R(r"buffers on the reset signal (%s)" % NET), self.h_buffers_reset),

            # delta questions ("how many X were merged/added/...") must precede
            # the transform rules, or "were merged as structural duplicates"
            # re-runs the merge instead of answering the recorded delta.
            (R(r"how many flip-?flops .*enable or hold"), self.h_enable_hold_count),
            (R(r"how many (?!.*\b(?:floating|unconnected|undriven)\b).*\b(?:was|were)\b.*\b(?:added|removed|eliminated|merged|collapsed|inserted|found|excised|swept|deleted|absorbed)\b"), self.h_delta),

            # the load rule requires from/file context in the same sentence so
            # a noun "load" ("each load of n2") can never trigger it.
            (R(r"\b(load|read)\b[^.]*\bdesign\b[^.]*\b(from|file)\b|design from (the )?file"), self.h_load),
            (R(r"\bwrite\b.*(design|netlist).*\b(file|to)\b|write out"), self.h_write),

            # optimize rules first: they mention "depth/cone" but must not be
            # captured by the depth/cone *query* rules.
            (R(r"optimize the depth of the cone of (%s)" % NET), self.h_opt_cone),
            (R(r"optimize the (logic )?cone of (?:output )?(%s)" % NET), self.h_opt_cone),
            (R(r"(reduce|minimize|minimise).*(critical path|maximum|max|critical).*depth"), self.h_opt_depth),
            (R(r"(depth optimization|perform depth optimization|reduce critical path)"), self.h_opt_depth),
            (R(r"minimize the maximum logic depth|minimize maximum (logic |path )?depth"), self.h_opt_depth),
            # broader verbs + the cost-function sentence itself; the buffer
            # rules above already claimed lines whose cost is gate count but
            # whose action is buffering (test36-style).
            (R(r"(shorten|reduce|minimi[sz]e|decrease|lower).*(worst-?case|critical|maximum|max).*(path|depth)"), self.h_opt_depth),
            (R(r"cost function is the maximum logic depth"), self.h_opt_depth),
            (R(r"minimi[sz]e (the )?total (number of gates|gate count)|(minimi[sz]e|reduce).*(number of gates|gate count).*without changing|cost function is the total gate count"), self.h_opt_area),
            # explicit resynthesis requests (the reverse-to-RTL/yosys/ABC
            # rebuild tool): scoped cone first, then area wording, then the
            # depth default.  "Reconstruct ... using only X" stays with the
            # basis-conversion rules below (different verb set).
            (R(r"re-?synthesi[sz]e.*cone of (?:output )?(%s)" % NET), self.h_opt_cone),
            (R(r"re-?synthesi[sz]e.*(area|gate count|number of gates)"), self.h_opt_area),
            (R(r"re-?synthesi[sz]e|resynthesis"), self.h_opt_depth),

            # transforms (imperative actions) — matched BEFORE analysis queries
            # so cost-function / "gates in the cone of" phrasing in a transform
            # request is never captured by a query rule.
            # XOR/XNOR-specific conversions must precede the generic basis rule:
            # "replace all XOR gates ... NAND-only implementations" is an
            # XOR->NAND request, not a whole-design NAND+NOT remap.
            (R(r"replace.*XNOR.*(NOR-only|NOR only|equivalent NOR)"), self.h_xnor_nor),
            (R(r"convert every XNOR.*NOR"), self.h_xnor_nor),
            # XNOR->NAND is its own targeted decomposition: without it the
            # request falls through to the whole-design basis remap, which
            # rewrites gates the prompt never named.
            (R(r"(replace|convert|rewrite|re-?implement)\b.*\bXNOR\b.*\bNAND\b"), self.h_xnor_nand),
            (R(r"(replace|convert).*XOR.*(NAND-only|4 ?NAND|4-NAND|NAND)"), self.h_xor_nand),
            (R(r"decompose all XOR.*(AND, OR, and NOT|AND.*OR.*NOT)"), self.h_xor_aoi),
            (R(r"convert every XOR.*4-?NAND"), self.h_xor_nand),
            (R(r"(remap|reconstruct|convert|restructure|replace|rebuild|rewrite|re-?implement|re-?express)\b.*(only|use only|using only).*(nand|nor|and).*(not|nand|nor)"), self.h_basis),
            (R(r"replace all 2-input NAND.*constant 1.*inverter"), self.h_nand_inv),
            (R(r"simplify the reported (\w+) gates|simplify the reported|propagating .*constant"), self.h_constprop),
            (R(r"(back-to-back|back to back).*(invert|NOT).*collapse|collapse them into.*wire|pairs of.*inverters"), self.h_collapse),
            (R(r"(remove|delete|trim|sweep|prune|eliminate).*(dangling|unused|floating|redundant)"), self.h_dangling),
            (R(r"(delete|remove|eliminate|prune).*gates?.*(do not|don't|not) (contribute|affect|connected)"), self.h_dangling),
            # "how many ... floating/unconnected" asks for the count, not a
            # re-run of the check -- it must win over the query rule below.
            # Removal phrasings are left to the delta rule further down.
            (R(r"how many (?!.*\b(?:removed|merged|deleted|swept|excised)\b)"
               r".*\b(?:floating|unconnected|undriven)\b"), self.h_floating_count),
            # port-level floating/unconnected questions must win before the
            # dangling-*gate* rules — different concept, query only (no mutation)
            (R(r"floating (inputs?|signals?)|unconnected (output )?ports?|undriven (inputs?|pins?|ports?)|unloaded output"), self.h_check_floating_ports),
            (R(r"are there any (redundant|dangling) gates"), self.h_check_dangling),
            (R(r"check.*dangling gates|check if there are any floating"), self.h_check_dangling),
            (R(r"(merge|find and merge).*(functionally equivalent|same function|structural duplicate|duplicate)"), self.h_merge),
            (R(r"(rename|change the identifier of|update the name of|change the name).*(gate|wire|signal)\s+(%s)\s+to\s+(%s)" % (NET, NET)), self.h_rename),
            (R(r"list all gates.*connect.*to the renamed signal (%s)" % NET), self.h_connected_renamed),
            # Anchored: "After that rename, what connects to renamed_sig?" is a
            # question ABOUT a past rename, and the loose form matched
            # "connects to renamed_sig" as the old/new pair.
            (R(r"^\W*(?:please\s+)?(rename|update the name of|change the identifier of).*?(%s)\s+to\s+(%s)" % (NET, NET)), self.h_rename2),

            # equivalence verification (imperative)
            (R(r"(verify|prove|confirm|check).*(equivalen|equivalent).*(original|pre-transformation|last loaded|as last loaded|loaded netlist)"), self.h_verify),
            (R(r"(verify|prove).*(transformed|current).*(equivalent|equivalence)"), self.h_verify),

            (R(r"count all the gates|broken down by gate type"), self.h_count_all),
            # Guarded: an optimisation request naming its cost function
            # ("minimize total gate count") is a transform, not a query, and
            # h_opt_area above owns it.
            (R(r"^(?!.*\b(minimi[sz]e|reduce|optimi[sz]e|shrink|lower|cut)\b).*(total gate count|compute the total gate count)"), self.h_total),
            # "Report only ... instance names" is an exact-set contract, so these
            # answer with a bare name list.  They must precede the count-style
            # cone/fanout rules below, which would otherwise capture a stray
            # word as the net and answer with a number.
            (R(r"list .*\bgates?\b.*\bthat\b\s+(?:primary input |input |signal |net )?(%s)\s+drives within (\d+) hop" % NET), self.h_hops_drives),
            (R(r"list .*\bwithin (\d+) hops?\b.*\bof\s+(?:primary input |input |signal |net |gate )?(%s)\s*[.,;]" % NET), self.h_hops_downstream),
            (R(r"list the instance names of all gates in the fan-?in cone of (?:the )?(?:net |signal )?(%s)" % NET), self.h_fanin_cone_names),
            (R(r"report the number of each gate type in the cone of (%s)" % NET), self.h_cone_type),
            (R(r"how many (\w+) gates? (are|were|is)"), self.h_count_or_delta),
            (R(r"how many (\w+) (were|gates were)? ?(added|removed|eliminated|merged|collapsed|inserted|found)"), self.h_delta),
            (R(r"how many (dangling|redundant|floating|duplicate|structural duplicate).*(removed|merged|found)"), self.h_delta),
            (R(r"how many (buf|buffer) gates were added"), self.h_delta),

            (R(r"what type of gate is (%s)" % NET), self.h_gate_info),
            (R(r"list all (\w+) gates? in this design"), self.h_list_type),
            (R(r"list all XOR gates"), self.h_list_xor),
            (R(r"list (all|every).*primary inputs?.*bit widths?"), self.h_list_pi),
            (R(r"list all primary outputs?.*bit widths?"), self.h_list_po),
            (R(r"(number of|how many) primary inputs? and (primary )?outputs?"), self.h_count_ports),
            (R(r"determine the number of primary inputs and outputs"), self.h_count_ports),
            # Requires the counting question.  Without it the rule fired inside
            # transform requests that merely scope themselves to a cone
            # ("Decompose every XOR gate in the fanin cone of n32[0] ...").
            (R(r"(how many|number of)\s+gates?\s+(are )?in the (fanin |logic )?cone of (primary output |output )?(%s)" % NET), self.h_cone_gate_count),
            (R(r"list all gates.*tied to 1'b1|inputs tied to 1'b1"), self.h_const1_gates),
            (R(r"report any (\w+) gates? with (a )?constant"), self.h_report_const),
            (R(r"report any (\w+) gates? with constant inputs"), self.h_report_const),

            # paths
            (R(r"path.*from (%s) to (%s).*(?:does not traverse|avoid\w*|without)\s+(?:node\s+)?(%s)" % (NET, NET, NET)), self.h_path_avoid),
            (R(r"path connecting (?:input )?(%s) to (?:output )?(%s).*avoiding\s+(?:node\s+)?(%s)" % (NET, NET, NET)), self.h_path_avoid2),
            (R(r"combinational path.*from (%s) to (%s).*avoids?\s+(?:node\s+)?(%s)" % (NET, NET, NET)), self.h_path_avoid3),
            (R(r"(does|is there|whether).*combinational path exist.*from (?:primary input )?(%s) to (?:primary output )?(%s)" % (NET, NET)), self.h_path_plain),
            (R(r"path.*from (?:primary input )?(%s) to (?:primary output )?(%s)\??\s*(?:report)?.*exist" % (NET, NET)), self.h_path_plain),
            # "immediate successors of gate X" must win before the enumerate-paths
            # rule below (whose 'enumerat' branch would otherwise capture the "e"
            # and "the" in "Enumerate the ..." as two path endpoints).
            (R(r"immediate successors of (?:gate )?(%s)" % NET), self.h_successors),
            # The bare "enumerat" branch used to fire on any "Enumerate X ... Y"
            # sentence, swallowing successors / connected_to_net / reg_to_reg
            # requests that merely start with the verb.  Require path wording:
            # a miss costs one LLM call, a false match costs the wrong answer.
            (R(r"(complete enumeration|list every path|find all combinational paths|enumerat\w*[^.]*\bpaths?\b).*?(%s).*?(%s)" % (NET, NET)), self.h_enum_paths),
            (R(r"paths? of length 0|direct wire connections from PI to PO"), self.h_len0),
            (R(r"does every path from (?:input )?(%s) to (?:output )?(%s) pass through (?:gate )?(%s)" % (NET, NET, NET)), self.h_dominator),
            (R(r"articulation points.*between (%s) and (%s)" % (NET, NET)), self.h_articulation),
            (R(r"(is |whether )?wire (%s) is a cut" % NET), self.h_cut),

            # depth
            (R(r"(maximum|max).*depth from (?:input )?(%s) to (?:output )?(%s)" % (NET, NET)), self.h_depth_ab),
            (R(r"longest combinational path depth from (%s) to (%s)" % (NET, NET)), self.h_depth_ab2),
            (R(r"critical path depth between (%s) and (%s)" % (NET, NET)), self.h_depth_ab3),
            (R(r"(maximum|max).*depth.*cone of (?:output )?(%s)|depth of the cone of (%s)" % (NET, NET)), self.h_cone_depth),
            (R(r"max(imum)? combinational (logic )?depth on any register-to-register path"), self.h_reg2reg_depth),
            (R(r"max(imum)? (combinational )?(logic )?depth from any primary input to any (DFF )?D-?pin"), self.h_pi2d_depth),
            (R(r"max(imum)? combinational (logic )?depth.*primary input to any primary output"), self.h_pi2po_depth),
            (R(r"max(imum)? combinational (logic )?depth.*(entire design|in the design)"), self.h_global_depth),
            (R(r"max(imum)? (combinational )?logic depth in the design now"), self.h_global_depth),
            (R(r"how many outputs have a logic depth greater than (\d+)"), self.h_depth_gt),
            (R(r"which output (bit )?has the (deepest|largest) (fanin )?(logic )?cone"), self.h_deepest),
            (R(r"(does|whether) gate (%s) lies? on any maximum-depth path" % NET), self.h_on_maxpath),

            # connectivity
            (R(r"fanout of (?:primary input )?(%s).*list (all|every) gate" % NET), self.h_fanout),
            (R(r"(number of gates driven by|gates? driven by) (%s)" % NET), self.h_driven_by),
            (R(r"transitive fanin cone of (?:output )?(%s)" % NET), self.h_tfanin),
            (R(r"transitive fanout (cone )?of (?:input |primary input )?(%s)" % NET), self.h_tfanout),
            (R(r"(all gates reachable from|determine all gates reachable from|reachable from) (%s)" % NET), self.h_reachable),
            (R(r"which primary input (has the highest fanout|drives the largest number of loads)"
               r"|highest fanout (in this design|over all nets)"
               r"|which (signal|net)\b.*(highest fanout|largest number of loads)"), self.h_highest_fanout),
            (R(r"(maximum|max) fanout of (%s) now" % NET), self.h_max_fanout),
            (R(r"gates shared between the fanin cones of (%s) and (%s)" % (NET, NET)), self.h_shared),
            (R(r"(every gate|report every gate) connected to the output of (%s)" % NET), self.h_connected_out),
            (R(r"compute the fanin (logic )?cone of (?:output )?(%s)" % NET), self.h_cone_gate_count),

            # functional
            (R(r"(functionally equivalent|identical logic values|functional equivalence between internal signals|equivalent for all input).*?(%s).*?(%s)" % (NET, NET)), self.h_sig_equiv),
            (R(r"signals? (%s) and (%s) are functionally equivalent" % (NET, NET)), self.h_sig_equiv2),
            (R(r"output (%s) (always 0|is always 0|恒)" % NET), self.h_const_out),
            (R(r"does output (%s) depend on input (%s)" % (NET, NET)), self.h_depends),
            (R(r"(boolean equation|logic expression|boolean function).*?(%s)" % NET), self.h_boolean),
            (R(r"symmetric with respect to inputs (%s) and (%s)" % (NET, NET)), self.h_symmetric),
            (R(r"exist.*\(?(%s)?,?\s*\)?.*NAND\(.*\).*equivalent to (%s)" % (NET, NET)), self.h_nand_pair),
            (R(r"NAND\(a, ?b\) is equivalent to (%s)" % NET), self.h_nand_pair2),

            # sequential
            (R(r"flip-?flops driven by clock (%s)" % NET), self.h_ffs_clock),
            (R(r"(dff\w*|flip-?flop\w*).*same clock domain"), self.h_same_clock),
            (R(r"register-to-register paths"), self.h_reg2reg_paths),
            (R(r"D input logic.*enable or hold|enable or hold structures"), self.h_enable_hold),
        ]
        return rules

    # ===================================================================
    # helpers
    # ===================================================================
    @property
    def nl(self) -> Optional[Netlist]:
        return self.state.current

    def _need_design(self) -> Optional[str]:
        if self.state.current is None:
            return "No design is currently loaded."
        return None

    def _commit(self, mutate, *, basis=None, max_fanout=None,
                max_fanout_pi=True, verify=True) -> Tuple[object, bool, str]:
        st = self.state
        st.begin_transform()
        info = mutate(st.current)
        st.current.touch()
        ok, reason = validators.check(st.pre, st.current, basis=basis,
                                      max_fanout=max_fanout,
                                      max_fanout_pi=max_fanout_pi,
                                      verify_equiv=verify)
        if not ok:
            st.rollback()
            return info, False, reason
        return info, True, "ok"

    @staticmethod
    def _names(insts) -> str:
        return ", ".join(insts) if insts else "(none)"

    # ---- answer shape -------------------------------------------------
    # A query and its answer's SHAPE are separate choices: "how many gates are
    # in the cone of X" and "list the gates in the cone of X" select the same
    # set and differ only in what is reported.  Modelling that as two intents
    # made the router pick between near-identical catalog entries, which is the
    # single largest source of misroutes measured against the reference set --
    # so the shape is a parameter and the set is the intent.
    _COUNT_WORDS = ("count", "how many", "number")
    _LIST_WORDS = ("list", "name", "enumerate", "report only", "which gates")

    def _shaped(self, form, names, noun: str, hint: str):
        """Render ``names`` per ``form``, or None if ``form`` says nothing.

        Returning None for an absent/unrecognised form lets the caller keep its
        original sentence verbatim.  That matters: evaluator/golden records the
        pre-parameter wording for these ops, so a request that does not ask for
        a shape must still answer exactly as it did before.
        """
        f = str(form).strip().lower() if form is not None else ""
        if f in ("count", "how_many", "number"):
            return f"{noun}: {len(list(names))} gate(s)."
        if f in ("list", "names", "enumerate"):
            return f"{noun}: " + self._names_or_file(names, hint)
        return None

    def _names_or_file(self, names, hint: str) -> str:
        """Render a name list for an answer.

        Inline while it is short enough, otherwise write the *complete* list to
        a file and point at it.  Q&A A16 requires full enumeration -- a
        truncated list is simply wrong, and one truncated silently is worse,
        because it reads as if it were complete.
        """
        names = list(names)
        if len(names) <= self.LIST_INLINE:
            return self._names(names)
        fname = f"{self.state.case_name or 'case'}_{self._san_name(hint)}.txt"
        path = self._list_to_file(fname, names)
        if path is None:
            return self._names(names)   # inline everything rather than mislead
        return f"the complete list has been written to {path}"

    # ===================================================================
    # IO / basic
    # ===================================================================
    def h_begin(self, m, line):
        mm = (
            re.search(r"case name is\s+(\S+?)[\.\s]*$", line, re.IGNORECASE)
            or re.search(r"testcase identifier is\s+(\S+?)[\.\s]*$", line, re.IGNORECASE)
            or re.search(r"(?:testcase|case)\s+(?:called|named)\s+(\S+?)[\.\s]*$", line, re.IGNORECASE)
            or re.search(r"(?:the )?identifier(?:\s+is|[:\s]+)\s*(\S+?)[\.\s]*$", line, re.IGNORECASE)
            or re.search(r"case\s+(?:id|title|label)[:\s]+\s*(\S+?)[\.\s]*$", line, re.IGNORECASE)
            or re.search(r"test.?case\s+(\S+?)[\.\s]*$", line, re.IGNORECASE)
        )
        name = mm.group(1).rstrip(".") if mm else (self.state.case_name or "case")
        self.state.case_name = name
        return (f'Acknowledged. Initialized testcase "{name}". All subsequent '
                f"responses will be recorded to {name}.log. Design state is "
                "empty and ready for commands.")

    def h_load(self, m, line):
        fm = (
            re.search(r"(?:file|from)\s+['\"]?(\S+\.v)['\"]?", line, re.IGNORECASE)
            or re.search(r"verilog\s+file\s+['\"]?(\S+\.v)['\"]?", line, re.IGNORECASE)
            or re.search(r"['\"]?(\S+\.v)['\"]?", line)
        )
        dm = (
            re.search(r"(?:under|in|inside|located in|from)\s+(?:the\s+)?(?:director(?:y|ies)|folder)\s+['\"]?([^'\"]+?)['\"]?[\.\s]*$", line, re.IGNORECASE)
            or re.search(r"(?:director(?:y|ies)|folder)\s+['\"]?([^'\"]+?)['\"]?[\.\s]*$", line, re.IGNORECASE)
        )
        fname = fm.group(1).strip("'\"") if fm else None
        d = dm.group(1).strip().strip("'\"") if dm else ""
        if fname is None:
            return "Could not determine the design file to load."
        return self.op_load_design(fname, d)

    def _list_to_file(self, basename: str, lines):
        """Write a long result list to a file (next to the design) and return
        its path — per official Q&A A16/A21.3 (large result sets go to a file)."""
        path = self._out_path(basename)
        try:
            with open(path, "w") as fh:
                fh.write("\n".join(lines))
                if lines:
                    fh.write("\n")
            return path
        except Exception:
            return None

    @staticmethod
    def _san_name(s: str) -> str:
        return re.sub(r"[^0-9A-Za-z_]", "_", s)

    def _out_path(self, fname: str) -> str:
        """Resolve an output file path. If the name has no directory component,
        place it in the loaded design's directory (the contest expects outputs
        in the same testcase directory as the input — Q&A A5.3)."""
        if os.path.dirname(fname):
            return fname
        d = self.state.design_dir or "."
        return os.path.join(d, fname)

    def h_write(self, m, line):
        fm = (
            re.search(r"(?:output\s+file|file|into|to|as)\s+['\"]?(\S+\.v)['\"]?", line, re.IGNORECASE)
            or re.search(r"['\"]?(\S+\.v)['\"]?", line)
        )
        fname = fm.group(1).strip("'\"") if fm else f"{self.state.case_name or 'out'}_out.v"
        return self.op_write_design(fname)

    # ===================================================================
    # counts / reports
    # ===================================================================
    def h_count_all(self, m, line):
        if self._need_design():
            return self._need_design()
        return counts.count_breakdown_text(self.state.current)

    def h_total(self, m, line):
        if self._need_design():
            return self._need_design()
        n = counts.total_gate_count(self.state.current)
        return f"The total gate count of the design is {n}."

    _ACTION_WORDS = ("added", "removed", "eliminated", "merged", "collapsed",
                     "inserted", "converted", "found")

    def h_count_or_delta(self, m, line):
        gtype = m.group(1).lower()
        low = line.lower()
        # "...were added / eliminated / merged ..." is a delta question, even
        # though it names a gate type; current-count only for "now/currently".
        if any(w in low for w in self._ACTION_WORDS):
            return self.h_delta(m, line)
        if gtype in ("and", "or", "not", "nand", "nor", "xor", "xnor", "buf", "dff"):
            if self._need_design():
                return self._need_design()
            # "in the (restructured) cone of X" scopes the count to X's fanin
            # cone; the whole-design number answers a different question.
            mm = re.search(r"cone of (?:primary output |output )?(%s)" % NET,
                           line, re.IGNORECASE)
            if mm:
                out = mm.group(1)
                gs = cones.fanin_cone_gates(self.state.current, out)
                n = sum(1 for g in gs if g.type == gtype)
                return (f"There are currently {n} {gtype.upper()} gates in "
                        f"the cone of {out}.")
            n = counts.count_of_type(self.state.current, gtype)
            return f"There are currently {n} {gtype.upper()} gates in the design."
        return self.h_delta(m, line)

    def h_delta(self, m, line):
        """Answer a 'how many X were added/removed/...' question from the delta
        recorded by the immediately-preceding transform."""
        low = line.lower()
        gtype = None
        for t in ("nand", "nor", "not", "xnor", "xor", "and", "or", "buf"):
            if re.search(r"\b%s\b" % t, low):
                gtype = t
                break
        d = self.state.deltas
        val = None
        if "buf" in low or "buffer" in low:
            val = d.get("buffers_added")
        elif "added" in low and gtype:
            val = d.get(f"{gtype}_added")
        elif ("eliminated" in low or "constant propagation" in low):
            val = d.get("const_eliminated")
        elif "merged" in low:
            val = d.get("merged")
        elif "collapsed" in low or "back-to-back" in low or "inverter" in low:
            val = d.get("collapsed")
        elif "removed" in low or "dangling" in low or "redundant" in low or "floating" in low:
            val = d.get("removed")
        elif "found" in low and "enable" in low:
            val = d.get("enable_hold")
        if val is None and d:
            val = d[list(d)[-1]]      # fall back to the most recent delta
        return f"{val if val is not None else 0}"

    def h_cone_type(self, m, line):
        if self._need_design():
            return self._need_design()
        out = m.group(1)
        return counts.cone_type_counts_text(self.state.current, out)

    def h_gate_info(self, m, line):
        if self._need_design():
            return self._need_design()
        name = m.group(1)
        inst = self.state.current.instance_by_name(name)
        if inst is None:
            return f"No gate named {name} exists in the design."
        kind, g = inst
        if kind == "gate":
            return (f"Gate {name} is a {g.type.upper()} gate. "
                    f"Output: {g.out}; inputs: {', '.join(g.ins)}.")
        return (f"{name} is a DFF. Q={g.q}, D={g.d}, CK={g.clk}, "
                f"RN={g.rn}, SN={g.sn}.")

    LIST_INLINE = 200          # list inline up to this many; else write a file

    def h_list_type(self, m, line):
        if self._need_design():
            return self._need_design()
        gtype = m.group(1).lower()
        if gtype == "dff":
            gs = self.state.current.dffs
            items = [f"{g.name} (Q={g.q}, D={g.d})" for g in gs]
        else:
            gs = counts.list_gates_of_type(self.state.current, gtype)
            items = [f"{g.name} (out={g.out}, in={', '.join(g.ins)})" for g in gs]
        if not gs:
            return f"There are 0 {gtype.upper()} gates in the design."
        if len(items) <= self.LIST_INLINE:
            return f"{len(gs)} {gtype.upper()} gate(s):\n" + "\n".join(items)
        # large list -> write to a file and report the path (Q&A A16)
        fname = f"{self.state.case_name or 'case'}_{gtype}_gates.txt"
        path = self._list_to_file(fname, items)
        if path:
            return (f"{len(gs)} {gtype.upper()} gates. The complete list has been "
                    f"written to {path}.")
        return f"{len(gs)} {gtype.upper()} gates:\n" + "\n".join(items[:self.LIST_INLINE])

    def h_list_xor(self, m, line):
        return self.h_list_type(re.match(r"(xor)", "xor"), line)

    def h_list_pi(self, m, line):
        if self._need_design():
            return self._need_design()
        nl = self.state.current
        items = []
        for name in nl.port_order:
            p = nl.ports.get(name)
            if p and p.direction == "input":
                items.append(f"{name} [{p.width}-bit]" if p.is_bus else f"{name} [1-bit]")
        return "Primary inputs:\n" + "\n".join(items)

    def h_list_po(self, m, line):
        if self._need_design():
            return self._need_design()
        nl = self.state.current
        items = []
        for name in nl.port_order:
            p = nl.ports.get(name)
            if p and p.direction == "output":
                items.append(f"{name} [{p.width}-bit]" if p.is_bus else f"{name} [1-bit]")
        return "Primary outputs:\n" + "\n".join(items)

    def h_count_ports(self, m, line):
        if self._need_design():
            return self._need_design()
        nl = self.state.current
        ni = sum(1 for n in nl.port_order if nl.ports.get(n) and nl.ports[n].direction == "input")
        no = sum(1 for n in nl.port_order if nl.ports.get(n) and nl.ports[n].direction == "output")
        bi = len(nl.pi)
        bo = len(nl.po)
        return (f"The design has {bi} primary inputs and {bo} primary outputs "
                f"(across {ni} input port(s) and {no} output port(s)).")

    def h_cone_gate_count(self, m, line):
        if self._need_design():
            return self._need_design()
        out = m.group(m.lastindex)
        n = counts.gates_in_fanin_cone(self.state.current, out)
        return f"The fan-in cone of {out} contains {n} gates."

    def h_const1_gates(self, m, line):
        if self._need_design():
            return self._need_design()
        nl = self.state.current
        names = [g.name for g in nl.gates if "1'b1" in g.ins]
        # A flip-flop is a primitive gate too (Q&A A2) and its .SN / .RN are
        # ordinary inputs (A12), which is where a tied constant actually shows
        # up in these benchmarks: test37 has 2324 such flip-flops and not one
        # combinational gate tied to 1'b1.
        names += [ff.name for ff in nl.dffs
                  if "1'b1" in (ff.d, ff.clk, ff.rn, ff.sn)]
        return (f"{len(names)} gate(s) have an input tied to 1'b1: " +
                self._names_or_file(names, "gates_tied_1b1"))

    _GATE_SYNONYMS = {
        "sheffer stroke": "nand", "pierce arrow": "nor",
        "inverter": "not", "inv": "not",
    }

    def h_report_const(self, m, line):
        if self._need_design():
            return self._need_design()
        gtype = m.group(1).lower()
        gtype = self._GATE_SYNONYMS.get(gtype, gtype)
        val = "1'b0" if "0" in line.split(gtype)[-1][:40] else None
        if "constant 1" in line.lower():
            val = "1'b1"
        elif "constant 0" in line.lower() or "constant-0" in line.lower():
            val = "1'b0"
        # "constant" = structural literal OR functionally constant (Q&A A21.1)
        self.const_nets = functional.constant_nets(self.state.current)
        gs = constprop.gates_with_const_input(
            self.state.current,
            gtype if gtype in ("and", "or", "nand", "nor") else None,
            val, extra_const=self.const_nets)
        self.state.last_report = [g.name for g in gs]
        self.state.last_report_kind = gtype
        if not gs:
            return f"No {gtype.upper()} gates with constant inputs were found."
        return (f"Found {len(gs)} {gtype.upper()} gate(s) with constant inputs: " +
                self._names_or_file([g.name for g in gs], f"{gtype}_const_inputs"))

    def h_report_const_synonym(self, m, line):
        """Handle patterns with non-standard gate synonyms (Sheffer stroke→NAND, etc.)."""
        import re
        gtype = m.group(1).lower().strip()
        canonical = self._GATE_SYNONYMS.get(gtype, gtype)
        fake_m = re.match(f"({canonical})", canonical, re.IGNORECASE)
        return self.h_report_const(fake_m, line)

    # ===================================================================
    # paths
    # ===================================================================
    def _path_avoid(self, a, b, avoid):
        if self._need_design():
            return self._need_design()
        ok = paths.path_exists(self.state.current, a, b, avoid={avoid})
        if ok:
            return (f"Yes. A combinational path from {a} to {b} that avoids "
                    f"{avoid} exists.")
        return (f"No. There is no combinational path from {a} to {b} that "
                f"avoids {avoid}.")

    def h_path_avoid(self, m, line):
        return self._path_avoid(m.group(1), m.group(2), m.group(3))

    def h_path_avoid2(self, m, line):
        return self._path_avoid(m.group(1), m.group(2), m.group(3))

    def h_path_avoid3(self, m, line):
        return self._path_avoid(m.group(1), m.group(2), m.group(3))

    def h_path_plain(self, m, line):
        if self._need_design():
            return self._need_design()
        a, b = m.group(m.lastindex - 1), m.group(m.lastindex)
        ok = paths.path_exists(self.state.current, a, b)
        if ok:
            return f"Yes. A combinational path from {a} to {b} exists."
        return f"No. There is no combinational path from {a} to {b}."

    PATH_INLINE = 50           # list inline up to this many paths
    PATH_FILE_CAP = 2_000_000  # above this, listing literally is infeasible

    def h_enum_paths(self, m, line):
        if self._need_design():
            return self._need_design()
        nl = self.state.current
        toks = re.findall(NET, line)
        cand = [t for t in toks if re.match(r"n\d", t)]
        if len(cand) < 2:
            return "Could not identify the two endpoints for path enumeration."
        a, b = cand[-2], cand[-1]
        count = paths.count_paths(nl, a, b)
        if count == 0:
            return f"There are no combinational paths from {a} to {b}."
        if count <= self.PATH_INLINE:
            _, plist = paths.enumerate_paths(nl, a, b, limit=self.PATH_INLINE)
            lines = [f"There are {count} path(s) from {a} to {b}:"]
            for p in plist:
                lines.append("  " + a + " -> " + " -> ".join(p) + " -> " + b
                             if p else "  " + a + " -> " + b)
            return "\n".join(lines)
        if count <= self.PATH_FILE_CAP:
            fname = f"{self.state.case_name or 'case'}_paths_{self._san_name(a)}_to_{self._san_name(b)}.txt"
            path = self._out_path(fname)
            try:
                with open(path, "w") as fh:
                    paths.stream_paths(nl, a, b, fh, count + 1)
                return (f"There are {count} combinational paths from {a} to {b}. "
                        f"The complete enumeration has been written to {path}.")
            except Exception as exc:
                return (f"There are {count} combinational paths from {a} to {b} "
                        f"(could not write the list file: {exc}).")
        return (f"There are {count} combinational paths from {a} to {b} — too "
                f"many to enumerate literally.")

    def h_len0(self, m, line):
        if self._need_design():
            return self._need_design()
        z = paths.length_zero_paths(self.state.current)
        if not z:
            return "No length-0 paths exist (no primary input directly drives a primary output)."
        return f"{len(z)} length-0 path(s): " + self._names(z)

    def h_dominator(self, m, line):
        if self._need_design():
            return self._need_design()
        a, b, g = m.group(1), m.group(2), m.group(3)
        res = paths.every_path_passes_through(self.state.current, a, b, g)
        if res is None:
            return f"No combinational path from {a} to {b} exists."
        return ("Yes." if res else "No.") + \
               f" {'Every' if res else 'Not every'} path from {a} to {b} passes through {g}."

    def h_articulation(self, m, line):
        if self._need_design():
            return self._need_design()
        a, b = m.group(1), m.group(2)
        pts = paths.articulation_points(self.state.current, a, b)
        if not pts:
            return f"There are no articulation points between {a} and {b}."
        return (f"{len(pts)} articulation point(s) between {a} and {b}: " +
                self._names_or_file(pts, f"articulation_{a}_{b}"))

    def h_cut(self, m, line):
        if self._need_design():
            return self._need_design()
        # group(1) is the optional "is |whether " prefix; the wire name is the
        # last capture group (taking group 1 printed "Wire whether ...").
        w = m.group(m.lastindex)
        res = paths.is_cut_pi_po(self.state.current, w)
        return ("Yes." if res else "No.") + \
               f" Wire {w} {'is' if res else 'is not'} a cut between a primary input and a primary output."

    # ===================================================================
    # depth
    # ===================================================================
    def _depth_ab(self, a, b):
        if self._need_design():
            return self._need_design()
        d = depth.max_depth_from_to(self.state.current, a, b)
        if d is None:
            return f"There is no combinational path from {a} to {b} (depth 0)."
        return f"The maximum combinational logic depth from {a} to {b} is {d}."

    def h_depth_ab(self, m, line):
        return self._depth_ab(m.group(2), m.group(3))

    def h_depth_ab2(self, m, line):
        return self._depth_ab(m.group(1), m.group(2))

    def h_depth_ab3(self, m, line):
        return self._depth_ab(m.group(1), m.group(2))

    def h_cone_depth(self, m, line):
        if self._need_design():
            return self._need_design()
        out = m.group(2) or m.group(3)
        d = depth.depth_of_cone(self.state.current, out)
        return f"The maximum logic depth of the cone of {out} is {d}."

    def h_reg2reg_depth(self, m, line):
        if self._need_design():
            return self._need_design()
        d = depth.reg_to_reg_max_depth(self.state.current)
        if d < 0:
            return "There are no register-to-register combinational paths."
        return f"The maximum combinational depth on any register-to-register path is {d}."

    def h_pi2d_depth(self, m, line):
        if self._need_design():
            return self._need_design()
        d = depth.pi_to_dff_d_max_depth(self.state.current)
        return f"The maximum logic depth from any primary input to any DFF D-pin is {max(d,0)}."

    def h_pi2po_depth(self, m, line):
        if self._need_design():
            return self._need_design()
        d = depth.pi_to_po_max_depth(self.state.current)
        if d < 0:
            return ("The maximum combinational logic depth from any primary input "
                    "to any primary output is 0 (no primary output is reachable "
                    "from a primary input by a purely combinational path).")
        return ("The maximum combinational logic depth from any primary input to "
                f"any primary output is {d}.")

    def h_global_depth(self, m, line):
        if self._need_design():
            return self._need_design()
        d = depth.global_max_depth(self.state.current)
        return f"The maximum combinational logic depth in the design is {d}."

    def h_depth_gt(self, m, line):
        if self._need_design():
            return self._need_design()
        k = int(m.group(1))
        outs = depth.outputs_depth_greater_than(self.state.current, k)
        return f"{len(outs)} output(s) have a logic depth greater than {k}."

    def h_deepest(self, m, line):
        if self._need_design():
            return self._need_design()
        word = (m.group(2) or "").lower() if m.lastindex and m.lastindex >= 2 else ""
        if "deep" in word or "deep" in line.lower():
            name, n = depth.deepest_output(self.state.current)
            return f"Output {name} has the deepest fan-in cone (depth {n})."
        name, n = cones.largest_fanin_output(self.state.current)
        return f"Output {name} has the largest fan-in cone ({n} gates)."

    def h_on_maxpath(self, m, line):
        if self._need_design():
            return self._need_design()
        # group(1) is the "does|whether" lead-in; the gate name is group 2
        # (taking group 1 answered "No gate named whether exists").
        g = m.group(2)
        res = depth.gate_on_max_depth_path(self.state.current, g)
        if res is None:
            return f"No gate named {g} exists."
        return ("Yes." if res else "No.") + f" Gate {g} {'lies' if res else 'does not lie'} on a maximum-depth path."

    # ===================================================================
    # connectivity
    # ===================================================================
    def h_fanout(self, m, line):
        if self._need_design():
            return self._need_design()
        net = m.group(1)
        return self._fanout_answer(net)

    def _fanout_answer(self, net: str) -> str:
        """'List every gate' must be complete: inline up to LIST_INLINE loads,
        else write the full list to a file and answer with its path (A16)."""
        f = connectivity.fanout_count(self.state.current, net)
        loads = connectivity.fanout_load_instances(self.state.current, net)
        if len(loads) > self.LIST_INLINE:
            fname = f"{self.state.case_name or 'case'}_{self._san_name(net)}_fanout.txt"
            path = self._list_to_file(fname, loads)
            if path:
                return (f"The fanout of {net} is {f}. The complete list of "
                        f"driven gates has been written to {path}.")
        return (f"The fanout of {net} is {f}. It directly drives: " +
                self._names(loads))

    def h_driven_by(self, m, line):
        if self._need_design():
            return self._need_design()
        g = m.group(2)
        ds = connectivity.gates_driven_by_gate(self.state.current, g)
        if ds is None:
            return f"No gate named {g} exists."
        return (f"Gate {g} drives {len(ds)} gate(s): "
                + self._names_or_file(ds, f"driven_by_{g}"))

    def h_successors(self, m, line):
        if self._need_design():
            return self._need_design()
        g = m.group(1)
        ds = connectivity.immediate_successors(self.state.current, g)
        if ds is None:
            return f"No instance named {g} exists."
        return ("Immediate successors of %s: " % g
                + self._names_or_file(ds, f"successors_{g}"))

    def h_tfanin(self, m, line):
        if self._need_design():
            return self._need_design()
        net = m.group(1)
        gs = cones.fanin_cone_gates(self.state.current, net)
        return f"The transitive fan-in cone of {net} contains {len(gs)} gates."

    def h_tfanout(self, m, line):
        if self._need_design():
            return self._need_design()
        net = m.group(2)
        gs = cones.fanout_cone_gates(self.state.current, net)
        return f"The transitive fan-out cone of {net} contains {len(gs)} gates."

    def h_reachable(self, m, line):
        if self._need_design():
            return self._need_design()
        net = m.group(2)
        gs = connectivity.reachable_gates_from(self.state.current, net)
        return (f"{len(gs)} gate(s) are reachable from {net}: "
                + self._names_or_file(gs, f"reachable_{net}"))

    def _bare_names(self, names, hint: str) -> str:
        """Answer a "report only ... names" request with just the names.

        These are graded as an exact identifier set, so any surrounding prose
        contributes stray words to the comparison.
        """
        if not names:
            return "(none)"
        return self._names_or_file(names, hint)

    def h_hops_downstream(self, m, line):
        """"... all gates within N hops downstream (in the fanout) of X"."""
        if self._need_design():
            return self._need_design()
        k, net = int(m.group(1)), m.group(2)
        gs = connectivity.gates_within_hops(self.state.current, net, k)
        return self._bare_names(gs, f"within_{k}_hops_of_{net}")

    def h_hops_drives(self, m, line):
        """"... all gates that <net> drives within N hops"."""
        if self._need_design():
            return self._need_design()
        net, k = m.group(1), int(m.group(2))
        gs = connectivity.gates_within_hops(self.state.current, net, k)
        return self._bare_names(gs, f"{net}_drives_{k}_hops")

    def h_fanin_cone_names(self, m, line):
        """"List the instance names of all gates in the fanin cone of X"."""
        if self._need_design():
            return self._need_design()
        net = m.group(1)
        gs = [g.name for g in cones.fanin_cone_gates(self.state.current, net)]
        return self._bare_names(gs, f"fanin_cone_{net}")

    def h_highest_fanout(self, m, line):
        if self._need_design():
            return self._need_design()
        # "which primary input ..." is PI-scoped; "which signal ..." / "over all
        # nets" ranges over every driven net, where the winner is usually an
        # internal one.  Answering the second with the PI-only maximum silently
        # reports a smaller number for a different net.
        if "primary input" in line.lower():
            name, f = connectivity.highest_fanout_pi(self.state.current)
            return f"Primary input {name} has the highest fanout ({f})."
        name, f = connectivity.highest_fanout_net(self.state.current)
        drv = self.state.current.driver(name)
        by = f", driven by {drv[1].name}" if drv and drv[0] == "gate" else ""
        return f"Signal {name} drives the largest number of loads ({f}){by}."

    def h_max_fanout(self, m, line):
        if self._need_design():
            return self._need_design()
        net = m.group(2)
        f = connectivity.max_fanout_of(self.state.current, net)
        return f"The maximum fanout of {net} is now {f}."

    def h_shared(self, m, line):
        if self._need_design():
            return self._need_design()
        a, b = m.group(1), m.group(2)
        gs = cones.shared_fanin_gates(self.state.current, a, b)
        return (f"{len(gs)} gate(s) are shared between the fan-in cones of "
                f"{a} and {b}: " + self._names_or_file([g.name for g in gs], f"shared_{a}_{b}"))

    def h_connected_out(self, m, line):
        if self._need_design():
            return self._need_design()
        g = m.group(2)
        ds = connectivity.gates_driven_by_gate(self.state.current, g)
        if ds is None:
            return f"No gate named {g} exists."
        return (f"Gates connected to the output of {g}: "
                + self._names_or_file(ds, f"connected_out_{g}"))

    # ===================================================================
    # functional
    # ===================================================================
    def _sig_equiv(self, a, b):
        if self._need_design():
            return self._need_design()
        r = functional.signals_equivalent(self.state.current, a, b)
        if r is None:
            return f"Could not determine the equivalence of {a} and {b}."
        return ("Yes." if r else "No.") + f" Signals {a} and {b} are {'' if r else 'not '}functionally equivalent."

    def h_sig_equiv(self, m, line):
        toks = [t for t in re.findall(NET, line) if re.match(r"n\d", t)]
        if len(toks) < 2:
            return "Could not identify the two signals."
        return self._sig_equiv(toks[-2], toks[-1])

    def h_sig_equiv2(self, m, line):
        return self._sig_equiv(m.group(1), m.group(2))

    def h_const_out(self, m, line):
        if self._need_design():
            return self._need_design()
        out = m.group(1)
        v = functional.output_always_constant(self.state.current, out)
        if v == 0:
            return f"Yes. Output {out} is always 0 regardless of the inputs."
        if v == 1:
            return f"No. Output {out} is constant but always 1, not 0."
        return f"No. Output {out} is not constant; it depends on the inputs."

    def h_depends(self, m, line):
        if self._need_design():
            return self._need_design()
        out, inp = m.group(1), m.group(2)
        r = functional.depends_on(self.state.current, out, inp)
        return ("Yes." if r else "No.") + f" Output {out} {'depends' if r else 'does not depend'} on input {inp}."

    def h_boolean(self, m, line):
        if self._need_design():
            return self._need_design()
        toks = [t for t in re.findall(NET, line) if re.match(r"n\d", t)]
        out = toks[-1] if toks else None
        if out is None:
            return "Could not identify the output."
        eq = functional.boolean_equation(self.state.current, out)
        if eq is None:
            return (f"The Boolean equation for {out} has too large a support to "
                    "express compactly in terms of the primary inputs.")
        return f"{out} = {eq}"

    def h_symmetric(self, m, line):
        if self._need_design():
            return self._need_design()
        toks = [t for t in re.findall(NET, line) if re.match(r"n\d", t)]
        a, b = m.group(1), m.group(2)
        # the function net is the first net mentioned
        fn = toks[0] if toks else None
        r = functional.is_symmetric(self.state.current, fn, a, b)
        if r is None:
            return f"Could not determine symmetry of {fn} in {a}, {b}."
        return ("Yes." if r else "No.") + f" The function at {fn} is {'' if r else 'not '}symmetric in {a} and {b}."

    def h_nand_pair(self, m, line):
        if self._need_design():
            return self._need_design()
        toks = [t for t in re.findall(NET, line) if re.match(r"n\d", t)]
        target = toks[-1] if toks else None
        r = functional.exists_nand_pair(self.state.current, target)
        if r is None:
            return f"No pair of internal signals (a, b) with NAND(a, b) equivalent to {target} was found."
        return f"Yes. NAND({r[0]}, {r[1]}) is functionally equivalent to {target}."

    def h_nand_pair2(self, m, line):
        return self.h_nand_pair(m, line)

    # ===================================================================
    # sequential
    # ===================================================================
    def h_ffs_clock(self, m, line):
        if self._need_design():
            return self._need_design()
        clk = m.group(1)
        ff = sequential.ffs_on_clock(self.state.current, clk)
        return (f"{len(ff)} flip-flop(s) are driven by clock {clk}: " +
                self._names_or_file([f.name for f in ff], f"ffs_clock_{clk}"))

    def h_same_clock(self, m, line):
        if self._need_design():
            return self._need_design()
        toks = re.findall(r"(?:dff|g)\w+", line, re.IGNORECASE)
        if len(toks) < 2:
            return "Could not identify the two flip-flops."
        r = sequential.same_clock_domain(self.state.current, toks[0], toks[1])
        if r is None:
            return "One or both flip-flops were not found."
        return ("Yes." if r else "No.") + " They are in the same clock domain." if r else \
               "No. They are in different clock domains."

    def h_reg2reg_paths(self, m, line):
        if self._need_design():
            return self._need_design()
        count, pairs = sequential.reg_to_reg_pairs(self.state.current)
        head = f"There are {count} register-to-register connections through combinational logic."
        if pairs:
            head += " Examples: " + self._names([f"{a}->{b}" for a, b in pairs[:30]])
        return head

    def h_enable_hold(self, m, line):
        if self._need_design():
            return self._need_design()
        eh = sequential.enable_hold_ffs(self.state.current)
        self.state.record_delta("enable_hold", len(eh))
        return (f"{len(eh)} flip-flop(s) implement an enable/hold structure in "
                f"their D-input logic (next state depends on the register's own "
                f"current state). Examples: " +
                self._names([f.name for f in eh[:30]]))

    def h_enable_hold_count(self, m, line):
        if self._need_design():
            return self._need_design()
        n = self.state.get_delta("enable_hold")
        if not n:
            eh = sequential.enable_hold_ffs(self.state.current)
            n = len(eh)
        return f"{n} flip-flops were found to have enable or hold structures in their D-input logic."

    # ===================================================================
    # transforms
    # ===================================================================
    def _scope_gates(self, line):
        """If the request scopes to a cone of X, return that gate-name set."""
        mm = re.search(r"cone of (?:output )?(%s)" % NET, line, re.IGNORECASE)
        if mm:
            out = mm.group(1)
            return {g.name for g in cones.fanin_cone_gates(self.state.current, out)}
        return None

    def h_basis(self, m, line):
        if self._need_design():
            return self._need_design()
        basis = _basis_from_text(line)
        if basis is None:
            return "Could not determine the target gate basis."
        scope = self._scope_gates(line)
        before = self.state.current.type_counts()
        # scoped (cone) conversion: only equivalence is required, not whole-design
        # basis purity; full-design conversion enforces basis purity.
        info, ok, reason = self._commit(
            lambda nl: rewrite.to_basis(nl, basis, scope),
            basis=(None if scope is not None else basis))
        if not ok:
            return f"The basis remap was reverted: {reason}."
        self.state.record_delta("basis_remap", info)
        b = basis.replace("_", "+")
        return (f"Remapped {'the cone' if scope is not None else 'the entire design'} to "
                f"{b} gates ({info} gates rewritten); functional equivalence verified.")

    def h_xnor_nor(self, m, line):
        if self._need_design():
            return self._need_design()
        scope = self._scope_gates(line)
        before = counts.count_of_type(self.state.current, "nor")
        info, ok, reason = self._commit(lambda nl: rewrite.xnor_to_nor(nl, scope))
        if not ok:
            return f"The XNOR->NOR conversion was reverted: {reason}."
        added = counts.count_of_type(self.state.current, "nor") - before
        self.state.record_delta("nor_added", added)
        self.state.record_delta("xnor_converted", info)
        return (f"Converted {info} XNOR gate(s) to NOR-only logic "
                f"({added} NOR gates added); functional equivalence verified.")

    def h_xnor_nand(self, m, line):
        if self._need_design():
            return self._need_design()
        scope = self._scope_gates(line)
        before = counts.count_of_type(self.state.current, "nand")
        info, ok, reason = self._commit(lambda nl: rewrite.xnor_to_nand(nl, scope))
        if not ok:
            return f"The XNOR->NAND conversion was reverted: {reason}."
        added = counts.count_of_type(self.state.current, "nand") - before
        self.state.record_delta("nand_added", added)
        self.state.record_delta("xnor_converted", info)
        return (f"Converted {info} XNOR gate(s) to NAND-based logic "
                f"({added} NAND gates added); functional equivalence verified.")

    def h_xor_nand(self, m, line):
        if self._need_design():
            return self._need_design()
        scope = self._scope_gates(line)
        before = counts.count_of_type(self.state.current, "nand")
        info, ok, reason = self._commit(lambda nl: rewrite.xor_to_nand(nl, scope))
        if not ok:
            return f"The XOR->NAND conversion was reverted: {reason}."
        added = counts.count_of_type(self.state.current, "nand") - before
        self.state.record_delta("nand_added", added)
        self.state.record_delta("xor_converted", info)
        return (f"Converted {info} XOR gate(s) to 4-NAND logic "
                f"({added} NAND gates added); functional equivalence verified.")

    def h_xor_aoi(self, m, line):
        if self._need_design():
            return self._need_design()
        scope = self._scope_gates(line)
        info, ok, reason = self._commit(lambda nl: rewrite.xor_to_aoi(nl, scope))
        if not ok:
            return f"The XOR decomposition was reverted: {reason}."
        self.state.record_delta("xor_converted", info)
        return f"Decomposed {info} XOR gate(s) into AND/OR/NOT logic; equivalence verified."

    def h_nand_inv(self, m, line):
        if self._need_design():
            return self._need_design()
        info, ok, reason = self._commit(lambda nl: rewrite.nand_const1_to_inv(nl))
        if not ok:
            return f"The NAND(const-1)->INV conversion was reverted: {reason}."
        self.state.record_delta("nand_to_inv", info)
        return f"Replaced {info} NAND gate(s) tied to constant 1 with inverters; equivalence verified."

    def h_constprop(self, m, line):
        if self._need_design():
            return self._need_design()
        rtype = None
        mm = re.search(r"reported (\w+) gates", line, re.IGNORECASE)
        if mm:
            rtype = mm.group(1).lower()
        elif self.state.last_report_kind:
            rtype = self.state.last_report_kind
        # reuse the functional-constant set from the preceding "report" (A21.1);
        # if absent (simplify without a prior report), detect now
        extra = self.const_nets if self.const_nets else functional.constant_nets(self.state.current)
        info, ok, reason = self._commit(
            lambda nl: constprop.const_propagate(nl, rtype, extra_const=extra))
        if not ok:
            return f"The constant propagation was reverted: {reason}."
        self.state.record_delta("const_eliminated", info)
        tlabel = (rtype.upper() + " ") if rtype else ""
        return f"Constant propagation eliminated {info} {tlabel}gate(s); equivalence verified."

    def h_collapse(self, m, line):
        if self._need_design():
            return self._need_design()
        info, ok, reason = self._commit(lambda nl: cleanup.collapse_double_inverters(nl))
        if not ok:
            return f"The inverter collapse was reverted: {reason}."
        self.state.record_delta("collapsed", info)
        return f"Collapsed {info} back-to-back inverter pair(s) into direct wires; equivalence verified."

    def h_dangling(self, m, line):
        if self._need_design():
            return self._need_design()
        info, ok, reason = self._commit(lambda nl: cleanup.remove_dangling(nl))
        if not ok:
            return f"The dangling-gate removal was reverted: {reason}."
        self.state.record_delta("removed", info)
        return f"Removed {info} dangling gate(s) that do not affect any primary output; equivalence verified."

    def h_check_dangling(self, m, line):
        """Pure query: report dangling gates without touching the design.
        Only when the request itself asks for removal ("Remove them if
        found") delegate to the removal transform."""
        if self._need_design():
            return self._need_design()
        if re.search(r"remove|delete|excise|prune|eliminate", line, re.I):
            return self.h_dangling(m, line)
        dead_g, dead_ff = cleanup.find_dangling(self.state.current)
        total = len(dead_g) + len(dead_ff)
        if not total:
            return ("No dangling gates were found; every gate contributes to a "
                    "primary output.")
        return (f"Found {total} dangling instance(s) that do not affect any "
                "primary output: " + self._names_or_file(dead_g + dead_ff, "dangling"))

    def h_floating_count(self, m, line):
        """"How many floating signals were found?" wants the number.

        Re-running the check and repeating its yes/no sentence answers a
        different question and leaves the count unstated.  Uses the figure
        recorded by the preceding check, or computes it if asked cold.
        """
        if self._need_design():
            return self._need_design()
        n = self.state.deltas.get("floating")
        if n is None:
            nl = self.state.current
            n = (len([b for b in nl.pi if not nl.loads(b) and b not in nl.po])
                 + len([b for b in nl.po if nl.driver(b)[0] == "undriven"]))
        return f"{n} floating signal(s) were found."

    def h_check_floating_ports(self, m, line):
        """Pure query: undriven input bits and unconnected output bits.
        Port-level question — distinct from dangling *gates*."""
        if self._need_design():
            return self._need_design()
        nl = self.state.current
        und_in = [b for b in sorted(nl.pi) if not nl.loads(b) and b not in nl.po]
        unc_out = [b for b in sorted(nl.po) if nl.driver(b)[0] == "undriven"]
        self.state.record_delta("floating", len(und_in) + len(unc_out))
        if not und_in and not unc_out:
            return ("No. There are no floating inputs or unconnected output "
                    "ports in this design.")
        parts = ["Yes."]
        if und_in:
            parts.append(f"{len(und_in)} floating input bit(s): "
                         + self._names_or_file(und_in, "floating_inputs"))
        if unc_out:
            parts.append(f"{len(unc_out)} unconnected output bit(s): "
                         + self._names_or_file(unc_out, "unconnected_outputs"))
        return " ".join(parts)

    def h_merge(self, m, line):
        if self._need_design():
            return self._need_design()
        info, ok, reason = self._commit(lambda nl: cleanup.merge_structural_duplicates(nl))
        if not ok:
            return f"The duplicate merge was reverted: {reason}."
        self.state.record_delta("merged", info)
        return f"Merged {info} structurally-duplicate gate(s); equivalence verified."

    def h_rename(self, m, line):
        if self._need_design():
            return self._need_design()
        kind = m.group(2).lower()
        old, new = m.group(3), m.group(4)
        if kind == "gate":
            ok = naming.rename_gate(self.state.current, old, new)
        else:
            ok = naming.rename_net(self.state.current, old, new)
        self.state.current.touch()
        if not ok:
            return f"No {kind} named {old} was found to rename."
        return f"Renamed {kind} {old} to {new} and updated all references."

    def h_rename2(self, m, line):
        if self._need_design():
            return self._need_design()
        old, new = m.group(2), m.group(3)
        kind = "gate" if old.startswith("g") and self.state.current.gate_by_name(old) else "signal"
        if kind == "gate":
            ok = naming.rename_gate(self.state.current, old, new)
        else:
            ok = naming.rename_net(self.state.current, old, new)
        self.state.current.touch()
        if not ok:
            return f"No object named {old} was found to rename."
        return f"Renamed {kind} {old} to {new} and updated all references."

    def h_connected_renamed(self, m, line):
        if self._need_design():
            return self._need_design()
        net = m.group(1)
        gs = naming.gates_connected_to_net(self.state.current, net)
        return (f"Gates connected to {net}: "
                + self._names_or_file(gs, f"connected_{net}"))

    def h_buffers_fanout(self, m, line):
        if self._need_design():
            return self._need_design()
        k = int(re.findall(r"(\d+)", line)[-1])
        # "no gate ..." bounds gate outputs and DFF.Q (a flip-flop is a gate,
        # Q&A A2).  "no signal/net ..." is broader and adds primary inputs.
        low = line.lower()
        include_pi = "signal" in low or "net" in low
        info, ok, reason = self._commit(
            lambda nl: buffering.limit_fanout(nl, k, include_pi=include_pi),
            max_fanout=k, max_fanout_pi=include_pi)
        if not ok:
            return f"The buffer insertion was reverted: {reason}."
        self.state.record_delta("buffers_added", info)
        return (f"Inserted {info} buffer(s) so that no driver exceeds {k} loads; "
                f"max-fanout bound and equivalence verified.")

    def h_buffers_signal(self, m, line):
        if self._need_design():
            return self._need_design()
        net = m.group(1)
        k = int(m.group(2))
        info, ok, reason = self._commit(
            lambda nl: buffering.limit_fanout(nl, k, only_nets={net}), max_fanout=None)
        if not ok:
            return f"The buffer insertion was reverted: {reason}."
        self.state.record_delta("buffers_added", info)
        return f"Inserted {info} buffer(s) on {net} so each driver has at most {k} loads; equivalence verified."

    def h_buffers_dedicated(self, m, line):
        if self._need_design():
            return self._need_design()
        net = m.group(1)
        info, ok, reason = self._commit(
            lambda nl: buffering.dedicated_buffer_per_load(nl, net))
        if not ok:
            return f"The buffer insertion was reverted: {reason}."
        self.state.record_delta("buffers_added", info)
        return f"Inserted {info} dedicated buffer(s), one per load of {net}; equivalence verified."

    def h_buffers_reset(self, m, line):
        if self._need_design():
            return self._need_design()
        net = m.group(1)
        km = re.findall(r"(\d+)", line)
        k = int(km[-1]) if km else 4
        info, ok, reason = self._commit(
            lambda nl: buffering.limit_fanout(nl, k, only_nets={net}))
        if not ok:
            return f"The buffer insertion was reverted: {reason}."
        self.state.record_delta("buffers_added", info)
        return f"Inserted {info} buffer(s) on reset signal {net} (max {k} loads per driver); equivalence verified."

    # ===================================================================
    # optimize
    # ===================================================================
    def h_opt_cone(self, m, line):
        if self._need_design():
            return self._need_design()
        out = m.group(m.lastindex)
        basis = _basis_from_text(line)
        before = depth.depth_of_cone(self.state.current, out)
        res, imp = abc_opt.optimize_cone_depth(self.state.current, out, basis=basis)
        self.state.current = res
        after = depth.depth_of_cone(res, out)
        if imp:
            return (f"Optimized the cone of {out}: depth reduced from {before} "
                    f"to {after}{' (basis preserved)' if basis else ''}; equivalence verified.")
        return (f"The cone of {out} is already optimal at depth {before}; "
                "reported the original (equivalence preserved).")

    def h_opt_depth(self, m, line):
        if self._need_design():
            return self._need_design()
        basis = _basis_from_text(line)
        before = depth.global_max_depth(self.state.current)
        res, imp = abc_opt.minimize_depth(self.state.current, basis=basis)
        self.state.current = res
        after = depth.global_max_depth(res)
        if imp:
            return (f"Reduced the maximum logic depth from {before} to {after}"
                    f"{' (basis preserved)' if basis else ''}; equivalence verified.")
        return (f"The design is already optimal at depth {before}; reported the "
                "original (equivalence preserved).")

    def h_opt_area(self, m, line):
        if self._need_design():
            return self._need_design()
        basis = _basis_from_text(line)
        before = len(self.state.current.gates)
        res, imp = abc_opt.minimize_area(self.state.current, basis=basis)
        self.state.current = res
        if imp:
            self.state.provably_equiv = getattr(res, '_provably_equiv', False)
        after = len(res.gates)
        if imp:
            return (f"Reduced the total gate count from {before} to {after}"
                    f"{' (basis preserved)' if basis else ''}; equivalence verified.")
        return (f"The design is already optimal at {before} gates; reported the "
                "original (equivalence preserved).")

    # ===================================================================
    # equivalence
    # ===================================================================
    def h_verify(self, m, line):
        if self._need_design():
            return self._need_design()
        t = line.lower()
        if "pre-transformation" in t or "pre transformation" in t:
            ref = self.state.pre or self.state.original
            label = "pre-transformation netlist"
        elif "last loaded" in t or "as last loaded" in t or "loaded from disk" in t:
            ref = self.state.last_loaded
            label = "netlist as last loaded from disk"
        else:
            ref = self.state.original
            label = "original loaded netlist"
        if ref is None:
            return "No reference design is available for comparison."
        r = equiv_gate.equivalent(ref, self.state.current)
        if r is True:
            return f"Verified: the current design is functionally equivalent to the {label}."
        if r is False:
            return f"The current design is NOT functionally equivalent to the {label}."
        return f"Could not conclusively verify equivalence to the {label}."

    # ===================================================================
    # intent dispatch (LLM fallback path)
    # ===================================================================
    def _dispatch_intent(self, intent, params, line):
        """Dispatch a structured LLM intent directly to op_* methods.

        The original natural-language line has already failed the regex router
        in handle(), so do not run the same rule table on that original line
        again.  The LLM fallback returns an intent plus extracted parameters;
        this method sends those parameters to the corresponding op_* function.
        """
        if intent in (None, "noop"):
            return self._default_ack(line)

        params = params or {}

        # Regex guards against known LLM misclassifications.  These read the
        # raw request text, so they are gated on use_rules: --no-rules must
        # measure the LLM's own routing, with no text-based patch rescuing it.
        if self.use_rules:
            # A basis-remap request ("rebuild the netlist using only AND and
            # NOT primitives") carries a basis and "only" phrasing but no
            # cost-function sentence — it is a conversion, not an
            # optimization, whatever the model chose.
            if intent in ("minimize_area", "minimize_depth", "optimize_cone"):
                basis = _basis_from_text(line)
                if (basis and re.search(r"\b(only|use only|using only)\b", line, re.I)
                        and not re.search(r"cost function|smaller is better", line, re.I)):
                    return self.op_convert_basis(basis=basis,
                                                 scope=params.get("scope") or params.get("output"))

            # "maximum depth from any primary input to any primary output"
            # WITHOUT the word "combinational": per Q&A A21.2 the graded value
            # is the design-wide maximum (DFF Q-pins count as PIs, D-pins as
            # POs).  Answer with that number but say what it actually measures
            # — labelling it the PI->PO combinational depth would be wrong
            # whenever the critical segment is register-bounded.
            if (intent in ("pi_to_po_depth", "global_max_depth")
                    and re.search(r"primary inputs?\b.*\bprimary outputs?", line, re.I)
                    and "combinational" not in line.lower()):
                if self._need_design():
                    return self._need_design()
                d = depth.global_max_depth(self.state.current)
                return ("The maximum logic depth from any primary input to any "
                        f"primary output is {d} (per the contest Q&A, DFF outputs "
                        "count as primary inputs and DFF D-pins as primary outputs, "
                        "so this design-wide maximum includes register-bounded "
                        "paths).")

            # A conversion that names the gate type to replace is targeted,
            # never a whole-design remap — remapping every gate answers a
            # different request and poisons every later response in the case.
            if intent == "convert_basis":
                scope = params.get("scope")
                if re.search(r"\bXNOR\b", line, re.I) and re.search(r"\bNOR\b", line, re.I):
                    return self.op_xnor_to_nor(scope=scope)
                if re.search(r"\bXOR\b", line, re.I) and re.search(r"\bNAND\b", line, re.I):
                    return self.op_xor_to_nand(scope=scope)
                if re.search(r"\bXOR\b", line, re.I) and re.search(r"\bAND\b.*\bOR\b.*\bNOT\b", line, re.I):
                    return self.op_xor_to_aoi(scope=scope)

        handler = getattr(self, f"op_{intent}", None)
        if handler is None:
            return (f'Acknowledged. The request was mapped to intent "{intent}", '
                    "but that operation is not implemented; the current design "
                    "is unchanged.")
        return handler(**params)

    # ===================================================================
    # op_* methods: parameter-based operations for LLM fallback
    # ===================================================================
    @staticmethod
    def _clean_opt(v, default=None):
        if v is None or v == "":
            return default
        return v

    @staticmethod
    def _norm_gate_type(t):
        return str(t).lower() if t is not None else ""

    @staticmethod
    def _norm_basis(basis):
        if not basis:
            return None
        b = str(basis).upper().replace("+", "_").replace(" ", "_").replace("-", "_")
        aliases = {
            "NAND_NOT": "NAND_NOT",
            "NAND_AND_NOT": "NAND_NOT",
            "NOR_NOT": "NOR_NOT",
            "NOR_AND_NOT": "NOR_NOT",
            "AND_NOT": "AND_NOT",
            "AND_AND_NOT": "AND_NOT",
            "AND_OR_NOT": "AND_OR_NOT",
            "AND_OR_AND_NOT": "AND_OR_NOT",
        }
        return aliases.get(b, b)

    # Values that mean "no scope" rather than naming a cone.  A model asked to
    # remap the ENTIRE design tends to fill the optional scope with a word for
    # the design itself instead of leaving it out, and that word is not a net:
    # the cone of it is empty, so the remap rewrites nothing and still reports
    # success.  Treating them as absent keeps a correct intent with a correct
    # basis from being turned into a no-op by a redundant parameter.
    _WHOLE_DESIGN = {"design", "entire design", "whole design", "the design",
                     "all", "everything", "netlist", "the netlist",
                     "entire netlist", "whole netlist", "global", "none"}

    def _scope_from_param(self, scope):
        """Convert an LLM scope/output parameter into a gate-name set.

        The rewrite helpers expect scope=None for whole design, or a set of gate
        instance names for a scoped cone rewrite.
        """
        scope = self._clean_opt(scope)
        if scope is None:
            return None
        if str(scope).strip().lower() in self._WHOLE_DESIGN:
            return None
        if self._need_design():
            return None
        gates = {g.name for g in cones.fanin_cone_gates(self.state.current, str(scope))}
        if not gates and self.state.current is not None \
                and str(scope) not in self.state.current.all_nets():
            # Names no net at all: scoping to it would silently rewrite nothing.
            return None
        return gates

    # ----- IO / testcase -------------------------------------------------
    def op_begin_case(self, name="case"):
        name = str(name).rstrip(".") if name else "case"
        self.state.case_name = name
        return (f'Acknowledged. Initialized testcase "{name}". All subsequent '
                f"responses will be recorded to {name}.log. Design state is "
                "empty and ready for commands.")

    def op_load_design(self, file, dir=""):
        fname = str(file).strip().strip("'\"")
        d = str(dir or "").strip().strip("'\"")
        path = os.path.join(d, fname) if d else fname
        candidates = [path, fname, os.path.join(self.state.design_dir, fname)]
        # If the LLM returned a directory with the file already appended, avoid
        # failing on testcase/test41/test41.v/test41.v-like paths.
        if d and d.endswith(fname):
            candidates.insert(0, d)
        for cand in candidates:
            if os.path.exists(cand):
                path = cand
                break
        try:
            nl = reader.parse_file(path)
        except Exception as exc:
            return f"Failed to load design from {path}: {exc}"
        self.state.set_loaded(nl)
        self.state.design_dir = os.path.dirname(path) or "."
        return (f'Loaded gate-level Verilog from "{path}" successfully. '
                f"Detected a single top module (flat netlist) with "
                f"{len(nl.gates)} combinational gates and {len(nl.dffs)} "
                f"flip-flops. Design state has been updated.")

    def op_write_design(self, file=None):
        if self._need_design():
            return self._need_design()
        fname = str(file or f"{self.state.case_name or 'out'}_out.v").strip().strip("'\"")
        out_path = self._out_path(fname)
        try:
            writer.write_file(self.state.current, out_path)
        except Exception as exc:
            return f"Failed to write design to {out_path}: {exc}"
        return f'Wrote the current netlist to "{out_path}" successfully.'

    # ----- counts / reports ---------------------------------------------
    def op_count_gates(self):
        if self._need_design():
            return self._need_design()
        return counts.count_breakdown_text(self.state.current)

    def op_total_gate_count(self):
        if self._need_design():
            return self._need_design()
        n = counts.total_gate_count(self.state.current)
        return f"The total gate count of the design is {n}."

    def op_count_type(self, type, scope=None):
        if self._need_design():
            return self._need_design()
        gtype = self._norm_gate_type(type)
        scope = self._clean_opt(scope)
        if scope is not None:
            out = str(scope)
            gs = cones.fanin_cone_gates(self.state.current, out)
            n = sum(1 for g in gs if g.type == gtype)
            return (f"There are currently {n} {gtype.upper()} gates in "
                    f"the cone of {out}.")
        n = counts.count_of_type(self.state.current, gtype)
        return f"There are currently {n} {gtype.upper()} gates in the design."

    def op_delta_count(self, kind=""):
        low = str(kind or "").lower()
        d = self.state.deltas
        val = None
        if low in d:
            # exact delta key ("nor_added", "xor_converted", ...)
            val = d[low]
        elif "buf" in low or "buffer" in low:
            val = d.get("buffers_added")
        elif "nand" in low and "added" in low:
            val = d.get("nand_added")
        elif "nor" in low and "xnor" not in low and "added" in low:
            val = d.get("nor_added")
        elif "const" in low or "eliminated" in low or "propagation" in low:
            val = d.get("const_eliminated")
        elif "merge" in low or "duplicate" in low:
            val = d.get("merged")
        elif "collapse" in low or "inverter" in low:
            val = d.get("collapsed")
        elif "enable" in low or "hold" in low:
            val = d.get("enable_hold")
        elif "remove" in low or "dangling" in low or "redundant" in low or "floating" in low:
            val = d.get("removed")
        if val is None and d:
            val = d[list(d)[-1]]
        return f"{val if val is not None else 0}"

    def op_gate_info(self, gate):
        if self._need_design():
            return self._need_design()
        name = str(gate)
        inst = self.state.current.instance_by_name(name)
        if inst is None:
            return f"No gate named {name} exists in the design."
        kind, g = inst
        if kind == "gate":
            return (f"Gate {name} is a {g.type.upper()} gate. "
                    f"Output: {g.out}; inputs: {', '.join(g.ins)}.")
        return (f"{name} is a DFF. Q={g.q}, D={g.d}, CK={g.clk}, "
                f"RN={g.rn}, SN={g.sn}.")

    def op_list_type(self, type):
        if self._need_design():
            return self._need_design()
        gtype = self._norm_gate_type(type)
        if gtype == "dff":
            gs = self.state.current.dffs
            items = [f"{g.name} (Q={g.q}, D={g.d})" for g in gs]
        else:
            gs = counts.list_gates_of_type(self.state.current, gtype)
            items = [f"{g.name} (out={g.out}, in={', '.join(g.ins)})" for g in gs]
        if not gs:
            return f"There are 0 {gtype.upper()} gates in the design."
        if len(items) <= self.LIST_INLINE:
            return f"{len(gs)} {gtype.upper()} gate(s):\n" + "\n".join(items)
        fname = f"{self.state.case_name or 'case'}_{gtype}_gates.txt"
        path = self._list_to_file(fname, items)
        if path:
            return (f"{len(gs)} {gtype.upper()} gates. The complete list has been "
                    f"written to {path}.")
        return f"{len(gs)} {gtype.upper()} gates:\n" + "\n".join(items[:self.LIST_INLINE])

    def op_cone_gate_count(self, output):
        if self._need_design():
            return self._need_design()
        out = str(output)
        n = counts.gates_in_fanin_cone(self.state.current, out)
        return f"The fan-in cone of {out} contains {n} gates."

    def op_cone_type_counts(self, output):
        if self._need_design():
            return self._need_design()
        return counts.cone_type_counts_text(self.state.current, str(output))

    def op_list_ports(self, dir="input"):
        if self._need_design():
            return self._need_design()
        nl = self.state.current
        direction = str(dir or "input").lower()
        items = []
        for name in nl.port_order:
            p = nl.ports.get(name)
            if p and p.direction == direction:
                items.append(f"{name} [{p.width}-bit]" if p.is_bus else f"{name} [1-bit]")
        label = "Primary inputs" if direction == "input" else "Primary outputs"
        return f"{label}:\n" + "\n".join(items)

    def op_count_ports(self):
        if self._need_design():
            return self._need_design()
        nl = self.state.current
        ni = sum(1 for n in nl.port_order if nl.ports.get(n) and nl.ports[n].direction == "input")
        no = sum(1 for n in nl.port_order if nl.ports.get(n) and nl.ports[n].direction == "output")
        bi = len(nl.pi)
        bo = len(nl.po)
        return (f"The design has {bi} primary inputs and {bo} primary outputs "
                f"(across {ni} input port(s) and {no} output port(s)).")

    # ----- connectivity --------------------------------------------------
    def op_fanout(self, net):
        if self._need_design():
            return self._need_design()
        return self._fanout_answer(str(net))

    def op_gates_driven_by(self, gate):
        if self._need_design():
            return self._need_design()
        g = str(gate)
        ds = connectivity.gates_driven_by_gate(self.state.current, g)
        if ds is None:
            return f"No gate named {g} exists."
        return (f"Gate {g} drives {len(ds)} gate(s): "
                + self._names_or_file(ds, f"driven_by_{g}"))

    def op_successors(self, gate, form=None):
        if self._need_design():
            return self._need_design()
        g = str(gate)
        ds = connectivity.immediate_successors(self.state.current, g)
        if ds is None:
            return f"No instance named {g} exists."
        out = self._shaped(form, ds, f"Immediate successors of {g}",
                           f"successors_{g}")
        return out if out is not None else (
            "Immediate successors of %s: " % g
            + self._names_or_file(ds, f"successors_{g}"))

    def op_transitive_fanin(self, net, form=None):
        if self._need_design():
            return self._need_design()
        net = str(net)
        gs = cones.fanin_cone_gates(self.state.current, net)
        out = self._shaped(form, [g.name for g in gs],
                           f"The fan-in cone of {net}", f"fanin_cone_{net}")
        return out if out is not None else (
            f"The transitive fan-in cone of {net} contains {len(gs)} gates.")

    def op_transitive_fanout(self, net, form=None):
        if self._need_design():
            return self._need_design()
        net = str(net)
        gs = cones.fanout_cone_gates(self.state.current, net)
        out = self._shaped(form, [g.name for g in gs],
                           f"The fan-out cone of {net}", f"fanout_cone_{net}")
        return out if out is not None else (
            f"The transitive fan-out cone of {net} contains {len(gs)} gates.")

    def op_reachable_from(self, net):
        if self._need_design():
            return self._need_design()
        net = str(net)
        gs = connectivity.reachable_gates_from(self.state.current, net)
        return (f"{len(gs)} gate(s) are reachable from {net}: "
                + self._names_or_file(gs, f"reachable_{net}"))

    def op_highest_fanout_pi(self):
        if self._need_design():
            return self._need_design()
        name, f = connectivity.highest_fanout_pi(self.state.current)
        return f"Primary input {name} has the highest fanout ({f})."

    def op_highest_fanout_net(self):
        """The busiest net in the whole design, not just among the PIs.

        Separate from op_highest_fanout_pi because the winner is usually an
        internal signal: on the reference design the PI maximum is 8 while the
        design maximum is 12, so answering a "which signal" question with the
        PI-scoped intent reports a smaller number for a different net -- and
        the number is what a threshold question then turns on.

        The driver comes along because the question is normally asked about a
        net whose driver the caller wants next.
        """
        if self._need_design():
            return self._need_design()
        nl = self.state.current
        name, f = connectivity.highest_fanout_net(nl)
        drv = nl.driver(name)
        by = f", driven by {drv[1].name}" if drv and drv[0] == "gate" else ""
        return f"Signal {name} drives the largest number of loads ({f}){by}."

    def op_max_fanout_of(self, net):
        if self._need_design():
            return self._need_design()
        net = str(net)
        f = connectivity.max_fanout_of(self.state.current, net)
        return f"The maximum fanout of {net} is now {f}."

    def op_shared_cone(self, a, b):
        if self._need_design():
            return self._need_design()
        a, b = str(a), str(b)
        gs = cones.shared_fanin_gates(self.state.current, a, b)
        return (f"{len(gs)} gate(s) are shared between the fan-in cones of "
                f"{a} and {b}: " + self._names_or_file([g.name for g in gs], f"shared_{a}_{b}"))

    def op_connected_to_output(self, gate):
        if self._need_design():
            return self._need_design()
        g = str(gate)
        ds = connectivity.gates_driven_by_gate(self.state.current, g)
        if ds is None:
            return f"No gate named {g} exists."
        return (f"Gates connected to the output of {g}: "
                + self._names_or_file(ds, f"connected_out_{g}"))

    # ----- paths ---------------------------------------------------------
    def op_path_exists(self, a, b, avoid=None):
        if self._need_design():
            return self._need_design()
        a, b = str(a), str(b)
        avoid = self._clean_opt(avoid)
        if avoid:
            ok = paths.path_exists(self.state.current, a, b, avoid={str(avoid)})
            if ok:
                return f"Yes. A combinational path from {a} to {b} that avoids {avoid} exists."
            return f"No. There is no combinational path from {a} to {b} that avoids {avoid}."
        ok = paths.path_exists(self.state.current, a, b)
        if ok:
            return f"Yes. A combinational path from {a} to {b} exists."
        return f"No. There is no combinational path from {a} to {b}."

    def op_enumerate_paths(self, a, b):
        if self._need_design():
            return self._need_design()
        a, b = str(a), str(b)
        nl = self.state.current
        count = paths.count_paths(nl, a, b)
        if count == 0:
            return f"There are no combinational paths from {a} to {b}."
        if count <= self.PATH_INLINE:
            _, plist = paths.enumerate_paths(nl, a, b, limit=self.PATH_INLINE)
            lines = [f"There are {count} path(s) from {a} to {b}:"]
            for p in plist:
                lines.append("  " + a + " -> " + " -> ".join(p) + " -> " + b
                             if p else "  " + a + " -> " + b)
            return "\n".join(lines)
        if count <= self.PATH_FILE_CAP:
            fname = f"{self.state.case_name or 'case'}_paths_{self._san_name(a)}_to_{self._san_name(b)}.txt"
            path = self._out_path(fname)
            try:
                with open(path, "w") as fh:
                    paths.stream_paths(nl, a, b, fh, count + 1)
                return (f"There are {count} combinational paths from {a} to {b}. "
                        f"The complete enumeration has been written to {path}.")
            except Exception as exc:
                return (f"There are {count} combinational paths from {a} to {b} "
                        f"(could not write the list file: {exc}).")
        return (f"There are {count} combinational paths from {a} to {b} — too "
                f"many to enumerate literally.")

    def op_length_zero_paths(self):
        if self._need_design():
            return self._need_design()
        z = paths.length_zero_paths(self.state.current)
        if not z:
            return "No length-0 paths exist (no primary input directly drives a primary output)."
        return f"{len(z)} length-0 path(s): " + self._names(z)

    def op_dominator(self, a, b, gate):
        if self._need_design():
            return self._need_design()
        a, b, g = str(a), str(b), str(gate)
        res = paths.every_path_passes_through(self.state.current, a, b, g)
        if res is None:
            return f"No combinational path from {a} to {b} exists."
        return (("Yes." if res else "No.") +
                f" {'Every' if res else 'Not every'} path from {a} to {b} passes through {g}.")

    def op_articulation(self, a, b):
        if self._need_design():
            return self._need_design()
        a, b = str(a), str(b)
        pts = paths.articulation_points(self.state.current, a, b)
        if not pts:
            return f"There are no articulation points between {a} and {b}."
        return (f"{len(pts)} articulation point(s) between {a} and {b}: " +
                self._names_or_file(pts, f"articulation_{a}_{b}"))

    def op_is_cut(self, wire):
        if self._need_design():
            return self._need_design()
        w = str(wire)
        res = paths.is_cut_pi_po(self.state.current, w)
        return (("Yes." if res else "No.") +
                f" Wire {w} {'is' if res else 'is not'} a cut between a primary input and a primary output.")

    # ----- depth ---------------------------------------------------------
    def op_max_depth_between(self, a, b):
        if self._need_design():
            return self._need_design()
        a, b = str(a), str(b)
        d = depth.max_depth_from_to(self.state.current, a, b)
        if d is None:
            return f"There is no combinational path from {a} to {b} (depth 0)."
        return f"The maximum combinational logic depth from {a} to {b} is {d}."

    def op_cone_depth(self, output):
        if self._need_design():
            return self._need_design()
        out = str(output)
        d = depth.depth_of_cone(self.state.current, out)
        return f"The maximum logic depth of the cone of {out} is {d}."

    def op_global_max_depth(self):
        if self._need_design():
            return self._need_design()
        d = depth.global_max_depth(self.state.current)
        return f"The maximum combinational logic depth in the design is {d}."

    def op_pi_to_dff_depth(self):
        if self._need_design():
            return self._need_design()
        d = depth.pi_to_dff_d_max_depth(self.state.current)
        return f"The maximum logic depth from any primary input to any DFF D-pin is {max(d,0)}."

    def op_reg_to_reg_depth(self):
        if self._need_design():
            return self._need_design()
        d = depth.reg_to_reg_max_depth(self.state.current)
        if d < 0:
            return "There are no register-to-register combinational paths."
        return f"The maximum combinational depth on any register-to-register path is {d}."

    def op_outputs_depth_gt(self, k):
        if self._need_design():
            return self._need_design()
        k = int(k)
        outs = depth.outputs_depth_greater_than(self.state.current, k)
        return f"{len(outs)} output(s) have a logic depth greater than {k}."

    def op_pi_to_po_depth(self):
        if self._need_design():
            return self._need_design()
        d = depth.pi_to_po_max_depth(self.state.current)
        if d < 0:
            return ("The maximum combinational logic depth from any primary input "
                    "to any primary output is 0 (no primary output is reachable "
                    "from a primary input by a purely combinational path).")
        return ("The maximum combinational logic depth from any primary input to "
                f"any primary output is {d}.")

    def op_reg_to_po_depth(self):
        if self._need_design():
            return self._need_design()
        d = depth.reg_to_po_max_depth(self.state.current)
        if d < 0:
            return ("The maximum combinational logic depth from any register "
                    "output to any primary output is 0 (no primary output is "
                    "reachable from a register output by a purely combinational "
                    "path).")
        return ("The maximum combinational logic depth from any register output "
                f"to any primary output is {d}.")

    def op_largest_fanin_cone(self):
        if self._need_design():
            return self._need_design()
        name, n = cones.largest_fanin_output(self.state.current)
        return f"Output {name} has the largest fan-in cone ({n} gates)."

    def op_check_floating(self):
        return self.h_check_floating_ports(None, "")

    def op_floating_count(self):
        return self.h_floating_count(None, "")

    def op_const1_gates(self):
        return self.h_const1_gates(None, "")

    def op_connected_to_net(self, net):
        if self._need_design():
            return self._need_design()
        net = str(net)
        gs = naming.gates_connected_to_net(self.state.current, net)
        return (f"Gates connected to {net}: "
                + self._names_or_file(gs, f"connected_{net}"))

    def op_deepest_output(self):
        if self._need_design():
            return self._need_design()
        name, n = depth.deepest_output(self.state.current)
        return f"Output {name} has the deepest fan-in cone (depth {n})."

    def op_gate_on_max_path(self, gate):
        if self._need_design():
            return self._need_design()
        g = str(gate)
        res = depth.gate_on_max_depth_path(self.state.current, g)
        if res is None:
            return f"No gate named {g} exists."
        return ("Yes." if res else "No.") + f" Gate {g} {'lies' if res else 'does not lie'} on a maximum-depth path."

    # ----- functional ----------------------------------------------------
    def op_signals_equivalent(self, a, b):
        return self._sig_equiv(str(a), str(b))

    def op_output_constant(self, output):
        if self._need_design():
            return self._need_design()
        out = str(output)
        v = functional.output_always_constant(self.state.current, out)
        if v == 0:
            return f"Yes. Output {out} is always 0 regardless of the inputs."
        if v == 1:
            return f"Output {out} is constant and always 1 regardless of the inputs."
        return f"No. Output {out} is not constant; it depends on the inputs."

    def op_depends_on(self, output, input, kind=None):
        """Does ``output`` depend on ``input``?  Functionally, by default.

        The catalog has always described this intent as "asking functional
        influence" ("will changing n4 ever change n32[0]"), but it called
        functional.depends_on, whose own docstring says STRUCTURAL: is the
        input anywhere in the fan-in cone.  Those differ exactly where the
        question is interesting -- a net can sit in the cone and still be
        masked, and a reference set asks precisely that, stating the
        structural fact in the prompt and asking for the functional one.

        Structural first, because it is cheap and one-directional: outside the
        cone there is no path, so no influence, and no SAT call is needed.
        Inside the cone the exact check decides.  If that cannot decide (no
        solver, or it gave up) the structural answer is reported and labelled
        as structural rather than silently passed off as functional.

        ``kind="structural"`` asks for cone membership explicitly.
        """
        if self._need_design():
            return self._need_design()
        out, inp = str(output), str(input)
        nl = self.state.current
        structural = functional.depends_on(nl, out, inp)
        want = str(kind).strip().lower() if kind is not None else "functional"

        if want.startswith("struct"):
            return (("Yes." if structural else "No.")
                    + f" {inp} is {'' if structural else 'not '}in the fan-in "
                      f"cone of {out} (structural).")
        if not structural:
            return f"No. Output {out} does not depend on input {inp}."

        exact = functional.truly_depends_on(nl, out, inp)
        if exact is None:
            return (f"Yes. {inp} is in the fan-in cone of {out} (structural); "
                    f"functional dependence could not be decided.")
        return (("Yes." if exact else "No.")
                + f" Output {out} {'depends' if exact else 'does not depend'} "
                  f"on input {inp}.")

    def op_boolean_equation(self, output):
        if self._need_design():
            return self._need_design()
        out = str(output)
        eq = functional.boolean_equation(self.state.current, out)
        if eq is None:
            return (f"The Boolean equation for {out} has too large a support to "
                    "express compactly in terms of the primary inputs.")
        return f"{out} = {eq}"

    def op_symmetric(self, output, a, b):
        if self._need_design():
            return self._need_design()
        out, a, b = str(output), str(a), str(b)
        r = functional.is_symmetric(self.state.current, out, a, b)
        if r is None:
            return f"Could not determine symmetry of {out} in {a}, {b}."
        return ("Yes." if r else "No.") + f" The function at {out} is {'' if r else 'not '}symmetric in {a} and {b}."

    def op_exists_nand_pair(self, target):
        if self._need_design():
            return self._need_design()
        target = str(target)
        r = functional.exists_nand_pair(self.state.current, target)
        if r is None:
            return f"No pair of internal signals (a, b) with NAND(a, b) equivalent to {target} was found."
        return f"Yes. NAND({r[0]}, {r[1]}) is functionally equivalent to {target}."

    # ----- sequential ----------------------------------------------------
    def op_ffs_on_clock(self, clk):
        if self._need_design():
            return self._need_design()
        clk = str(clk)
        ff = sequential.ffs_on_clock(self.state.current, clk)
        return (f"{len(ff)} flip-flop(s) are driven by clock {clk}: " +
                self._names_or_file([f.name for f in ff], f"ffs_clock_{clk}"))

    def op_same_clock(self, a, b):
        if self._need_design():
            return self._need_design()
        r = sequential.same_clock_domain(self.state.current, str(a), str(b))
        if r is None:
            return "One or both flip-flops were not found."
        return "Yes. They are in the same clock domain." if r else "No. They are in different clock domains."

    def op_reg_to_reg_paths(self):
        if self._need_design():
            return self._need_design()
        count, pairs = sequential.reg_to_reg_pairs(self.state.current)
        head = f"There are {count} register-to-register connections through combinational logic."
        if pairs:
            head += " Examples: " + self._names([f"{a}->{b}" for a, b in pairs[:30]])
        return head

    def op_enable_hold_report(self):
        if self._need_design():
            return self._need_design()
        eh = sequential.enable_hold_ffs(self.state.current)
        self.state.record_delta("enable_hold", len(eh))
        return (f"{len(eh)} flip-flop(s) implement an enable/hold structure in "
                f"their D-input logic (next state depends on the register's own "
                f"current state). Examples: " +
                self._names([f.name for f in eh[:30]]))

    def op_enable_hold_count(self):
        if self._need_design():
            return self._need_design()
        n = self.state.get_delta("enable_hold")
        if not n:
            eh = sequential.enable_hold_ffs(self.state.current)
            n = len(eh)
        return f"{n} flip-flops were found to have enable or hold structures in their D-input logic."

    # ----- transforms ----------------------------------------------------
    def op_convert_basis(self, basis=None, scope=None):
        if self._need_design():
            return self._need_design()
        basis = self._norm_basis(basis)
        if basis is None:
            return "Could not determine the target gate basis."
        scope_gates = self._scope_from_param(scope)
        info, ok, reason = self._commit(
            lambda nl: rewrite.to_basis(nl, basis, scope_gates),
            basis=(None if scope_gates is not None else basis))
        if not ok:
            return f"The basis remap was reverted: {reason}."
        self.state.record_delta("basis_remap", info)
        b = basis.replace("_", "+")
        return (f"Remapped {'the cone' if scope_gates is not None else 'the entire design'} to "
                f"{b} gates ({info} gates rewritten); functional equivalence verified.")

    def op_xor_to_nand(self, scope=None):
        if self._need_design():
            return self._need_design()
        scope_gates = self._scope_from_param(scope)
        before = counts.count_of_type(self.state.current, "nand")
        info, ok, reason = self._commit(lambda nl: rewrite.xor_to_nand(nl, scope_gates))
        if not ok:
            return f"The XOR->NAND conversion was reverted: {reason}."
        added = counts.count_of_type(self.state.current, "nand") - before
        self.state.record_delta("nand_added", added)
        self.state.record_delta("xor_converted", info)
        return (f"Converted {info} XOR gate(s) to 4-NAND logic "
                f"({added} NAND gates added); functional equivalence verified.")

    def op_xnor_to_nor(self, scope=None):
        if self._need_design():
            return self._need_design()
        scope_gates = self._scope_from_param(scope)
        before = counts.count_of_type(self.state.current, "nor")
        info, ok, reason = self._commit(lambda nl: rewrite.xnor_to_nor(nl, scope_gates))
        if not ok:
            return f"The XNOR->NOR conversion was reverted: {reason}."
        added = counts.count_of_type(self.state.current, "nor") - before
        self.state.record_delta("nor_added", added)
        self.state.record_delta("xnor_converted", info)
        return (f"Converted {info} XNOR gate(s) to NOR-only logic "
                f"({added} NOR gates added); functional equivalence verified.")

    def op_xor_to_aoi(self, scope=None):
        if self._need_design():
            return self._need_design()
        scope_gates = self._scope_from_param(scope)
        info, ok, reason = self._commit(lambda nl: rewrite.xor_to_aoi(nl, scope_gates))
        if not ok:
            return f"The XOR decomposition was reverted: {reason}."
        self.state.record_delta("xor_converted", info)
        return f"Decomposed {info} XOR gate(s) into AND/OR/NOT logic; equivalence verified."

    def op_nand_const1_to_inv(self):
        if self._need_design():
            return self._need_design()
        info, ok, reason = self._commit(lambda nl: rewrite.nand_const1_to_inv(nl))
        if not ok:
            return f"The NAND(const-1)->INV conversion was reverted: {reason}."
        self.state.record_delta("nand_to_inv", info)
        return f"Replaced {info} NAND gate(s) tied to constant 1 with inverters; equivalence verified."

    def op_report_const_gates(self, type="nand", value=None):
        if self._need_design():
            return self._need_design()
        gtype = self._norm_gate_type(type)
        val = None
        if value in (0, "0", "1'b0"):
            val = "1'b0"
        elif value in (1, "1", "1'b1"):
            val = "1'b1"
        self.const_nets = functional.constant_nets(self.state.current)
        gs = constprop.gates_with_const_input(
            self.state.current,
            gtype if gtype in ("and", "or", "nand", "nor") else None,
            val, extra_const=self.const_nets)
        self.state.last_report = [g.name for g in gs]
        self.state.last_report_kind = gtype
        if not gs:
            return f"No {gtype.upper()} gates with constant inputs were found."
        return (f"Found {len(gs)} {gtype.upper()} gate(s) with constant inputs: " +
                self._names_or_file([g.name for g in gs], f"{gtype}_const_inputs"))

    def op_const_propagate(self, type=None):
        if self._need_design():
            return self._need_design()
        rtype = self._norm_gate_type(type) if type else self.state.last_report_kind
        extra = self.const_nets if self.const_nets else functional.constant_nets(self.state.current)
        info, ok, reason = self._commit(
            lambda nl: constprop.const_propagate(nl, rtype, extra_const=extra))
        if not ok:
            return f"The constant propagation was reverted: {reason}."
        self.state.record_delta("const_eliminated", info)
        tlabel = (rtype.upper() + " ") if rtype else ""
        return f"Constant propagation eliminated {info} {tlabel}gate(s); equivalence verified."

    def op_remove_buffers(self):
        """Delete every BUF, rewiring around it.  A transform, not a query."""
        if self._need_design():
            return self._need_design()
        info, ok, reason = self._commit(lambda nl: cleanup.remove_buffers(nl))
        if not ok:
            return f"The buffer removal was reverted: {reason}."
        self.state.record_delta("buffers_removed", info)
        left = len(cleanup.buffers_remaining(self.state.current))
        tail = (f" {left} buffer(s) remain, each driving a primary output "
                f"straight from a primary input." if left else "")
        return (f"Removed {info} buffer(s), connecting each buffered signal "
                f"directly; equivalence verified.{tail}")

    def op_collapse_inverters(self):
        if self._need_design():
            return self._need_design()
        info, ok, reason = self._commit(lambda nl: cleanup.collapse_double_inverters(nl))
        if not ok:
            return f"The inverter collapse was reverted: {reason}."
        self.state.record_delta("collapsed", info)
        return f"Collapsed {info} back-to-back inverter pair(s) into direct wires; equivalence verified."

    def op_remove_dangling(self):
        if self._need_design():
            return self._need_design()
        info, ok, reason = self._commit(lambda nl: cleanup.remove_dangling(nl))
        if not ok:
            return f"The dangling-gate removal was reverted: {reason}."
        self.state.record_delta("removed", info)
        return f"Removed {info} dangling gate(s) that do not affect any primary output; equivalence verified."

    def op_merge_duplicates(self):
        if self._need_design():
            return self._need_design()
        info, ok, reason = self._commit(lambda nl: cleanup.merge_structural_duplicates(nl))
        if not ok:
            return f"The duplicate merge was reverted: {reason}."
        self.state.record_delta("merged", info)
        return f"Merged {info} structurally-duplicate gate(s); equivalence verified."

    def op_rename(self, old, new, kind="signal"):
        if self._need_design():
            return self._need_design()
        old, new = str(old), str(new)
        kind = str(kind or "signal").lower()
        # wires/nets rename like signals, but the reply echoes the caller's noun
        word = kind if kind in ("gate", "wire", "signal") else "signal"
        if kind == "gate":
            ok = naming.rename_gate(self.state.current, old, new)
        else:
            ok = naming.rename_net(self.state.current, old, new)
        self.state.current.touch()
        if not ok:
            return f"No {word} named {old} was found to rename."
        return f"Renamed {word} {old} to {new} and updated all references."

    def op_insert_buffers(self, k=4, net=None, mode="fanout", scope=None):
        if self._need_design():
            return self._need_design()
        k = int(k)
        net = self._clean_opt(net)
        mode = str(mode or "fanout").lower()
        # "no gate drives more than k" bounds gate outputs and DFF.Q (a
        # flip-flop is a gate, Q&A A2); "no signal/net" also bounds PIs.
        include_pi = str(scope or "gate").lower() == "signal"
        if mode == "dedicated" and net:
            info, ok, reason = self._commit(lambda nl: buffering.dedicated_buffer_per_load(nl, str(net)))
            if not ok:
                return f"The buffer insertion was reverted: {reason}."
            self.state.record_delta("buffers_added", info)
            return f"Inserted {info} dedicated buffer(s), one per load of {net}; equivalence verified."
        if net:
            info, ok, reason = self._commit(
                lambda nl: buffering.limit_fanout(nl, k, only_nets={str(net)}), max_fanout=None)
            if not ok:
                return f"The buffer insertion was reverted: {reason}."
            self.state.record_delta("buffers_added", info)
            return f"Inserted {info} buffer(s) on {net} so each driver has at most {k} loads; equivalence verified."
        info, ok, reason = self._commit(
            lambda nl: buffering.limit_fanout(nl, k, include_pi=include_pi),
            max_fanout=k, max_fanout_pi=include_pi)
        if not ok:
            return f"The buffer insertion was reverted: {reason}."
        self.state.record_delta("buffers_added", info)
        return (f"Inserted {info} buffer(s) so that no driver exceeds {k} loads; "
                f"max-fanout bound and equivalence verified.")

    # ----- optimize ------------------------------------------------------
    def op_minimize_depth(self, basis=None):
        if self._need_design():
            return self._need_design()
        basis = self._norm_basis(basis)
        before = depth.global_max_depth(self.state.current)
        res, imp = abc_opt.minimize_depth(self.state.current, basis=basis)
        self.state.current = res
        after = depth.global_max_depth(res)
        if imp:
            return (f"Reduced the maximum logic depth from {before} to {after}"
                    f"{' (basis preserved)' if basis else ''}; equivalence verified.")
        return (f"The design is already optimal at depth {before}; reported the "
                "original (equivalence preserved).")

    def op_minimize_area(self, basis=None):
        if self._need_design():
            return self._need_design()
        if not hasattr(abc_opt, "minimize_area"):
            return "Area minimization is not implemented in this build; the current design is unchanged."
        basis = self._norm_basis(basis)
        before = len(self.state.current.gates)
        res, imp = abc_opt.minimize_area(self.state.current, basis=basis)
        self.state.current = res
        after = len(res.gates)
        if imp:
            return (f"Reduced the gate count from {before} to {after}"
                    f"{' (basis preserved)' if basis else ''}; equivalence verified.")
        return (f"The design is already optimal at gate count {before}; reported the "
                "original (equivalence preserved).")

    def op_optimize_cone(self, output, basis=None):
        if self._need_design():
            return self._need_design()
        out = str(output)
        basis = self._norm_basis(basis)
        before = depth.depth_of_cone(self.state.current, out)
        res, imp = abc_opt.optimize_cone_depth(self.state.current, out, basis=basis)
        self.state.current = res
        after = depth.depth_of_cone(res, out)
        if imp:
            return (f"Optimized the cone of {out}: depth reduced from {before} "
                    f"to {after}{' (basis preserved)' if basis else ''}; equivalence verified.")
        return (f"The cone of {out} is already optimal at depth {before}; "
                "reported the original (equivalence preserved).")

    # ----- equivalence ---------------------------------------------------
    def op_verify_equivalence(self, against="original"):
        if self._need_design():
            return self._need_design()
        target = str(against or "original").lower()
        if target in ("pre", "pre_transformation", "pre-transformation"):
            ref = self.state.pre or self.state.original
            label = "pre-transformation netlist"
        elif target in ("last_loaded", "last-loaded", "last loaded"):
            ref = self.state.last_loaded
            label = "netlist as last loaded from disk"
        else:
            ref = self.state.original
            label = "original loaded netlist"
        if ref is None:
            return "No reference design is available for comparison."
        r = equiv_gate.equivalent(ref, self.state.current)
        if r is True:
            return f"Verified: the current design is functionally equivalent to the {label}."
        if r is False:
            return f"The current design is NOT functionally equivalent to the {label}."
        return f"Could not conclusively verify equivalence to the {label}."
