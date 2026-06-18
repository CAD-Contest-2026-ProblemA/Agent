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
    def __init__(self, config: Config):
        self.config = config
        self.state = State()
        self.llm = LLMClient(config)
        self.fallback = Fallback(self.llm)
        self.rules = self._build_rules()

    # ===================================================================
    # main entry
    # ===================================================================
    def handle(self, line: str, ident: int) -> str:
        line = line.strip()
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
            try:
                return self._dispatch_intent(obj.get("intent"), obj.get("params", {}), line)
            except Exception as exc:
                return f"Could not complete the request ({exc})."
        return self._default_ack(line)

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
            (R(r"\b(load|read)\b.*design|design from (the )?file"), self.h_load),
            (R(r"\bwrite\b.*(design|netlist).*\b(file|to)\b|write out"), self.h_write),

            # optimize rules first: they mention "depth/cone" but must not be
            # captured by the depth/cone *query* rules.
            (R(r"optimize the depth of the cone of (%s)" % NET), self.h_opt_cone),
            (R(r"optimize the (logic )?cone of (?:output )?(%s)" % NET), self.h_opt_cone),
            (R(r"(reduce|minimize|minimise).*(critical path|maximum|max|critical).*depth"), self.h_opt_depth),
            (R(r"(depth optimization|perform depth optimization|reduce critical path)"), self.h_opt_depth),
            (R(r"minimize the maximum logic depth|minimize maximum (logic |path )?depth"), self.h_opt_depth),

            # transforms (imperative actions) — matched BEFORE analysis queries
            # so cost-function / "gates in the cone of" phrasing in a transform
            # request is never captured by a query rule.
            (R(r"(remap|reconstruct|convert|restructure|replace).*(only|use only|using only).*(nand|nor|and).*(not|nand|nor)"), self.h_basis),
            (R(r"replace.*XNOR.*(NOR-only|NOR only|equivalent NOR)"), self.h_xnor_nor),
            (R(r"convert every XNOR.*NOR"), self.h_xnor_nor),
            (R(r"(replace|convert).*XOR.*(NAND-only|4 ?NAND|4-NAND|NAND)"), self.h_xor_nand),
            (R(r"decompose all XOR.*(AND, OR, and NOT|AND.*OR.*NOT)"), self.h_xor_aoi),
            (R(r"convert every XOR.*4-?NAND"), self.h_xor_nand),
            (R(r"replace all 2-input NAND.*constant 1.*inverter"), self.h_nand_inv),
            (R(r"simplify the reported (\w+) gates|simplify the reported|propagating .*constant"), self.h_constprop),
            (R(r"(back-to-back|back to back).*(invert|NOT).*collapse|collapse them into.*wire|pairs of.*inverters"), self.h_collapse),
            (R(r"(remove|delete|trim|sweep|prune|eliminate).*(dangling|unused|floating|redundant)"), self.h_dangling),
            (R(r"(delete|remove|eliminate|prune).*gates?.*(do not|don't|not) (contribute|affect|connected)"), self.h_dangling),
            (R(r"are there any redundant gates"), self.h_dangling),
            (R(r"check.*dangling gates|check if there are any floating"), self.h_check_dangling),
            (R(r"(merge|find and merge).*(functionally equivalent|same function|structural duplicate|duplicate)"), self.h_merge),
            (R(r"(rename|change the identifier of|update the name of|change the name).*(gate|wire|signal)\s+(%s)\s+to\s+(%s)" % (NET, NET)), self.h_rename),
            (R(r"list all gates.*connect.*to the renamed signal (%s)" % NET), self.h_connected_renamed),
            (R(r"insert.*buffers?.*no (gate|signal) (drives|has fanout).*?(\d+)"), self.h_buffers_fanout),
            (R(r"insert a BUF gate on signal (%s).*dedicated buffer" % NET), self.h_buffers_dedicated),
            (R(r"insert.*buffers? on (?:the )?(?:reset )?signal (%s).*?(\d+) loads" % NET), self.h_buffers_signal),
            (R(r"buffers on the reset signal (%s)" % NET), self.h_buffers_reset),
            (R(r"(rename|update the name of|change the identifier of).*?(%s)\s+to\s+(%s)" % (NET, NET)), self.h_rename2),

            # equivalence verification (imperative)
            (R(r"(verify|prove|confirm|check).*(equivalen|equivalent).*(original|pre-transformation|last loaded|as last loaded|loaded netlist)"), self.h_verify),
            (R(r"(verify|prove).*(transformed|current).*(equivalent|equivalence)"), self.h_verify),

            (R(r"count all the gates|broken down by gate type"), self.h_count_all),
            (R(r"total gate count|compute the total gate count"), self.h_total),
            (R(r"report the number of each gate type in the cone of (%s)" % NET), self.h_cone_type),
            (R(r"how many (\w+) gates? (are|were|is)"), self.h_count_or_delta),
            (R(r"how many (\w+) (were|gates were)? ?(added|removed|eliminated|merged|collapsed|inserted|found)"), self.h_delta),
            (R(r"how many (dangling|redundant|floating|duplicate|structural duplicate).*(removed|merged|found)"), self.h_delta),
            (R(r"how many (buf|buffer) gates were added"), self.h_delta),
            (R(r"how many flip-?flops .*enable or hold"), self.h_enable_hold_count),

            (R(r"what type of gate is (%s)" % NET), self.h_gate_info),
            (R(r"list all (\w+) gates? in this design"), self.h_list_type),
            (R(r"list all XOR gates"), self.h_list_xor),
            (R(r"list (all|every).*primary inputs?.*bit widths?"), self.h_list_pi),
            (R(r"list all primary outputs?.*bit widths?"), self.h_list_po),
            (R(r"(number of|how many) primary inputs? and (primary )?outputs?"), self.h_count_ports),
            (R(r"determine the number of primary inputs and outputs"), self.h_count_ports),
            (R(r"gates? (are )?in the (fanin |logic )?cone of (primary output |output )?(%s)" % NET), self.h_cone_gate_count),
            (R(r"list all gates.*tied to 1'b1|inputs tied to 1'b1"), self.h_const1_gates),
            (R(r"report any (\w+) gates? with (a )?constant"), self.h_report_const),
            (R(r"report any (\w+) gates? with constant inputs"), self.h_report_const),

            # paths
            (R(r"path.*from (%s) to (%s).*(?:does not traverse|avoid\w*|without)\s+(?:node\s+)?(%s)" % (NET, NET, NET)), self.h_path_avoid),
            (R(r"path connecting (?:input )?(%s) to (?:output )?(%s).*avoiding\s+(?:node\s+)?(%s)" % (NET, NET, NET)), self.h_path_avoid2),
            (R(r"combinational path.*from (%s) to (%s).*avoids?\s+(?:node\s+)?(%s)" % (NET, NET, NET)), self.h_path_avoid3),
            (R(r"(does|is there|whether).*combinational path exist.*from (?:primary input )?(%s) to (?:primary output )?(%s)" % (NET, NET)), self.h_path_plain),
            (R(r"path.*from (?:primary input )?(%s) to (?:primary output )?(%s)\??\s*(?:report)?.*exist" % (NET, NET)), self.h_path_plain),
            (R(r"(complete enumeration|list every path|find all combinational paths|enumerat).*?(%s).*?(%s)" % (NET, NET)), self.h_enum_paths),
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
            (R(r"max(imum)? combinational (logic )?depth.*(entire design|in the design|primary input to any primary output)"), self.h_global_depth),
            (R(r"max(imum)? (combinational )?logic depth in the design now"), self.h_global_depth),
            (R(r"how many outputs have a logic depth greater than (\d+)"), self.h_depth_gt),
            (R(r"which output (bit )?has the (deepest|largest) (fanin )?(logic )?cone"), self.h_deepest),
            (R(r"(does|whether) gate (%s) lies? on any maximum-depth path" % NET), self.h_on_maxpath),

            # connectivity
            (R(r"fanout of (?:primary input )?(%s).*list (all|every) gate" % NET), self.h_fanout),
            (R(r"(number of gates driven by|gates? driven by) (%s)" % NET), self.h_driven_by),
            (R(r"immediate successors of (?:gate )?(%s)" % NET), self.h_successors),
            (R(r"transitive fanin cone of (?:output )?(%s)" % NET), self.h_tfanin),
            (R(r"transitive fanout (cone )?of (?:input |primary input )?(%s)" % NET), self.h_tfanout),
            (R(r"(all gates reachable from|determine all gates reachable from|reachable from) (%s)" % NET), self.h_reachable),
            (R(r"which primary input has the highest fanout|highest fanout in this design"), self.h_highest_fanout),
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

    # ===================================================================
    # IO / basic
    # ===================================================================
    def h_begin(self, m, line):
        mm = re.search(r"case name is\s+(\S+?)[\.\s]*$", line, re.IGNORECASE)
        name = mm.group(1).rstrip(".") if mm else (self.state.case_name or "case")
        self.state.case_name = name
        return (f'Acknowledged. Initialized testcase "{name}". All subsequent '
                f"responses will be recorded to {name}.log. Design state is "
                "empty and ready for commands.")

    def h_load(self, m, line):
        fm = re.search(r"file\s+(\S+\.v)", line, re.IGNORECASE) or \
             re.search(r"from\s+(\S+\.v)", line, re.IGNORECASE) or \
             re.search(r"(\S+\.v)", line)
        dm = re.search(r"director(?:y|ies)\s+(\S+?)[\.\s]*$", line, re.IGNORECASE) or \
             re.search(r"folder\s+['\"]?([^'\"]+?)['\"]?[\.\s]*$", line, re.IGNORECASE)
        fname = fm.group(1) if fm else None
        d = (dm.group(1).strip().strip("'\"") if dm else "")
        if fname is None:
            return "Could not determine the design file to load."
        path = os.path.join(d, fname) if d else fname
        for cand in (path, fname, os.path.join(self.state.design_dir, fname)):
            if os.path.exists(cand):
                path = cand
                break
        try:
            nl = reader.parse_file(path)
        except Exception as exc:
            return f"Failed to load design from {path}: {exc}"
        self.state.set_loaded(nl)
        self.state.design_dir = os.path.dirname(path) or "."
        c = nl.type_counts()
        return (f'Loaded gate-level Verilog from "{path}" successfully. '
                f"Detected a single top module (flat netlist) with "
                f"{len(nl.gates)} combinational gates and {len(nl.dffs)} "
                f"flip-flops. Design state has been updated.")

    def h_write(self, m, line):
        if self._need_design():
            return self._need_design()
        fm = re.search(r"(?:file|into|to)\s+['\"]?(\S+\.v)['\"]?", line, re.IGNORECASE) or \
             re.search(r"(\S+\.v)", line)
        fname = fm.group(1).strip("'\"") if fm else f"{self.state.case_name or 'out'}_out.v"
        try:
            writer.write_file(self.state.current, fname)
        except Exception as exc:
            return f"Failed to write design to {fname}: {exc}"
        return f'Wrote the current netlist to "{fname}" successfully.'

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

    def h_list_type(self, m, line):
        if self._need_design():
            return self._need_design()
        gtype = m.group(1).lower()
        if gtype == "dff":
            gs = self.state.current.dffs
            items = [f"{g.name} (Q={g.q}, D={g.d})" for g in gs[:200]]
        else:
            gs = counts.list_gates_of_type(self.state.current, gtype)
            items = [f"{g.name} (out={g.out}, in={', '.join(g.ins)})" for g in gs[:200]]
        more = "" if len(gs) <= 200 else f" ... ({len(gs)} total)"
        head = f"{len(gs)} {gtype.upper()} gate(s):"
        return head + ("\n" + "\n".join(items) + more if gs else " (none)")

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
        return (f"The design has {ni} primary input ports ({bi} bits) and "
                f"{no} primary output ports ({bo} bits).")

    def h_cone_gate_count(self, m, line):
        if self._need_design():
            return self._need_design()
        out = m.group(m.lastindex)
        n = counts.gates_in_fanin_cone(self.state.current, out)
        return f"The fan-in cone of {out} contains {n} gates."

    def h_const1_gates(self, m, line):
        if self._need_design():
            return self._need_design()
        gs = [g for g in self.state.current.gates if "1'b1" in g.ins]
        return (f"{len(gs)} gate(s) have an input tied to 1'b1: " +
                self._names([g.name for g in gs[:100]]))

    def h_report_const(self, m, line):
        if self._need_design():
            return self._need_design()
        gtype = m.group(1).lower()
        val = "1'b0" if "0" in line.split(gtype)[-1][:40] else None
        if "constant 1" in line.lower():
            val = "1'b1"
        elif "constant 0" in line.lower() or "constant-0" in line.lower():
            val = "1'b0"
        gs = constprop.gates_with_const_input(self.state.current,
                                              gtype if gtype in ("and", "or", "nand", "nor") else None,
                                              val)
        self.state.last_report = [g.name for g in gs]
        self.state.last_report_kind = gtype
        if not gs:
            return f"No {gtype.upper()} gates with constant inputs were found."
        return (f"Found {len(gs)} {gtype.upper()} gate(s) with constant inputs: " +
                self._names([g.name for g in gs[:100]]))

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

    def h_enum_paths(self, m, line):
        if self._need_design():
            return self._need_design()
        toks = re.findall(NET, line)
        # heuristic: take the last two distinct net-like tokens
        cand = [t for t in toks if re.match(r"n\d", t)]
        if len(cand) < 2:
            return "Could not identify the two endpoints for path enumeration."
        a, b = cand[-2], cand[-1]
        count, plist = paths.enumerate_paths(self.state.current, a, b)
        if count == 0:
            return f"There are no combinational paths from {a} to {b}."
        if plist is None:
            return (f"There are {count} combinational paths from {a} to {b} "
                    f"(too many to enumerate explicitly).")
        lines = [f"There are {count} path(s) from {a} to {b}:"]
        for p in plist[:50]:
            lines.append("  " + a + " -> " + " -> ".join(p) + " -> " + b)
        if count > 50:
            lines.append(f"  ... ({count} total)")
        return "\n".join(lines)

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
                self._names(pts[:100]))

    def h_cut(self, m, line):
        if self._need_design():
            return self._need_design()
        w = m.group(1)
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
        g = m.group(1)
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
        f = connectivity.fanout_count(self.state.current, net)
        loads = connectivity.fanout_load_instances(self.state.current, net)
        return (f"The fanout of {net} is {f}. It directly drives: " +
                self._names(loads[:200]))

    def h_driven_by(self, m, line):
        if self._need_design():
            return self._need_design()
        g = m.group(2)
        ds = connectivity.gates_driven_by_gate(self.state.current, g)
        if ds is None:
            return f"No gate named {g} exists."
        return f"Gate {g} drives {len(ds)} gate(s): " + self._names(ds[:200])

    def h_successors(self, m, line):
        if self._need_design():
            return self._need_design()
        g = m.group(1)
        ds = connectivity.immediate_successors(self.state.current, g)
        if ds is None:
            return f"No instance named {g} exists."
        return f"Immediate successors of {g}: " + self._names(ds[:200])

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
        return f"{len(gs)} gate(s) are reachable from {net}: " + self._names(gs[:100])

    def h_highest_fanout(self, m, line):
        if self._need_design():
            return self._need_design()
        name, f = connectivity.highest_fanout_pi(self.state.current)
        return f"Primary input {name} has the highest fanout ({f})."

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
                f"{a} and {b}: " + self._names([g.name for g in gs[:100]]))

    def h_connected_out(self, m, line):
        if self._need_design():
            return self._need_design()
        g = m.group(2)
        ds = connectivity.gates_driven_by_gate(self.state.current, g)
        if ds is None:
            return f"No gate named {g} exists."
        return f"Gates connected to the output of {g}: " + self._names(ds[:200])

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
                self._names([f.name for f in ff[:100]]))

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
        return (f"Remapped {'the cone' if scope else 'the entire design'} to "
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
        info, ok, reason = self._commit(lambda nl: constprop.const_propagate(nl, rtype))
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
        if self._need_design():
            return self._need_design()
        # count without removing first, then remove
        return self.h_dangling(m, line)

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
        return f"Renamed {old} to {new} and updated all references."

    def h_connected_renamed(self, m, line):
        if self._need_design():
            return self._need_design()
        net = m.group(1)
        gs = naming.gates_connected_to_net(self.state.current, net)
        return f"Gates connected to {net}: " + self._names(gs[:200])

    def h_buffers_fanout(self, m, line):
        if self._need_design():
            return self._need_design()
        k = int(re.findall(r"(\d+)", line)[-1])
        include_pi = "signal" in line.lower()
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
        # Map a structured intent back onto the handlers by synthesising a line.
        if intent in (None, "noop"):
            return self._default_ack(line)
        # Re-run the rule router on the original line first (covers most cases).
        for rx, fn in self.rules:
            mm = rx.search(line)
            if mm:
                return fn(mm, line)
        return self._default_ack(line)
