#!/usr/bin/env python3
"""Independently verify golden answers for BASIC (analysis) queries.

Three-way check per query line, excluding transform/optimization actions:

  1. REPLAY   — drive the real Agent through each testcase so every query is
                answered on the same evolving design state as during golden
                generation (transforms themselves are covered by the HARD
                equivalence checks, not re-judged here).
  2. INDEPEND — at each query, serialize the current state and recompute the
                expected answer with a from-scratch implementation (val-harness
                parser + iterative graph algorithms written here; nothing from
                cada.analysis is imported).
  3. VERDICT  — golden vs live-replay vs independent value:
                  live != golden   -> NONDET  (state/answer not reproducible)
                  indep != golden  -> FAIL    (suspect wrong golden answer)
                  else             -> PASS
                unimplementable families -> SKIP with a reason.

Usage:
    .venv/bin/python scripts/verify_basic_golden.py [test01 test02 ...]
    (no args = all cases that have a golden file; ABC_BIN should be set)

Report: /tmp/golden_check/report.txt (+ report.json)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VH = os.path.join(os.path.dirname(ROOT), "val-harness")
sys.path.insert(0, ROOT)
sys.path.insert(0, VH)

from cada.io_.config import load_config                    # noqa: E402
from cada.agent.agent import Agent                         # noqa: E402
from harness import netlist as H                           # noqa: E402

OUT_DIR = "/tmp/golden_check"
NET = r"[A-Za-z_]\w*(?:\[\d+\])?"


# ============================================================================
# snapshot emitter (cada IR -> minimal Verilog the val-harness parser accepts)
# ============================================================================
def emit_snapshot(nl, path):
    lines = ["module top(" + ", ".join(nl.port_order) + ");"]
    for name in nl.port_order:
        p = nl.ports.get(name)
        if p is None:
            continue
        rng = f"[{p.msb}:{p.lsb}] " if p.is_bus else ""
        lines.append(f"  {p.direction} {rng}{name};")
    for g in nl.gates:
        lines.append(f"  {g.type} {g.name} ( {g.out}, {', '.join(g.ins)} );")
    for ff in nl.dffs:
        conns = []
        for port, net in (("CK", ff.clk), ("RN", ff.rn), ("SN", ff.sn),
                          ("D", ff.d), ("Q", ff.q)):
            if net is not None:
                conns.append(f".{port}({net})")
        lines.append(f"  dff {ff.name} ( {', '.join(conns)} );")
    lines.append("endmodule")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def state_hash(nl):
    acc = 0
    for g in nl.gates:
        acc ^= hash((g.type, g.name, g.out, tuple(g.ins)))
    for ff in nl.dffs:
        acc ^= hash((ff.name, ff.clk, ff.d, ff.q, ff.rn, ff.sn))
    return acc


# ============================================================================
# independent engine over a harness-parsed snapshot (iterative, from scratch)
# ============================================================================
class Indep:
    def __init__(self, path):
        self.hn = H.parse_file(path)
        self.hn.index()
        hn = self.hn
        self.pi_bits = set()
        for base, w in hn.inputs.items():
            self.pi_bits |= {base} if w == 1 else {f"{base}[{i}]" for i in range(w)}
        self.po_bits = set()
        for base, w in hn.outputs.items():
            self.po_bits |= {base} if w == 1 else {f"{base}[{i}]" for i in range(w)}
        self.gates = hn.gates
        for g in self.gates:            # normalize field name (harness uses .kind)
            g.type = g.kind
        self.dffs = hn.dffs
        self.by_name = {g.name: g for g in hn.gates}
        self.dff_by_name = {f.name: f for f in hn.dffs}
        self.driver = dict(hn._driver)          # net -> Gate
        self.qs = {f.q for f in hn.dffs if f.q}
        self.ds = {f.d for f in hn.dffs if f.d}
        # loads: net -> [(kind, instname)] (gate ins + dff pins), from harness
        self.loads = {k: list(v) for k, v in hn._loads.items()}
        self._topo = None
        self._levels = None

    # ---- topo order over gates (Kahn) ----
    def topo(self):
        if self._topo is not None:
            return self._topo
        remaining, use = {}, {}
        gate_out = {g.out for g in self.gates}
        for g in self.gates:
            remaining[g.name] = sum(1 for i in g.ins if i in gate_out)
            for i in g.ins:
                use.setdefault(i, []).append(g)
        ready = deque(g for g in self.gates if remaining[g.name] == 0)
        order = []
        while ready:
            g = ready.popleft()
            order.append(g)
            for h in use.get(g.out, []):
                remaining[h.name] -= 1
                if remaining[h.name] == 0:
                    ready.append(h)
        self._topo = order
        return order

    # ---- longest-path levels; sources = every net without a gate driver ----
    def levels(self):
        if self._levels is None:
            dist = {}
            for g in self.topo():
                dist[g.out] = 1 + max((dist.get(i, 0) for i in g.ins), default=0)
            self._levels = dist
        return self._levels

    def seeded_depth(self, seeds):
        """Longest path counting gates, starting ONLY from `seeds`."""
        dist = {s: 0 for s in seeds}
        out = {}
        for g in self.topo():
            best = None
            for i in g.ins:
                v = out.get(i, dist.get(i))
                if v is not None:
                    best = v + 1 if best is None else max(best, v + 1)
            if best is not None:
                out[g.out] = best
        return out

    def cone_gates(self, net):
        seen_nets, seen_gates, stack = set(), set(), [net]
        while stack:
            n = stack.pop()
            if n in seen_nets:
                continue
            seen_nets.add(n)
            g = self.driver.get(n)
            if g is None:
                continue
            seen_gates.add(g.name)
            stack.extend(g.ins)
        return seen_gates

    def fanout(self, net):
        return len(self.loads.get(net, [])) + (1 if net in self.po_bits else 0)

    def load_names(self, net):
        seen, out = set(), []
        for kind, name in self.loads.get(net, []):
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def reachable_gates(self, net, include_dffs=False):
        """Instances transitively reachable forward from `net`.  Traversal
        continues only through gate outputs (combinational model); touched DFF
        instances are included when include_dffs=True (the convention both
        teams use for e.g. the transitive fanout of a clock net)."""
        seen_nets, seen_inst, stack = set(), set(), [net]
        while stack:
            n = stack.pop()
            if n in seen_nets:
                continue
            seen_nets.add(n)
            for kind, name in self.loads.get(n, []):
                if kind == "gate":
                    if name not in seen_inst:
                        seen_inst.add(name)
                        stack.append(self.by_name[name].out)
                elif include_dffs:
                    seen_inst.add(name)
        return seen_inst

    def path_exists(self, a, b, avoid=None):
        seen, stack = set(), [a]
        while stack:
            n = stack.pop()
            if n == b:
                return True
            if n in seen or n == avoid:
                continue
            seen.add(n)
            for kind, name in self.loads.get(n, []):
                if kind != "gate":
                    continue
                out = self.by_name[name].out
                if out not in seen and out != avoid:
                    stack.append(out)
        return False

    def count_paths(self, a, b):
        """Exact number of combinational paths a->b (big ints, DP on DAG)."""
        cnt = {a: 1}
        total = 0
        for g in self.topo():
            c = sum(cnt.get(i, 0) for i in g.ins)
            if c:
                cnt[g.out] = cnt.get(g.out, 0) + c
        return cnt.get(b, 0) if b != a else 1

    def support(self, nets):
        """Source bits (PI/Q/floating) feeding the cones of `nets`."""
        seen, sup, stack = set(), set(), list(nets)
        while stack:
            n = stack.pop()
            if n in seen or H.is_const(n):
                continue
            seen.add(n)
            g = self.driver.get(n)
            if g is None:
                sup.add(n)
                continue
            stack.extend(g.ins)
        return sup

    def eval_nets(self, assign, targets):
        val = dict(assign)
        memo = {}
        order = self.topo()
        for g in order:
            ins = []
            for i in g.ins:
                if H.is_const(i):
                    ins.append(1 if i in H.CONST1 else 0)
                else:
                    ins.append(val.get(i, 0))
            val[g.out] = H._apply_gate(g.type if hasattr(g, "type") else g.kind, ins)
        return {t: val.get(t, 0) for t in targets}


# ============================================================================
# golden-text extractors
# ============================================================================
def ints(s):
    # standalone numbers only — digits inside identifiers/bit-selects
    # (n2, n63[0], g13209) must not count as answer values
    return [int(x) for x in re.findall(r"(?<![\w\[\]])-?\d+(?![\w\]])", s)]


def kv(s):
    return {k.upper(): int(v) for k, v in re.findall(r"([A-Za-z_]\w*)\s*[:=]\s*(-?\d+)", s)}


def yesno(s):
    m = re.search(r"\b(yes|no)\b", s.lower())
    return m.group(1) if m else None


def name_list(s):
    """Instance names after the last ':' (the convention of our list answers)."""
    tail = s.rsplit(":", 1)[-1]
    return [t for t in re.findall(r"[A-Za-z_]\w*(?:\[\d+\])?", tail)
            if t.lower() not in {"none", "and", "or", "not"}]


# ============================================================================
# query families: (name, regex, checker(ind, m, golden) -> (ok, expected, note))
# ============================================================================
def fam_counts(ind, m, gold):
    got = kv(gold)
    exp = ind.hn.gate_type_counts()
    bad = {k: v for k, v in exp.items() if got.get(k) != v}
    return (not bad, exp, "" if not bad else f"golden={got}")


def fam_total(ind, m, gold):
    exp = ind.hn.total_gate_count()
    return (exp in ints(gold), exp, "")


def fam_pipo(ind, m, gold):
    bi, bo = len(ind.pi_bits), len(ind.po_bits)
    g = ints(gold)
    return (len(g) >= 2 and g[0] == bi and g[1] == bo, (bi, bo), "")


def fam_cone_gates(ind, m, gold):
    exp = len(ind.cone_gates(m.group(1)))
    return (exp in ints(gold), exp, "")


def fam_cone_depth(ind, m, gold):
    exp = ind.levels().get(m.group(1), 0)
    return (exp in ints(gold), exp, "")


def fam_global_depth(ind, m, gold):
    lv = ind.levels()
    sinks = ind.po_bits | ind.ds
    exp = max((lv.get(s, 0) for s in sinks), default=0)
    return (exp in ints(gold), exp, "")


def fam_pi2po_depth(ind, m, gold):
    d = ind.seeded_depth(ind.pi_bits)
    vals = [d[p] for p in ind.po_bits if p in d]
    exp = max(vals) if vals else 0
    return (exp in ints(gold), exp, "")


def fam_pi2d_depth(ind, m, gold):
    lv = ind.levels()
    exp = max((lv.get(d, 0) for d in ind.ds), default=0)
    return (exp in ints(gold), exp, "")


def fam_reg2reg(ind, m, gold):
    d = ind.seeded_depth(ind.qs)
    vals = [d[x] for x in ind.ds if x in d]
    exp = max(vals) if vals else -1
    ok = (exp in ints(gold)) if exp >= 0 else ("no register" in gold.lower())
    return (ok, exp, "")


def fam_depth_ab(ind, m, gold):
    d = ind.seeded_depth({m.group(1)}).get(m.group(2))
    gi = ints(gold)
    if d is None:
        return (not gi or 0 in gi, None, "no path")
    return (d in gi, d, "")


def fam_fanout(ind, m, gold):
    net = m.group(1)
    exp = ind.fanout(net)
    ok = exp in ints(gold)
    note = ""
    if ok and ":" in gold:              # only judge names when a list is given
        names = ind.load_names(net)
        gl = name_list(gold)
        if 0 < len(gl) < 200 and set(gl) != set(names[:200]):
            ok, note = False, f"name-set differs ({len(gl)} vs {len(names)})"
    return (ok, exp, note)


def fam_max_fanout_of(ind, m, gold):
    base = m.group(1)
    hn = ind.hn
    w = hn.inputs.get(base) or hn.outputs.get(base)
    bits = [base] if not w or w == 1 else [f"{base}[{i}]" for i in range(w)]
    if base in ind.driver or ind.loads.get(base):
        bits = [base]                    # scalar net that just looks like a base
    exp = max((ind.fanout(b) for b in bits), default=0)
    return (exp in ints(gold), exp, "max over bus bits")


def fam_connected_net(ind, m, gold):
    net = m.group(1)
    exp = set(ind.load_names(net))
    g = ind.driver.get(net)
    if g is not None:
        exp.add(g.name)                  # driver ∪ loads convention
    gl = set(name_list(gold))
    ok = gl == exp or (not exp and ("none" in gold.lower() or not gl))
    return (ok, sorted(exp), "driver ∪ loads")


def fam_depends(ind, m, gold):
    out, inp = m.group(1), m.group(2)
    sup = ind.support([out])
    if inp not in sup:
        return (yesno(gold) == "no", "no", "not in structural support")
    if len(sup) > 18:
        return (True, None, f"SKIP functional: support={len(sup)}>18")
    import itertools
    rest = sorted(sup - {inp})
    for combo in itertools.product((0, 1), repeat=len(rest)):
        base = dict(zip(rest, combo))
        r0 = ind.eval_nets({**base, inp: 0}, [out])[out]
        r1 = ind.eval_nets({**base, inp: 1}, [out])[out]
        if r0 != r1:
            return (yesno(gold) == "yes", "yes", "exhaustive")
    return (yesno(gold) == "no", "no", "exhaustive")


def fam_articulation(ind, m, gold):
    a, b = m.group(1), m.group(2)
    if not ind.path_exists(a, b):
        return ("no " in gold.lower() or "none" in gold.lower(), [], "no path at all")
    fwd = {a}
    stack = [a]
    while stack:
        n = stack.pop()
        for kind, name in ind.loads.get(n, []):
            if kind == "gate":
                o = ind.by_name[name].out
                if o not in fwd:
                    fwd.add(o)
                    stack.append(o)
    cand = {n for n in fwd if n not in (a, b)}
    arts = sorted(w for w in cand if not ind.path_exists(a, b, avoid=w))
    if not arts:
        return ("none" in gold.lower() or "no articulation" in gold.lower(), [], "")
    gl = set(name_list(gold))
    return (gl == set(arts) or len(arts) in ints(gold), arts, "")


def fam_list_type(ind, m, gold):
    t = m.group(1).lower()
    exp = sum(1 for g in ind.gates if g.type == t)
    if exp == 0:
        return ("no " in gold.lower() or "none" in gold.lower() or 0 in ints(gold), 0, "")
    if exp in ints(gold):
        return (True, exp, "")
    toks = [x for x in name_list(gold) if x in ind.by_name]
    return (len(toks) == exp or exp > 200, exp, "count via listed names")


def fam_successors(ind, m, gold):
    name = m.group(1)
    inst = ind.by_name.get(name) or ind.dff_by_name.get(name)
    if inst is None:
        return ("no instance" in gold.lower() or "no gate" in gold.lower(), None, "absent")
    out = inst.out if hasattr(inst, "out") else inst.q
    exp = ind.load_names(out)
    gl = name_list(gold)
    if not exp:
        return ("none" in gold.lower() or not gl, [], "")
    return (set(gl) == set(exp) if len(gl) < 200 else len(exp) >= 200, exp, "")


def fam_driven_by(ind, m, gold):
    g = ind.by_name.get(m.group(2))
    if g is None:
        return ("no gate" in gold.lower(), None, "absent")
    exp = ind.load_names(g.out)
    return (len(exp) in ints(gold) or set(name_list(gold)) == set(exp), exp, "")


def fam_connected_out(ind, m, gold):
    g = ind.by_name.get(m.group(2))
    if g is None:
        return ("no gate" in gold.lower(), None, "absent")
    exp = set(ind.load_names(g.out))
    return (set(name_list(gold)) == exp if exp else "none" in gold.lower() or not name_list(gold),
            sorted(exp), "")


def fam_highest_fanout_pi(ind, m, gold):
    best = max(ind.pi_bits, key=lambda p: ind.fanout(p))
    bv = ind.fanout(best)
    args = {p for p in ind.pi_bits if ind.fanout(p) == bv}
    gl = ints(gold)
    ok = bv in gl and any(a in gold for a in args)
    return (ok, (sorted(args)[0], bv), "")


def fam_reachable(ind, m, gold):
    a = len(ind.reachable_gates(m.group(2)))
    b = len(ind.reachable_gates(m.group(2), include_dffs=True))
    gi = ints(gold)
    return (a in gi or b in gi, (a, b), "gates-only / incl-DFF")


def fam_tfanin(ind, m, gold):
    exp = len(ind.cone_gates(m.group(1)))
    return (exp in ints(gold) or set(name_list(gold)) == ind.cone_gates(m.group(1)), exp, "")


def fam_tfanout(ind, m, gold):
    a = ind.reachable_gates(m.group(1))
    b = ind.reachable_gates(m.group(1), include_dffs=True)
    gi = ints(gold)
    ok = len(a) in gi or len(b) in gi or set(name_list(gold)) in (a, b)
    return (ok, (len(a), len(b)), "gates-only / incl-DFF")


def fam_shared(ind, m, gold):
    exp = ind.cone_gates(m.group(1)) & ind.cone_gates(m.group(2))
    if not exp:
        return ("none" in gold.lower() or "no gates" in gold.lower() or "0" in ints(gold) == [0], set(), "")
    return (set(name_list(gold)) == exp or len(exp) in ints(gold), sorted(exp), "")


def fam_path_plain(ind, m, gold):
    exp = "yes" if ind.path_exists(m.group(1), m.group(2)) else "no"
    g = yesno(gold) or ("yes" if "exists" in gold.lower() else None)
    return (g == exp, exp, "")


def fam_path_avoid(ind, m, gold):
    a, b, c = m.group(1), m.group(2), m.group(3)
    exp = "yes" if ind.path_exists(a, b, avoid=c) else "no"
    g = yesno(gold) or ("yes" if "exists" in gold.lower() else None)
    return (g == exp, exp, "")


def fam_dominator(ind, m, gold):
    a, b, c = m.group(1), m.group(2), m.group(3)
    if not ind.path_exists(a, b):
        exp = "no"        # no path at all -> "every path passes" vacuous; engine says?
        return (yesno(gold) is not None, exp, "vacuous — review manually")
    exp = "no" if ind.path_exists(a, b, avoid=c) else "yes"
    return (yesno(gold) == exp, exp, "")


def fam_count_paths(ind, m, gold):
    nets = [g for g in m.groups() if g is not None][-2:]
    exp = ind.count_paths(nets[0], nets[1])
    if exp == 0:
        return ("no " in gold.lower() or 0 in ints(gold), 0, "")
    return (exp in ints(gold), exp, "")


def fam_cone_type_counts(ind, m, gold):
    from collections import Counter
    c = Counter(ind.by_name[n].type.upper() for n in ind.cone_gates(m.group(1)))
    got = kv(gold)
    bad = {k: v for k, v in got.items() if k != "TOTAL" and c.get(k, 0) != v}
    return (not bad, dict(c), "" if not bad else f"mismatch={bad}")


def fam_port_widths(ind, m, gold):
    which = "output" if "output" in m.group(0).lower() else "input"
    src = ind.hn.outputs if which == "output" else ind.hn.inputs
    ok = all(f"{name} [{w}-bit]" in gold for name, w in src.items()) and \
        gold.count("-bit]") == len(src)
    return (ok, {n: w for n, w in src.items()}, "")


def fam_floating_check(ind, m, gold):
    und_in = sorted(p for p in ind.pi_bits
                    if not ind.loads.get(p) and p not in ind.po_bits)
    unc_out = sorted(p for p in ind.po_bits if p not in ind.driver
                     and p not in ind.pi_bits and p not in ind.qs)
    responsive = bool(re.search(r"floating|undriven|unconnected|unloaded",
                                gold, re.I))
    exp = {"undriven_inputs": len(und_in), "unconnected_outputs": len(unc_out)}
    if not responsive:
        return (False, exp, "MISROUTE: answer does not address floating/unconnected "
                            "(h_check_dangling delegates to the removal transform)")
    return (True, exp, "responsiveness only; definition not strictly judged")


def fam_symmetric(ind, m, gold):
    net, a, b = m.group(1), m.group(2), m.group(3)
    sup = ind.support([net])
    if len(sup) > 18:
        return (True, None, f"SKIP functional: support={len(sup)}>18")
    if a not in sup and b not in sup:
        exp = "yes"     # function depends on neither -> trivially symmetric
    else:
        import itertools
        rest = sorted(sup - {a, b})
        exp = "yes"
        for combo in itertools.product((0, 1), repeat=len(rest)):
            base = dict(zip(rest, combo))
            for va, vb in ((0, 1), (1, 0)):
                r1 = ind.eval_nets({**base, a: va, b: vb}, [net])[net]
                r2 = ind.eval_nets({**base, a: vb, b: va}, [net])[net]
                if r1 != r2:
                    exp = "no"
                    break
            if exp == "no":
                break
    return (yesno(gold) == exp, exp, "exhaustive")


def fam_on_maxpath(ind, m, gold):
    g = ind.by_name.get(m.group(1))
    if g is None:
        return ("no gate" in gold.lower(), None, "absent")
    lv = ind.levels()
    back = {}
    for gg in reversed(ind.topo()):
        best = 0
        for kind, name in ind.loads.get(gg.out, []):
            if kind == "gate":
                best = max(best, 1 + back.get(ind.by_name[name].out, 0))
        back[gg.out] = best
    sinks = ind.po_bits | ind.ds
    gmax = max((lv.get(s, 0) for s in sinks), default=0)
    exp = "yes" if lv.get(g.out, 0) + back.get(g.out, 0) == gmax else "no"
    return (yesno(gold) == exp, exp, "")


def fam_skip_semantic(ind, m, gold):
    return (True, None, "SKIP semantic/listing — not machine-checkable here")


def fam_cut(ind, m, gold):
    w = m.group(m.lastindex)
    # cut = removing net w disconnects some previously PI-reachable PO
    def po_reach(avoid=None):
        seen, stack = set(), [p for p in ind.pi_bits if p != avoid]
        reached = set()
        while stack:
            n = stack.pop()
            if n in seen or n == avoid:
                continue
            seen.add(n)
            if n in ind.po_bits:
                reached.add(n)
            for kind, name in ind.loads.get(n, []):
                if kind == "gate":
                    out = ind.by_name[name].out
                    if out not in seen and out != avoid:
                        stack.append(out)
        return reached
    exp = "yes" if (po_reach() - po_reach(avoid=w)) else "no"
    return (yesno(gold) == exp, exp, "definition-sensitive")


def fam_zero_hop(ind, m, gold):
    exp = sorted(ind.pi_bits & ind.po_bits)
    if not exp:
        return ("no" in gold.lower() or "none" in gold.lower() or 0 in ints(gold), [], "")
    return (len(exp) in ints(gold) or all(e in gold for e in exp), exp, "")


def fam_type_count(ind, m, gold):
    t = m.group(1).upper()
    syn = {"INVERTER": "NOT", "INV": "NOT"}
    t = syn.get(t, t)
    exp = ind.hn.gate_type_counts().get(t)
    if exp is None:
        return (True, None, f"unknown type {t} — skip")
    return (exp in ints(gold), exp, "")


def fam_const1_gates(ind, m, gold):
    exp = [g.name for g in ind.gates if "1'b1" in g.ins]
    ok = (len(exp) in ints(gold)) or (not exp and ("none" in gold.lower() or "0" in gold))
    return (ok, len(exp), "structural scan; A21.1 functional constants not covered")


def fam_const_report(ind, m, gold):
    t = m.group(1).lower()
    cv = "1'b0" if "0" in m.group(0) else ("1'b1" if "1" in m.group(0) else None)
    hits = [g.name for g in ind.gates
            if g.type == t and any(H.is_const(i) and (cv is None or i in (cv, cv.replace("b", "h"))) for i in g.ins)]
    if not hits:
        ok = "no " in gold.lower() or "none" in gold.lower() or 0 in ints(gold)
    else:
        ok = len(hits) in ints(gold) or set(name_list(gold)) >= set(hits)
    return (ok, len(hits), "structural scan; A21.1 functional constants not covered")


def fam_gate_info(ind, m, gold):
    name = m.group(1)
    g = ind.by_name.get(name)
    if g is None:
        ff = ind.dff_by_name.get(name)
        if ff is None:
            return ("no " in gold.lower(), None, "absent")
        ok = "dff" in gold.lower() and (ff.q or "") in gold
        return (ok, ("DFF", ff.q), "")
    ok = g.type.upper() in gold.upper() and g.out in gold and all(i in gold for i in g.ins if not H.is_const(i))
    return (ok, (g.type.upper(), g.out, tuple(g.ins)), "")


def fam_ffs_clock(ind, m, gold):
    net = m.group(1)
    exp = [f.name for f in ind.dffs if f.ck == net]
    return (len(exp) in ints(gold) or set(name_list(gold)) == set(exp), len(exp), "")


def fam_depth_gt(ind, m, gold):
    k = int(m.group(1))
    lv = ind.levels()
    exp = sum(1 for p in ind.po_bits if max((lv.get(s, 0) for s in [p]), default=0) > k)
    return (exp in ints(gold), exp, "")


def fam_deepest_output(ind, m, gold):
    lv = ind.levels()
    if "deep" in m.group(0).lower():
        score = {p: lv.get(p, 0) for p in ind.po_bits}
    else:
        score = {p: len(ind.cone_gates(p)) for p in ind.po_bits}
    best = max(score.values(), default=0)
    args = {p for p, v in score.items() if v == best}
    ok = best in ints(gold) and any(a in gold for a in args)
    return (ok, (sorted(args)[0] if args else None, best), "")


def fam_const_out(ind, m, gold):
    net = m.group(1)
    sup = ind.support([net])
    if len(sup) > 18:
        return (True, None, f"SKIP functional: support={len(sup)}>18")
    import itertools
    sup = sorted(sup)
    vals = set()
    for combo in itertools.product((0, 1), repeat=len(sup)):
        vals.add(ind.eval_nets(dict(zip(sup, combo)), [net])[net])
        if len(vals) > 1:
            break
    exp = "yes" if vals == {0} else "no"
    return (yesno(gold) == exp, exp, "exhaustive")


def fam_sig_equiv(ind, m, gold):
    a, b = m.group(m.lastindex - 1), m.group(m.lastindex)
    sup = ind.support([a, b])
    if len(sup) > 18:
        return (True, None, f"SKIP functional: support={len(sup)}>18")
    import itertools
    sup = sorted(sup)
    for combo in itertools.product((0, 1), repeat=len(sup)):
        r = ind.eval_nets(dict(zip(sup, combo)), [a, b])
        if r[a] != r[b]:
            return (yesno(gold) == "no", "no", "exhaustive")
    return (yesno(gold) == "yes", "yes", "exhaustive")


FAMILIES = [
    ("cone_types", re.compile(r"number of each gate type in the cone of (%s)|(?:primitive-type|gate-type) distribution within the (?:fanin )?cone of (?:output )?(%s)" % (NET, NET), re.I),
     lambda ind, m, g: fam_cone_type_counts(ind, _first_group(m), g)),
    ("counts", re.compile(r"count all the gates|broken down by gate type|breakdown of gate primitives", re.I), fam_counts),
    ("port_widths", re.compile(r"list all (?:the )?primary (inputs?|outputs?).*bit widths?", re.I), fam_port_widths),
    ("floating", re.compile(r"(?:check|does).*(?:floating|undriven).*(?:input|signal)|unconnected output ports|unloaded output", re.I), fam_floating_check),
    ("symmetric", re.compile(r"function at (?:output )?(%s) is symmetric with respect to (?:inputs )?(%s) and (%s)" % (NET, NET, NET), re.I), fam_symmetric),
    ("total", re.compile(r"total gate count|how many gates does (the|this) design contain", re.I), fam_total),
    ("pipo", re.compile(r"(number of|how many) primary inputs? and", re.I), fam_pipo),
    ("successors", re.compile(r"immediate successors of (?:gate )?(%s)" % NET, re.I), fam_successors),
    ("cone_gates", re.compile(r"gates? (?:are |belong )?(?:to |in )?the (?:fanin |logic )?cone of (?:primary output |output )?(%s)|compute the fanin (?:logic )?cone of (?:output )?(%s)" % (NET, NET), re.I),
     lambda ind, m, g: fam_cone_gates(ind, _first_group(m), g)),
    ("tfanin", re.compile(r"transitive fanin (?:cone )?of (?:output )?(%s)" % NET, re.I), fam_tfanin),
    ("tfanout", re.compile(r"transitive fanout (?:cone )?of (?:input |primary input )?(%s)" % NET, re.I), fam_tfanout),
    ("path_avoid", re.compile(r"path.*?from (?:input |primary input )?(%s) to (?:output |primary output )?(%s).*?(?:does not traverse|bypass\w*|avoid\w*|skips?|without)\s+(?:node\s+|wire\s+)?(%s)" % (NET, NET, NET), re.I), fam_path_avoid),
    ("dominator", re.compile(r"does every path from (?:input )?(%s) to (?:output )?(%s) pass through (?:gate )?(%s)" % (NET, NET, NET), re.I), fam_dominator),
    ("count_paths", re.compile(
        r"(?:complete enumeration|enumerate all|list every|list all|find all)[^.]*?paths?[^.]*?"
        r"(?:between|from)\s+(?:primary input )?(%s)\s+(?:and|to)\s+(?:primary output )?(%s)"
        r"|paths? originating at (?:primary input )?(%s) and terminating at (?:primary output )?(%s)"
        % (NET, NET, NET, NET), re.I), fam_count_paths),
    ("on_maxpath", re.compile(r"gate (%s) lies? on any maximum-depth path" % NET, re.I), fam_on_maxpath),
    ("bool_expr", re.compile(r"logic expression|boolean function|boolean equation|express the", re.I), fam_skip_semantic),
    ("r2r_list", re.compile(r"register-to-register (?:combinational )?paths", re.I), fam_skip_semantic),
    ("enable_hold", re.compile(r"enable or hold", re.I), fam_skip_semantic),
    ("path_plain", re.compile(r"(?:is there|does|whether).*?path.*?from (?:input |primary input )?(%s) to (?:output |primary output )?(%s)" % (NET, NET), re.I), fam_path_plain),
    ("depth_ab", re.compile(r"(?:max\w*|longest|critical).*?(?:depth|path).*?(?:from|between)\s+(?:input )?(%s)\s+(?:to|and)\s+(?:output )?(%s)" % (NET, NET), re.I), fam_depth_ab),
    ("pi2d", re.compile(r"depth from any primary input to any (?:DFF )?D-?pin", re.I), fam_pi2d_depth),
    ("pi2po", re.compile(r"depth.*primary input to any primary output", re.I), fam_pi2po_depth),
    ("reg2reg_d", re.compile(r"depth on any register-to-register", re.I), fam_reg2reg),
    ("global_d", re.compile(r"max\w* (?:combinational )?(?:logic )?depth (?:in|of) (?:the|this) design|deepest combinational path anywhere|(?:depth|path)[^.]*entire design|entire design[^.]*(?:depth|path)", re.I), fam_global_depth),
    ("cone_depth", re.compile(r"(?:max\w*|maximum) (?:logic )?depth of the (?:fanin )?cone (?:rooted at |of )(?:output )?(%s)|depth of the cone of (%s)" % (NET, NET), re.I),
     lambda ind, m, g: fam_cone_depth(ind, _first_group(m), g)),
    ("depth_gt", re.compile(r"how many (?:primary )?outputs? (?:bits? )?have a (?:fanin )?logic depth (?:greater than|exceeding) (\d+)", re.I), fam_depth_gt),
    ("deepest", re.compile(r"which (?:primary )?output (?:bit )?has the (deepest|largest) (?:fanin )?(?:logic )?cone", re.I), fam_deepest_output),
    ("max_fanout", re.compile(r"max\w* fanout of (%s) now" % NET, re.I), fam_max_fanout_of),
    ("conn_net", re.compile(r"connect to the (?:renamed )?signal (%s)" % NET, re.I), fam_connected_net),
    ("depends", re.compile(r"does (?:primary )?output (%s) (?:functionally )?depend on (?:primary )?input (%s)" % (NET, NET), re.I), fam_depends),
    ("articulation", re.compile(r"articulation points?.*between (%s) and (%s)" % (NET, NET), re.I), fam_articulation),
    ("list_type", re.compile(r"list all (\w+) gates in this design", re.I), fam_list_type),
    ("fanout", re.compile(r"fanout of (?:primary input |signal |net )?(%s)" % NET, re.I), fam_fanout),
    ("highest_pi", re.compile(r"highest fanout", re.I), fam_highest_fanout_pi),
    ("driven_by", re.compile(r"(number of gates driven by|gates? driven by) (%s)" % NET, re.I), fam_driven_by),
    ("connected", re.compile(r"(gates? |every gate |cells? )connected to the output (?:net )?of (?:gate )?(%s)" % NET, re.I), fam_connected_out),
    ("reachable", re.compile(r"(gates? reachable from|reachable from) (%s)" % NET, re.I), fam_reachable),
    ("shared", re.compile(r"(?:gates )?(?:shared between|in both) the fanin cones? of (%s) and (?:of )?(%s)" % (NET, NET), re.I), fam_shared),
    ("cut", re.compile(r"(?:whether |is )wire (%s) is a (?:structural )?cut" % NET, re.I), fam_cut),
    ("zero_hop", re.compile(r"zero-hop|length 0|direct wire connections from", re.I), fam_zero_hop),
    ("const_report", re.compile(r"report any (\w+) gates? with (?:a )?constant", re.I), fam_const_report),
    ("const1", re.compile(r"tied to 1'b1", re.I), fam_const1_gates),
    ("gate_info", re.compile(r"what (?:type of gate|cell type) is (?:gate )?(%s)" % NET, re.I), fam_gate_info),
    ("ffs_clock", re.compile(r"flip-?flops driven by clock (?:signal )?(%s)" % NET, re.I), fam_ffs_clock),
    ("const_out", re.compile(r"output (%s).*?(?:always 0|constant-0 function)" % NET, re.I), fam_const_out),
    ("sig_equiv", re.compile(r"(?:signals? |nets? )?(%s) and (%s) (?:are )?(?:logically|computationally|functionally) equivalent" % (NET, NET), re.I), fam_sig_equiv),
    ("type_count", re.compile(r"how many (\w+) (?:gates?|cells?) are (?:currently |now )?in", re.I), fam_type_count),
]


def _first_group(m):
    class M:  # tiny adapter: first non-None group as group(1)
        def group(self, i):
            for g in m.groups():
                if g is not None:
                    return g
    return M()


SETUP_RX = re.compile(r"beginning of|case name is|\bload\b|\bread\b.*design|write out|save the|"
                      r"output the design|write the (?:current|resulting|modified)", re.I)
MUTATOR_RX = re.compile(r"insert|remove|delete|excise|replace|convert|rebuild|reconstruct|"
                        r"restructure|rebalance|simplify|collapse|dissolve|merge|rename|"
                        r"relabel|reassign|buffer|minimi[sz]e|optimi[sz]e|reduce|shorten|"
                        r"apply|fold|substitute|transform|translate|prune|sweep|eliminate all|"
                        r"limit the fanout|add buffer|remap|trim", re.I)
VERIFY_RX = re.compile(r"verify|assert|confirm|equivalen", re.I)
DELTA_RX = re.compile(r"how many .*(were|was) .*(removed|added|inserted|eliminated|merged|"
                      r"collapsed|pruned|excised|rewritten|found)|previous (step|buffering)|"
                      r"how many .* (did|were) (the|this)", re.I)


_BLOCK = re.compile(r"\breport\b|\bhow many\b|\bwhat\b|\bwhich\b|\blist\b", re.I)


def classify(line):
    if SETUP_RX.search(line):
        return "SETUP"
    if DELTA_RX.search(line):
        return "DELTA"
    if VERIFY_RX.search(line) and not _BLOCK.search(line):
        return "VERIFY"
    if MUTATOR_RX.search(line) and not _BLOCK.search(line):
        return "MUTATOR"
    for name, rx, fn in FAMILIES:
        m = rx.search(line)
        if m:
            return ("Q", name, m, fn)
    return "UNMATCHED"


def norm(s):
    return " ".join((s or "").split())


# ============================================================================
def run_case(case, cfg, results):
    case_dir = os.path.join(ROOT, "testcase", case)
    gold_path = os.path.join(ROOT, "evaluator", "golden", f"{case}.txt")
    if not (os.path.isfile(os.path.join(case_dir, "prompt.txt")) and os.path.isfile(gold_path)):
        return
    prompts = [l.rstrip("\n") for l in open(os.path.join(case_dir, "prompt.txt")) if l.strip()]
    goldens = open(gold_path).read().split("\n")

    sb = tempfile.mkdtemp(prefix=f"gc_{case}_")
    shutil.copytree(case_dir, os.path.join(sb, "testcase", case))
    cwd = os.getcwd()
    os.chdir(sb)
    snap_dir = os.path.join(OUT_DIR, case)
    os.makedirs(snap_dir, exist_ok=True)
    jobs = []
    try:
        agent = Agent(cfg)
        if getattr(agent, "fallback", None) is not None:      # never touch the LLM
            agent.fallback.translate = lambda line: None
        version, snapped, h = 0, {}, None
        for i, line in enumerate(prompts, start=1):
            cls = classify(line)
            gold = goldens[i - 1] if i - 1 < len(goldens) else ""
            snap = None
            if isinstance(cls, tuple) and agent.state.current is not None:
                if version not in snapped:
                    p = os.path.join(snap_dir, f"v{version}.v")
                    emit_snapshot(agent.state.current, p)
                    snapped[version] = p
                snap = snapped[version]
            live = agent.handle(line, i)
            live_ok = norm(live) == norm(gold)
            if agent.state.current is not None:
                nh = state_hash(agent.state.current)
                if nh != h:
                    h, version = nh, version + 1
            if isinstance(cls, tuple):
                _, fam, m, fn = cls
                jobs.append((i, fam, m, fn, snap, gold, live_ok, line))
            else:
                results.append({"case": case, "id": i, "family": cls, "status": "SKIP",
                                "note": cls, "line": line[:70]})
    finally:
        os.chdir(cwd)
        shutil.rmtree(sb, ignore_errors=True)

    cache = {}
    for i, fam, m, fn, snap, gold, live_ok, line in jobs:
        if snap is None:
            results.append({"case": case, "id": i, "family": fam, "status": "SKIP",
                            "note": "no design state", "line": line[:70]})
            continue
        if snap not in cache:
            cache[snap] = Indep(snap)
        try:
            ok, exp, note = fn(cache[snap], m, gold)
        except Exception as e:
            results.append({"case": case, "id": i, "family": fam, "status": "ERROR",
                            "note": f"{type(e).__name__}: {e}", "line": line[:70]})
            continue
        if note.startswith("SKIP"):
            status = "SKIP"
        elif not live_ok:
            status = "NONDET"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
        results.append({"case": case, "id": i, "family": fam, "status": status,
                        "expected": repr(exp)[:90], "golden": gold[:90],
                        "note": note, "line": line[:70]})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_config(os.path.join(ROOT, "configs", "api_key.yaml"))
    cases = sys.argv[1:] or sorted(
        d for d in os.listdir(os.path.join(ROOT, "testcase"))
        if re.fullmatch(r"test\d+", d)
        and os.path.isfile(os.path.join(ROOT, "evaluator", "golden", f"{d}.txt")))
    results = []
    for c in cases:
        print(f"[{c}] ...", flush=True)
        try:
            run_case(c, cfg, results)
        except Exception as e:
            results.append({"case": c, "id": 0, "family": "-", "status": "ERROR",
                            "note": f"case-level {type(e).__name__}: {e}", "line": ""})
    with open(os.path.join(OUT_DIR, "report.json"), "w") as fh:
        json.dump(results, fh, indent=1)

    counts = {}
    lines = []
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] in ("FAIL", "NONDET", "ERROR"):
            lines.append(f"{r['case']} r{r['id']:<3} {r['family']:<12} {r['status']:<7} "
                         f"exp={r.get('expected','')} golden={r.get('golden','')} {r.get('note','')}")
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as fh:
        fh.write("\n".join(lines) + f"\n\nsummary: {counts}\n")
    print("\n".join(lines[-40:]))
    print(f"\nsummary: {counts}  -> {OUT_DIR}/report.txt")


if __name__ == "__main__":
    main()
