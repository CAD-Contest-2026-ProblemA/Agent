"""Template-based reverse engineering of word-level functions (ALS-style).

Ported methodology from ALS_Final_Project's ``detect.py``: *guess* what
well-known function a block of logic computes, *verify* the guess exactly, then
*rebuild* it from a depth-optimal template.  Adapted for gate-level netlists
(where exhaustive truth tables are unavailable):

1. **Sample**: simulate the whole combinational core on a few hundred random
   vectors (bit-packed, one pass in topo order).
2. **Group**: view the register-transfer structure as words — PI bus ports and
   DFF register banks (grouped by their Q bus) are the operand words; PO bus
   ports and each register bank's D vector are the target words.
3. **Guess**: for every target word, try a bank of candidate functions
   (add/sub/inc, multiply, bitwise, comparisons, ...) over operand words drawn
   from its structural support.  A candidate survives only if it agrees with
   the sampled behaviour on every sample (false-positive odds ~2^-samples).
4. **Verify**: each surviving guess is proved exact with an ABC ``cec`` of the
   old fan-in cone vs the freshly built template cone (never trust a sample).
5. **Rebuild**: verified targets are re-driven by depth-optimal structures
   (Sklansky prefix adders, Wallace-tree multipliers, balanced comparator
   trees) emitted in generic gates; the surrounding optimiser then maps them
   into whatever gate basis the request demands.

Everything here is *opportunistic*: no match simply means no template
candidate, and the caller falls back to the generic ABC portfolio.  A wrong
guess can never ship — step 4 discards it, and the final winner is cec-checked
against the whole design again by the caller.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..netlist.ir import Gate, Netlist, is_const
from ..netlist.blif_export import _TT
from ..analysis import graph
from ..transform.base import Emitter
from ..equiv import abc_bridge

# guard rails: skip template detection entirely on huge designs
MAX_GATES = 120_000
MAX_TARGET_BUSES = 96
MAX_CONE_GATES = 40_000       # per-target cone bound for support DFS
SIM_WORDS = 8                 # 8 x 64 = 512 random samples
_M64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# 1. sampling simulation (bit-packed, whole comb core in one pass)
# ---------------------------------------------------------------------------

def simulate(nl: Netlist, words: int = SIM_WORDS, seed: int = 2026,
             ) -> Optional[Dict[str, List[int]]]:
    """Random-vector simulation of the combinational core.

    Returns net -> list of ``words`` 64-bit packed sample words, or None if the
    netlist contains an unexpected gate type / combinational cycle.
    """
    rng = random.Random(seed)
    val: Dict[str, List[int]] = {
        "1'b0": [0] * words,
        "1'b1": [_M64] * words,
    }
    for src in graph.comb_sources(nl):
        if src not in val:
            val[src] = [rng.getrandbits(64) for _ in range(words)]

    order = graph.topo_nets(nl)
    nl.driver("__force_build__")
    driver = nl._driver
    for net in order:
        if net in val:
            continue
        drv = driver.get(net)
        if drv is None or drv[0] != "gate":
            # undriven net: treat as a free input (matches X-safe behaviour)
            val[net] = [rng.getrandbits(64) for _ in range(words)]
            continue
        g = drv[1]
        ins = [val.get(i) for i in g.ins]
        if any(v is None for v in ins):
            return None            # cycle / ordering failure: bail out safely
        a = ins[0]
        b = ins[1] if len(ins) > 1 else None
        t = g.type
        if t == "and":
            val[net] = [x & y for x, y in zip(a, b)]
        elif t == "or":
            val[net] = [x | y for x, y in zip(a, b)]
        elif t == "nand":
            val[net] = [~(x & y) & _M64 for x, y in zip(a, b)]
        elif t == "nor":
            val[net] = [~(x | y) & _M64 for x, y in zip(a, b)]
        elif t == "xor":
            val[net] = [x ^ y for x, y in zip(a, b)]
        elif t == "xnor":
            val[net] = [~(x ^ y) & _M64 for x, y in zip(a, b)]
        elif t == "not":
            val[net] = [~x & _M64 for x in a]
        elif t == "buf":
            val[net] = list(a)
        else:
            return None
    return val


def _word_samples(val: Dict[str, List[int]], bits: Sequence[str],
                  words: int = SIM_WORDS) -> Optional[List[int]]:
    """Assemble per-sample integers from LSB-first bit nets."""
    cols = []
    for b in bits:
        v = val.get(b)
        if v is None:
            return None
        cols.append(v)
    out: List[int] = []
    for w in range(words):
        for s in range(64):
            x = 0
            for i, col in enumerate(cols):
                x |= ((col[w] >> s) & 1) << i
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# 2. word (bus) discovery
# ---------------------------------------------------------------------------

class Word:
    """A named word: LSB-first bit nets."""
    __slots__ = ("name", "bits", "kind")

    def __init__(self, name: str, bits: List[str], kind: str):
        self.name = name
        self.bits = bits          # LSB first
        self.kind = kind          # pi | q | po | d

    @property
    def width(self) -> int:
        return len(self.bits)

    def __repr__(self):
        return f"<{self.kind} {self.name}[{self.width}]>"


_BIT_RE = re.compile(r"^(.*)\[(\d+)\]$")


def _group_bits(nets: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """Group ``name[i]`` nets into complete 0..w-1 buses (LSB first)."""
    by_base: Dict[str, Dict[int, str]] = {}
    for n in nets:
        m = _BIT_RE.match(n)
        if m:
            by_base.setdefault(m.group(1), {})[int(m.group(2))] = n
    out = []
    for base, idx in by_base.items():
        w = len(idx)
        if w >= 2 and set(idx) == set(range(w)):
            out.append((base, [idx[i] for i in range(w)]))
    return out


def operand_words(nl: Netlist) -> Tuple[List[Word], List[str]]:
    """Candidate operand words (PI buses + register banks by Q bus) and scalar
    nets (1-bit PIs/Qs, usable as carry-ins)."""
    words: List[Word] = []
    pi = sorted(nl.pi)
    for base, bits in _group_bits(pi):
        words.append(Word(base, bits, "pi"))
    q_nets = sorted({ff.q for ff in nl.dffs})
    for base, bits in _group_bits(q_nets):
        words.append(Word(base, bits, "q"))
    grouped = {b for w in words for b in w.bits}
    scalars = [n for n in pi + q_nets if n not in grouped]
    return words, scalars


def target_words(nl: Netlist) -> List[Word]:
    """Target words to reverse-engineer: PO bus ports driven by gates, and each
    register bank's D vector (ordered by its Q bus index)."""
    nl.driver("__force_build__")
    driver = nl._driver

    def all_gate_driven(bits: Sequence[str]) -> bool:
        return all((driver.get(b) or ("undriven",))[0] == "gate" for b in bits)

    out: List[Word] = []
    for base, bits in _group_bits(sorted(nl.po)):
        if all_gate_driven(bits):
            out.append(Word(base, bits, "po"))

    # register banks: Q bus name -> D nets in bit order
    d_by_q: Dict[str, Dict[int, str]] = {}
    for ff in nl.dffs:
        m = _BIT_RE.match(ff.q)
        if m:
            d_by_q.setdefault(m.group(1), {})[int(m.group(2))] = ff.d
    for base, idx in sorted(d_by_q.items()):
        w = len(idx)
        if w >= 2 and set(idx) == set(range(w)):
            bits = [idx[i] for i in range(w)]
            if all_gate_driven(bits) and len(set(bits)) == w:
                out.append(Word(base + "$D", bits, "d"))
    return out


def _support_sources(nl: Netlist, bits: Sequence[str],
                     max_gates: int = MAX_CONE_GATES) -> Optional[Set[str]]:
    """Source nets (PI / Q / const / undriven) feeding the cone of ``bits``.
    None if the cone is too large to bother."""
    nl.driver("__force_build__")
    driver = nl._driver
    seen: Set[str] = set()
    srcs: Set[str] = set()
    stack = list(bits)
    gates = 0
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        drv = driver.get(n)
        if drv is not None and drv[0] == "gate":
            gates += 1
            if gates > max_gates:
                return None
            for i in drv[1].ins:
                if i not in seen:
                    stack.append(i)
        else:
            srcs.add(n)
    return srcs


# ---------------------------------------------------------------------------
# 3. the function bank (guess on samples)
# ---------------------------------------------------------------------------
# Each entry: name, kind ('unary'|'binary'|'binary_cin'|'mul'|'cmp'),
# model(a, b, cin, w) -> int   (word-level reference semantics, mod 2^w)

def _mask(w: int) -> int:
    return (1 << w) - 1


FUNCS_BINARY = {
    "add": lambda a, b, w: (a + b) & _mask(w),
    "addc1": lambda a, b, w: (a + b + 1) & _mask(w),
    "sub": lambda a, b, w: (a - b) & _mask(w),
    "rsub": lambda a, b, w: (b - a) & _mask(w),
    "band": lambda a, b, w: (a & b) & _mask(w),
    "bor": lambda a, b, w: (a | b) & _mask(w),
    "bxor": lambda a, b, w: (a ^ b) & _mask(w),
    "bnand": lambda a, b, w: (~(a & b)) & _mask(w),
    "bnor": lambda a, b, w: (~(a | b)) & _mask(w),
    "bxnor": lambda a, b, w: (~(a ^ b)) & _mask(w),
}

FUNCS_UNARY = {
    "inc": lambda a, w: (a + 1) & _mask(w),
    "dec": lambda a, w: (a - 1) & _mask(w),
    "neg": lambda a, w: (-a) & _mask(w),
    "bnot": lambda a, w: (~a) & _mask(w),
}

FUNCS_CMP = {
    "eq": lambda a, b: int(a == b),
    "ne": lambda a, b: int(a != b),
    "ult": lambda a, b: int(a < b),
    "ule": lambda a, b: int(a <= b),
    "ugt": lambda a, b: int(a > b),
    "uge": lambda a, b: int(a >= b),
}


class Match:
    __slots__ = ("target", "func", "ops", "cin", "extra", "note")

    def __init__(self, target: Word, func: str, ops: List[Word],
                 cin: Optional[str] = None, extra=None):
        self.target = target
        self.func = func
        self.ops = ops
        self.cin = cin          # scalar operand: carry-in / select / shift...
        self.extra = extra      # per-bit wiring plan for "perm"
        self.note = (f"{target.name} = {func}(" +
                     ", ".join(o.name for o in ops) +
                     (f", s={cin}" if cin else "") + ")")

    def __repr__(self):
        return f"<Match {self.note}>"


def _match_projection(tgt: Word, val: Dict[str, List[int]],
                      sup: Set[str]) -> Optional[Match]:
    """Pure rewiring: every target bit equals a support-source bit, its
    complement, or a constant.  Subsumes constant shifts, rotates, byteswap,
    bitreverse, shift-register D vectors, zero/sign fills — depth <= 1."""
    zero = tuple([0] * SIM_WORDS)
    ones = tuple([_M64] * SIM_WORDS)
    src_cols: Dict[tuple, str] = {}
    inv_cols: Dict[tuple, str] = {}
    for s in sup:
        v = val.get(s)
        if v is None or is_const(s):
            continue
        src_cols.setdefault(tuple(v), s)
        inv_cols.setdefault(tuple(~x & _M64 for x in v), s)
    plan = []
    for b in tgt.bits:
        v = val.get(b)
        if v is None:
            return None
        col = tuple(v)
        if col == zero:
            plan.append(("const", "1'b0"))
        elif col == ones:
            plan.append(("const", "1'b1"))
        elif col in src_cols:
            plan.append(("wire", src_cols[col], False))
        elif col in inv_cols:
            plan.append(("wire", inv_cols[col], True))
        else:
            return None
    return Match(tgt, "perm", [], extra=plan)


def detect(nl: Netlist, val: Dict[str, List[int]]) -> List[Match]:
    """Sample-match every target word against the function bank."""
    ops, scalars = operand_words(nl)
    targets = target_words(nl)
    if not ops or not targets or len(targets) > MAX_TARGET_BUSES:
        return []

    op_samples: Dict[str, List[int]] = {}
    for o in ops:
        s = _word_samples(val, o.bits)
        if s is not None:
            op_samples[o.name] = s
    sc_samples: Dict[str, List[int]] = {}
    for s in scalars:
        v = val.get(s)
        if v is not None:
            sc_samples[s] = _word_samples(val, [s])

    matches: List[Match] = []
    for tgt in targets:
        tgt_s = _word_samples(val, tgt.bits)
        if tgt_s is None:
            continue
        sup = _support_sources(nl, tgt.bits)
        if sup is None:
            continue
        # pure rewiring first: strictly the shallowest possible rebuild
        m = _match_projection(tgt, val, sup)
        if m is None:
            # operand words whose bits intersect the target's support
            cand_ops = [o for o in ops if o.name in op_samples
                        and not sup.isdisjoint(o.bits)
                        and set(o.bits) != set(tgt.bits)]
            cand_cins = [s for s in scalars if s in sup and s in sc_samples]
            w = tgt.width
            m = _match_target(tgt, tgt_s, w, cand_ops, op_samples,
                              cand_cins, sc_samples)
        if m is not None:
            matches.append(m)
    return matches


def _match_target(tgt: Word, tgt_s: List[int], w: int,
                  cand_ops: List[Word], op_samples: Dict[str, List[int]],
                  cand_cins: List[str], sc_samples: Dict[str, List[int]],
                  ) -> Optional[Match]:
    """Operands narrower than the target are zero-extended (their sampled
    values already are); the 512-sample agreement decides validity, and the
    builders implement the identical zero-extend / mod-2^w semantics."""
    mk = _mask(w)

    # unary
    for a in cand_ops:
        av = op_samples[a.name]
        for fname, fn in FUNCS_UNARY.items():
            if all(fn(x, w) == y for x, y in zip(av, tgt_s)):
                return Match(tgt, fname, [a])

    # binary (multiply, arithmetic, bitwise, carry-in adds)
    for i, a in enumerate(cand_ops):
        for b in cand_ops[i:]:
            av, bv = op_samples[a.name], op_samples[b.name]
            if a is b:
                if all(((x * x) & mk) == t for x, t in zip(av, tgt_s)):
                    return Match(tgt, "mul", [a, a])
                continue
            if all(((x * y) & mk) == t for x, y, t in zip(av, bv, tgt_s)):
                return Match(tgt, "mul", [a, b])
            for fname, fn in FUNCS_BINARY.items():
                if all(fn(x, y, w) == t for x, y, t in zip(av, bv, tgt_s)):
                    return Match(tgt, fname, [a, b])
            # add with a scalar carry-in
            for cin in cand_cins:
                cv = sc_samples[cin]
                if all(((x + y + c) & mk) == t
                       for x, y, c, t in zip(av, bv, cv, tgt_s)):
                    return Match(tgt, "addcin", [a, b], cin=cin)

    # single-bit comparisons (unsigned, zero-extended)
    if w == 1:
        for i, a in enumerate(cand_ops):
            for b in cand_ops[i + 1:]:
                av, bv = op_samples[a.name], op_samples[b.name]
                for fname, fn in FUNCS_CMP.items():
                    if all(fn(x, y) == t for x, y, t in zip(av, bv, tgt_s)):
                        return Match(tgt, fname, [a, b])
    return None


# ---------------------------------------------------------------------------
# 4. depth-optimal builders (generic gates; the mapper handles the basis)
# ---------------------------------------------------------------------------

def _xor(em: Emitter, out: str, a: str, b: str) -> str:
    em.emit("xor", out, [a, b])
    return out


def _w(em: Emitter) -> str:
    return em.fresh_wire()


def _build_prefix_carries(em: Emitter, g: List[str], p: List[str],
                          cin: Optional[str]) -> List[str]:
    """Sklansky parallel-prefix: return carries c[0..w] (c[0]=cin or const0).

    (G, P) combine: (G2, P2) o (G1, P1) = (G2 + P2*G1, P2*P1).
    """
    w = len(g)
    # prefix[i] = (G, P) spanning bits [0..i]
    G = list(g)
    P = list(p)
    span = 1
    while span < w:
        for i in range(w):
            if (i // span) % 2 == 1:
                j = (i // span) * span - 1     # top of the previous block
                t_and = _w(em)
                em.and_(t_and, P[i], G[j])
                ng = _w(em)
                em.or_(ng, G[i], t_and)
                np_ = _w(em)
                em.and_(np_, P[i], P[j])
                G[i], P[i] = ng, np_
        span *= 2
    carries = ["1'b0" if cin is None else cin]
    for i in range(w):
        if cin is None:
            carries.append(G[i])
        else:
            t = _w(em)
            em.and_(t, P[i], cin)
            c = _w(em)
            em.or_(c, G[i], t)
            carries.append(c)
    return carries


def _addsub_bits(em: Emitter, a: List[str], b: List[str],
                 sub: bool, cin: Optional[str]) -> Tuple[List[str], str]:
    """Build a + b (+cin) or a - b; returns (sum bit wires, carry-out wire)."""
    w = len(a)
    bb = b
    if sub:
        bb = []
        for x in b:
            nx = _w(em)
            em.not_(nx, x)
            bb.append(nx)
        cin = "1'b1" if cin is None else cin   # a - b = a + ~b + 1
    g = []
    p = []
    for x, y in zip(a, bb):
        gi = _w(em)
        em.and_(gi, x, y)
        pi = _w(em)
        _xor(em, pi, x, y)
        g.append(gi)
        p.append(pi)
    carries = _build_prefix_carries(em, g, p, None if cin in (None, "1'b0") else cin)
    sums = []
    for i in range(w):
        s = _w(em)
        _xor(em, s, p[i], carries[i])
        sums.append(s)
    return sums, carries[w]


def _extend(em: Emitter, bits: List[str], w: int) -> List[str]:
    return list(bits[:w]) + ["1'b0"] * max(0, w - len(bits))


def _balanced(em: Emitter, op: str, xs: List[str]) -> str:
    """Balanced reduction tree; returns the root wire."""
    layer = list(xs)
    while len(layer) > 1:
        nxt = []
        for i in range(0, len(layer) - 1, 2):
            t = _w(em)
            em.emit(op, t, [layer[i], layer[i + 1]])
            nxt.append(t)
        if len(layer) % 2:
            nxt.append(layer[-1])
        layer = nxt
    return layer[0]


def _build_mul(em: Emitter, a: List[str], b: List[str], outs: List[str]):
    """Wallace-tree multiplier truncated to len(outs) bits."""
    w = len(outs)
    cols: List[List[str]] = [[] for _ in range(w)]
    for i, x in enumerate(a):
        if i >= w:
            break
        for j, y in enumerate(b):
            if i + j >= w:
                break
            t = _w(em)
            em.and_(t, x, y)
            cols[i + j].append(t)
    # pure 3:2 (full-adder) reduction until every column has <= 2 entries;
    # leftovers pass through, so each round strictly shrinks any column >= 3
    while any(len(c) > 2 for c in cols):
        nxt: List[List[str]] = [[] for _ in range(w)]
        for i, col in enumerate(cols):
            k = 0
            while len(col) - k >= 3:
                x, y, z = col[k:k + 3]
                k += 3
                axy = _w(em)
                _xor(em, axy, x, y)
                s = _w(em)
                _xor(em, s, axy, z)
                nxt[i].append(s)
                if i + 1 < w:
                    t1 = _w(em)
                    em.and_(t1, x, y)
                    t2 = _w(em)
                    em.and_(t2, axy, z)
                    c = _w(em)
                    em.or_(c, t1, t2)
                    nxt[i + 1].append(c)
            nxt[i].extend(col[k:])
        cols = nxt
    # final carry-propagate add of the two remaining rows
    row_a = [c[0] if len(c) > 0 else "1'b0" for c in cols]
    row_b = [c[1] if len(c) > 1 else "1'b0" for c in cols]
    sums, _ = _addsub_bits(em, row_a, row_b, sub=False, cin=None)
    for s, o in zip(sums, outs):
        em.emit("buf", o, [s])


def build_match(em: Emitter, m: Match):
    """Emit gates driving m.target.bits from the matched operands."""
    tgt = m.target.bits
    w = len(tgt)
    f = m.func

    def opbits(i: int) -> List[str]:
        return _extend(em, m.ops[i].bits, w)

    if f == "perm":
        for plan, o in zip(m.extra, tgt):
            if plan[0] == "const":
                em.emit("buf", o, [plan[1]])
            else:
                _kind, src, inv = plan
                if inv:
                    em.not_(o, src)
                else:
                    em.emit("buf", o, [src])
    elif f in ("add", "addc1", "sub", "rsub", "addcin"):
        a, b = opbits(0), opbits(1)
        if f == "rsub":
            a, b = b, a
        cin = None
        if f == "addc1":
            cin = "1'b1"
        elif f == "addcin":
            cin = m.cin
        sums, _ = _addsub_bits(em, a, b, sub=f in ("sub", "rsub"), cin=cin)
        for s, o in zip(sums, tgt):
            em.emit("buf", o, [s])
    elif f in ("inc", "dec", "neg", "bnot"):
        a = opbits(0)
        if f == "bnot":
            for x, o in zip(a, tgt):
                em.not_(o, x)
            return
        if f == "inc":
            sums, _ = _addsub_bits(em, a, ["1'b0"] * w, sub=False, cin="1'b1")
        elif f == "dec":
            sums, _ = _addsub_bits(em, a, ["1'b1"] * w, sub=False, cin=None)
        else:  # neg = ~a + 1
            na = []
            for x in a:
                nx = _w(em)
                em.not_(nx, x)
                na.append(nx)
            sums, _ = _addsub_bits(em, na, ["1'b0"] * w, sub=False, cin="1'b1")
        for s, o in zip(sums, tgt):
            em.emit("buf", o, [s])
    elif f in ("band", "bor", "bxor", "bnand", "bnor", "bxnor"):
        op = {"band": "and", "bor": "or", "bxor": "xor",
              "bnand": "nand", "bnor": "nor", "bxnor": "xnor"}[f]
        a, b = opbits(0), opbits(1)
        for x, y, o in zip(a, b, tgt):
            em.emit(op, o, [x, y])
    elif f == "mul":
        _build_mul(em, m.ops[0].bits, m.ops[1].bits, tgt)
    elif f in FUNCS_CMP:
        wc = max(m.ops[0].width, m.ops[1].width)
        a = _extend(em, m.ops[0].bits, wc)
        b = _extend(em, m.ops[1].bits, wc)
        out = tgt[0]
        if f in ("eq", "ne"):
            bits = []
            for x, y in zip(a, b):
                t = _w(em)
                em.emit("xnor" if f == "eq" else "xor", t, [x, y])
                bits.append(t)
            root = _balanced(em, "and" if f == "eq" else "or", bits)
            em.emit("buf", out, [root])
        else:
            # a<b  = NOT carry_out(a + ~b + 1); a>=b = carry_out(...)
            swap = f in ("ugt", "ule")     # a>b == b<a ; a<=b == not(b<a)
            x, y = (b, a) if swap else (a, b)
            _, cout = _addsub_bits(em, x, y, sub=True, cin=None)
            if f in ("ult", "ugt"):
                em.not_(out, cout)
            else:                          # uge / ule
                em.emit("buf", out, [cout])
    else:
        raise ValueError(f)


# ---------------------------------------------------------------------------
# 5. exact verification (cone-vs-template cec) and application
# ---------------------------------------------------------------------------

def _cone_blif(nl: Netlist, out_bits: Sequence[str], inputs: Sequence[str],
               gates: Sequence[Gate], model: str) -> str:
    lines = [f".model {model}",
             ".inputs " + " ".join(inputs),
             ".outputs " + " ".join(f"T{i}" for i in range(len(out_bits))),
             ".names __const0",
             ".names __const1", "1"]

    def ref(n: str) -> str:
        if n == "1'b0":
            return "__const0"
        if n == "1'b1":
            return "__const1"
        return n

    for g in gates:
        tt = _TT.get(g.type)
        if tt is None:
            continue
        lines.append(".names " + " ".join([ref(i) for i in g.ins] + [g.out]))
        lines.extend(tt)
    for i, o in enumerate(out_bits):
        lines.append(f".names {ref(o)} T{i}")
        lines.append("1 1")
    return "\n".join(lines) + "\n.end\n"


def _verify_match(nl: Netlist, m: Match, timeout: int = 60) -> Optional[List[Gate]]:
    """Build the template cone and cec it against the existing cone.
    Returns the new gates on success, else None."""
    scratch = Netlist()
    em = Emitter(scratch)
    try:
        build_match(em, m)
    except Exception:
        return None
    # rename intermediate wires to a namespace no real design uses, so the
    # template cone can never collide with nets of the design under check
    targets = set(m.target.bits)
    ren: Dict[str, str] = {}
    for k, g in enumerate(em.out):
        if g.out not in targets:
            ren[g.out] = f"__tpl_{k}"
    new_gates = [Gate(g.type, g.name, ren.get(g.out, g.out),
                      [ren.get(i, i) for i in g.ins]) for g in em.out]

    inputs = sorted({b for o in m.ops for b in o.bits}
                    | ({m.cin} if m.cin else set()))
    old_sup = _support_sources(nl, m.target.bits)
    if old_sup is None:
        return None
    inputs = sorted(set(inputs)
                    | {s for s in old_sup if not is_const(s)})
    inputs = [i for i in inputs if not is_const(i)]
    # target nets must not appear among the inputs (would alias)
    if set(m.target.bits) & set(inputs):
        return None

    old_gates = graph.fanin_cone_gates(nl, m.target.bits)
    blif_old = _cone_blif(nl, m.target.bits, inputs, old_gates, "cone_old")
    blif_new = _cone_blif(nl, m.target.bits, inputs, new_gates, "cone_new")
    if abc_bridge.cec_blif(blif_old, blif_new, timeout=timeout) is True:
        return new_gates
    return None


def rebuild_via_templates(nl: Netlist, timeout: int = 120,
                          ) -> Optional[Tuple[Netlist, List[str]]]:
    """Detect known word-level functions and rebuild their cones from
    depth-optimal templates.  Returns (candidate netlist, notes) or None when
    nothing matched."""
    if len(nl.gates) > MAX_GATES:
        return None
    val = simulate(nl)
    if val is None:
        return None
    matches = detect(nl, val)
    if not matches:
        return None

    cand = nl.snapshot()
    cand.touch()
    applied: List[str] = []
    per_match_to = max(10, timeout // max(1, len(matches)))
    for m in matches:
        # verify against (and splice into) the evolving candidate so the
        # sampled match stays valid even after earlier replacements
        new_gates = _verify_match(cand, m, timeout=per_match_to)
        if new_gates is None:
            continue
        targets = set(m.target.bits)
        cand.gates = [g for g in cand.gates if g.out not in targets]
        # re-emit through the candidate's own namespace to keep names unique
        em = Emitter(cand)
        remap: Dict[str, str] = {}
        for g in new_gates:
            ins = [remap.get(i, i) for i in g.ins]
            out = g.out
            if out not in targets:
                nw = cand.fresh_name("tw")
                remap[out] = nw
                out = nw
            em.emit(g.type, out, ins)
        cand.gates.extend(em.out)
        cand.touch()
        applied.append(m.note)
    if not applied:
        return None
    return cand, applied
