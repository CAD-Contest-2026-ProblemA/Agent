"""Netlist cleanup: dangling removal, back-to-back inverter collapse, and
fixpoint structural-duplicate merging.

All three are equivalence-preserving by construction.  Duplicate merging uses a
to-a-fixpoint union-find: merging one pair can expose new duplicates upstream
(naive single pass under-counts, e.g. test33 966 vs 1970).
"""

from __future__ import annotations

from typing import Dict, List, Set

from ..netlist.ir import Gate, Netlist, is_const
from ..analysis import graph


# ---- dangling / unused removal -----------------------------------------
def _live_set(nl: Netlist) -> Set[str]:
    """Nets reachable backward from a primary output (the liveness rule shared
    by find_dangling and remove_dangling — keep them in lock-step)."""
    # map net -> flip-flops driving it (handles multi-driver Q)
    q_to_dffs: Dict[str, List] = {}
    for ff in nl.dffs:
        q_to_dffs.setdefault(ff.q, []).append(ff)
    gate_by_out = {g.out: g for g in nl.gates}

    live: Set[str] = set()
    stack: List[str] = list(nl.po)
    while stack:
        n = stack.pop()
        if n in live or is_const(n):
            continue
        live.add(n)
        g = gate_by_out.get(n)
        if g is not None:
            for i in g.ins:
                if i not in live:
                    stack.append(i)
        for ff in q_to_dffs.get(n, []):
            for i in (ff.d, ff.clk, ff.rn, ff.sn):
                if i not in live:
                    stack.append(i)
    return live


def find_dangling(nl: Netlist):
    """Query only: names of gates / flip-flops that cannot affect any primary
    output.  Same liveness rule as remove_dangling, but the netlist is NOT
    modified — use this for 'check/are there any dangling' questions."""
    live = _live_set(nl)
    dead_gates = [g.name for g in nl.gates if g.out not in live]
    dead_dffs = [ff.name for ff in nl.dffs if ff.q not in live]
    return dead_gates, dead_dffs


def remove_dangling(nl: Netlist) -> int:
    """Remove gates (and flip-flops) that cannot affect any primary output.

    A net is *live* iff it is reachable backward from a primary output.  A gate
    is kept iff its output is live; a flip-flop is kept iff its Q is live.
    Multi-driver Q nets (several flip-flops sharing a Q) are handled atomically:
    all flip-flops on a live Q are kept, all on a dead Q dropped.

    Returns the number of combinational gates removed.
    """
    live = _live_set(nl)
    new_gates = [g for g in nl.gates if g.out in live]
    removed = len(nl.gates) - len(new_gates)
    nl.gates = new_gates
    nl.dffs = [ff for ff in nl.dffs if ff.q in live]
    nl.touch()
    return removed


# ---- BUF removal -------------------------------------------------------
def remove_buffers(nl: Netlist) -> int:
    """Delete every BUF, rewiring around it.  Returns #removed.

    A BUF computes the identity, so one of its two nets can absorb the other.
    Which one depends on where the buffer sits, and getting that backwards is
    why a first attempt removed nothing on a design whose buffers all drive
    primary outputs:

    * output is internal -- the output net is redundant, so loads move onto
      the input net and the instance goes.
    * output is a PRIMARY OUTPUT -- the output net cannot be dropped, it is a
      port.  Substitute the other way instead: the gate feeding the buffer
      takes the port's name and drives it directly.  This is what the
      reference netlist does (its NAND2_384 drives N1324, and the internal
      N1292 that fed the buffer is gone).

    A buffer is kept only when neither direction is legal -- the input is a
    port too (PI straight to PO, which structurally needs the buffer), or
    either net is already spoken for by an earlier substitution.  Two buffers
    sharing an input cannot both hand it a different port name.
    """
    nl.driver("__force_build__")
    po, pi = set(nl.po), set(nl.pi)
    subst: Dict[str, str] = {}
    touched: Set[str] = set()          # nets already renamed, or renamed onto
    keep, removed = [], 0

    def free(*nets):
        return all(n not in touched for n in nets)

    for g in nl.gates:
        if g.type != "buf":
            keep.append(g)
            continue
        src, dst = g.ins[0], g.out
        if src == dst:
            removed += 1
            continue
        if dst not in po and free(src, dst):
            subst[dst] = src                       # loads move to the input
        elif src not in po and src not in pi and free(src, dst):
            subst[src] = dst                       # driver takes the port name
        else:
            keep.append(g)
            continue
        touched.update((src, dst))
        removed += 1

    if not subst:
        return removed

    def resolve(n):
        seen = 0
        while n in subst and seen < 1_000_000:
            n = subst[n]
            seen += 1
        return n

    nl.gates = keep
    for g in nl.gates:
        g.ins = [resolve(i) for i in g.ins]
        g.out = resolve(g.out)                     # the reverse case renames outputs
    for ff in nl.dffs:
        ff.d = resolve(ff.d)
        ff.clk = resolve(ff.clk)
        ff.rn = resolve(ff.rn)
        ff.sn = resolve(ff.sn)
        ff.q = resolve(ff.q)
    nl.touch()
    return removed


def buffers_remaining(nl: Netlist):
    """BUF instances remove_buffers() cannot legally remove (PI straight to PO)."""
    return [g for g in nl.gates if g.type == "buf"]


# ---- back-to-back inverter collapse ------------------------------------
def collapse_double_inverters(nl: Netlist) -> int:
    """Collapse NOT(NOT(a)) chains.  Returns #pairs collapsed.

    Which of the two nets absorbs the other depends on where the pair sits,
    the same asymmetry remove_buffers() has:

    * the outer NOT's output is internal -- it is redundant, so its loads move
      onto ``a`` and the gate goes.
    * the outer NOT drives a PRIMARY OUTPUT -- that net is a port and cannot be
      dropped.  Rename the other way instead: whatever drives ``a`` takes the
      port's name and drives it directly.  No buffer is inserted, so a
      "NAND/NOT only" basis constraint still holds afterwards (Q&A A63).

    A pair is left in place only when neither direction is legal: the source is
    itself a port (a PI wired through two inverters to a PO structurally needs
    them), or it was already renamed onto some other port -- one net cannot
    take two port names.

    A DFF.Q source IS renameable.  The A30 register cut identifies a REGISTER,
    not the net it happens to drive, so giving that net the port's name leaves
    the cut intact; both our BLIF export and the reference judge label the cut
    __q_<instance>.  (An earlier version excluded it, because the equivalence
    check compared Q nets rather than instances and rejected its own correct
    output.)  Note the asymmetry in that guard: many pairs may
    share a source in the FORWARD direction, because they all substitute
    *towards* ``a`` and resolve() follows the chain; only the reverse direction
    is exclusive.  Callers report what remains rather than implying "all pairs"
    were collapsed; see pairs_remaining().
    """
    nl.driver("__force_build__")
    subst: Dict[str, str] = {}
    collapsed = 0
    remove: Set[int] = set()
    po, pi = set(nl.po), set(nl.pi)

    # map net -> driving NOT gate
    not_out = {g.out: g for g in nl.gates if g.type == "not"}
    g_index = {id(g): i for i, g in enumerate(nl.gates)}

    for g in nl.gates:
        if g.type != "not":
            continue
        x = g.ins[0]
        first = not_out.get(x)
        if first is None:
            continue
        a = first.ins[0]                 # g.out == NOT(NOT(a)) == a
        y = g.out
        if a == y:
            continue
        if y not in po:
            if y in subst:
                continue
            subst[y] = a                 # loads move onto the source
        elif a not in po and a not in pi and a not in subst:
            subst[a] = y                 # the source takes the port's name
        else:
            continue
        remove.add(g_index[id(g)])
        collapsed += 1

    if subst or remove:
        def resolve(n):
            seen = 0
            while n in subst and seen < 1_000_000:
                n = subst[n]
                seen += 1
            return n
        nl.gates = [g for i, g in enumerate(nl.gates) if i not in remove]
        for g in nl.gates:
            g.ins = [resolve(i) for i in g.ins]
            g.out = resolve(g.out)       # the reverse case renames an output
        for ff in nl.dffs:
            ff.d = resolve(ff.d)
            ff.clk = resolve(ff.clk)
            ff.rn = resolve(ff.rn)
            ff.sn = resolve(ff.sn)
            ff.q = resolve(ff.q)
        nl.touch()
        # a leading inverter that became unused is swept by remove_dangling
    return collapsed


def pairs_remaining(nl: Netlist):
    """Back-to-back NOT pairs collapse_double_inverters() could not remove."""
    not_out = {g.out: g for g in nl.gates if g.type == "not"}
    return [(g.name, not_out[g.ins[0]].name)
            for g in nl.gates
            if g.type == "not" and g.ins[0] in not_out]


# ---- structural duplicate merge (fixpoint) -----------------------------
_COMMUTATIVE = {"and", "or", "nand", "nor", "xor", "xnor"}


def merge_structural_duplicates(nl: Netlist) -> int:
    total = 0
    while True:
        merged = _merge_pass(nl)
        total += merged
        if merged == 0:
            break
    return total


def _merge_pass(nl: Netlist) -> int:
    topo = graph.topo_nets(nl)
    pos = {net: i for i, net in enumerate(topo)}
    gates_sorted = sorted(nl.gates, key=lambda g: pos.get(g.out, 1 << 60))

    subst: Dict[str, str] = {}

    def resolve(n):
        seen = 0
        while n in subst and seen < 1_000_000:
            n = subst[n]
            seen += 1
        return n

    seen: Dict[tuple, str] = {}
    remove_names: Set[str] = set()
    po = nl.po
    merged = 0

    for g in gates_sorted:
        rins = [resolve(i) for i in g.ins]
        g.ins = rins
        if g.type in _COMMUTATIVE:
            key = (g.type, tuple(sorted(rins)))
        else:
            key = (g.type, tuple(rins))
        canon = seen.get(key)
        if canon is None:
            seen[key] = g.out
            continue
        # duplicate of canon
        if g.out in po:
            # keep PO-driving gate; only merge if canon could redirect to it
            continue
        subst[g.out] = canon
        remove_names.add(g.name)
        merged += 1

    if merged:
        nl.gates = [g for g in nl.gates if g.name not in remove_names]
        for g in nl.gates:
            g.ins = [resolve(i) for i in g.ins]
        for ff in nl.dffs:
            ff.d = resolve(ff.d)
            ff.clk = resolve(ff.clk)
            ff.rn = resolve(ff.rn)
            ff.sn = resolve(ff.sn)
        nl.touch()
    return merged
