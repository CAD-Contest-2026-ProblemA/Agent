"""Resynthesis harness: cost-ranked portfolio over multiple seeds.

The engine behind ``minimize_depth`` / ``minimize_area`` / ``optimize_cone_depth``.
It follows the ALS_Final_Project "monotonic refine" methodology:

* **Seeds** — the current design; plus (opportunistically) a template-rebuilt
  variant (:mod:`.templates` — reverse-engineered word-level functions rebuilt
  from depth-optimal structures) and a yosys re-synthesis of the comb core
  (:mod:`.yosys_synth`).
* **Adaptive search** — generic ABC/Yosys/template transform families race in
  a serial beam search.  Cut sizes, effort, mapper tie mode, horizons, and
  stochastic seeds come from a name-independent topology fingerprint.  Online
  gain/second feedback decides which family receives the next deadline slice;
  there is no testcase dispatch or preselected winner chain.
* **Selection** — candidates are ranked by the true IR cost (recomputed on
  the rebuilt netlist, never trusted from ABC), the basis purity is enforced,
  and the best strictly-improving candidate that passes a full ``cec``
  against the current design wins.  No candidate, no change — the original is
  reported as already optimal.

Wrong template guesses are eliminated twice (per-cone cec at detection, whole
design cec here); ABC transforms are equivalence-preserving by construction
but still gated by the final cec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

from ..netlist.ir import Netlist, is_const
from ..analysis import cones, depth as depth_mod, graph
from ..transform import rewrite
from ..equiv import gate as equiv_gate
from . import abc_opt, templates, yosys_synth

# ---- adaptive search ------------------------------------------------------

# These are systematic ABC parameter domains, not per-design recipes.  The
# scheduler below explores them in a topology-derived permutation, learns each
# transform family's gain/second online, and applies promising families to a
# small beam of the best structures seen so far.
# Low-cost center-out expansion of ABC's cut domain.  K=6 is ABC's conventional
# first probe; subsequent attempts widen symmetrically, with topology choosing
# which side is visited first.  This prevents one unlucky large-K/high-C probe
# from consuming an entire short budget before another family is sampled.
_CUT_SIZES = (6, 5, 7, 4, 8, 3, 9, 10, 12, 11, 14, 13, 16, 15)
_CUT_BUDGETS = (8, 16, 32, 64)
_HORIZONS = (1, 2, 3, 4)
_SEED_COUNT = 101                 # ABC accepts -S 0..100
_BEAM_WIDTH = 4
_ARCHIVE_WIDTH = 12


@dataclass
class _Arm:
    """One generic transform family in the serial anytime scheduler."""

    name: str
    kind: str
    complexity: float
    attempts: int = 0
    successes: int = 0
    total_gain: float = 0.0
    total_seconds: float = 0.0
    consecutive_failures: int = 0
    # Parameter cursors are per exact parent topology.  Keeping parent and
    # parameter rotation separate avoids silently skipping half of a beam.
    next_by_parent: Dict[str, int] = field(default_factory=dict)
    parent_cursor: int = 0
    seen: Set[Tuple[str, Tuple[object, ...]]] = field(default_factory=set)
    # A successful local operator is immediately retried on its verified child
    # before broad exploration resumes (generic evolutionary continuation).
    promotions: List[Tuple[object, object]] = field(default_factory=list)
    blocked_epoch: int = -1
    exhausted: bool = False


@dataclass
class _Candidate:
    serial: int
    label: str
    nl: Netlist
    value: int
    rank: Tuple[int, ...]
    shape: Tuple[int, ...]
    key: str


@dataclass(frozen=True)
class _Trial:
    label: str
    params: Tuple[object, ...]
    recipe: Tuple[str, ...] = ()
    unit_area: Optional[bool] = None
    area_mode: bool = False
    normalize: bool = False
    template: bool = False
    effort: int = 1
    cut: int = 0
    cut_budget: int = 0
    mapper: str = ""


def _mix32(acc: int, value: int) -> int:
    """Small stable integer mixer (unlike Python's process-randomized hash)."""
    acc ^= int(value) & 0xFFFFFFFF
    return (acc * 16777619) & 0xFFFFFFFF


def _shape_signature(nl: Netlist, value: int,
                     output: Optional[str] = None) -> Tuple[int, ...]:
    """Name-independent structural summary used for diversity and seeding."""
    order = ("and", "or", "not", "nand", "nor", "xor", "xnor", "buf")
    counts = nl.type_counts()
    levels = graph.forward_levels(nl)
    # Include every possible fanout-boundary class in the feature vector.  The
    # objective decides which class is charged, but stochastic seeding should
    # still distinguish a PI/Q-heavy topology from a gate-output-heavy one.
    fanouts = ([len(nl.loads(g.out)) for g in nl.gates]
               + [len(nl.loads(ff.q)) for ff in nl.dffs]
               + [len(nl.loads(pi)) for pi in nl.pi])
    cone_size = (len(cones.fanin_cone_gates(nl, output))
                 if output is not None and output in nl.all_nets() else 0)
    # Two commutative accumulators over local topology tokens make collisions
    # between equal-size/equal-depth but structurally different candidates far
    # less likely, without depending on arbitrary signal or instance names.
    type_code = {name: idx + 1 for idx, name in enumerate(order)}
    kind_code = {"gate": 11, "dff": 12, "pi": 13,
                 "const": 14, "undriven": 15}
    topo_sum = 0
    topo_xor = 0
    level_sum = 0
    level_sq = 0
    for gate in nl.gates:
        level = levels.get(gate.out, 0)
        token = type_code.get(gate.type, 0)
        token = _mix32(token, level)
        token = _mix32(token, len(nl.loads(gate.out)))
        for inp in gate.ins:
            drv = nl.driver(inp)
            token = _mix32(token, kind_code.get(drv[0], 0))
            if drv[0] == "gate":
                token = _mix32(token, type_code.get(drv[1].type, 0))
            token = _mix32(token, levels.get(inp, 0))
        topo_sum = (topo_sum + token) & 0xFFFFFFFF
        topo_xor ^= ((token << (level & 15))
                     | (token >> ((32 - (level & 15)) & 31))) & 0xFFFFFFFF
        level_sum += level
        level_sq += level * level
    return (value, len(nl.gates), len(nl.dffs), len(nl.pi), len(nl.po),
            depth_mod.global_max_depth(nl), cone_size,
            max(fanouts, default=0), sum(fanouts),
            sum(1 for f in fanouts if f > 4), level_sum, level_sq,
            topo_sum, topo_xor,
            *(counts.get(t, 0) for t in order))


def _topology_key(nl: Netlist) -> str:
    """Canonical-ish digest for exact scheduler identity.

    The cheap aggregate :func:`_shape_signature` is intentionally retained for
    name-independent seeding and runtime features, but aggregate sums can
    collide.  Beam deduplication and the parent×parameter grid therefore use a
    stronger digest of the complete combinational DAG.  Internal net names are
    excluded; fixed interface/source identities remain so two different input
    assignments are never conflated.
    """
    memo: Dict[str, bytes] = {
        "1'b0": b"C0",
        "1'b1": b"C1",
    }

    q_instances: Dict[str, List[str]] = {}
    for ff in nl.dffs:
        q_instances.setdefault(ff.q, []).append(ff.name)

    def source_token(net: str) -> bytes:
        if net in memo:
            return memo[net]
        drv = nl.driver(net)
        if drv[0] == "pi":
            token = ("PI:" + net).encode()
        elif drv[0] == "dff":
            token = ("Q:" + ",".join(sorted(q_instances.get(net, [net])))).encode()
        elif drv[0] == "undriven":
            token = ("U:" + net).encode()
        else:
            token = ("S:" + net).encode()
        memo[net] = hashlib.blake2b(token, digest_size=16).digest()
        return memo[net]

    def token_for(net: str) -> bytes:
        token = memo.get(net)
        if token is not None:
            return token
        return source_token(net)

    # Each gate digest depends only on already-computed input digests.
    for out_net in graph.topo_nets(nl):
        drv = nl.driver(out_net)
        if drv[0] != "gate":
            continue
        gate = drv[1]
        ins = []
        for net in gate.ins:
            ins.append(token_for(net))
        if len(ins) == 2:  # every legal binary primitive is commutative
            ins.sort()
        payload = gate.type.encode() + b"(" + b",".join(ins) + b")"
        memo[gate.out] = hashlib.blake2b(payload, digest_size=16).digest()

    outputs: List[bytes] = []
    for po in sorted(nl.po):
        outputs.append(b"PO:" + po.encode() + b"=" + token_for(po))
    for ff in sorted(nl.dffs, key=lambda item: item.name):
        for attr in ("d", "clk", "rn", "sn"):
            net = getattr(ff, attr)
            outputs.append((f"FF:{ff.name}:{attr}=".encode()
                            + token_for(net)))
    # Include the multiset of all nodes as well as observable endpoints so dead
    # or duplicated logic cannot masquerade as the same evolutionary parent.
    nodes = sorted(memo.get(g.out, b"") for g in nl.gates)
    h = hashlib.blake2b(digest_size=20)
    h.update(str(len(nl.gates)).encode())
    for token in nodes + outputs:
        h.update(len(token).to_bytes(2, "little"))
        h.update(token)
    return h.hexdigest()


def _stable_seed(shape: Tuple[int, ...], family: str = "") -> int:
    acc = 2166136261
    for value in shape:
        acc = _mix32(acc, value)
    for ch in family:
        acc = _mix32(acc, ord(ch))
    return acc


def _cycle(values: Tuple[int, ...], seed: int, cursor: int) -> int:
    """Visit a finite parameter domain completely before repeating it."""
    n = len(values)
    if n == 1:
        return values[0]
    step = 1 + ((seed >> 8) % (n - 1))
    while math.gcd(step, n) != 1:
        step = 1 + (step % (n - 1))
    return values[((seed % n) + cursor * step) % n]


def _depth_grid_promoted(seed: int, index: int) -> int:
    """Promotion-first, complete ordering of the deterministic depth grid.

    A one-round probe predicts neither a two/three/four-round ABC invocation
    nor a remap of its already-mapped output.  Visit the conventional K6/C8
    restart at every horizon first, then expose every cut-budget tier and tie
    mode at K6.  The remaining Cartesian points are still visited exactly once
    in a topology-derived permutation.  This is a generic successive-halving
    schedule, not a circuit-specific recipe.
    """
    def encode(cut_i: int, budget_i: int, horizon_i: int,
               mapper_i: int) -> int:
        return (horizon_i
                + len(_HORIZONS) * mapper_i
                + len(_HORIZONS) * 2 * cut_i
                + len(_HORIZONS) * 2 * len(_CUT_SIZES) * budget_i)

    anchors: List[int] = []
    # Same-parent horizon promotion at the cheap conventional point.
    for mapper_i in range(2):
        for horizon_i in range(len(_HORIZONS)):
            anchors.append(encode(0, 0, horizon_i, mapper_i))
    # Make C16/C32/C64 practically reachable, including their promoted forms.
    for horizon_i in range(len(_HORIZONS)):
        for budget_i in range(1, len(_CUT_BUDGETS)):
            for mapper_i in range(2):
                anchors.append(encode(0, budget_i, horizon_i, mapper_i))
    # Cheap probes immediately to either side of K6 before the long tail.
    for cut_i in range(1, min(3, len(_CUT_SIZES))):
        for mapper_i in range(2):
            anchors.append(encode(cut_i, 0, 0, mapper_i))

    if index < len(anchors):
        return anchors[index]
    total = (2 * len(_CUT_SIZES) * len(_HORIZONS)
             * len(_CUT_BUDGETS))
    anchor_set = set(anchors)
    remaining = tuple(point for point in range(total)
                      if point not in anchor_set)
    return _cycle(remaining, seed ^ 0x85EBCA6B, index - len(anchors))


def _depth_grid_budget_first(index: int) -> int:
    """Encode the depth grid with cut budget as the fastest dimension."""
    budget_index = index % len(_CUT_BUDGETS)
    quotient = index // len(_CUT_BUDGETS)
    horizon_index = quotient % len(_HORIZONS)
    quotient //= len(_HORIZONS)
    mapper_index = quotient % 2
    cut_index = (quotient // 2) % len(_CUT_SIZES)
    # Convert back to the decoder's horizon→mapper→cut→budget radix order.
    return (horizon_index
            + len(_HORIZONS) * mapper_index
            + len(_HORIZONS) * 2 * cut_index
            + len(_HORIZONS) * 2 * len(_CUT_SIZES) * budget_index)


def _make_arms(area_objective: bool, use_templates: bool,
               use_yosys: bool) -> List[_Arm]:
    if area_objective:
        specs = [
            ("compress", "local", 1.0),
            ("amp-compress", "local", 1.2),
            ("dc2", "local", 1.0),
            ("resyn", "local", 1.0),
            ("randsyn", "stochastic", 1.4),
            ("deepsyn", "stochastic", 2.0),
        ]
    else:
        specs = [
            ("if-g", "local", 1.0),
            ("if-x", "local", 1.0),
            ("dc2", "local", 1.2),
            ("resyn", "local", 1.0),
            ("randsyn", "stochastic", 1.4),
            ("deepsyn", "stochastic", 2.0),
        ]
    if use_templates:
        specs.append(("template", "structural", 2.0))
    if use_yosys:
        specs.append(("yosys", "normalize", 2.5))
    return [_Arm(*spec) for spec in specs]


def _trial_for(arm: _Arm, parent: _Candidate, objective: str,
               seconds: float, index: int) -> _Trial:
    """Decode one independent parent×parameter-grid point.

    The mixed-radix dimensions are intentionally independent: K, cut budget,
    in-process horizon, mapper tie mode, and stochastic seed all eventually
    cross.  Index zero is the same cheap probe for every base-racing family;
    topology only permutes the later choices.
    """
    area_objective = objective in ("area", "buffered_area")
    buffered_objective = objective == "buffered_area"
    seed0 = _stable_seed(parent.shape, arm.name)

    if arm.name == "template":
        return _Trial("template", ("template",), template=True)

    # Stochastic seeds are the fast dimension so local plateaus can see several
    # genuinely different structures.  A coprime diagonal walk also rotates
    # cleanup dimensions immediately; over the finite limit it still covers the
    # complete seed×cleanup Cartesian product exactly once.
    if arm.kind == "stochastic":
        seed_index = index % _SEED_COUNT
        cleanup_grid = (len(_HORIZONS)
                        * (2 if objective == "buffered_area" else 1)
                        if area_objective else
                        2 * len(_CUT_SIZES) * len(_HORIZONS)
                        * len(_CUT_BUDGETS))
        if area_objective:
            grid_index = index % cleanup_grid
        else:
            # Seeds already provide topology-derived diversity.  Interleave
            # C8/C16/C32/C64 explicitly so no stochastic family spends its
            # entire practical lifetime in only the first cut-budget tier.
            grid_index = _depth_grid_budget_first(index % cleanup_grid)
        random_seed = _cycle(tuple(range(_SEED_COUNT)), seed0, seed_index)
    else:
        grid_index = index
        random_seed = -1

    if not area_objective and arm.kind != "stochastic":
        grid_index = _depth_grid_promoted(seed0, index)

    if area_objective:
        map_modes = (True, False) if buffered_objective else (True,)
        # Mapper mode is fastest for buffered area, so area and delay mapping
        # race at the same cheap horizon before either consumes longer slices.
        map_mode = map_modes[grid_index % len(map_modes)]
        quotient = grid_index // len(map_modes)
        horizon = _HORIZONS[quotient % len(_HORIZONS)]
        if arm.name == "compress":
            recipe = tuple(["compress2rs"] * horizon + ["resyn2rs"])
        elif arm.name == "amp-compress":
            recipe = tuple(["&get -n"] + ["&compress3rs"] * horizon
                           + ["&put", "compress2rs"])
        elif arm.name == "dc2":
            recipe = tuple(["dc2", "resyn2rs"] * horizon)
        elif arm.name == "resyn":
            recipe = tuple(["resyn2"] * horizon + ["resyn2rs"])
        elif arm.name == "randsyn":
            recipe = tuple(["&get -n", f"&randsyn -S {random_seed}"]
                           + ["&compress3rs"] * horizon
                           + ["&put", "compress2rs"])
        elif arm.name == "deepsyn":
            deep_time = max(1, min(12, int(seconds * 0.45)))
            recipe = tuple([
                "&get -n",
                f"&deepsyn -T {deep_time} -S {random_seed} -o",
                "&put"] + ["compress2rs"] * horizon)
        else:
            recipe = tuple(["compress2rs"] * horizon + ["resyn2rs"])
        normalize = arm.name == "yosys"
        params = (arm.name, horizon, "area" if map_mode else "delay",
                  random_seed)
        label = (f"{arm.name}[{index}]/r{horizon}/"
                 f"{'area-map' if map_mode else 'delay-map'}"
                 + (f"/s{random_seed}" if random_seed >= 0 else ""))
        return _Trial(label, params, recipe, unit_area=True,
                      area_mode=map_mode, normalize=normalize,
                      effort=horizon,
                      mapper="area" if map_mode else "delay")

    horizon = _HORIZONS[grid_index % len(_HORIZONS)]
    quotient = grid_index // len(_HORIZONS)
    mapper_index = quotient % 2
    quotient //= 2
    cut_index = quotient % len(_CUT_SIZES)
    quotient //= len(_CUT_SIZES)
    cut_budget = _CUT_BUDGETS[quotient % len(_CUT_BUDGETS)]
    # Every family starts at K6; topology permutes only the late expansion.
    if cut_index == 0:
        cut = 6
    else:
        late = tuple(k for k in _CUT_SIZES if k != 6)
        cut = _cycle(late, seed0, cut_index - 1)
    unit_area = bool(mapper_index)
    mode = "x" if arm.name == "if-x" else "g"
    mapper = "unit" if unit_area else "delay-tie"
    local = f"if -{mode} -K {cut} -C {cut_budget}"
    rounds: List[str] = []
    for _ in range(horizon):
        rounds.extend(("dch -f", local, "strash"))
    rounds.append("dch -f")
    if arm.name in ("if-g", "if-x"):
        recipe = tuple(rounds)
    elif arm.name == "dc2":
        recipe = tuple(["dc2"] + rounds)
    elif arm.name == "resyn":
        recipe = tuple(["resyn2"] + rounds)
    elif arm.name == "randsyn":
        recipe = tuple(["&get -n", f"&randsyn -S {random_seed}", "&put",
                        "strash"] + rounds)
    elif arm.name == "deepsyn":
        deep_time = max(1, min(12, int(seconds * 0.45)))
        recipe = tuple([
            "&get -n", f"&deepsyn -T {deep_time} -S {random_seed}",
            "&put", "strash"] + rounds)
    else:
        recipe = tuple(rounds)
    normalize = arm.name == "yosys"
    params = (arm.name, cut, cut_budget, horizon, mapper, random_seed)
    label = (f"{arm.name}[{index}]/k{cut}/c{cut_budget}/r{horizon}/{mapper}"
             + (f"/s{random_seed}" if random_seed >= 0 else ""))
    return _Trial(label, params, recipe, unit_area=unit_area,
                  area_mode=False, normalize=normalize, effort=horizon,
                  cut=cut, cut_budget=cut_budget, mapper=mapper)


def _arm_score(arm: _Arm, total_attempts: int, base_cost: int,
               stagnation: int = 0) -> float:
    """UCB-like gain/second score with one compulsory sample per family."""
    if arm.promotions:
        return 2_000_000.0
    if arm.attempts == 0:
        # Local arms participate in a fair base race.  Expensive/kick arms are
        # introduced by the plateau gate in resynthesize(), not forced merely
        # because they have not run yet.
        if arm.kind == "local":
            return 1_000_000.0 - 100.0 * arm.complexity
        kick = 1.0 + min(3.0, stagnation / 4.0)
        return 0.015 * kick / arm.complexity
    avg = max(0.01, arm.total_seconds / arm.attempts)
    # Cumulative reward divided by cumulative wall time.  Dividing by only the
    # average attempt time lets one early win retain a constant score forever
    # while thousands of duplicate failures accumulate.
    useful = ((arm.total_gain / max(1.0, float(base_cost)))
              + 0.02 * arm.successes) / max(0.01, arm.total_seconds)
    explore = (0.03 * math.sqrt(math.log(total_attempts + 2.0)
                                / arm.attempts) / math.sqrt(avg))
    stall = 1.0 / (1.0 + 0.25 * arm.consecutive_failures)
    kick = 1.0
    if arm.kind in ("stochastic", "structural", "normalize"):
        kick += min(3.0, stagnation / 4.0)
    elif stagnation >= 6:
        kick *= 0.65
    return ((useful + explore) * stall * kick
            / math.sqrt(arm.complexity))


def _proves_depth_two_optimal(nl: Netlist, output: Optional[str],
                              basis: Optional[str]) -> bool:
    """Prove that a depth-2 cone has no legal depth-0/1 implementation.

    For a small support, compute its complete truth table, then enumerate every
    function realizable by a wire/constant or one legal primitive gate.  If the
    target is absent, depth two is a mathematical lower bound.  This avoids
    spending an anytime budget searching below a proven bound while remaining
    conservative for large or ambiguous cones.
    """
    if output is None or output not in nl.all_nets():
        return False

    memo = {"1'b0": 0}
    visiting = set()
    sources = set()

    def collect(net: str) -> bool:
        if is_const(net):
            return True
        drv = nl.driver(net)
        if drv[0] != "gate":
            sources.add(net)
            return len(sources) <= 8
        if net in visiting:
            return False
        visiting.add(net)
        ok = all(collect(i) for i in drv[1].ins)
        visiting.remove(net)
        return ok

    if not collect(output) or len(sources) > 8:
        return False
    ordered = sorted(sources)
    patterns = 1 << len(ordered)
    full = (1 << patterns) - 1
    memo["1'b1"] = full
    for bit, name in enumerate(ordered):
        column = 0
        for pattern in range(patterns):
            if (pattern >> bit) & 1:
                column |= 1 << pattern
        memo[name] = column

    def value(net: str):
        if net in memo:
            return memo[net]
        drv = nl.driver(net)
        if drv[0] != "gate":
            return None
        g = drv[1]
        iv = [value(i) for i in g.ins]
        if any(v is None for v in iv):
            return None
        if g.type == "and":
            result = iv[0] & iv[1]
        elif g.type == "or":
            result = iv[0] | iv[1]
        elif g.type == "nand":
            result = full ^ (iv[0] & iv[1])
        elif g.type == "nor":
            result = full ^ (iv[0] | iv[1])
        elif g.type == "xor":
            result = iv[0] ^ iv[1]
        elif g.type == "xnor":
            result = full ^ (iv[0] ^ iv[1])
        elif g.type == "not":
            result = full ^ iv[0]
        elif g.type == "buf":
            result = iv[0]
        else:
            return None
        memo[net] = result
        return result

    target = value(output)
    if target is None:
        return False
    atoms = [0, full] + [memo[name] for name in ordered]
    allowed = (rewrite.BASES.get(basis, set()) if basis else
               {"and", "or", "nand", "nor", "xor", "xnor", "not", "buf"})
    reachable = set(atoms)
    for a in atoms:
        if "not" in allowed:
            reachable.add(full ^ a)
        if "buf" in allowed:
            reachable.add(a)
        for b in atoms:
            if "and" in allowed:
                reachable.add(a & b)
            if "or" in allowed:
                reachable.add(a | b)
            if "nand" in allowed:
                reachable.add(full ^ (a & b))
            if "nor" in allowed:
                reachable.add(full ^ (a | b))
            if "xor" in allowed:
                reachable.add(a ^ b)
            if "xnor" in allowed:
                reachable.add(full ^ (a ^ b))
    return target not in reachable


def _buffered_gate_count(nl: Netlist, k: int, include_pi: bool) -> int:
    """Exact final gate count after the minimal fanout tree insertion.

    A K-way buffer replaces K pending loads by one pending load, reducing the
    driver's load count by K-1.  Thus ``ceil((loads-K)/(K-1))`` is both the
    lower bound and what :func:`buffering.limit_fanout` constructs.
    """
    if k < 2:
        return len(nl.gates)
    targets = {g.out for g in nl.gates}
    if include_pi:
        targets.update(nl.pi)
        targets.update(ff.q for ff in nl.dffs)
    extra = 0
    for net in targets:
        if net in ("1'b0", "1'b1"):
            continue
        loads = len(nl.loads(net))
        if loads > k:
            extra += (loads - k + (k - 2)) // (k - 1)
    return len(nl.gates) + extra


def _cost_fn(objective: str, output: Optional[str],
             fanout_limit: Optional[int] = None,
             fanout_include_pi: bool = False) -> Callable[[Netlist], int]:
    if objective == "area":
        return lambda nl: len(nl.gates)
    if objective == "buffered_area" and fanout_limit is not None:
        return lambda nl: _buffered_gate_count(
            nl, fanout_limit, fanout_include_pi)
    if objective == "cone_depth":
        return lambda nl: depth_mod.depth_of_cone(nl, output)
    return depth_mod.global_max_depth


def _ensure_basis(cand: Netlist, basis: Optional[str],
                  basis_output: Optional[str] = None) -> Netlist:
    """Purity before costing: stray cells (e.g. mapper ``buf``) are rewritten
    into the basis so the ranking sees the netlist the checks will see."""
    if basis:
        want = rewrite.BASES[basis]
        scoped = (cones.fanin_cone_gates(cand, basis_output)
                  if basis_output is not None else cand.gates)
        if any(g.type not in want for g in scoped):
            scope_gates = ({g.name for g in scoped}
                           if basis_output is not None else None)
            rewrite.to_basis(cand, basis, scope_gates=scope_gates)
    return cand


def _basis_compliant(nl: Netlist, basis: Optional[str],
                     basis_output: Optional[str]) -> bool:
    if not basis:
        return True
    want = rewrite.BASES[basis]
    scoped = (cones.fanin_cone_gates(nl, basis_output)
              if basis_output is not None else nl.gates)
    return all(g.type in want for g in scoped)


def _trial_limit(arm: _Arm, objective: str) -> int:
    if arm.name == "template":
        return 1
    if objective in ("area", "buffered_area"):
        grid = len(_HORIZONS) * (2 if objective == "buffered_area" else 1)
    else:
        grid = (2 * len(_CUT_SIZES) * len(_HORIZONS)
                * len(_CUT_BUDGETS))
    return grid * (_SEED_COUNT if arm.kind == "stochastic" else 1)


def resynthesize(nl: Netlist, objective: str = "depth",
                 output: Optional[str] = None, basis: Optional[str] = None,
                 basis_output: Optional[str] = None,
                 fanout_limit: Optional[int] = None,
                 fanout_include_pi: bool = False,
                 timeout: int = 290, use_templates: bool = True,
                 use_yosys: bool = True) -> Tuple[Netlist, bool, dict]:
    """Best-effort resynthesis of the combinational core.

    Returns ``(netlist, improved, info)``.  Every evolutionary parent and every
    returned result has passed a whole-register-cut CEC against ``nl``.
    """
    t0 = time.monotonic()
    budget = max(0.5, float(timeout))
    hard_deadline = t0 + budget
    # Reserve enough wall time for one size-predicted final proof, rather than
    # blindly discarding 20% of every request.  Small/medium designs therefore
    # search almost to the contest deadline; very large designs retain a wider
    # CEC reserve.  The cap for short API calls still leaves useful search time.
    predicted_final_cec = 2.0 * (0.5 + len(nl.gates) / 15_000.0)
    final_reserve = min(
        30.0, max(1.0, predicted_final_cec), budget * 0.25)
    output_reserve = min(2.0, max(0.1, budget * 0.02), budget * 0.10)
    search_deadline = hard_deadline - final_reserve
    validation_deadline = hard_deadline - output_reserve

    def search_remaining() -> float:
        return search_deadline - time.monotonic()

    def hard_remaining() -> float:
        return hard_deadline - time.monotonic()

    def validation_remaining() -> float:
        return validation_deadline - time.monotonic()

    if objective == "buffered_area" and (
            fanout_limit is None or fanout_limit < 2):
        raise ValueError("buffered-area optimization requires fanout_limit >= 2")

    cost = _cost_fn(objective, output, fanout_limit, fanout_include_pi)
    original_cost = cost(nl)
    # A cone-only basis constraint is independent of the optimization metric.
    # Validate the name here: an invalid scope must never silently weaken a
    # whole-design constraint, while a valid DFF.Q scope is intentionally an
    # empty cone under the contest's register-boundary convention.
    if basis_output is not None and basis_output not in nl.all_nets():
        raise ValueError(f'unknown basis-scope net "{basis_output}"')

    info: dict = {"objective": objective, "base_cost": original_cost,
                  "basis_output": basis_output,
                  "tried": [], "templates": None, "winner": None}

    # A hard basis requirement needs a legal fallback even when mapping into
    # that basis costs more than the unconstrained input.  Build and prove one
    # before optimization; the original remains an additional restart seed.
    fallback = nl
    fallback_changed = False
    fallback_compliant = _basis_compliant(nl, basis, basis_output)
    basis_pending: Optional[Netlist] = None
    if basis and not fallback_compliant:
        feasible = nl.snapshot()
        _ensure_basis(feasible, basis, basis_output)
        if (_basis_compliant(feasible, basis, basis_output)
                and validation_remaining() > 1.0):
            per_tool = max(1, int(min(15.0, validation_remaining() / 2.0)))
            proof = equiv_gate.equivalent(nl, feasible, timeout=per_tool)
        elif _basis_compliant(feasible, basis, basis_output):
            proof = None
        else:
            proof = False
        if proof is True:
            fallback = feasible
            fallback_changed = True
            fallback_compliant = True
            info["basis_fallback"] = "verified"
        elif proof is None:
            # Keep an inconclusive mandatory conversion for the final proof
            # reserve.  False is rejected; None is not evidence of inequivalence.
            basis_pending = feasible
            info["basis_fallback"] = "pending"
        else:
            info["basis_fallback"] = "rejected"

    base = cost(fallback)
    info["feasible_cost"] = base
    if base <= 0 and fallback_compliant:
        if fallback_changed:
            info["winner"] = "basis-feasible"
        return fallback, fallback_changed, info
    if (objective == "cone_depth" and base == 2 and fallback_compliant
            and (basis_output is None or basis_output == output)
            and _proves_depth_two_optimal(fallback, output, basis)):
        info["winner"] = "proven-depth-lower-bound"
        return fallback, fallback_changed, info

    area_objective = objective in ("area", "buffered_area")
    arms = _make_arms(area_objective, use_templates, use_yosys)
    info["scheduler"] = "adaptive-serial-ucb"

    def candidate_rank(cand: Netlist, value: int) -> Tuple[int, ...]:
        # Primary objective is exact.  Deterministic secondary terms make the
        # beam stable while preferring simpler structures at equal cost.
        if objective == "buffered_area":
            return (value, value - len(cand.gates), len(cand.gates),
                    depth_mod.global_max_depth(cand))
        if area_objective:
            return (value, depth_mod.global_max_depth(cand), len(cand.gates))
        return (value, len(cand.gates), depth_mod.global_max_depth(cand))

    base_shape = _shape_signature(fallback, base, output)
    base_candidate = _Candidate(
        0, "base", fallback, base, candidate_rank(fallback, base), base_shape,
        _topology_key(fallback))
    restart_candidate = base_candidate
    if fallback is not nl:
        restart_candidate = _Candidate(
            -1, "original-restart", nl, original_cost,
            candidate_rank(nl, original_cost),
            _shape_signature(nl, original_cost, output), _topology_key(nl))
    population: List[_Candidate] = [base_candidate]
    archive: List[_Candidate] = []
    pending: List[_Candidate] = []
    serial = 0
    epoch = 0

    def make_candidate(label: str, cand: Netlist) -> _Candidate:
        nonlocal serial
        _ensure_basis(cand, basis, basis_output)
        value = cost(cand)
        serial += 1
        return _Candidate(
            serial, label, cand, value, candidate_rank(cand, value),
            _shape_signature(cand, value, output), _topology_key(cand))

    def consider(item: _Candidate) -> Tuple[_Candidate, bool, int]:
        """Admit a *verified* result and report real frontier progress."""
        nonlocal population, archive, epoch
        before_best = min([base] + [entry.value for entry in archive])
        old_keys = tuple(entry.key for entry in population)

        best_by_key: Dict[str, _Candidate] = {}
        for entry in population + [item]:
            prev = best_by_key.get(entry.key)
            if prev is None or entry.rank < prev.rank:
                best_by_key[entry.key] = entry
        ranked = sorted(best_by_key.values(), key=lambda c: c.rank)
        # Preserve a small Pareto frontier rather than cloning only the current
        # scalar champion.  A slightly deeper but much smaller structure (or a
        # slightly larger but shallower area candidate) can be the better seed
        # for the next generic transform family.
        pareto: List[_Candidate] = []
        for entry in ranked:
            point = (entry.rank[0], entry.rank[1])
            dominated = any(
                other.rank[0] <= point[0] and other.rank[1] <= point[1]
                and (other.rank[0] < point[0] or other.rank[1] < point[1])
                for other in ranked)
            if not dominated:
                pareto.append(entry)
        selected = sorted(pareto, key=lambda c: c.rank)[:_BEAM_WIDTH]
        if len(selected) < _BEAM_WIDTH:
            selected_ids = {c.serial for c in selected}
            selected.extend(c for c in ranked if c.serial not in selected_ids)
        population = sorted(selected[:_BEAM_WIDTH], key=lambda c: c.rank)
        # Multi-round synthesis inside one ABC invocation is not equivalent to
        # repeatedly mapping an already-mapped winner.  Keep the original base
        # as a permanent restart lane so later horizons/parameters can race
        # from it even after several improving candidates enter the beam.
        if all(entry.key != base_candidate.key for entry in population):
            population = sorted(
                population[:max(0, _BEAM_WIDTH - 1)] + [base_candidate],
                key=lambda c: c.rank)
        if tuple(entry.key for entry in population) != old_keys:
            epoch += 1

        # When no legal fallback could be proved, any verified basis-compliant
        # result is returnable.  Otherwise preserve strict primary monotonicity.
        returnable = (not fallback_compliant or item.value < base)
        if returnable:
            by_key: Dict[str, _Candidate] = {entry.key: entry for entry in archive}
            prev = by_key.get(item.key)
            if prev is None or item.rank < prev.rank:
                by_key[item.key] = item
            archive = sorted(by_key.values(), key=lambda c: c.rank)[
                :_ARCHIVE_WIDTH]
        admitted = (any(entry.serial == item.serial for entry in population)
                    or any(entry.serial == item.serial for entry in archive))
        after_best = min([base] + [entry.value for entry in archive])
        canonical = next((entry for entry in population + archive
                          if entry.key == item.key), item)
        return canonical, admitted, max(0, before_best - after_best)

    prepared_cache: Dict[str, tuple] = {}
    yosys_cache: Dict[str, str] = {}
    runtime_model: Dict[Tuple[object, ...], float] = {}
    cec_model: Dict[int, float] = {}
    # Reuse whole-design proofs for exact topology duplicates.  Saturated ABC
    # families often rediscover one mapped graph through many K/C settings;
    # proving each copy again wastes the validation budget on large designs.
    verified_keys: Set[str] = {base_candidate.key, restart_candidate.key}
    rejected_keys: Set[str] = set()
    pending_keys: Set[str] = set()

    def parents_for(arm: _Arm) -> List[_Candidate]:
        if arm.attempts == 0:
            # Fair base racing: every family gets its first evidence from the
            # same raw structure, never from a list-order predecessor's winner.
            return [restart_candidate]
        ordered = [base_candidate, restart_candidate] + population
        unique: Dict[str, _Candidate] = {}
        for entry in ordered:
            unique.setdefault(entry.key, entry)
        return list(unique.values())

    def choose_grid_point(
            arm: _Arm,
            ) -> Optional[Tuple[_Candidate, int, Optional[_Trial], bool]]:
        if arm.promotions:
            promoted_parent, promoted_trial = arm.promotions.pop(0)
            return (promoted_parent, -1, promoted_trial, False)
        parents = parents_for(arm)
        limit = _trial_limit(arm, objective)
        for offset in range(len(parents)):
            pos = (arm.parent_cursor + offset) % len(parents)
            parent = parents[pos]
            index = arm.next_by_parent.get(parent.key, 0)
            if index >= limit:
                continue
            arm.next_by_parent[parent.key] = index + 1
            arm.parent_cursor = (pos + 1) % len(parents)
            return parent, index, None, True
        return None

    def has_grid_point(arm: _Arm) -> bool:
        if arm.promotions:
            return True
        limit = _trial_limit(arm, objective)
        return any(arm.next_by_parent.get(parent.key, 0) < limit
                   for parent in parents_for(arm))

    def add_pending(item: _Candidate) -> None:
        nonlocal pending
        by_key = {entry.key: entry for entry in pending}
        prev = by_key.get(item.key)
        if prev is None or item.rank < prev.rank:
            by_key[item.key] = item
        pending = sorted(by_key.values(), key=lambda entry: entry.rank)[:6]
        pending_keys.clear()
        pending_keys.update(entry.key for entry in pending)

    total_attempts = 0
    last_improvement_attempt = 0

    def minimum_slice(arm: _Arm) -> float:
        if arm.kind == "normalize":
            return 6.0
        if arm.kind == "stochastic":
            return 3.0
        return 2.5

    while search_remaining() > 1.0:
        local_arms = [arm for arm in arms if arm.kind == "local"]
        local_ready = [arm for arm in local_arms
                       if (not arm.exhausted and arm.blocked_epoch != epoch
                           and has_grid_point(arm))]
        unsampled_local = [arm for arm in local_ready if arm.attempts == 0]
        if unsampled_local:
            # Phase zero is a fair, same-parent race among cheap transforms.
            eligible = unsampled_local
        else:
            stalled = total_attempts - last_improvement_attempt
            # If the finite local grid is exhausted/temporarily blocked, do
            # not deadlock merely because its last point happened to improve:
            # stochastic/structural restarts are the only remaining progress.
            allow_kicks = (stalled >= max(2, len(local_arms))
                           or not local_ready)
            eligible = local_ready + [
                arm for arm in arms
                if (arm.kind != "local" and allow_kicks
                    and not arm.exhausted and arm.blocked_epoch != epoch
                    and has_grid_point(arm))]
        available = [arm for arm in eligible
                     if (not arm.exhausted and arm.blocked_epoch != epoch
                         and search_remaining() >= minimum_slice(arm))]
        if not available:
            break
        arm = max(available,
                  key=lambda a: _arm_score(
                      a, total_attempts, base,
                      total_attempts - last_improvement_attempt))

        point = choose_grid_point(arm)
        if point is None:
            arm.blocked_epoch = epoch
            continue
        parent, trial_index, promoted_trial, from_grid = point

        untried = sum(1 for candidate_arm in available
                      if candidate_arm.attempts == 0)
        topology_est = arm.complexity * (
            2.0 + len(parent.nl.gates) / 1_500.0)
        if arm.attempts == 0:
            # Prevent one large first probe from starving every other family.
            fair = search_remaining() / max(2.0, untried + 1.0)
            predicted = min(20.0, fair, topology_est)
        else:
            predicted = topology_est
        trial = (promoted_trial if promoted_trial is not None else
                 _trial_for(arm, parent, objective,
                            max(1.0, predicted), trial_index))
        runtime_key = (
            arm.name, int(math.log2(max(1, len(parent.nl.gates)))),
            trial.effort, trial.normalize, trial.area_mode,
            trial.cut, trial.cut_budget, trial.mapper)
        learned = runtime_model.get(runtime_key, predicted)
        min_slice = minimum_slice(arm)
        trial_budget = min(
            90.0, search_remaining(), max(min_slice, learned * 1.8))
        if trial_budget < min_slice:
            # No operator can be launched safely in the remaining search wall.
            # Put the grid point back; consuming an unexecuted point corrupts
            # both the finite product enumeration and runtime feedback.
            if from_grid:
                arm.next_by_parent[parent.key] = trial_index
            else:
                arm.promotions.insert(0, (parent, trial))
            arm.blocked_epoch = epoch
            continue
        # DeepSyn's internal T should reflect the actual scheduled slice.
        if promoted_trial is None:
            trial = _trial_for(
                arm, parent, objective, trial_budget, trial_index)
        seen_key = (parent.key, trial.params)
        if seen_key in arm.seen:
            continue
        arm.seen.add(seen_key)

        started = time.monotonic()
        attempt_deadline = min(search_deadline, started + trial_budget)
        cand = None
        launched = False
        label = f"p{parent.serial}/{trial.label}"
        try:
            if trial.template:
                launched = True
                tr = templates.rebuild_via_templates(
                    parent.nl,
                    timeout=max(1, int(attempt_deadline - time.monotonic())))
                if tr is not None:
                    cand = tr[0]
                    info["templates"] = tr[1]
            else:
                prepared = prepared_cache.get(parent.key)
                if prepared is None:
                    prepared = abc_opt._opt_blif(parent.nl)
                    prepared_cache[parent.key] = prepared
                if trial.normalize:
                    ys = yosys_cache.get(parent.key)
                    left = attempt_deadline - time.monotonic()
                    if ys is None and left > 2:
                        launched = True
                        ys = yosys_synth.blif_roundtrip(
                            prepared[0], timeout=max(1, int(left * 0.55)))
                        if ys is not None:
                            yosys_cache[parent.key] = ys
                    if ys is not None:
                        prepared = (ys, prepared[1], prepared[2])
                    else:
                        prepared = None
                left = attempt_deadline - time.monotonic()
                if prepared is not None and left > 1:
                    launched = True
                    cand = abc_opt.optimize_comb(
                        parent.nl, list(trial.recipe),
                        # A cone-only basis is enforced after unrestricted
                        # multi-output mapping; a global basis maps directly.
                        basis=None if basis_output is not None else basis,
                        timeout=max(1, int(left)), area_mode=trial.area_mode,
                        unit_area=trial.unit_area, prepared=prepared)
        except Exception:
            cand = None

        if not launched:
            arm.seen.discard(seen_key)
            if from_grid:
                arm.next_by_parent[parent.key] = trial_index
            else:
                arm.promotions.insert(0, (parent, trial))
            arm.blocked_epoch = epoch
            continue

        transform_elapsed = max(0.001, time.monotonic() - started)
        old_runtime = runtime_model.get(runtime_key)
        runtime_model[runtime_key] = (transform_elapsed if old_runtime is None
                                      else 0.7 * old_runtime
                                      + 0.3 * transform_elapsed)
        arm.attempts += 1
        total_attempts += 1

        value = None
        proof = None
        item = None
        if cand is not None:
            item = make_candidate(label, cand)
            value = item.value
            # Every frontier promotion gets a whole-register CEC.  A transform
            # that cannot be proved equivalent may remain a measured trial but
            # can never become an evolutionary parent and contaminate all of
            # its descendants.
            if item.key in verified_keys:
                proof = True
            elif item.key in rejected_keys:
                proof = False
            elif item.key in pending_keys:
                proof = None
            else:
                left = validation_remaining()
            if (item.key not in verified_keys
                    and item.key not in rejected_keys
                    and item.key not in pending_keys
                    and left > 1.0):
                size_bucket = int(math.log2(max(1, len(cand.gates))))
                predicted_cec = cec_model.get(
                    size_bucket, 0.5 + len(cand.gates) / 15_000.0)
                # equivalent() may run ABC then Yosys with this timeout; half
                # the remaining validation wall bounds both attempts.
                cec_slice = max(1, int(min(30.0, left / 2.0,
                                           max(2.0, predicted_cec * 2.0))))
                proof_started = time.monotonic()
                proof = equiv_gate.equivalent(nl, cand, timeout=cec_slice)
                proof_elapsed = max(0.001, time.monotonic() - proof_started)
                old_cec = cec_model.get(size_bucket)
                cec_model[size_bucket] = (proof_elapsed if old_cec is None
                                          else 0.7 * old_cec
                                          + 0.3 * proof_elapsed)
                if proof is True:
                    verified_keys.add(item.key)
                elif proof is False:
                    rejected_keys.add(item.key)

        elapsed = max(0.001, time.monotonic() - started)
        arm.total_seconds += elapsed
        if item is not None and proof is True:
            canonical, admitted, gain = consider(item)
            if admitted:
                arm.successes += 1
                arm.total_gain += gain
            if gain > 0:
                arm.consecutive_failures = 0
                last_improvement_attempt = total_attempts
                if arm.kind == "local" and canonical.key != parent.key:
                    arm.promotions.append((canonical, trial))
            else:
                arm.consecutive_failures += 1
            info["tried"].append((label, canonical.value))
        else:
            arm.consecutive_failures += 1
            if item is not None and proof is None and (
                    not fallback_compliant or item.value < base):
                add_pending(item)
            info["tried"].append((label, value))

    info["arms"] = {
        arm.name: {
            "attempts": arm.attempts,
            "successes": arm.successes,
            "gain": arm.total_gain,
            "seconds": round(arm.total_seconds, 3),
        }
        for arm in arms
    }

    # A mandatory basis conversion gets first claim on the proof reserve.  It
    # must not be starved by optional improving candidates.
    if (not archive and basis_pending is not None
            and validation_remaining() > 1.0):
        per_tool = max(1, int(min(30.0, validation_remaining() / 2.0)))
        if equiv_gate.equivalent(nl, basis_pending, timeout=per_tool) is True:
            info["basis_fallback"] = "verified-on-retry"
            info["winner"] = "basis-feasible"
            return basis_pending, True, info

    # Retry only the best candidates whose first proof was inconclusive.  They
    # were never evolutionary parents.  A verified archive entry is immutable
    # and is not pointlessly re-proved (a later timeout cannot revoke True).
    verified_best = min([base] + [entry.value for entry in archive])
    for entry in sorted(pending, key=lambda candidate: candidate.rank):
        if fallback_compliant and entry.value >= verified_best:
            continue
        left = validation_remaining()
        if left <= 1.0:
            break
        per_tool = max(1, int(min(30.0, left / 2.0)))
        if equiv_gate.equivalent(nl, entry.nl, timeout=per_tool) is True:
            consider(entry)
            verified_best = min(verified_best, entry.value)

    if archive:
        winner = min(archive, key=lambda entry: entry.rank)
        info["winner"] = winner.label
        return winner.nl, True, info
    if fallback_changed:
        info["winner"] = "basis-feasible"
    if basis and not fallback_compliant:
        raise TimeoutError(
            "could not verify a netlist satisfying the mandatory basis constraint")
    info["hard_remaining"] = round(max(0.0, hard_remaining()), 3)
    return fallback, fallback_changed, info
