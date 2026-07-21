"""Resynthesis harness: cost-ranked portfolio over multiple seeds.

The engine behind ``minimize_depth`` / ``minimize_area`` / ``optimize_cone_depth``.
It follows the ALS_Final_Project "monotonic refine" methodology:

* **Seeds** — the current design; plus (opportunistically) a template-rebuilt
  variant (:mod:`.templates` — reverse-engineered word-level functions rebuilt
  from depth-optimal structures) and a yosys re-synthesis of the comb core
  (:mod:`.yosys_synth`).
* **Portfolio** — each seed is optimised by several ABC recipes.  Depth uses
  the delay-oriented ``dch -f; if -g`` choice-mapping loops (empirically 2-4x
  shallower than ``resyn2`` on the contest netlists), area the
  ``compress2rs`` family; every result is technology-mapped onto the
  unit-delay gate library of the requested basis so the *contest* cost
  (1 gate = 1 level, inverters included) is what ABC minimises.
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

import time
from typing import Callable, List, Optional, Tuple

from ..netlist.ir import Netlist
from ..analysis import depth as depth_mod
from ..transform import rewrite
from ..equiv import gate as equiv_gate
from . import abc_opt, templates, yosys_synth

# ---- recipe portfolios (distilled from the ALS_Final_Project corpus) ------

DEPTH_PORTFOLIO: List[Tuple[str, List[str]]] = [
    ("ifg-dch", ["dch -f", "if -g -K 6", "strash",
                 "dch -f", "if -g -K 6", "strash", "dch -f"]),
    ("dc2-ifg", ["dc2", "dch -f", "if -g -K 6", "strash",
                 "dc2", "dch -f", "if -g -K 6", "strash", "dch -f"]),
    ("resyn2", ["resyn2", "resyn2"]),
]

AREA_PORTFOLIO: List[Tuple[str, List[str]]] = [
    ("c2rs", ["compress2rs", "compress2rs", "resyn2rs"]),
    ("resyn2-dc2", ["resyn2", "dc2", "resyn2"]),
]

# skip the optional seeds on very large cores (keep runtime bounded)
YOSYS_MAX_GATES = 80_000


def _cost_fn(objective: str, output: Optional[str]) -> Callable[[Netlist], int]:
    if objective == "area":
        return lambda nl: len(nl.gates)
    if objective == "cone_depth":
        return lambda nl: depth_mod.depth_of_cone(nl, output)
    return depth_mod.global_max_depth


def _ensure_basis(cand: Netlist, basis: Optional[str]) -> Netlist:
    """Purity before costing: stray cells (e.g. mapper ``buf``) are rewritten
    into the basis so the ranking sees the netlist the checks will see."""
    if basis:
        want = rewrite.BASES[basis]
        if any(g.type not in want for g in cand.gates):
            rewrite.to_basis(cand, basis)
    return cand


def resynthesize(nl: Netlist, objective: str = "depth",
                 output: Optional[str] = None, basis: Optional[str] = None,
                 timeout: int = 280, use_templates: bool = True,
                 use_yosys: bool = True) -> Tuple[Netlist, bool, dict]:
    """Best-effort resynthesis of the combinational core.

    Returns ``(netlist, improved, info)``; the input netlist is returned
    unchanged when no strictly-improving, equivalence-verified candidate was
    found.  ``info`` records seeds/recipes tried and the winner.
    """
    t0 = time.monotonic()
    deadline = t0 + max(30, timeout) * 0.8       # reserve tail for final cec

    def remaining() -> float:
        return deadline - time.monotonic()

    cost = _cost_fn(objective, output)
    base = cost(nl)
    info: dict = {"objective": objective, "base_cost": base,
                  "tried": [], "templates": None, "winner": None}
    if base <= 0:
        return nl, False, info

    portfolio = AREA_PORTFOLIO if objective == "area" else DEPTH_PORTFOLIO

    # ---- seeds ----
    seeds: List[Tuple[str, Netlist]] = [("base", nl)]
    if use_templates:
        try:
            tr = templates.rebuild_via_templates(
                nl, timeout=int(max(20, min(90, remaining() / 3))))
        except Exception:
            tr = None
        if tr is not None:
            seeds.append(("tpl", tr[0]))
            info["templates"] = tr[1]

    # ---- candidates ----
    candidates: List[Tuple[str, Netlist]] = []
    for tag, seed in seeds:
        if remaining() < 15:
            break
        prepared = abc_opt._opt_blif(seed)
        variants: List[Tuple[str, str]] = [("abc", prepared[0])]
        if (use_yosys and tag == "base"
                and len(seed.gates) <= YOSYS_MAX_GATES and remaining() > 60):
            ys = yosys_synth.blif_roundtrip(
                prepared[0], timeout=int(min(60, remaining() / 3)))
            if ys is not None:
                variants.append(("ys", ys))
        for vtag, blif_text in variants:
            for rname, recipe in portfolio:
                if remaining() < 15:
                    break
                label = f"{tag}/{vtag}/{rname}"
                cand = abc_opt.optimize_comb(
                    seed, recipe, basis=basis,
                    timeout=int(max(10, min(90, remaining()))),
                    prepared=(blif_text, prepared[1], prepared[2]))
                if cand is None:
                    info["tried"].append((label, None))
                    continue
                _ensure_basis(cand, basis)
                c = cost(cand)
                info["tried"].append((label, c))
                candidates.append((label, cand))

    # ---- monotonic selection: best first, cec-gated ----
    candidates.sort(key=lambda tc: (cost(tc[1]), len(tc[1].gates)))
    for label, cand in candidates:
        if cost(cand) >= base:
            break                      # ranked ascending: nothing better left
        cec_to = int(max(30, (t0 + timeout) - time.monotonic()))
        if equiv_gate.equivalent(nl, cand, timeout=cec_to) is True:
            info["winner"] = label
            return cand, True, info
    return nl, False, info
