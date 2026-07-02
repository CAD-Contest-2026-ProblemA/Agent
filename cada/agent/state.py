"""Mutable design state for one testcase.

Holds the current netlist plus the snapshots the equivalence questions refer
to, the per-transform delta counters that later "how many ..." questions read,
and the most recent "report" result that a following "simplify the reported
..." instruction consumes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..netlist.ir import Netlist


class State:
    def __init__(self):
        self.current: Optional[Netlist] = None
        self.original: Optional[Netlist] = None      # first load (== "last loaded")
        self.last_loaded: Optional[Netlist] = None   # most recent load from disk
        self.pre: Optional[Netlist] = None           # snapshot before last transform
        self.case_name: Optional[str] = None
        self.design_dir: str = "."
        # Delta counters keyed by a tag; read by follow-up count queries.
        self.deltas: Dict[str, int] = {}
        # The last "report ... gates" result (list of gate names), consumed by
        # a following "simplify the reported ..." instruction.
        self.last_report: List[str] = []
        self.last_report_kind: Optional[str] = None
        # True iff every structural op since the last load is provably equivalent
        # by construction (buf insertion, rename, dangling removal, const_prop,
        # collapse_inverters, nand_const1_to_inv, merge_duplicates).
        # Cleared to False by any resynthesis op (basis remap, minimize_depth, etc.).
        self.provably_equiv: bool = True

    # ----- snapshots -----------------------------------------------------
    def set_loaded(self, nl: Netlist):
        self.current = nl
        self.original = nl.snapshot()
        self.last_loaded = nl.snapshot()
        self.pre = None
        self.provably_equiv = True

    def begin_transform(self):
        """Snapshot the current design before a structural edit (for rollback
        and for 'equivalent to pre-transformation netlist' queries)."""
        if self.current is not None:
            self.pre = self.current.snapshot()

    def rollback(self):
        if self.pre is not None:
            self.current = self.pre
            self.pre = None

    # ----- deltas --------------------------------------------------------
    def record_delta(self, tag: str, n: int):
        self.deltas[tag] = n

    def get_delta(self, tag: str, default: int = 0) -> int:
        return self.deltas.get(tag, default)
