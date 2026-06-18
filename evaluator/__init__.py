"""Local evaluator for the 40 Problem-A testcases.

Because the contest does not ship golden answers, this evaluator focuses on what
can be checked *independently* and rigorously:

* HARD requirements (the binary 0-score gate), auto-derived from each request:
  - functional equivalence to the original (ABC cec — an external oracle),
  - structural bounds ("no gate/signal drives > K"),
  - gate-basis purity ("only NAND and NOT", whole design or a cone),
  - the written netlist is valid (round-trips through our own parser).
* DERIVED answers that have an objective value computable a second way
  (gate-type counts and PI/PO counts via the raw .v file).
* GOLDEN regression: every response is diffed against a pinned baseline so
  drift is caught; drop official answers into evaluator/golden/ to grade for real.
* OPTIMIZE cost: the achieved cost (max depth / gate count) is reported per
  optimize request so you can compare runs (lower = better rank).
"""
