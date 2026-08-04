"""Routing-only probe: report which rule/handler each sentence routes to,
WITHOUT executing the handler body (no EDA ops, no design needed).

Usage:
    python3 scripts/route_check.py                 # reads route_test.txt
    python3 scripts/route_check.py my_sentences.txt

Input: one natural-language request per line; blank lines and lines starting
with '#' are skipped.  Lines that match no regex rule are reported as
"NO_REGEX (would go to LLM fallback)" — the LLM is never called here.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cada.io_.config import Config
from cada.agent.agent import Agent


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    path = args[0] if args else os.path.join(ROOT, "route_test.txt")
    if not os.path.isfile(path):
        sys.stderr.write(f"error: sentence file not found: {path}\n")
        return 2

    # Empty config: no API key -> LLM unavailable; only the rule table is used.
    agent = Agent(Config())

    width = 24
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        routed = next((fn.__name__ for rx, fn in agent.rules if rx.search(line)),
                      "NO_REGEX (would go to LLM fallback)")
        print(f"{routed:<{width}s} {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
