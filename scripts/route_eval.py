#!/usr/bin/env python3
"""Evaluate pure-LLM routing accuracy against the labelled sentence set.

Calls only ``Fallback.translate()`` — no handler runs, no netlist is loaded, so
this works on the routing-only testcases (test91-171 ship no .v designs).

Failures are split into the two kinds that need different fixes:

  * gave up   — answered ``noop`` or produced nothing, i.e. the model did not
                recognise the phrasing even though the op was in the catalog
  * wrong op  — picked a different op, i.e. the model could not separate two
                neighbouring ops

THIS SPENDS API CREDIT: one call per sentence.  Start with --limit.

Usage:
    python3 scripts/route_eval.py --limit 100
    python3 scripts/route_eval.py                      # full test101-171
    python3 scripts/route_eval.py --cases test91-100   # the other split
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cada.io_.config import load_config
from cada.llm.client import LLMClient
from cada.llm.fallback import Fallback
from cada.llm.retrieval import build_retriever

BANK = os.path.join(ROOT, "cada", "llm", "examples.jsonl")
# USD per million tokens (input, output), for the pre-run estimate only.
PRICES = {"claude-haiku-4-5": (1.00, 5.00), "claude-sonnet-5": (3.00, 15.00),
          "claude-opus-5": (5.00, 25.00), "gpt-4o-mini": (0.15, 0.60)}


def load_bank(path=BANK):
    if not os.path.isfile(path):
        raise SystemExit(f"error: {path} missing — run scripts/export_examples.py first")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def select(rows, spec: str):
    """spec is an inclusive testcase range like 'test101-171' or 'test91-100'."""
    m = re.fullmatch(r"test(\d+)-(\d+)", spec)
    if not m:
        raise SystemExit(f"error: bad --cases {spec!r}; expected e.g. test101-171")
    lo, hi = int(m.group(1)), int(m.group(2))
    out = []
    for r in rows:
        n = re.fullmatch(r"test(\d+)", r["case"])
        if n and lo <= int(n.group(1)) <= hi:
            out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="test101-171")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N")
    ap.add_argument("--sample", type=int, default=None,
                    help="evenly-spread sample of N (deterministic; covers every op, "
                         "unlike --limit which takes the first N testcases only)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="configs/api_key.yaml")
    ap.add_argument("--provider", choices=("openai", "anthropic"), default=None,
                    help="override the config's provider for this run")
    ap.add_argument("--out", default=None, help="write per-sentence results as JSONL")
    ap.add_argument("--retriever", choices=("none","bm25","onnx","union","union-rrf","qdrant","qdrant-hybrid","auto"),
                    default="none", help="append retrieved examples (default: none)")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.provider:
        cfg.provider = args.provider
    client = LLMClient(cfg)
    if not client.available:
        raise SystemExit("error: LLM client unavailable — check provider/api_key.")

    rows = select(load_bank(), args.cases)
    if args.sample and args.sample < len(rows):
        stride = len(rows) / args.sample
        rows = [rows[int(i * stride)] for i in range(args.sample)]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"error: no sentences matched --cases {args.cases}")

    p_in, p_out = PRICES.get(cfg.model, (None, None))
    est = ""
    if p_in:
        # ~7k prompt tokens per call; caching makes repeats ~0.1x.
        est = f"  (rough cost with caching: ${len(rows) * 7000 * 0.1 / 1e6 * p_in:.2f})"
    retriever = build_retriever(args.retriever) if args.retriever != "none" else None
    print(f"model={cfg.model}  cases={args.cases}  sentences={len(rows)}"
          f"  workers={args.workers}  retriever="
          f"{retriever.name if retriever else 'none'}"
          f"{'/k=%d' % args.top_k if retriever else ''}{est}\n", flush=True)

    local = threading.local()

    def route(row):
        # One Fallback per worker: translate() writes instance state
        # (cache, last_error), so sharing one across threads would race.
        fb = getattr(local, "fb", None)
        if fb is None:
            fb = local.fb = Fallback(client, retriever=retriever, top_k=args.top_k)
        # Leave-one-testcase-out: the bank contains this very sentence, so
        # without the exclusion the run measures lookup, not routing.
        fb.exclude_case = row["case"]
        try:
            obj = fb.translate(row["text"])
        except Exception as exc:
            return {**row, "pred": None, "error": str(exc)}
        return {**row, "pred": (obj or {}).get("intent"), "error": None}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for i, res in enumerate(pool.map(route, rows), 1):
            results.append(res)
            if i % 50 == 0:
                print(f"  ... {i}/{len(rows)}", flush=True)

    ok = [r for r in results if r["pred"] == r["op"]]
    fails = [r for r in results if r["pred"] != r["op"]]
    gave_up = [r for r in fails if r["pred"] in (None, "noop")]
    wrong = [r for r in fails if r not in gave_up]

    def rate(subset, kind=None):
        s = [r for r in subset if kind is None or r["kind"] == kind]
        t = [r for r in results if kind is None or r["kind"] == kind]
        return f"{len(s)}/{len(t)} = {100*len(s)/len(t):.1f}%" if t else "n/a"

    print(f"\n{'='*64}")
    print(f"success        {rate(ok)}")
    print(f"  safe         {rate(ok, 'safe')}")
    print(f"  hard         {rate(ok, 'hard')}")
    print(f"\nfailures {len(fails)}")
    if fails:
        print(f"  gave up (noop/none)  {len(gave_up):4d}  ({100*len(gave_up)/len(fails):.0f}%)")
        print(f"  wrong op             {len(wrong):4d}  ({100*len(wrong)/len(fails):.0f}%)")

    if gave_up:
        print("\ngave-up by op:")
        for op, n in collections.Counter(r["op"] for r in gave_up).most_common(10):
            print(f"  {n:4d}  {op}")
    if wrong:
        print("\nconfusion pairs (truth -> predicted):")
        for (t, p), n in collections.Counter(
                (r["op"], r["pred"]) for r in wrong).most_common(15):
            print(f"  {n:4d}  {t:22s} -> {p}")

    errs = [r for r in results if r["error"]]
    if errs:
        print(f"\nWARNING: {len(errs)} sentence(s) raised; first: {errs[0]['error']}",
              file=sys.stderr)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
