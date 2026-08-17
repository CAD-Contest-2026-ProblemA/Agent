#!/usr/bin/env python3
"""Evaluate pure-LLM routing accuracy against a labelled sentence set.

Calls only ``Fallback.translate()`` — no handler runs and no netlist is loaded,
so it can score labels from either routing-only or ordinary testcases.

The evaluation target and retrieval bank are deliberately separate.  This
allows a public testcase export to be scored while retrieval still uses the
original examples.jsonl bank (and without changing runtime retrieval).

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
    python3 scripts/route_eval.py --eval-bank cada/llm/public_examples.jsonl \
        --cases test01-40 --sample 20 --retriever bm25
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
from cada.llm.allowed_intents import validate_intent_object
from cada.llm.client import LLMClient
from cada.llm.fallback import Fallback
from cada.llm.retrieval import build_retriever

BANK = os.path.join(ROOT, "cada", "llm", "examples.jsonl")
# USD per million tokens (input, output), for the pre-run estimate only.
PRICES = {"claude-haiku-4-5": (1.00, 5.00), "claude-sonnet-5": (3.00, 15.00),
          "claude-opus-5": (5.00, 25.00), "gpt-4o-mini": (0.15, 0.60)}


def load_bank(path=BANK):
    if not os.path.isfile(path):
        raise SystemExit(f"error: evaluation/retrieval bank missing: {path}")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def select(rows, spec: str):
    """Select an inclusive range such as ``test101-171`` or ``test01-40``."""
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


def spread_sample(rows, count: int):
    """Return a deterministic sample spread across the complete selection."""
    if count >= len(rows):
        return rows
    if count == 1:
        return rows[:1]
    last = len(rows) - 1
    return [rows[round(i * last / (count - 1))] for i in range(count)]


def expected_object(row):
    """Return the normalized labelled object, or None for intent-only labels.

    ``Fallback.translate`` returns the output of ``validate_intent_object``.
    Running labelled params through that same validator makes params accuracy
    an exact Python-dict comparison after the production normalization rules
    (for example, gate types are lowercase, bases uppercase, and ``k`` is an
    integer).  A row with no ``params`` key is an intent-only legacy label and
    is omitted from both the params and complete-object denominators.
    """
    if "params" not in row:
        return None
    obj, err = validate_intent_object({"intent": row.get("op"),
                                       "params": row["params"]})
    if err:
        where = f'{row.get("case", "?")}:{row.get("line", "?")}'
        raise SystemExit(f"error: invalid evaluation label at {where}: {err}")
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="test101-171")
    ap.add_argument("--eval-bank", "--evaluation-bank", "--bank", dest="eval_bank",
                    default=BANK,
                    help="JSONL to score (default: cada/llm/examples.jsonl)")
    ap.add_argument("--retrieval-bank", default=BANK,
                    help="independent JSONL used only for retrieval "
                         "(default: cada/llm/examples.jsonl)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N")
    ap.add_argument("--sample", type=int, default=None,
                    help="deterministic sample of N spread across the selected rows "
                         "(unlike --limit, which takes the first N only)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="configs/api_key.yaml")
    ap.add_argument("--provider", choices=("openai", "anthropic"), default=None,
                    help="override the config's provider for this run")
    ap.add_argument("--out", default=None, help="write per-sentence results as JSONL")
    ap.add_argument("--retriever", choices=("none","bm25","onnx","union","union-rrf","qdrant","qdrant-hybrid","auto"),
                    default="none", help="append retrieved examples (default: none)")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    if args.sample is not None and args.sample < 1:
        ap.error("--sample must be a positive integer")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be a positive integer")

    cfg = load_config(args.config)
    if args.provider:
        cfg.provider = args.provider
    client = LLMClient(cfg)
    if not client.available:
        raise SystemExit("error: LLM client unavailable — check provider/api_key.")

    rows = select(load_bank(args.eval_bank), args.cases)
    if args.sample is not None:
        rows = spread_sample(rows, args.sample)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit(f"error: no sentences matched --cases {args.cases}")
    targets = [(row, expected_object(row)) for row in rows]

    p_in, p_out = PRICES.get(cfg.model, (None, None))
    est = ""
    if p_in:
        # ~7k prompt tokens per call; caching makes repeats ~0.1x.
        est = f"  (rough cost with caching: ${len(rows) * 7000 * 0.1 / 1e6 * p_in:.2f})"
    retriever = None
    if args.retriever != "none":
        retrieval_rows = load_bank(args.retrieval_bank)
        retriever = build_retriever(args.retriever, bank=retrieval_rows)
    print(f"model={cfg.model}  cases={args.cases}  sentences={len(rows)}"
          f"  workers={args.workers}  retriever="
          f"{retriever.name if retriever else 'none'}"
          f"{'/k=%d' % args.top_k if retriever else ''}{est}\n", flush=True)
    print(f"evaluation_bank={args.eval_bank}", flush=True)
    if args.retriever != "none":
        print(f"retrieval_bank={args.retrieval_bank}\n", flush=True)

    local = threading.local()
    stop_event = threading.Event()

    class EvaluationClient:
        """Stop corrective/queued calls after the first infrastructure failure."""

        @property
        def available(self):
            return client.available

        def complete(self, *complete_args, **complete_kwargs):
            if stop_event.is_set():
                return None
            out = client.complete(*complete_args, **complete_kwargs)
            if client.degraded:
                stop_event.set()
            return out

    evaluation_client = EvaluationClient()

    def result(row, truth, *, obj=None, error=None, fallback_error=None,
               aborted=False):
        pred_intent = (obj or {}).get("intent")
        pred_params = obj.get("params") if obj is not None else None
        params_scorable = truth is not None
        return {
            **row,
            # Keep ``pred`` as the scalar intent for compatibility with old
            # result consumers; the remaining fields expose full routing.
            "pred": pred_intent,
            "pred_params": pred_params,
            "pred_object": obj,
            "expected_object": truth,
            "intent_ok": pred_intent == row["op"],
            "params_ok": ((pred_params == truth["params"])
                          if params_scorable and obj is not None
                          else (False if params_scorable else None)),
            "object_ok": ((obj == truth)
                          if params_scorable and obj is not None
                          else (False if params_scorable else None)),
            "fallback_error": fallback_error,
            "error": error,
            "aborted": aborted,
        }

    def route(target):
        row, truth = target
        if stop_event.is_set():
            return result(row, truth, aborted=True)

        # Fallback's cache key is only the sentence text.  Keep a separate
        # instance per testcase so a cached answer can never bypass a changed
        # leave-one-case-out exclusion context.
        fb_by_case = getattr(local, "fb_by_case", None)
        if fb_by_case is None:
            fb_by_case = local.fb_by_case = {}
        fb = fb_by_case.get(row["case"])
        if fb is None:
            fb = fb_by_case[row["case"]] = Fallback(
                evaluation_client, retriever=retriever, top_k=args.top_k,
                exclude_case=row["case"])
        # Leave-one-testcase-out when the target and retrieval bank overlap;
        # for independent banks this is harmless and usually excludes none.
        try:
            obj = fb.translate(row["text"])
        except Exception as exc:
            stop_event.set()
            error = f"{type(exc).__name__}: {exc}"
            return result(row, truth, error=error)
        return result(row, truth, obj=obj, fallback_error=fb.last_error)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for i, res in enumerate(pool.map(route, targets), 1):
            results.append(res)
            if i % 50 == 0:
                print(f"  ... {i}/{len(rows)}", flush=True)

    errs = [r for r in results if r["error"] is not None]
    aborted = [r for r in results if r["aborted"]]
    invalid_reasons = []
    if client.degraded:
        invalid_reasons.append(
            f"LLM client reported {client.degraded} exhausted/fatal completion(s)")
    if errs:
        invalid_reasons.append(f"{len(errs)} evaluator worker exception(s)")
    if aborted:
        invalid_reasons.append(
            f"{len(aborted)} target(s) not attempted after fail-fast")
    run_valid = not invalid_reasons
    invalid_reason = "; ".join(invalid_reasons) or None
    for result in results:
        result["run_valid"] = run_valid
        result["run_invalid_reason"] = invalid_reason

    ok = [r for r in results if r["intent_ok"]]
    fails = [r for r in results if not r["intent_ok"]]
    gave_up = [r for r in fails if r["pred"] in (None, "noop")]
    wrong = [r for r in fails if r not in gave_up]

    def accuracy(field, kind=None):
        scored = [r for r in results
                  if r[field] is not None and
                  (kind is None or r.get("kind") == kind)]
        correct = [r for r in scored if r[field]]
        return (f"{len(correct)}/{len(scored)} = "
                f"{100*len(correct)/len(scored):.1f}%") if scored else "n/a"

    print(f"\n{'='*64}")
    if run_valid:
        print(f"intent accuracy  {accuracy('intent_ok')}")
        print(f"  safe           {accuracy('intent_ok', 'safe')}")
        print(f"  hard           {accuracy('intent_ok', 'hard')}")
        print(f"params accuracy  {accuracy('params_ok')}  (exact, normalized params)")
        print(f"object accuracy  {accuracy('object_ok')}  (exact intent + params)")
        unlabelled = sum(r["params_ok"] is None for r in results)
        if unlabelled:
            print(f"  omitted        {unlabelled} intent-only row(s) without params labels")
        print(f"\nintent failures {len(fails)}")
        if fails:
            print(f"  gave up (noop/none)  {len(gave_up):4d}  "
                  f"({100*len(gave_up)/len(fails):.0f}%)")
            print(f"  wrong op             {len(wrong):4d}  "
                  f"({100*len(wrong)/len(fails):.0f}%)")

        if gave_up:
            print("\ngave-up by op:")
            for op, n in collections.Counter(r["op"] for r in gave_up).most_common(10):
                print(f"  {n:4d}  {op}")
        if wrong:
            print("\nconfusion pairs (truth -> predicted):")
            for (truth, pred), n in collections.Counter(
                    (r["op"], r["pred"]) for r in wrong).most_common(15):
                print(f"  {n:4d}  {truth:22s} -> {pred}")
    else:
        print("accuracy        NOT REPORTED (invalid run)")
        print(f"\n{'!'*64}\nINVALID RUN: {invalid_reason}.\n"
              "No accuracy is reported because infrastructure failures must not "
              "be counted as routing mistakes. Fix the API/rate-limit issue and "
              f"re-run before continuing from a sample to the full set.\n{'!'*64}",
              file=sys.stderr)
        if errs:
            print(f"first worker exception: {errs[0]['error']}", file=sys.stderr)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")
    # A sample can be chained to the full run with ``&&``; infrastructure
    # degradation stops the chain instead of blessing a meaningless score.
    return 0 if run_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
