#!/usr/bin/env python3
"""Verify canonical routing params and duplicate labels without an API call."""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cada.llm.allowed_intents import (  # noqa: E402
    infer_delta_kind, validate_intent_object,
)
from cada.llm.example_contract import (  # noqa: E402
    INTENT_ONLY_EXCEPTIONS, ExampleContractError, validate_example_banks,
)

DEFAULT_BANKS = (
    os.path.join(ROOT, "cada", "llm", "examples.jsonl"),
    os.path.join(ROOT, "cada", "llm", "public_examples.jsonl"),
)


def _load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ExampleContractError(f"{path}:{line_no}: invalid JSON: {exc}")
    return rows


def _expect(params, intent, expected):
    obj, err = validate_intent_object({"intent": intent, "params": params})
    if err or obj != {"intent": intent, "params": expected}:
        raise ExampleContractError(
            f"normalizer regression for {intent}: params={params!r}, "
            f"expected={expected!r}, got={obj!r}, error={err!r}")


def _verify_normalizers():
    _expect({"file": "x.v", "dir": "testcase/x/"}, "load_design",
            {"file": "x.v", "dir": "testcase/x"})
    _expect({"file": "x.v", "dir": "/"}, "load_design",
            {"file": "x.v", "dir": "/"})
    _expect({"dir": "OUTPUT"}, "list_ports", {"dir": "output"})
    _expect({"k": 4}, "insert_buffers", {"k": 4, "scope": "gate"})
    _expect({"k": 4, "scope": "SIGNAL"}, "insert_buffers",
            {"k": 4, "scope": "signal"})
    _expect({"k": 4, "net": "n1"}, "insert_buffers", {"k": 4, "net": "n1"})
    _expect({"mode": "Dedicated", "net": "n1"}, "insert_buffers",
            {"mode": "dedicated", "net": "n1"})
    _expect({"kind": "dangling"}, "delta_count", {"kind": "removed"})
    _expect({"kind": "or_eliminated"}, "delta_count",
            {"kind": "const_eliminated"})
    obj, err = validate_intent_object({
        "intent": "delta_count", "params": {"kind": "frobnicate"},
    })
    if obj is not None or not err:
        raise ExampleContractError("unknown delta_count.kind was not rejected")
    delta_cases = {
        "How many BUF gates were added?": "buffers_added",
        "How many buffers were removed?": "buffers_removed",
        "How many gates did buffer removal eliminate?": "buffers_removed",
        "How many buffers were eliminated?": "buffers_removed",
        "How many buffer gates did that operation eliminate?":
            "buffers_removed",
        "How many buffers were pruned?": "buffers_removed",
        "How many buffers were swept away?": "buffers_removed",
        "Were buffers added or removed?": "net_change",
        "How many buffers did constant propagation eliminate?":
            "const_eliminated",
    }
    for text, expected in delta_cases.items():
        actual = infer_delta_kind(text)
        if actual != expected:
            raise ExampleContractError(
                f"delta inference regression: {text!r} -> {actual!r}, "
                f"expected {expected!r}")


def _verify_duplicate_guards():
    def rejected(rows, expected):
        try:
            validate_example_banks({"synthetic": rows})
        except ExampleContractError as exc:
            if expected in str(exc):
                return
            raise ExampleContractError(
                f"duplicate guard raised the wrong error: {exc}") from exc
        raise ExampleContractError(
            f"duplicate guard did not reject {expected}")

    count = {
        "case": "check1", "line": 1, "text": "Count every gate.",
        "op": "count_gates", "kind": "safe", "params": {},
    }
    count_intent_conflict = {
        "case": "check2", "line": 1, "text": "  Count   every gate.  ",
        "op": "total_gate_count", "kind": "safe", "params": {},
    }
    rejected([count, count_intent_conflict], "different intents")

    buffers_gate = {
        "case": "check3", "line": 1, "text": "Apply a four-load bound.",
        "op": "insert_buffers", "kind": "safe",
        "params": {"k": 4, "scope": "gate"},
    }
    buffers_signal = {
        "case": "check4", "line": 1, "text": "Apply  a four-load bound.",
        "op": "insert_buffers", "kind": "safe",
        "params": {"k": 4, "scope": "signal"},
    }
    rejected([buffers_gate, buffers_signal], "different objects")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("banks", nargs="*", default=list(DEFAULT_BANKS))
    args = parser.parse_args()
    try:
        _verify_normalizers()
        _verify_duplicate_guards()
        banks = {os.path.relpath(path, ROOT): _load(path) for path in args.banks}
        includes_main = any(
            os.path.abspath(path) == os.path.abspath(DEFAULT_BANKS[0])
            for path in args.banks)
        stats = validate_example_banks(
            banks,
            allow_intent_only=(INTENT_ONLY_EXCEPTIONS if includes_main else ()))
    except (OSError, ExampleContractError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("normalizers: PASS")
    print(f"banks:       PASS ({stats['rows']} rows / "
          f"{stats['unique_texts']} normalized texts / "
          f"{stats['duplicate_groups']} duplicate groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
