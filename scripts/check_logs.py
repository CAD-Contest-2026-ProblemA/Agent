#!/usr/bin/env python3
"""Validate the agent's <case>.log files against protocol framing and golden.

    python3 scripts/check_logs.py --log-dir /tmp/logs test01 test02 ...

For each case:
  * the log parses as #RESPONSE i .. #END i frames with ids exactly 1..N,
    where N is the number of non-empty prompt lines;
  * every frame body, normalised the same way the evaluator normalises
    responses, matches the corresponding golden baseline line.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.evaluate import _norm, GOLDEN_DIR, TC_ROOT

FRAME_RE = re.compile(r"#RESPONSE\s+(\d+)\s*\n(.*?)\n#END\s+\1\n", re.DOTALL)


def check_case(case: str, log_dir: str):
    problems = []
    log_path = os.path.join(log_dir, f"{case}.log")
    if not os.path.exists(log_path):
        return [f"missing log file {log_path}"]
    text = open(log_path).read()

    frames = {}
    for m in FRAME_RE.finditer(text):
        i = int(m.group(1))
        if i in frames:
            problems.append(f"duplicate frame id {i}")
        frames[i] = m.group(2)

    leftover = FRAME_RE.sub("", text).strip()
    if leftover:
        problems.append(f"unframed text in log ({len(leftover)} chars)")

    with open(os.path.join(TC_ROOT, case, "prompt.txt")) as fh:
        n_prompt = sum(1 for ln in fh if ln.strip())
    if sorted(frames) != list(range(1, n_prompt + 1)):
        problems.append(f"frame ids {sorted(frames)} != 1..{n_prompt}")

    gpath = os.path.join(GOLDEN_DIR, f"{case}.txt")
    if os.path.exists(gpath):
        gold = open(gpath).read().split("\n")
        for i in range(1, n_prompt + 1):
            body = _norm(frames.get(i, ""))
            want = gold[i - 1] if i - 1 < len(gold) else "<missing>"
            if body != want:
                problems.append(f"L{i} differs from golden:\n"
                                f"    log:    {body[:160]}\n"
                                f"    golden: {want[:160]}")
    else:
        problems.append("no golden baseline to compare against")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--log-dir", required=True)
    args = ap.parse_args()

    bad = 0
    for case in args.cases:
        probs = check_case(case, args.log_dir)
        if probs:
            bad += 1
            print(f"{case}  FAIL")
            for p in probs:
                print(f"  - {p}")
        else:
            print(f"{case}  OK")
    print(f"\n{len(args.cases) - bad}/{len(args.cases)} logs clean")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
