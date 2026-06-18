#!/usr/bin/env python3
"""Local driver: feed a testcase's prompt.txt to the agent and show the log.

Usage:
    python3 scripts/run_local.py testcase/test01            # one case
    python3 scripts/run_local.py --all                      # every case
    python3 scripts/run_local.py testcase/test22 --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cada.io_.config import load_config
from cada.io_.protocol import Protocol
from cada.agent.agent import Agent


def run_case(case_dir: str, config_path: str, out_dir: str) -> str:
    prompt = os.path.join(case_dir, "prompt.txt")
    if not os.path.exists(prompt):
        return f"(no prompt.txt in {case_dir})"
    cfg = load_config(config_path)
    agent = Agent(cfg)
    buf = io.StringIO()
    proto = Protocol(handler=agent.handle,
                     on_case_name=lambda n: setattr(agent.state, "case_name", n),
                     out=buf, log_dir=out_dir)
    with open(prompt) as fh:
        proto.run(fh)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", help="path to a testcase directory")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--testcase-root", default="testcase")
    args = ap.parse_args()

    if args.all:
        for name in sorted(os.listdir(args.testcase_root)):
            d = os.path.join(args.testcase_root, name)
            if os.path.isdir(d):
                print(f"########## {name} ##########")
                print(run_case(d, args.config, args.out_dir))
    elif args.case:
        print(run_case(args.case, args.config, args.out_dir))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
