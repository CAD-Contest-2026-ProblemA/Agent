"""Entry point: ``cadaXXXX_alpha -config <config_file>``.

Reads natural-language requests from stdin, processes each on the current
design state, and emits #RESPONSE/#END frames to stdout (mirrored to
<case_name>.log).
"""

from __future__ import annotations

import argparse
import sys

from .io_.config import load_config
from .io_.protocol import Protocol
from .agent.agent import Agent


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-assisted netlist EDA agent")
    p.add_argument("-config", "--config", dest="config", default=None,
                   help="Path to the LLM configuration YAML file")
    p.add_argument("--log-dir", dest="log_dir", default=".",
                   help="Directory for <case_name>.log files")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    agent = Agent(config)

    def on_case_name(name: str):
        agent.state.case_name = name

    proto = Protocol(handler=agent.handle, on_case_name=on_case_name,
                     out=sys.stdout, log_dir=args.log_dir)
    proto.run(sys.stdin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
