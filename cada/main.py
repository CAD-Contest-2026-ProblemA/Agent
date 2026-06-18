"""Entry point: ``cadaXXXX_alpha -config <config_file>``.

Reads natural-language requests from stdin, processes each on the current
design state, and emits #RESPONSE/#END frames to stdout (mirrored to
<case_name>.log).
"""

from __future__ import annotations

import argparse
import os
import sys

from .io_.config import load_config, load_tools_file
from .io_.protocol import Protocol
from .agent.agent import Agent
from . import toolpaths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM-assisted netlist EDA agent")
    p.add_argument("-config", "--config", dest="config", default=None,
                   help="Path to the LLM configuration YAML file")
    p.add_argument("-tools", "--tools", dest="tools", default=None,
                   help="Path to a tools file pinning abc/yosys/... locations")
    p.add_argument("--log-dir", dest="log_dir", default=".",
                   help="Directory for <case_name>.log files")
    return p


def _configure_tools(config, tools_file):
    """Register external-tool paths, lowest priority first so later wins."""
    # 1. auto-discovered configs/tools.yaml next to the project root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(root, "configs", "tools.yaml"),
                 os.path.join(root, "configs", "tools.yml")):
        toolpaths.register_many(load_tools_file(cand))
    # 2. explicit --tools file
    if tools_file:
        toolpaths.register_many(load_tools_file(tools_file))
    # 3. tools: section inside the -config file (highest priority)
    toolpaths.register_many(getattr(config, "tools", {}) or {})


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    _configure_tools(config, args.tools)
    agent = Agent(config)

    def on_case_name(name: str):
        agent.state.case_name = name

    proto = Protocol(handler=agent.handle, on_case_name=on_case_name,
                     out=sys.stdout, log_dir=args.log_dir)
    proto.run(sys.stdin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
