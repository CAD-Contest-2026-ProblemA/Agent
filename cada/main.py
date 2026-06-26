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
    p.add_argument("--doctor", dest="doctor", action="store_true",
                   help="Run environment pre-flight checks and exit")
    p.add_argument("--no-llm", dest="no_llm", action="store_true",
                   help="Rules-only mode: do not require an API key (no LLM fallback)")
    return p


def _config_search_dirs():
    """Where to look for configs/tools.yaml, lowest priority first.

    Works both from source and from a PyInstaller one-file binary: when frozen,
    look inside the bundle (sys._MEIPASS) and next to the executable; from source,
    look at the project root.
    """
    dirs = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            dirs.append(meipass)                       # bundled defaults
        dirs.append(os.path.dirname(os.path.abspath(sys.executable)))  # next to binary
    else:
        dirs.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return dirs


def _configure_tools(config, tools_file):
    """Register external-tool paths, lowest priority first so later wins."""
    # 1. auto-discovered configs/tools.yaml (bundle / next to exe / project root)
    for base in _config_search_dirs():
        for cand in (os.path.join(base, "configs", "tools.yaml"),
                     os.path.join(base, "configs", "tools.yml")):
            toolpaths.register_many(load_tools_file(cand))
    # 2. explicit --tools file
    if tools_file:
        toolpaths.register_many(load_tools_file(tools_file))
    # 3. tools: section inside the -config file (highest priority)
    toolpaths.register_many(getattr(config, "tools", {}) or {})


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.doctor:
        from .doctor import main as doctor_main
        return doctor_main()

    # Fail loud on misconfiguration instead of silently degrading to defaults.
    if not args.config:
        sys.stderr.write("error: -config <file> is required (path to the LLM config YAML)\n")
        return 2
    if not os.path.isfile(args.config):
        sys.stderr.write(f"error: -config file not found: {args.config}\n")
        return 2
    config = load_config(args.config)
    if not args.no_llm and not config.api_key:
        sys.stderr.write(
            f"error: no API key for provider '{config.provider}' in {args.config}\n"
            f"       set {config.provider}.api_key, or pass --no-llm for rules-only mode\n")
        return 2
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
