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


# The standard build ships with the rule table off by default: scripts/build.sh
# uses NO_RULES=1 and bundles this marker, which the flag below reads.  A file
# rather than a baked-in constant so the source and frozen paths share one
# mechanism, and so an existing binary can be flipped by dropping the marker
# next to it instead of being rebuilt.
NO_RULES_MARKER = "no_rules.flag"
NO_BM25_MARKER = "no_bm25.flag"


def _default_no_rules() -> bool:
    """Whether this build routes through the LLM unless told otherwise.

    CADA_NO_RULES wins over the marker, so a pure-LLM binary can be put back on
    the rules for one run without a rebuild (and the reverse).
    """
    env = os.environ.get("CADA_NO_RULES")
    if env is not None:
        return env.strip().lower() not in ("", "0", "false", "no", "off")
    return any(os.path.isfile(os.path.join(d, NO_RULES_MARKER))
               for d in _config_search_dirs())


def _default_bm25() -> bool:
    """Whether BM25 examples are appended to LLM requests by default.

    CADA_BM25 wins over the build marker.  The example bank remains bundled in
    a BM25-off build, so ``--bm25`` can enable it for one run without rebuilding.
    """
    env = os.environ.get("CADA_BM25")
    if env is not None:
        return env.strip().lower() not in ("", "0", "false", "no", "off")
    return not any(os.path.isfile(os.path.join(d, NO_BM25_MARKER))
                   for d in _config_search_dirs())


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
    default_no_rules = _default_no_rules()
    p.add_argument("--no-rules", dest="no_rules", action="store_true",
                   default=default_no_rules,
                   help="Skip the deterministic regex rules and route every "
                        "request through the LLM (for testing LLM coverage)"
                        + (" [this build's default]" if default_no_rules else ""))
    # The counter-flag is what keeps a pure-LLM build usable: without it the
    # baked default cannot be undone at runtime, so the two modes could not be
    # compared on one binary and the evaluator's rules-on run would silently be
    # a rules-off run.
    p.add_argument("--rules", dest="no_rules", action="store_false",
                   help="Force the regex rules back on"
                        + (" (this build defaults to --no-rules)" if default_no_rules
                           else " [default]"))
    default_bm25 = _default_bm25()
    p.add_argument("--bm25", dest="bm25", action="store_true",
                   default=default_bm25,
                   help="Append BM25-retrieved public examples to LLM requests"
                        + (" [default]" if default_bm25 else ""))
    p.add_argument("--no-bm25", dest="bm25", action="store_false",
                   help="Do not append retrieved examples to LLM requests"
                        + (" [this build's default]" if not default_bm25 else ""))
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
        return doctor_main(use_rules=not args.no_rules, use_bm25=args.bm25)

    # Fail loud on misconfiguration instead of silently degrading to defaults.
    if not args.config:
        sys.stderr.write("error: -config <file> is required (path to the LLM config YAML)\n")
        return 2
    if not os.path.isfile(args.config):
        sys.stderr.write(f"error: -config file not found: {args.config}\n")
        return 2
    config = load_config(args.config)
    if not config.api_key:
        sys.stderr.write(
            f"error: no API key for provider '{config.provider}' in {args.config}\n"
            f"       set {config.provider}.api_key\n")
        return 2
    _configure_tools(config, args.tools)
    agent = Agent(config, use_rules=not args.no_rules, use_bm25=args.bm25)

    # Announce the routing mode.  A rules-off run answers every line from the
    # LLM, so it has a different accuracy, a different cost, and a different
    # failure mode from a rules-on run -- and once a build can bake the default
    # in, the two are indistinguishable from the command line.  Refuse the one
    # combination that cannot work at all: with no reachable LLM, rules-off has
    # nothing left to route with and would answer every line with the same ack.
    if args.no_rules:
        if not agent.llm.available:
            sys.stderr.write(
                "error: --no-rules needs a reachable LLM, and none is configured "
                f"(provider '{config.provider}'; check the api_key and that the "
                "provider SDK is installed in this build)\n")
            return 2
    route_mode = "regex-first" if not args.no_rules else "pure-LLM (no-rules)"
    bm25_mode = "enabled" if args.bm25 else "disabled (static catalog only)"
    sys.stderr.write(f"cada: routing={route_mode}; BM25={bm25_mode}\n")

    def on_case_name(name: str):
        agent.state.case_name = name

    proto = Protocol(handler=agent.handle, on_case_name=on_case_name,
                     out=sys.stdout, log_dir=args.log_dir)
    proto.run(sys.stdin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
