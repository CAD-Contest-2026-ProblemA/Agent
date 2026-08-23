"""Environment doctor — pre-flight check for the CADA agent.

Run it before an evaluation to see what's installed/missing:

    python3 -m cada.doctor          # or:  python3 scripts/doctor.py
    ./cada0001_alpha --doctor

Checks, in order:
  * Effective runtime modes: pure LLM vs regex-first, and BM25 on vs off.
  * Python: uv installed?  -> .venv present?  -> packages inside .venv.
    If uv or .venv is missing, the whole Python section FAILS and the host
    Python is intentionally NOT inspected (by design).
  * External tools: abc (required) and yosys (fallback) resolved via the same
    precedence the agent uses (configs/tools.yaml, env, $PATH), then actually
    executed (a cec smoke test catches a present-but-broken binary, e.g. glibc).
  * Config files and (optional) LLM key.
  * The agent package itself: import + parse a real testcase.

Exit code is 0 when there are no hard failures, 1 otherwise.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----- pretty printing ---------------------------------------------------
def _tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _tty() else s


def _green(s): return _c("32", s)
def _red(s): return _c("31", s)
def _yellow(s): return _c("33", s)
def _dim(s): return _c("2", s)
def _bold(s): return _c("1", s)


class Report:
    def __init__(self):
        self.fails = 0
        self.warns = 0

    def section(self, title: str):
        print("\n" + _bold(title))

    def ok(self, label: str, detail: str = ""):
        print(f"  {_green('✓')} {label}" + (f"  {_dim(detail)}" if detail else ""))

    def warn(self, label: str, hint: str = ""):
        self.warns += 1
        print(f"  {_yellow('!')} {label}")
        if hint:
            print(f"      {_dim('→ ' + hint)}")

    def fail(self, label: str, hint: str = ""):
        self.fails += 1
        print(f"  {_red('✗')} {label}")
        if hint:
            print(f"      {_dim('→ ' + hint)}")

    def skip(self, label: str):
        print(f"  {_dim('· ' + label)}")


# ----- helpers -----------------------------------------------------------
def _run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        return 999, str(exc)


def _find_uv():
    p = shutil.which("uv")
    if p:
        return p
    for c in (os.path.expanduser("~/.local/bin/uv"), "/usr/local/bin/uv"):
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None


def _venv_python():
    return os.path.join(ROOT, ".venv", "bin", "python")


def _pkg_version(python: str, module: str):
    code = (f"import {module} as m; "
            f"print(getattr(m, '__version__', '?'))")
    rc, out = _run([python, "-c", code], timeout=30)
    return (rc == 0), out.strip()


# ----- sections ----------------------------------------------------------
def check_runtime_modes(rep: Report, use_rules: bool, use_bm25: bool):
    """Report the modes that would apply to this invocation."""
    rep.section("Runtime modes")
    if use_rules:
        rep.ok("routing: regex-first",
               "matching rules run first; the LLM handles unmatched requests")
    else:
        rep.ok("routing: pure LLM (no-rules)",
               "every request is classified by the LLM")
    if use_bm25:
        rep.ok("BM25: enabled",
               "up to 50 public examples are appended to requests sent to the LLM")
    else:
        rep.ok("BM25: disabled",
               "LLM requests use the static intent catalog without retrieved examples")


def check_python(rep: Report):
    rep.section("Python environment (uv + .venv)")
    if getattr(sys, "frozen", False):
        rep.ok("running as a standalone binary",
               f"Python {sys.version.split()[0]} is bundled; uv/.venv not needed")
        return
    uv = _find_uv()
    if not uv:
        rep.fail("uv is not installed",
                 "curl -LsSf https://astral.sh/uv/install.sh | sh")
        rep.skip("skipping .venv / package checks (host Python not inspected by design)")
        return
    rc, out = _run([uv, "--version"])
    rep.ok("uv installed", f"{uv}  ({out.strip()})" if rc == 0 else uv)

    venv_py = _venv_python()
    if not os.path.exists(venv_py):
        rep.fail(".venv not found at " + os.path.join(ROOT, ".venv"),
                 "run:  bash setup.sh")
        rep.skip("skipping package checks (no .venv; host Python not inspected by design)")
        return
    rc, out = _run([venv_py, "--version"])
    rep.ok(".venv present", f"{venv_py}  ({out.strip()})")

    # packages (all optional — the benchmark runs without them)
    for pkg, mod in (("PyYAML", "yaml"), ("openai", "openai"),
                     ("anthropic", "anthropic")):
        ok, ver = _pkg_version(venv_py, mod)
        if ok:
            rep.ok(f"package {pkg}", ver)
        else:
            rep.warn(f"package {pkg} not installed in .venv (optional)",
                     "uv pip install --python .venv/bin/python -r requirements.txt")


def _configure_tool_registry():
    from .io_.config import load_config, load_tools_file
    from . import toolpaths
    for cand in (os.path.join(ROOT, "configs", "tools.yaml"),
                 os.path.join(ROOT, "configs", "tools.yml")):
        toolpaths.register_many(load_tools_file(cand))
    cfg = load_config(os.path.join(ROOT, "configs", "api_key.yaml"))
    toolpaths.register_many(getattr(cfg, "tools", {}) or {})
    return cfg


_ABC_A = ".model t\n.inputs a b\n.outputs o\n.names a b o\n11 1\n.end\n"
_ABC_B = ".model t\n.inputs a b\n.outputs o\n.names a b t\n11 1\n.names t o\n0 0\n.end\n"


def _abc_smoke(abc: str) -> bool:
    d = tempfile.mkdtemp(prefix="cada_doctor_")
    pa, pb = os.path.join(d, "a.blif"), os.path.join(d, "b.blif")
    try:
        open(pa, "w").write(_ABC_A)
        open(pb, "w").write(_ABC_B)
        rc, out = _run([abc, "-q", f'cec "{pa}" "{pb}"'], timeout=30)
        return "Networks are equivalent" in out
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _abc_resyn2_smoke(abc: str, rc_path: str) -> bool:
    d = tempfile.mkdtemp(prefix="cada_doctor_")
    pa = os.path.join(d, "a.blif")
    try:
        open(pa, "w").write(_ABC_A)
        rc, out = _run([abc, "-q",
                        f'source "{rc_path}"; read_blif "{pa}"; strash; resyn2; print_stats'],
                       timeout=30)
        return "unknown command" not in out.lower() and rc == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def check_tools(rep: Report):
    rep.section("External EDA tools (abc / yosys)")
    try:
        _configure_tool_registry()
        from .equiv.abc_bridge import find_abc
        from .equiv.yosys_bridge import find_yosys
    except Exception as exc:
        rep.fail(f"could not load tool resolution ({exc})")
        return

    abc = find_abc()
    if not abc:
        rep.fail("abc NOT found (required for equivalence & optimization)",
                 "set it in configs/tools.yaml:\n          tools:\n            abc: /path/to/abc/abc")
    else:
        rep.ok("abc found", abc)
        rc_path = os.path.join(os.path.dirname(abc), "abc.rc")
        if os.path.exists(rc_path):
            rep.ok("abc.rc present (resyn2/dc2 aliases)", rc_path)
        else:
            rep.warn("abc.rc not found next to the abc binary",
                     "place abc.rc beside abc, else resyn2/dc2 optimization is unavailable")
        if _abc_smoke(abc):
            rep.ok("abc runs", "cec smoke test passed")
        else:
            rep.fail("abc is present but failed to execute (glibc / arch mismatch?)",
                     "use an abc built for this machine's OS")
        if os.path.exists(rc_path) and not _abc_resyn2_smoke(abc, rc_path):
            rep.warn("resyn2 alias did not resolve (optimization may be degraded)")

    ys = find_yosys()
    if not ys:
        rep.warn("yosys not found (fallback only; the benchmark works without it)",
                 "optional: set tools.yosys in configs/tools.yaml")
    else:
        rep.ok("yosys found", ys)
        rc, out = _run([ys, "--version"], timeout=30)
        if rc == 0:
            rep.ok("yosys runs", out.strip().splitlines()[0] if out.strip() else "")
        else:
            rep.warn("yosys present but failed to run")


def check_config(rep: Report):
    rep.section("Configuration files")
    p = os.path.join(ROOT, "configs", "api_key.yaml")
    if os.path.exists(p):
        rep.ok("configs/api_key.yaml present")
    else:
        rep.warn("configs/api_key.yaml missing (the -config file; git-ignored)",
                 "copy the template: cp configs/example.api_key.yaml configs/api_key.yaml")
    pt = os.path.join(ROOT, "configs", "tools.yaml")
    if os.path.exists(pt):
        rep.ok("configs/tools.yaml present")
    else:
        rep.warn("configs/tools.yaml missing (git-ignored per-machine file)",
                 "copy the template: cp configs/example.tools.yaml configs/tools.yaml")
    try:
        from .io_.config import load_config
        cfg = load_config(p)
        if cfg.tools:
            rep.ok("tool paths configured", ", ".join(f"{k}={v}" for k, v in cfg.tools.items()))
        if cfg.api_key:
            rep.ok(f"LLM api_key set for provider '{cfg.provider}'")
        else:
            rep.warn("no LLM api_key set (the agent errors out at startup)",
                     "paste your real key into configs/api_key.yaml")
    except Exception as exc:
        rep.warn(f"could not parse configs/api_key.yaml ({exc})")


def check_package(rep: Report):
    rep.section("Agent package")

    # Frozen binary: cada is bundled — check in-process. (Do NOT spawn
    # sys.executable, which is the binary itself; it would re-enter the agent
    # and block on stdin.)  testcase/ isn't bundled, so parse one from cwd if
    # present.
    if getattr(sys, "frozen", False):
        try:
            from cada.netlist.reader import parse_file
        except Exception as exc:
            rep.fail("cada failed to import", str(exc)[:200])
            return
        for cand in ("testcase/test01/test01.v",):
            if os.path.exists(cand):
                try:
                    nl = parse_file(cand)
                    rep.ok("cada imports and parses a testcase",
                           f"test01 -> {len(nl.gates)} gates")
                    return
                except Exception as exc:
                    rep.fail("cada failed to parse a testcase", str(exc)[:200])
                    return
        rep.ok("cada package imports OK", "(no testcase in cwd to parse)")
        return

    # Source/uv layout: run the check under the .venv python.
    py = _venv_python()
    if not os.path.exists(py):
        py = sys.executable
    tcase = os.path.join(ROOT, "testcase", "test01", "test01.v")
    code = ("import sys; sys.path.insert(0, %r);"
            "from cada.netlist.reader import parse_file;"
            "nl=parse_file(%r);"
            "print(len(nl.gates))" % (ROOT, tcase))
    rc, out = _run([py, "-c", code], timeout=60)
    if rc == 0 and out.strip().isdigit():
        rep.ok("cada imports and parses a testcase", f"test01 -> {out.strip()} gates")
    elif not os.path.exists(tcase):
        rep.warn("testcase/test01 not present, skipped parse check")
    else:
        rep.fail("cada failed to import / parse", out.strip()[:200])


def check_retrieval(rep: Report, use_bm25: bool = True):
    """Is the example bank actually reachable, and does it carry params?

    A build that drops cada/llm/public_examples.jsonl does not crash: retrieval
    turns itself off and the agent keeps answering, just less accurately and
    with nothing in the output to say why.  That failure shipped once already,
    so it gets a check rather than trust.
    """
    rep.section("Example retrieval")
    if not use_bm25:
        rep.skip("BM25 bank probe skipped (disabled for this run)")
        return
    try:
        from cada.llm.retrieval import build_retriever, format_block
    except Exception as exc:
        rep.fail("cada.llm.retrieval failed to import", str(exc)[:200])
        return
    r = build_retriever("bm25")
    if r is None:
        rep.fail("no example bank found — retrieval is DISABLED",
                 "the binary was built without cada/llm/public_examples.jsonl; "
                 "run scripts/export_public_examples.py and rebuild")
        return
    probe = "Does every path from n2 to n40 pass through gate g0?"
    hits = r.top_k(probe, k=50)
    if not hits:
        rep.fail("example bank present but returned no hits", "bank may be empty")
        return
    withp = sum(1 for h in hits if h.get("params") is not None)
    rep.ok(f"{type(r).__name__} loaded",
           f"{len(hits)} examples for a probe query, {withp} with params")
    block = format_block(hits)
    if '"params"' not in block and "→ {" not in block:
        rep.warn("retrieved examples render without params",
                 "the prompt will teach intent names but not their arguments")


def main(argv=None, use_rules=None, use_bm25=None) -> int:
    if use_rules is None or use_bm25 is None:
        # Direct ``python -m cada.doctor`` calls do not pass through the agent
        # argument parser.  Resolve the same marker/env defaults here; calls
        # from cada.main pass the already-parsed effective values instead.
        from .main import _default_bm25, _default_no_rules
        if use_rules is None:
            use_rules = not _default_no_rules()
        if use_bm25 is None:
            use_bm25 = _default_bm25()
    print(_bold("CADA environment doctor"))
    print(_dim(f"project root: {ROOT}"))
    rep = Report()
    check_runtime_modes(rep, use_rules=use_rules, use_bm25=use_bm25)
    check_python(rep)
    check_tools(rep)
    check_config(rep)
    check_package(rep)
    check_retrieval(rep, use_bm25=use_bm25)

    print("\n" + _bold("Summary"))
    if rep.fails == 0 and rep.warns == 0:
        print("  " + _green("All checks passed."))
    else:
        status = []
        if rep.fails:
            status.append(_red(f"{rep.fails} failure(s)"))
        if rep.warns:
            status.append(_yellow(f"{rep.warns} warning(s)"))
        print("  " + ", ".join(status))
        if rep.fails == 0:
            print("  " + _green("No hard failures — the agent should run."))
    return 1 if rep.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
