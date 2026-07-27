"""Run the agent over the testcases and report independently-checkable results.

    python3 evaluator/evaluate.py                 # all cases
    python3 evaluator/evaluate.py test21 test33   # selected cases
    python3 evaluator/evaluate.py --update-golden  # snapshot responses as baseline
    python3 evaluator/evaluate.py --verbose        # show every check

Exit code is non-zero if any HARD requirement fails.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import subprocess                                           # noqa: E402

from cada.io_.config import load_config                     # noqa: E402
from cada.agent.agent import Agent                          # noqa: E402
from cada.main import _configure_tools                      # noqa: E402
from cada.analysis import depth as depth_mod                # noqa: E402
from cada.netlist import reader                             # noqa: E402
from evaluator import checks                                # noqa: E402
from evaluator.requirements import derive                   # noqa: E402

TC_ROOT = os.path.join(ROOT, "testcase")
GOLDEN_DIR = os.path.join(ROOT, "evaluator", "golden")


def _tty():
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _tty() else s


GREEN = lambda s: _c("32", s)
RED = lambda s: _c("31", s)
YEL = lambda s: _c("33", s)
DIM = lambda s: _c("2", s)
BOLD = lambda s: _c("1", s)


# ----- run one case, capturing per-step responses and snapshots ----------
def run_case(case: str, config_path, use_rules: bool = True):
    case_dir = os.path.join(TC_ROOT, case)
    prompt = os.path.join(case_dir, "prompt.txt")
    cfg = load_config(config_path)
    _configure_tools(cfg, None)
    agent = Agent(cfg, use_rules=use_rules)

    lines, responses, specs, snaps = [], [], [], {}
    transform_seen_before = []  # whether a transform happened before line i
    seen_transform = False
    with open(prompt) as fh:
        raw = [ln.strip() for ln in fh if ln.strip()]
    for i, line in enumerate(raw, start=1):
        spec = derive(line)
        resp = agent.handle(line, i)
        lines.append(line)
        responses.append(resp)
        specs.append(spec)
        transform_seen_before.append(seen_transform)
        if any(r.kind in ("basis", "gate_absent", "max_fanout", "max_fanout_net")
               for r in spec.requirements) and agent.state.current is not None:
            snaps[i] = agent.state.current.snapshot()
        if spec.is_transform:
            seen_transform = True

    vfile = os.path.join(case_dir, f"{case}.v")
    return {
        "case": case, "lines": lines, "responses": responses, "specs": specs,
        "snaps": snaps, "transform_before": transform_seen_before,
        "original": agent.state.original, "final": agent.state.current,
        "vfile": vfile,
    }


def _parse_frames(stdout: str, n: int):
    """Pull each #RESPONSE <id> .. #END <id> body out of an executable's stdout,
    returned as a list indexed by id (1..n)."""
    bodies = {}
    for m in re.finditer(r"#RESPONSE\s+(\d+)\s*\n(.*?)\n?#END\s+\1",
                         stdout, re.DOTALL):
        bodies[int(m.group(1))] = m.group(2)
    return [bodies.get(i, "") for i in range(1, n + 1)], len(bodies)


def run_case_exe(case: str, exe: str, config_path: str, timeout: int = 320,
                 use_rules: bool = True):
    """Drive the REAL executable (the PyInstaller binary or the wrapper) as a
    subprocess and evaluate its actual stdout + written netlist.

    No per-step netlist snapshots are available here (we only see stdout and the
    final written *_out.v), so the structural checks run on the final design —
    which is exactly what the contest grades.
    """
    case_dir = os.path.join(TC_ROOT, case)
    with open(os.path.join(case_dir, "prompt.txt")) as fh:
        raw = [ln.strip() for ln in fh if ln.strip()]
    specs, tb, seen = [], [], False
    for line in raw:
        s = derive(line)
        specs.append(s)
        tb.append(seen)
        if s.is_transform:
            seen = True

    # run the executable in the current (sandbox) cwd; out.v lands here
    try:
        cmd = [os.path.abspath(exe), "-config", config_path]
        if not use_rules:
            cmd.append("--no-rules")
        proc = subprocess.run(cmd,
                              input="\n".join(raw) + "\n",
                              capture_output=True, text=True, timeout=timeout)
        stdout = proc.stdout
    except Exception as exc:
        stdout = ""
    responses, nframes = _parse_frames(stdout, len(raw))

    final = None
    # the agent writes the output next to the input (testcase/<case>/), and
    # historically also cwd-root; check both
    for outv in (os.path.join("testcase", case, f"{case}_out.v"),
                 f"{case}_out.v"):
        if os.path.exists(outv):
            try:
                final = reader.parse_file(outv)
            except Exception:
                final = None
            break

    original = reader.parse_file(os.path.join(case_dir, f"{case}.v"))
    return {
        "case": case, "lines": raw, "responses": responses, "specs": specs,
        "snaps": {}, "transform_before": tb,
        "original": original, "final": final,
        "vfile": os.path.join(case_dir, f"{case}.v"),
        "exe_frames": (nframes, len(raw)),
    }


# ----- evaluate one case -------------------------------------------------
class Check:
    def __init__(self, group, label, status, detail=""):
        self.group = group       # HARD | DERIVED | OPT | GOLDEN
        self.label = label
        self.status = status     # PASS | FAIL | INFO | DRIFT | NA
        self.detail = detail


def _cost(final, opt):
    if opt.metric == "gate_count":
        return len(final.gates)
    if opt.metric == "cone_depth" and opt.output:
        return depth_mod.depth_of_cone(final, opt.output)
    return depth_mod.global_max_depth(final)


def evaluate_case(run, golden_dir, update_golden=False):
    out = []
    specs, lines, resps = run["specs"], run["lines"], run["responses"]
    has_transform = any(s.is_transform for s in specs)

    # HARD (exe mode only): the executable produced one frame per request
    if "exe_frames" in run:
        got, want = run["exe_frames"]
        out.append(Check("HARD", f"protocol frames {got}/{want}",
                         "PASS" if got == want else "FAIL"))

    # HARD: functional equivalence of the submitted design to the original
    if has_transform and run["original"] is not None and run["final"] is not None:
        eq = checks.equiv(run["original"], run["final"])
        st = "PASS" if eq is True else ("FAIL" if eq is False else "NA")
        out.append(Check("HARD", "equiv-to-original",
                         st, "" if eq is True else "ABC could not decide" if eq is None else "NOT EQUIVALENT"))

    # HARD: the written netlist is valid / round-trips
    if run["final"] is not None:
        ok = checks.valid_netlist(run["final"])
        out.append(Check("HARD", "valid-output-netlist", "PASS" if ok else "FAIL"))

    # HARD: per-step basis / fanout bounds
    for i, spec in enumerate(specs, start=1):
        nl = run["snaps"].get(i) or run["final"]
        if nl is None:
            continue
        for r in spec.requirements:
            if r.kind == "basis":
                ok = checks.basis_pure(nl, r.basis, r.scope_output)
                got = sorted(checks.basis_types(nl, r.scope_output))
                scope = f" (cone {r.scope_output})" if r.scope_output else ""
                out.append(Check("HARD", f"L{i} basis {r.basis}{scope}",
                                 "PASS" if ok else "FAIL",
                                 "" if ok else f"types={got}"))
            elif r.kind == "max_fanout":
                mx = checks.max_fanout(nl, r.include_pi)
                out.append(Check("HARD", f"L{i} max-fanout<={r.k}"
                                 f"{' (incl PI)' if r.include_pi else ' (gates)'}",
                                 "PASS" if mx <= r.k else "FAIL", f"got {mx}"))
            elif r.kind == "max_fanout_net":
                mx = checks.fanout_of(nl, r.net)
                out.append(Check("HARD", f"L{i} fanout({r.net})<={r.k}",
                                 "PASS" if mx <= r.k else "FAIL", f"got {mx}"))
            elif r.kind == "gate_absent":
                n = checks.type_count_in(nl, r.gate_type, r.scope_output)
                scope = f" in cone {r.scope_output}" if r.scope_output else ""
                out.append(Check("HARD", f"L{i} no {r.gate_type.upper()}{scope}",
                                 "PASS" if n == 0 else "FAIL",
                                 "" if n == 0 else f"{n} remain"))

    # DERIVED: objectively-computable answers (counts, ports)
    for i, (line, resp) in enumerate(zip(lines, resps), start=1):
        low = line.lower()
        pre_transform = not run["transform_before"][i - 1]
        if ("broken down by gate type" in low or "count all the gates" in low) and pre_transform:
            exp = checks.independent_type_counts(run["vfile"])
            got = checks.response_counts(resp)
            mism = [t for t in exp if t in got and got[t] != exp[t]]
            missing = [t for t in exp if exp[t] and t not in got]
            ok = not mism and not missing
            out.append(Check("DERIVED", f"L{i} gate-type counts",
                             "PASS" if ok else "FAIL",
                             "" if ok else f"mismatch={mism} missing={missing}"))
        if re.search(r"(number of|how many) primary inputs? and", low):
            ni, no = checks.independent_port_counts(run["vfile"])
            ok = re.search(r"\b%d\b" % ni, resp) and re.search(r"\b%d\b" % no, resp)
            out.append(Check("DERIVED", f"L{i} PI/PO counts ({ni}/{no})",
                             "PASS" if ok else "FAIL"))

    # OPT: report achieved cost
    for i, spec in enumerate(specs, start=1):
        if spec.optimize and run["final"] is not None:
            c = _cost(run["final"], spec.optimize)
            out.append(Check("OPT", f"L{i} cost[{spec.optimize.metric}]", "INFO", str(c)))

    # GOLDEN: regression vs pinned baseline
    gpath = os.path.join(golden_dir, f"{run['case']}.txt")
    if update_golden:
        os.makedirs(golden_dir, exist_ok=True)
        with open(gpath, "w") as fh:
            fh.write("\n".join(_norm(r) for r in resps))
    elif os.path.exists(gpath):
        gold = open(gpath).read().split("\n")
        cur = [_norm(r) for r in resps]
        drift = sum(1 for a, b in zip(gold, cur) if a != b) + abs(len(gold) - len(cur))
        out.append(Check("GOLDEN", "responses vs baseline",
                         "PASS" if drift == 0 else "DRIFT",
                         "" if drift == 0 else f"{drift} line(s) changed"))
    return out


def _norm(s: str) -> str:
    return " ".join(s.split())


# ----- reporting ---------------------------------------------------------
def _mark(status):
    return {"PASS": GREEN("✓"), "FAIL": RED("✗"), "DRIFT": YEL("~"),
            "NA": YEL("?"), "INFO": DIM("·")}.get(status, status)


def _enter_sandbox():
    """Run from a throwaway dir so the agent's `testNN_out.v` writes never land
    in the repo.  Code / testcases / golden are reached by absolute paths;
    only the working directory changes (the testcase symlink lets the prompts'
    relative load paths resolve).  Returns the temp dir or None if unavailable.
    """
    tmp = tempfile.mkdtemp(prefix="cada_eval_")
    try:
        # copy (not symlink) testcase/ so the agent's output files (written next
        # to the input) land inside the sandbox and never touch the repo
        shutil.copytree(os.path.join(ROOT, "testcase"),
                        os.path.join(tmp, "testcase"))
        os.chdir(tmp)
        return tmp
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="testcase names (default: all)")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "api_key.yaml"))
    ap.add_argument("--update-golden", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-sandbox", action="store_true",
                    help="write outputs in the current directory instead of a temp dir")
    ap.add_argument("--exe", default=None,
                    help="run this executable (e.g. dist/cada1125_alpha or "
                         "./cada1125_alpha) as a subprocess instead of in-process")
    ap.add_argument("--no-rules", dest="no_rules", action="store_true",
                    help="route every request through the LLM (skip the regex rules)")
    args = ap.parse_args(argv)
    if args.exe:
        args.exe = os.path.abspath(args.exe)

    # absolutise the config path before any chdir into the sandbox
    if args.config:
        args.config = os.path.abspath(args.config)

    cases = args.cases or sorted(d for d in os.listdir(TC_ROOT)
                                 if os.path.isdir(os.path.join(TC_ROOT, d)))

    orig_cwd = os.getcwd()
    sandbox = None if args.no_sandbox else _enter_sandbox()
    try:
        return _run_cases(args, cases)
    finally:
        if sandbox:
            os.chdir(orig_cwd)
            shutil.rmtree(sandbox, ignore_errors=True)


def _run_cases(args, cases) -> int:
    total_hard = total_hard_fail = 0
    failed_cases = []

    for case in cases:
        try:
            if args.exe:
                run = run_case_exe(case, args.exe, args.config,
                                   use_rules=not args.no_rules)
            else:
                run = run_case(case, args.config, use_rules=not args.no_rules)
            results = evaluate_case(run, GOLDEN_DIR, args.update_golden)
        except Exception as exc:
            print(f"{BOLD(case)}  {RED('ERROR')} {exc}")
            failed_cases.append(case)
            continue

        groups = {}
        for r in results:
            groups.setdefault(r.group, []).append(r)
        hard = groups.get("HARD", [])
        hp = sum(1 for r in hard if r.status == "PASS")
        hf = sum(1 for r in hard if r.status == "FAIL")
        total_hard += len(hard)
        total_hard_fail += hf
        if hf:
            failed_cases.append(case)

        der = groups.get("DERIVED", [])
        dp = sum(1 for r in der if r.status == "PASS")
        opt = groups.get("OPT", [])
        gold = groups.get("GOLDEN", [])
        gstat = gold[0].status if gold else "-"

        summary = (f"{BOLD(case):<22} "
                   f"HARD {hp}/{len(hard)}"
                   + (RED(f"  {hf} FAIL") if hf else "")
                   + (f"  DERIVED {dp}/{len(der)}" if der else "")
                   + (f"  GOLDEN {_mark(gstat) if gold else '-'}" if gold else "")
                   + ("  OPT " + ",".join(o.detail for o in opt) if opt else ""))
        print(summary)

        for r in results:
            show = args.verbose or r.status in ("FAIL", "DRIFT", "NA")
            if show and r.group != "OPT":
                line = f"    {_mark(r.status)} [{r.group}] {r.label}"
                if r.detail:
                    line += DIM(f"  ({r.detail})")
                print(line)

    print("\n" + BOLD("Summary"))
    if args.update_golden:
        print(f"  golden baseline written for {len(cases)} case(s)")
    print(f"  HARD checks: {total_hard - total_hard_fail}/{total_hard} passed")
    if total_hard_fail:
        print("  " + RED(f"{total_hard_fail} HARD failure(s) in: {', '.join(sorted(set(failed_cases)))}"))
    else:
        print("  " + GREEN("no HARD-requirement failures"))
    return 1 if total_hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
