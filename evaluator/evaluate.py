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
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cada.io_.config import load_config                     # noqa: E402
from cada.agent.agent import Agent                          # noqa: E402
from cada.main import _configure_tools                      # noqa: E402
from cada.analysis import depth as depth_mod                # noqa: E402
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
def run_case(case: str, config_path):
    case_dir = os.path.join(TC_ROOT, case)
    prompt = os.path.join(case_dir, "prompt.txt")
    cfg = load_config(config_path)
    _configure_tools(cfg, None)
    agent = Agent(cfg)

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="testcase names (default: all)")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--update-golden", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    cases = args.cases or sorted(d for d in os.listdir(TC_ROOT)
                                 if os.path.isdir(os.path.join(TC_ROOT, d)))

    total_hard = total_hard_fail = 0
    failed_cases = []

    for case in cases:
        try:
            run = run_case(case, args.config)
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
