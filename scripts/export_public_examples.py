#!/usr/bin/env python3
"""Export the public test01-test40 requests as a labelled JSONL bank.

The labels deliberately describe the deterministic router's *current* first
match, including historical behaviours whose natural-language interpretation
could reasonably differ.  They are therefore suitable as a reproducible
regression/evaluation bank, not as a replacement for ``examples.jsonl``.

Every generated param object is checked by both the conservative labeller in
``scripts/label_params.py`` and the runtime intent validator.  The exporter
also replays every testcase in an isolated temporary directory and compares
all 459 normalized responses with ``evaluator/golden/testNN.txt``.

Usage:
    python3 scripts/export_public_examples.py
    python3 scripts/export_public_examples.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Match, Tuple
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cada.agent.agent import Agent  # noqa: E402
from cada.agent.state import State  # noqa: E402
from cada.analysis import paths, sequential  # noqa: E402
from cada.io_.config import Config  # noqa: E402
from cada.llm.allowed_intents import (  # noqa: E402
    ALLOWED_INTENTS,
    GATE_TYPES,
    validate_intent_object,
)
from cada.llm.example_contract import (  # noqa: E402
    INTENT_ONLY_EXCEPTIONS, validate_example_banks,
)
from cada.main import _configure_tools  # noqa: E402
from scripts.label_params import extract as extract_params  # noqa: E402
from scripts.label_params import verify as verify_params  # noqa: E402


OUT = ROOT / "cada" / "llm" / "public_examples.jsonl"
CASES = tuple(f"test{i:02d}" for i in range(1, 41))
EXPECTED_ROWS = 459

# A first-match handler normally has a single canonical LLM intent.  The four
# handlers resolved in _intent_for() instead inspect the sentence before doing
# their work, so their canonical intent has to follow the same branch.
HANDLER_TO_INTENT = {
    "h_begin": "begin_case",
    "h_load": "load_design",
    "h_write": "write_design",
    "h_count_all": "count_gates",
    "h_total": "total_gate_count",
    "h_count_ports": "count_ports",
    "h_cone_gate_count": "cone_gate_count",
    "h_cone_type": "cone_type_counts",
    "h_cone_depth": "cone_depth",
    "h_gate_info": "gate_info",
    "h_list_type": "list_type",
    "h_list_pi": "list_ports",
    "h_list_po": "list_ports",
    "h_const1_gates": "const1_gates",
    "h_report_const": "report_const_gates",
    "h_path_plain": "path_exists",
    "h_path_avoid": "path_exists",
    "h_path_avoid2": "path_exists",
    "h_path_avoid3": "path_exists",
    "h_enum_paths": "enumerate_paths",
    "h_len0": "length_zero_paths",
    "h_dominator": "dominator",
    "h_articulation": "articulation",
    "h_cut": "is_cut",
    "h_depth_ab": "max_depth_between",
    "h_depth_ab2": "max_depth_between",
    "h_depth_ab3": "max_depth_between",
    "h_reg2reg_depth": "reg_to_reg_depth",
    "h_pi2d_depth": "pi_to_dff_depth",
    "h_pi2po_depth": "pi_to_po_depth",
    "h_global_depth": "global_max_depth",
    "h_depth_gt": "outputs_depth_gt",
    "h_on_maxpath": "gate_on_max_path",
    "h_fanout": "fanout",
    "h_driven_by": "gates_driven_by",
    "h_successors": "successors",
    "h_tfanin": "transitive_fanin",
    "h_tfanout": "transitive_fanout",
    "h_reachable": "reachable_from",
    "h_max_fanout": "max_fanout_of",
    "h_shared": "shared_cone",
    "h_connected_out": "connected_to_output",
    "h_connected_renamed": "connected_to_net",
    "h_sig_equiv": "signals_equivalent",
    "h_sig_equiv2": "signals_equivalent",
    "h_const_out": "output_constant",
    "h_depends": "depends_on",
    "h_boolean": "boolean_equation",
    "h_symmetric": "symmetric",
    "h_nand_pair": "exists_nand_pair",
    "h_ffs_clock": "ffs_on_clock",
    "h_reg2reg_paths": "reg_to_reg_paths",
    "h_enable_hold": "enable_hold_report",
    "h_enable_hold_count": "enable_hold_count",
    "h_basis": "convert_basis",
    "h_xnor_nor": "xnor_to_nor",
    "h_xor_nand": "xor_to_nand",
    "h_xor_aoi": "xor_to_aoi",
    "h_nand_inv": "nand_const1_to_inv",
    "h_constprop": "const_propagate",
    "h_collapse": "collapse_inverters",
    "h_dangling": "remove_dangling",
    "h_floating_count": "floating_count",
    "h_check_floating_ports": "check_floating",
    "h_merge": "merge_duplicates",
    "h_rename": "rename",
    "h_buffers_fanout": "insert_buffers",
    "h_buffers_signal": "insert_buffers",
    "h_buffers_dedicated": "insert_buffers",
    "h_opt_cone": "optimize_cone",
    "h_opt_depth": "minimize_depth",
    "h_verify": "verify_equivalence",
    "h_delta": "delta_count",
}

# These are intentionally pinned to current handler/Golden behaviour.  If a
# future router changes one, --check should fail loudly rather than silently
# recasting this regression dataset as semantic ground truth.
KNOWN_CURRENT_BEHAVIOUR = {
    ("test25", 4): ("h_basis", "convert_basis"),
    ("test32", 12): ("h_fanout", "fanout"),
    ("test38", 9): ("h_check_dangling", "remove_dangling"),
    ("test38", 19): ("h_cone_gate_count", "cone_gate_count"),
}

# The committed Golden files predate several deliberate answer/engine fixes.
# A replay must still compare every response, but these exact coordinates are
# reported as pinned baseline drift instead of making the exporter unusable.
# Comparing the complete coordinate set means a new drift, or the disappearance
# of one after Golden is refreshed, both force this list to be reviewed.
# Values are (normalized Golden SHA-256, normalized actual SHA-256,
# explanation).  Pinning both texts as well as their coordinates keeps
# a later change at an already-known line from passing unnoticed.
KNOWN_GOLDEN_DRIFTS = {
    ("test24", 7): ("4bf854fb66ddfd8a0cdda5c0c49a5b9489baa672e065f31583e2a135f6c8166e",
                     "04990c2a2812ea52b05024fdc895cbf9e6dc865a6e0af2453fcbd0f189326d54",
                     "write now reports restoration of renamed_gate"),
    ("test25", 9): ("1be07fbf53faa3ad752a017f4ca6058f7062253c6faf3543d80ca226f29b2dd9",
                     "a32282ca6eccd6596307e65c98399125c256a4918263384e004693e248e40930",
                     "write now reports restoration of both renamed identifiers"),
    ("test29", 6): ("7a8725de4142cde213c0ce6cfc8aca0a761898b1e7b2540b06f07aa78a06dec5",
                     "76e18dc0fcc904038b983db58b7b3decbd6da43cccd67ac58e3eec63f349457c",
                     "current basis rewrite exposes 1228 rather than 1216 collapses"),
    ("test29", 7): ("a16f306b81320aaa8c1824ecda005e9a597ae1263156f852d9ec75a1342d9e40",
                     "796f2bae63eb7f09110a534e1fd4156a07b550aedcbc7069387cea1f9c103554",
                     "the preceding collapse leaves 31 rather than 43 duplicates"),
    ("test32", 6): ("e81b50019cb7e565aae89f011876df80dc627e5fd3137dccc15126c8b6d93ba0",
                     "f73d9f96516d2542d430e6447d5eef9476b7dfba2364c08f0fea1b37f67128a0",
                     "register query now enumerates complete paths, not connections"),
    ("test34", 10): ("d4cfecae0d28318b849ddb08feabc12dcfc91c3c3103c6b2ce90661a55b01015",
                       "17e2ff574b856df6219f473e421c0020d0ef3e70f5a3d54c4298fa7b088723de",
                       "superlative answer now reports every tied output"),
    ("test35", 8): ("88dc62d3203ae4b0c685c984c6f8a07539b10e1b20787af844b525be7f08467b",
                      "7e4c69fca2ce96c0ac4033b61d8f18cf26b3bb2d8b270eb98fdc6ff67f7a57e7",
                      "superlative answer now reports every tied output"),
    ("test35", 14): ("0cc4f098d438c1da8dfed0b9db529e9f835440be8e806921c1d4a6a71f415661",
                       "211ced19e6157b70f3150c20ea19a54798bb4dd0f299eeab7f479da6697e7196",
                       "current transforms expose 54 rather than 51 collapses"),
    ("test37", 9): ("dcb1be2e186db73613d6c36a0767d599495289b5be0c374fa6dd2c7d02268116",
                      "eef58742578a13bd4893a40e272a109eb91027c6168c860e105fb47c85ec7311",
                      "register query now enumerates complete paths, not connections"),
    ("test40", 7): ("79b5d2389e2ec09375dc97da5070ac7801a9b582f34bd57233137ed5ca245f40",
                      "dcccca25e2bd36b46dcd76510e728197c7e71aa15a161ae6e3b709b3d67dae6a",
                      "current basis rewrite exposes one additional collapse"),
    ("test40", 12): ("17c5830842ffe493a5fea3dce274d6b979985bf94f04f2817eb7b72f49e94541",
                       "10d88cf52fe213a259892e26ce020cf731079e1e8bdf2f51e11f5b982ed397b5",
                       "large enable/hold result now uses a complete list file"),
    ("test40", 15): ("1e4dc2e56291b4469b5abaa767ccef1262c6821084e19c25c862788b6506e5b5",
                       "61d9ca25aa74f8aac55ea994e90b2789b7cfe7b8233e4d168298ed7870e268cc",
                       "superlative answer now reports every tied output"),
}

_ACTION_WORDS = (
    "added", "removed", "eliminated", "merged", "collapsed", "inserted",
    "converted", "found",
)


class ExportError(RuntimeError):
    pass


def _first_match(agent: Agent, text: str) -> Tuple[str, Match[str]]:
    for rx, fn in agent.rules:
        match = rx.search(text)
        if match:
            return fn.__name__, match
    raise ExportError(f"no deterministic regex handler for: {text!r}")


def _intent_for(handler: str, text: str) -> str:
    """Mirror handler branches that select between canonical operations."""
    low = text.lower()
    if handler == "h_deepest":
        intent = "deepest_output" if "deep" in low else "largest_fanin_cone"
    elif handler == "h_highest_fanout":
        intent = "highest_fanout_pi" if "primary input" in low else "highest_fanout_net"
    elif handler == "h_check_dangling":
        intent = ("remove_dangling" if re.search(
            r"remove|delete|excise|prune|eliminate", text, re.I)
                  else "check_dangling")
    elif handler == "h_count_or_delta":
        gate = re.search(r"how many\s+(\w+)\s+gates?", text, re.I)
        current_count = gate and gate.group(1).lower() in GATE_TYPES
        intent = ("delta_count" if any(w in low for w in _ACTION_WORDS)
                  or not current_count else "count_type")
    else:
        try:
            intent = HANDLER_TO_INTENT[handler]
        except KeyError as exc:
            raise ExportError(f"unmapped deterministic handler {handler!r}") from exc
    if intent not in ALLOWED_INTENTS:
        raise ExportError(f"handler {handler!r} maps to unknown intent {intent!r}")
    return intent


def _params_for(text: str, handler: str, intent: str,
                match: Match[str]) -> Dict:
    """Extract params, overriding only values dictated by handler mechanics."""
    if handler == "h_count_or_delta" and intent == "count_type":
        m = re.search(r"how many\s+(\w+)\s+gates?", text, re.I)
        params = {"type": m.group(1).lower()} if m else None
        # The handler scopes a present-tense count when the sentence names a
        # cone; label_params owns the conservative cone-name extraction.
        labelled = extract_params(text, intent)
        if labelled and "scope" in labelled:
            params["scope"] = labelled["scope"]
    elif handler == "h_path_plain":
        params = {
            "a": match.group(match.lastindex - 1),
            "b": match.group(match.lastindex),
        }
    elif handler == "h_rename":
        # h_rename echoes and dispatches on the actual noun in regex group 2;
        # preserving "signal" rather than inferring "wire" matters to a
        # params-level routing evaluation even though both rename a net.
        params = {
            "kind": match.group(2).lower(),
            "old": match.group(3),
            "new": match.group(4),
        }
    elif handler == "h_buffers_fanout":
        params = {
            "k": int(match.group(3)),
            "scope": "signal" if match.group(1).lower() in ("signal", "net") else "gate",
        }
    elif handler == "h_buffers_signal":
        params = {"net": match.group(1), "k": int(match.group(2))}
    elif handler == "h_buffers_dedicated":
        params = {"net": match.group(1), "mode": "dedicated"}
    else:
        params = extract_params(text, intent)

    if params is None:
        raise ExportError(
            f"could not extract trustworthy params for {handler}/{intent}: {text!r}")
    err = verify_params(text, intent, params)
    if err:
        raise ExportError(f"invalid params for {handler}/{intent}: {err}; {text!r}")
    clean, err = validate_intent_object({"intent": intent, "params": params})
    if err or clean is None:
        raise ExportError(f"runtime validation failed for {intent}: {err}; {text!r}")
    dropped = sorted(set(params) - set(clean["params"]))
    if dropped:
        raise ExportError(f"runtime validator silently dropped params {dropped} for {intent}")
    # Preserve the extractor's deterministic insertion order, then append any
    # canonical defaults materialized by production validation.
    ordered = list(params) + sorted(set(clean["params"]) - set(params))
    return {key: clean["params"][key] for key in ordered}


def build_rows() -> List[dict]:
    agent = Agent(Config())
    rows: List[dict] = []
    observed_known = {}
    handlers = set()

    for case in CASES:
        prompt = ROOT / "testcase" / case / "prompt.txt"
        if not prompt.is_file():
            raise ExportError(f"missing prompt: {prompt}")
        for line_no, raw in enumerate(prompt.read_text(encoding="utf-8").splitlines(), 1):
            text = raw.strip()
            if not text:
                continue
            handler, match = _first_match(agent, text)
            intent = _intent_for(handler, text)
            params = _params_for(text, handler, intent, match)
            handlers.add(handler)
            key = (case, line_no)
            if key in KNOWN_CURRENT_BEHAVIOUR:
                observed_known[key] = (handler, intent)
            rows.append({
                "case": case,
                "line": line_no,
                "text": text,
                "op": intent,
                "kind": "safe",
                "params": params,
            })

    if len(rows) != EXPECTED_ROWS:
        raise ExportError(f"expected {EXPECTED_ROWS} rows, generated {len(rows)}")
    if {row["case"] for row in rows} != set(CASES):
        raise ExportError("generated rows do not cover exactly test01-test40")
    if observed_known != KNOWN_CURRENT_BEHAVIOUR:
        raise ExportError(
            "known current-behaviour routes changed:\n"
            f"expected {KNOWN_CURRENT_BEHAVIOUR}\nobserved {observed_known}")

    print(f"routes: {len(rows)} rows / {len(CASES)} cases / {len(handlers)} handlers")
    return rows


def _serialize(rows: List[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _check_contract(rows: List[dict]) -> None:
    main_path = ROOT / "cada" / "llm" / "examples.jsonl"
    try:
        main_rows = [json.loads(line) for line in
                     main_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        stats = validate_example_banks(
            {"examples": main_rows, "public_examples": rows},
            allow_intent_only=INTENT_ONLY_EXCEPTIONS)
    except (OSError, ValueError) as exc:
        raise ExportError(f"example contract failed: {exc}") from exc
    print(f"contract: PASS ({stats['rows']} rows / "
          f"{stats['unique_texts']} normalized texts)")


def _check_bank(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ExportError(f"missing generated bank: {path}")
    actual = path.read_text(encoding="utf-8")
    if actual == expected:
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        print(f"bank:   PASS ({display_path})")
        return
    exp_lines, got_lines = expected.splitlines(), actual.splitlines()
    first = next((i for i, pair in enumerate(zip(exp_lines, got_lines), 1)
                  if pair[0] != pair[1]), min(len(exp_lines), len(got_lines)) + 1)
    raise ExportError(
        f"generated bank is stale at line {first} "
        f"(expected {len(exp_lines)} rows, found {len(got_lines)}); "
        "run scripts/export_public_examples.py")


def _norm(text: str) -> str:
    return " ".join(text.split())


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replay_golden(rows: List[dict]) -> None:
    """Replay both rule and canonical dispatch without touching the repo."""
    cfg = Config()  # every row must hit a regex; no API key is needed or used
    _configure_tools(cfg, None)
    rule_agent = Agent(cfg)
    canonical_agent = Agent(cfg, use_rules=False)
    grouped = {case: [row for row in rows if row["case"] == case]
               for case in CASES}
    checked = canonical_checked = 0
    golden_drifts = {}

    # Some public designs contain millions of paths.  The response depends on
    # the exact dynamic-programming count and whether the complete stream
    # succeeds, but not on the attachment bytes.  During this routing-label
    # replay, simulate a successful large stream so --check cannot consume
    # several GB of temporary disk.  Small inline register-path answers still
    # use the real streamer because their path text is part of the response.
    real_reg_stream = sequential.stream_reg_to_reg_paths

    def complete_path_stream(_nl, _a, _b, _fh, cap):
        return max(0, cap - 1)

    def complete_reg_stream(nl, out_fh, cap):
        if cap <= Agent.PATH_INLINE + 1:
            return real_reg_stream(nl, out_fh, cap)
        return max(0, cap - 1)

    with patch.object(paths, "stream_paths", complete_path_stream), \
            patch.object(sequential, "stream_reg_to_reg_paths", complete_reg_stream), \
            tempfile.TemporaryDirectory(prefix="cada_public_examples_") as tmp:
        tmp_root = Path(tmp)
        for case in CASES:
            case_dir = tmp_root / "testcase" / case
            case_dir.mkdir(parents=True)
            source = ROOT / "testcase" / case / f"{case}.v"
            if not source.is_file():
                raise ExportError(f"missing design: {source}")
            shutil.copy2(source, case_dir / source.name)

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_root)
            for case in CASES:
                # Fresh testcase state with one shared immutable rule table and
                # BM25 retriever.  No fallback call is permitted below.
                rule_agent.state = State()
                rule_agent.const_nets = {}
                canonical_agent.state = State()
                canonical_agent.const_nets = {}
                rule_responses = []
                for ident, row in enumerate(grouped[case], 1):
                    text = row["text"]
                    _first_match(rule_agent, text)  # assert no LLM fallback
                    rule_responses.append(_norm(rule_agent.handle(text, ident)))

                # A second fresh run proves that each exported canonical op and
                # params object reproduces the first-match handler behaviour.
                # use_rules=False also disables _dispatch_intent's wording
                # guards, so the bank cannot pass by being rescued by regex.
                # Drop the rule run's transformed netlist before loading the
                # same large design for the canonical pass.
                rule_agent.state = State()
                rule_agent.const_nets = {}
                canonical_responses = []
                for row in grouped[case]:
                    try:
                        response = canonical_agent._dispatch_intent(
                            row["op"], row["params"], row["text"])
                    except Exception as exc:
                        raise ExportError(
                            f"{row['case']}:{row['line']} canonical dispatch "
                            f"raised {type(exc).__name__}: {exc}") from exc
                    canonical_responses.append(_norm(response))

                for row, ruled, canonical in zip(
                        grouped[case], rule_responses, canonical_responses):
                    if ruled != canonical:
                        raise ExportError(
                            f"{case}:{row['line']} canonical dispatch drift\n"
                            f"op/params: {row['op']} {row['params']}\n"
                            f"rule:      {ruled}\ncanonical: {canonical}")

                golden = ROOT / "evaluator" / "golden" / f"{case}.txt"
                if not golden.is_file():
                    raise ExportError(f"missing Golden baseline: {golden}")
                expected = golden.read_text(encoding="utf-8").splitlines()
                if len(expected) != len(rule_responses):
                    raise ExportError(
                        f"{case} Golden length mismatch: "
                        f"expected {len(expected)}, replayed {len(rule_responses)}")
                for row, want, got in zip(grouped[case], expected, rule_responses):
                    if want != got:
                        golden_drifts[(case, row["line"])] = (want, got)
                checked += len(rule_responses)
                canonical_checked += len(canonical_responses)
        finally:
            os.chdir(original_cwd)

    if checked != EXPECTED_ROWS or canonical_checked != EXPECTED_ROWS:
        raise ExportError(
            f"replay checked rule={checked}, canonical={canonical_checked}; "
            f"expected {EXPECTED_ROWS} each")
    observed = {key: (_fingerprint(want), _fingerprint(got))
                for key, (want, got) in golden_drifts.items()}
    pinned = {key: values[:2] for key, values in KNOWN_GOLDEN_DRIFTS.items()}
    if observed != pinned:
        unexpected = sorted(set(observed) - set(pinned))
        resolved = sorted(set(pinned) - set(observed))
        changed = sorted(key for key in set(observed) & set(pinned)
                         if observed[key] != pinned[key])
        detail = []
        if unexpected:
            detail.append(f"unexpected={unexpected}")
        if resolved:
            detail.append(f"resolved-but-still-pinned={resolved}")
        if changed:
            detail.append("changed=" + repr([
                (key, pinned[key], observed[key]) for key in changed]))
        raise ExportError("Golden drift set changed: " + "; ".join(detail))

    print(f"dispatch: PASS ({canonical_checked}/{EXPECTED_ROWS} canonical == rule)")
    exact = checked - len(golden_drifts)
    print(f"golden: VERIFIED ({exact} exact; {len(golden_drifts)} pinned drifts)")
    for key in sorted(golden_drifts):
        print(f"  {key[0]}:{key[1]} {KNOWN_GOLDEN_DRIFTS[key][2]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the committed JSONL instead of rewriting it")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output path (default: cada/llm/public_examples.jsonl)")
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else ROOT / args.out

    try:
        rows = build_rows()
        _check_contract(rows)
        serialized = _serialize(rows)
        if args.check:
            _check_bank(out, serialized)
        replay_golden(rows)
        if not args.check:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(serialized, encoding="utf-8")
            print(f"wrote:  {out}")
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
