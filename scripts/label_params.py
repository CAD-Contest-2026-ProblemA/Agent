#!/usr/bin/env python3
"""Label the example bank with params, emitting only what can be verified.

The retrieved examples appear in the prompt as the model's most recent picture
of what an answer looks like.  Rendering them as bare op names teaches it to
answer with a bare op name; rendering them with WRONG params teaches it
something worse.  So every value here has to survive a check, and anything
that does not survive is left out rather than guessed — an example with no
params is merely unhelpful, one with the wrong params is a counter-example.

Checks applied to every candidate:

  * name-valued params (net/gate/output/a/b/...) must appear verbatim in the
    sentence — a name that is not a substring of the text cannot be right
  * enum-valued params (type/dir/kind/basis/mode/scope/against) must be in the
    set allowed_intents accepts
  * k must be an integer that occurs in the sentence
  * the assembled object must pass validate_intent_object
  * where the regex router recognises the sentence, its own extraction is used
    as an independent cross-check and a disagreement drops the label

Usage:
    python3 scripts/label_params.py --dry-run     # report coverage, write nothing
    python3 scripts/label_params.py               # rewrite examples.jsonl in place
    python3 scripts/label_params.py --sample 40   # print a sample for eyeballing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cada.llm.allowed_intents import (  # noqa: E402
    BASIS_VALUES, GATE_TYPES, LIST_PORT_DIRS, OPTIONAL_PARAMS, REQUIRED_PARAMS,
    RENAME_KINDS, validate_intent_object,
)

BANK = os.path.join(ROOT, "cada", "llm", "examples.jsonl")

NAME_KEYS = {"net", "gate", "output", "wire", "input", "a", "b", "target",
             "clk", "old", "new", "scope"}
# Identifiers as the netlists spell them: n5, n31[1], g868, renamed_sig.
IDENT = re.compile(r"\b[A-Za-z_]\w*(?:\[\d+\])?")
# Words that look like identifiers but are English, not nets.
STOPWORDS = set("""a an the of to from in on at by for with and or not is are was
were be been do does did this that these those which what who whom whose how
many much any all each every both either neither one two three no yes it its
if then than as into onto out up down over under between among within without
list name names report give show tell find check verify determine compute
count enumerate identify locate return please make sure ensure design netlist
gate gates cell cells wire wires net nets signal signals node nodes output
outputs input inputs primary pin pins path paths logic depth level levels
fanin fanout cone total number size set does exist exists still can could
would should will shall may might must have has had get take put let me my
you your we our they them their there here now after before while when where
why hop hops away downstream upstream directly indirectly first last next
previous same different other another new old current final initial""".split())


def idents(text: str):
    """Identifier-shaped tokens from a sentence, in order, minus English words."""
    out = []
    for m in IDENT.finditer(text):
        tok = m.group(0)
        if tok.lower() in STOPWORDS:
            continue
        # A bare English word with no digit and no bracket is not a net name.
        if not re.search(r"\d", tok):
            continue
        out.append(tok)
    return out


def named_idents(text: str):
    """Distinct identifier-shaped names, in first-appearance order.

    Deduplicated because a sentence naming one net twice ("the fanout of n5?
    List the gates n5 drives") is not ambiguous — counting raw occurrences
    made the single-slot guard reject exactly the clearest sentences.
    """
    out = idents(text)
    # Renamed nets (renamed_sig, sig_alias) carry no digit, so idents() misses
    # them.  An underscore is the discriminator: these sentences are English
    # prose, and English words do not contain one.  Matching on a prefix like
    # "sig" instead swallowed the word "signal" and broke the ambiguity guard.
    for m in re.finditer(r"\b\w+_\w+\b", text):
        out.append(m.group(0))
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _int_in(text: str) -> Optional[int]:
    nums = re.findall(r"\b(\d+)\b", text)
    return int(nums[0]) if len(nums) == 1 else None


def _gate_type(text: str) -> Optional[str]:
    low = text.lower()
    hits = [t for t in GATE_TYPES if re.search(rf"\b{t}\b", low)]
    # "NAND gates" -> nand.  Two candidates means the sentence is ambiguous.
    return hits[0] if len(hits) == 1 else None


def _basis(text: str) -> Optional[str]:
    low = text.lower()
    has = lambda *w: all(re.search(rf"\b{x}\b", low) for x in w)
    if has("nand", "not"):
        return "NAND_NOT"
    if has("nor", "not"):
        return "NOR_NOT"
    if has("and", "or", "not"):
        return "AND_OR_NOT"
    if has("and", "not") and "nand" not in low and "nor" not in low:
        return "AND_NOT"
    return None


def _dir(text: str) -> Optional[str]:
    low = text.lower()
    i, o = "input" in low, "output" in low
    return "input" if i and not o else ("output" if o and not i else None)


def extract(text: str, op: str) -> Optional[Dict]:
    """Best-effort params for one (sentence, op), or None if not confident."""
    req = set(REQUIRED_PARAMS.get(op, ()))
    if not req:
        return {}                      # ops that legitimately take no params
    names = named_idents(text)
    p: Dict = {}

    # --- name-valued -----------------------------------------------------
    name_req = sorted(req & NAME_KEYS)
    if name_req:
        if op == "rename":
            # Order carries meaning: "rename X to Y".  Only trust the explicit
            # form; anything else could invert old/new and teach the reverse.
            m = re.search(rf"({IDENT.pattern})\s+to\s+({IDENT.pattern})", text)
            if not m:
                return None
            p["old"], p["new"] = m.group(1), m.group(2)
        elif op == "depends_on":
            # Direction is semantic, not positional: "does X influence Y" puts
            # the input first, "does output X depend on input Y" puts it last.
            # Reading position alone labelled all 30 of these backwards, so
            # only the forms that name the roles are trusted.
            m = re.search(rf"output\s+({IDENT.pattern}).*?input\s+({IDENT.pattern})",
                          text, re.I)
            if m:
                p["output"], p["input"] = m.group(1), m.group(2)
            else:
                m = re.search(rf"input\s+({IDENT.pattern}).*?output\s+({IDENT.pattern})",
                              text, re.I)
                if not m:
                    return None
                p["input"], p["output"] = m.group(1), m.group(2)
        elif op == "symmetric":
            # Three names, and the odd one out is the observation point rather
            # than the swapped pair.  Positional order put the pair's first
            # member in `output` on every sentence, so require the explicit
            # "at OUTPUT ... inputs A and B" shape and skip the rest.
            m = re.search(
                rf"(?:at|of)\s+(?:output\s+)?({IDENT.pattern}).*?"
                rf"(?:inputs?|respect to)\s+(?:the\s+)?({IDENT.pattern})"
                rf"\s+and\s+({IDENT.pattern})", text, re.I)
            if not m:
                return None
            p["output"], p["a"], p["b"] = m.group(1), m.group(2), m.group(3)
        elif op in ("dominator", "path_exists"):
            # Both are "from A to B, around C".  What separates them is what C
            # is — a gate for dominator, a net for path_exists — which is also
            # what the catalog's disambiguation rule turns on, so the same
            # signal decides the label.
            gates = [n for n in names if re.fullmatch(r"g\d+\w*", n)]
            nets = [n for n in names if n not in gates]
            if op == "dominator":
                if len(gates) != 1 or len(nets) != 2:
                    return None
                p["a"], p["b"], p["gate"] = nets[0], nets[1], gates[0]
            else:
                if gates or len(nets) not in (2, 3):
                    return None
                # The avoided net is not reliably last: "Is there a n701-free
                # route from n3 to n41?" names it first.  Take the endpoints
                # from the explicit connecting phrase and let the leftover net
                # be the one being avoided.
                m = re.search(
                    rf"(?:from|connecting|between)\s+(?:input\s+|output\s+)?"
                    rf"({IDENT.pattern})\s+(?:to|and)\s+(?:input\s+|output\s+)?"
                    rf"({IDENT.pattern})", text, re.I)
                if not m:
                    return None
                a, b = m.group(1), m.group(2)
                if a not in nets or b not in nets:
                    return None
                p["a"], p["b"] = a, b
                rest = [n for n in nets if n not in (a, b)]
                if len(nets) == 3:
                    if len(rest) != 1:
                        return None
                    p["avoid"] = rest[0]
        elif len(name_req) == 1:
            # One slot, but several candidates -> which one is unclear.
            if len(names) != 1:
                return None
            p[name_req[0]] = names[0]
        elif name_req == ["a", "b"]:
            # Both roles are either symmetric (shared_cone, same_clock) or
            # written "from a to b", so sentence order is the right reading.
            if len(names) != 2:
                return None
            p["a"], p["b"] = names[0], names[1]
        else:
            return None

    # --- enum / scalar ---------------------------------------------------
    for key in sorted(req - NAME_KEYS):
        if key == "type":
            v = _gate_type(text)
        elif key == "dir":
            v = _dir(text)
        elif key == "k":
            v = _int_in(text)
        elif key == "basis":
            v = _basis(text)
        else:
            # name / kind / file — semantic, not reliably derivable.
            v = None
        if v is None:
            return None
        p[key] = v

    # --- optional params, only when unambiguous --------------------------
    opt = set(OPTIONAL_PARAMS.get(op, ()))
    if "basis" in opt:
        b = _basis(text)
        if b:
            p["basis"] = b

    return p


def verify(text: str, op: str, params: Dict) -> Optional[str]:
    """Return an error string if the labelled params are not trustworthy."""
    clean, err = validate_intent_object({"intent": op, "params": params})
    if err:
        return err
    for key, val in params.items():
        if key in NAME_KEYS and isinstance(val, str):
            if val not in text:
                return f"{key}={val!r} does not occur in the sentence"
        if key == "type" and val not in GATE_TYPES:
            return f"bad type {val!r}"
        if key == "dir" and val not in LIST_PORT_DIRS:
            return f"bad dir {val!r}"
        if key == "basis" and val not in BASIS_VALUES:
            return f"bad basis {val!r}"
        if key == "kind" and op == "rename" and val not in RENAME_KINDS:
            return f"bad rename kind {val!r}"
        if key == "k" and (not isinstance(val, int) or val < 1):
            return f"bad k {val!r}"
    # Direction-carrying pairs: "from a to b" fixes which is which, so a label
    # whose `a` sits after its `b` in the sentence has them swapped.
    if op in ("enumerate_paths", "path_exists", "max_depth_between",
              "articulation", "dominator") and {"a", "b"} <= set(params):
        m = re.search(r"\bfrom\b.*?\bto\b", text, re.I)
        if m and text.find(params["a"]) > text.find(params["b"]):
            return f"a={params['a']!r} occurs after b={params['b']!r} in a from/to sentence"

    # rename must not label a no-op or an inverted pair.
    if op == "rename" and params.get("old") == params.get("new"):
        return "old and new are the same name"

    missing = set(REQUIRED_PARAMS.get(op, ())) - set(params)
    if missing and not (op == "insert_buffers" and params.get("mode") == "dedicated"):
        return f"missing required {sorted(missing)}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=BANK)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.bank, encoding="utf-8") if l.strip()]
    labelled = skipped = empty = 0
    reasons: Dict[str, int] = {}
    out = []
    for r in rows:
        r.pop("params", None)
        p = extract(r["text"], r["op"])
        if p is None:
            skipped += 1
            reasons[r["op"]] = reasons.get(r["op"], 0) + 1
        else:
            err = verify(r["text"], r["op"], p)
            if err:
                skipped += 1
                reasons[f"{r['op']}: {err[:40]}"] = reasons.get(r["op"], 0) + 1
            else:
                r["params"] = p
                labelled += 1
                if not p:
                    empty += 1
        out.append(r)

    need = sum(1 for r in rows if REQUIRED_PARAMS.get(r["op"]))
    with_vals = labelled - empty
    print(f"總計 {len(rows)} 筆")
    print(f"  有必填 params 的 op          {need}")
    print(f"  成功標註(含空 params)       {labelled}")
    print(f"    其中真的有值               {with_vals} / {need} = "
          f"{100*with_vals/need:.1f}%")
    print(f"  無法確信、留空不標           {skipped}")
    if reasons:
        print("\n未標註原因(前 12):")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {v:5d}  {k}")

    if args.sample:
        import random
        random.seed(11)
        pool = [r for r in out if r.get("params")]
        print(f"\n抽樣 {args.sample} 筆已標註的:")
        for r in random.sample(pool, min(args.sample, len(pool))):
            print(f"  {r['op']:20s} {json.dumps(r['params'], ensure_ascii=False):46s} "
                  f"| {r['text'][:58]}")

    if not args.dry_run:
        with open(args.bank, "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.bank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
