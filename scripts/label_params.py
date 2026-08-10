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
  * enum-valued params (type/dir/basis/mode/scope/against) must be in the set
    allowed_intents accepts
  * k must be an integer that occurs in the sentence
  * the assembled object must pass validate_intent_object
  * delta_count's free-form `kind` is pushed through a mirror of
    op_delta_count's own dispatch, so the label is checked against the code
    that will consume it rather than against my reading of that code
  * the direction-carrying ops (depends_on / symmetric / path_exists, plus the
    a/b pairs) are labelled twice: once by the pattern tables below, once by
    hand in HAND_CHECK, and the bank is not written unless the two agree

Verbatim-and-valid is a weaker claim than correct: "Henceforth n1039 answers
to renamed_sig" once yielded old="answers", which is in the sentence and
passes validation and is still nonsense.  That class of error is what the
hand-check and the netlist-shape rules are for.

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
# Keys whose value must occur verbatim in the sentence.  "scope" is excluded
# because insert_buffers spells it as an enum (gate|signal) rather than a net.
VERBATIM_KEYS = (NAME_KEYS | {"name", "file", "dir", "avoid"}) - {"scope"}
# Identifiers as the netlists spell them: n5, n31[1], g868, renamed_sig.
IDENT = re.compile(r"\b[A-Za-z_]\w*(?:\[\d+\])?")
# The same shape, for embedding inside a larger pattern.
N = r"[A-Za-z_]\w*(?:\[\d+\])?"
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


# Longest first, so "xnor" is never read as "nor".
_TYPE_ALT = "xnor|nand|nor|xor|buf|dff|and|or|not"
_TYPE_SYNONYM = {
    "inverter": "not", "inverters": "not", "invertor": "not",
    "buffer": "buf", "buffers": "buf", "repeater": "buf", "repeaters": "buf",
    "flip-flop": "dff", "flip-flops": "dff", "flipflop": "dff",
    "register": "dff", "registers": "dff",
}


def _gate_type(text: str) -> Optional[str]:
    """The one gate type a sentence is about, or None if it is not decidable.

    Reading a bare ``\\band\\b`` as the AND gate is the trap here: half these
    sentences end in "with their input and output signals", so the English
    conjunction outvoted the real type and every one of them was dropped.  A
    type word only counts when it is wearing a gate's clothes -- followed by
    gate/cell, pluralised (NORs), spelled in capitals, or written as a synonym.
    """
    low = text.lower()
    hits = set()
    # "NAND gates", "NOT cells"
    hits.update(m.group(1) for m in
                re.finditer(rf"\b({_TYPE_ALT})s?\s+(?:gate|cell|instance)", low))
    # "inverters", "buffers"
    for word, t in _TYPE_SYNONYM.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            hits.add(t)
    # "Where are the NORs?" -- a pluralised type word is never English.
    hits.update(m.group(1) for m in re.finditer(rf"\b({_TYPE_ALT})s\b", low))
    # "The NAND contingent" -- capitals mark it as a cell name, not a word.
    hits.update(m.group(1).lower() for m in
                re.finditer(rf"\b({_TYPE_ALT.upper()})\b", text))
    if len(hits) == 1:
        t = hits.pop()
        return t if t in GATE_TYPES else None
    return None


_DIR_WORDS = {
    "input": r"entry point|entry pin|way[s]? into|bring[s]? data in|incoming|"
             r"inbound|feed[s]? in|driven in|\binput",
    "output": r"departing|exit port|outgoing|outbound|leaving|driven out|"
              r"emitted|\boutput",
}


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


# A cone-scoped request names the cone's output; anything else is design-wide.
# Anaphora ("just that output's logic", "the logic under n11[0]") is deliberately
# not matched — the net is not being named as a cone here, and guessing turns a
# scoped request into a whole-design remap or the reverse.
_CONE_PATTERNS = (
    rf"\b(?:fan-?in\s+)?cone\s+of\s+(?:primary\s+)?(?:output\s+)?({N})",
    rf"\bwithin\s+({N})'s\s+(?:fan-?in\s+)?cone",
    rf"\b({N})'s\s+(?:fan-?in\s+)?cone",
    rf"\b(?:the\s+)?logic\s+(?:of|under|behind|feeding)\s+({N})",
    rf"\beverything\s+(?:under|feeding|driving)\s+({N})",
)


def _cone_scope(text: str) -> Optional[str]:
    """The net a conversion is scoped to, or None for a design-wide request."""
    hits = set()
    for rx in _CONE_PATTERNS:
        for m in re.finditer(rx, text, re.I):
            if _looks_like_net(m.group(1)):
                hits.add(m.group(1))
    return hits.pop() if len(hits) == 1 else None


def _scope_is_decidable(text: str, scope: Optional[str]) -> bool:
    """False when a sentence names a net but no phrase says it is the scope.

    "Reforge the logic of n8 ..." is scoped; a label that carries the basis and
    omits the scope does not merely under-specify it, it asserts a design-wide
    remap. Whichever way that guess goes it becomes a counter-example, so the
    row is dropped instead.  Anaphoric scoping ("just that output's logic")
    names no net and is left alone -- there is nothing there to get wrong.
    """
    return scope is not None or not any(_looks_like_net(t) for t in named_idents(text))


# Ordered: "the netlist before the transformation" also contains "netlist", and
# "as last loaded" also reads as an origin, so the narrower reference wins.
_AGAINST_PATTERNS = (
    ("pre", r"before the transformation|before that (?:last )?(?:step|transform)|"
            r"pre-?transformation|prior to the (?:last )?transform|immediately before"),
    ("last_loaded", r"last loaded|as last loaded|loaded from disk|as read from disk"),
    ("original", r"\boriginal\b|as originally loaded|first loaded"),
)


def _equiv_target(text: str) -> Optional[str]:
    """pre|last_loaded|original, only when exactly one reference is named."""
    hits = [t for t, rx in _AGAINST_PATTERNS if re.search(rx, text, re.I)]
    return hits[0] if len(hits) == 1 else None


def _const_value(text: str) -> Optional[str]:
    """The constant level a report is restricted to, or None for either level.

    "with constant inputs (0 or 1)" names both and restricts nothing, so it has
    to come back None — labelling it "0" would teach the model to narrow a
    request that was deliberately broad.
    """
    # "a hardwired 0 or 1" names both levels but only the first sits next to
    # the constant word, so match the enumeration before matching a single one.
    if re.search(r"\b[01]\s+or\s+[01]\b", text):
        return None
    found = set(re.findall(
        r"(?:constant|fixed|hardwired|hard-wired|frozen|stuck|tied|hard)\W{0,12}?\b([01])\b",
        text, re.I))
    return found.pop() if len(found) == 1 else None


def _dir(text: str) -> Optional[str]:
    """input|output for list_ports, from whichever side's vocabulary appears.

    Many of these sentences never say "input" or "output" at all -- they say
    "entry points", "departing signals", "the exit ports".  Matching only the
    literal words left them unlabelled; matching the vocabulary of one side and
    requiring the other side to be silent keeps that from becoming a coin flip.
    """
    low = text.lower()
    hits = [d for d, rx in _DIR_WORDS.items() if re.search(rx, low)]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# delta_count
#
# `kind` is not an enum: op_delta_count() substring-matches it against the tags
# a transform recorded, and falls back to the most recent delta when nothing
# matches.  That fallback is a feature -- for a question that names no
# particular transform ("how many gates disappeared?") the most recent delta IS
# the answer, so a neutral kind is RIGHT and a specific one like "removed" is
# wrong: it would pin the lookup to a tag the preceding step never wrote.
# ---------------------------------------------------------------------------
DELTA_TAGS = {"buffers_added", "nand_added", "nor_added", "xor_converted",
              "xnor_converted", "nand_to_inv", "const_eliminated", "merged",
              "collapsed", "removed", "floating", "enable_hold", "basis_remap"}
NEUTRAL_KIND = "net_change"


def _kind_tag(kind: str) -> Optional[str]:
    """Which delta tag op_delta_count() would resolve `kind` to (None = fallback).

    Mirrors cada/agent/agent.py::op_delta_count so a label can be checked
    against the code that will consume it instead of against my reading of it.
    """
    low = kind.lower()
    if low in DELTA_TAGS:
        return low
    if "buf" in low or "buffer" in low:
        return "buffers_added"
    if "nand" in low and "added" in low:
        return "nand_added"
    if "nor" in low and "xnor" not in low and "added" in low:
        return "nor_added"
    if "const" in low or "eliminated" in low or "propagation" in low:
        return "const_eliminated"
    if "merge" in low or "duplicate" in low:
        return "merged"
    if "collapse" in low or "inverter" in low:
        return "collapsed"
    if "enable" in low or "hold" in low:
        return "enable_hold"
    if ("remove" in low or "dangling" in low or "redundant" in low
            or "floating" in low):
        return "removed"
    return None


def _delta_kind(text: str) -> str:
    """The kind word for a "how many were added/removed" question."""
    low = text.lower()
    if "buf" in low or "buffer" in low or "repeater" in low:
        return "buffers_added"
    # "How many NOR gates were added by replacing the XNOR gates?" names two
    # types; only the one attached to "added" was actually added.
    m = re.search(rf"\b({_TYPE_ALT})\s+(?:gate|cell)s?\s+(?:\w+\s+)?added", low)
    if m and f"{m.group(1)}_added" in DELTA_TAGS:
        return f"{m.group(1)}_added"
    if ("eliminated" in low or "constant propagation" in low
            or "constant folding" in low or "constant-folding" in low):
        return "const_eliminated"
    if "merge" in low or "duplicate" in low:
        return "merged"
    if "collapse" in low or "inverter" in low or "back-to-back" in low:
        return "collapsed"
    if "enable" in low and ("hold" in low or "found" in low):
        return "enable_hold"
    if "dangling" in low:
        return "dangling"
    # "did the last operation add or remove?" names removal but does not mean
    # it: the question is two-sided, so pinning kind to the removal tag would
    # answer the wrong half whenever the step added gates.  A sentence that
    # mentions both directions gets the neutral kind and the recent-delta path.
    two_sided = re.search(r"\badd\w*\b.*\bremov|\bremov\w*\b.*\badd|"
                          r"grow or shrink|gain or shed|up or down", low)
    if not two_sided and ("redundant" in low or "floating" in low
                          or re.search(r"\bremov", low)):
        return "removed"
    return NEUTRAL_KIND


# ---------------------------------------------------------------------------
# direction-carrying forms
#
# depends_on / symmetric / path_exists all name two or three nets whose ROLES
# are fixed by the wording, not by their order: "does n2 influence n30" and "is
# n2 sensitive to n30" put the same two names in the same places and mean
# opposite things.  So each accepted phrasing is written out, and a sentence
# matching none of them is left unlabelled rather than read positionally.
# ---------------------------------------------------------------------------
DEP_FORWARD = [                                   # (input, output)
    rf"({N})\s+ha(?:s|ve)\s+any influence over\s+({N})",
    rf"changing\s+({N})\b.*?\bchange\s+({N})",
    rf"wiggle\s+({N})\b.*?\bcan\s+({N})",
    rf"({N})\s+were frozen\b.*?\bcould\s+({N})",
    rf"influence flows?\s+from\s+({N})\s+to\s+({N})",
    rf"does\s+({N})\s+matter to\s+({N})",
    rf"({N})\s+part of the story\s+({N})\s+tells",
]
DEP_BACKWARD = [                                  # (output, input)
    rf"dependence of\s+({N})\s+on\s+({N})",
    rf"({N})\s+(?:is\s+)?sensitive to\s+({N})",
    rf"({N})\s+is a function of\s+({N})",
    rf"value at\s+({N})\s+ever hinge on what\s+({N})",
    rf"({N})\s+even listening to\s+({N})",
    rf"truth table of\s+({N})\s+involves?\s+({N})",
    rf"cone of\s+({N})\s+gives?\s+({N})\s+any say",
]
# Which of the three names is the observation point.  a and b are then the two
# leftovers in sentence order -- they are interchangeable by definition, so
# only the output slot can actually be got wrong.
SYM_OUTPUT = [
    rf"without\s+({N})\s+notic",
    rf"as far as\s+({N})\s+is concerned",
    rf"in the eyes of\s+({N})",
    rf"behavior of\s+({N})",
    rf"as seen from\s+({N})",
    rf"would\s+({N})\s+be unchanged",
    rf"does\s+({N})\s+care",
    rf"function at\s+({N})",
    rf"does\s+({N})\s+treat",
    rf"is\s+({N})\s+indifferent",
    rf"is\s+({N})\s+blind",
    rf"would\s+({N})\s+even notice",
    rf"test\s+({N})\s+for symmetry",
    rf"(?:at|of)\s+(?:output\s+)?({N})",
]
PATH_AVOID = [
    rf"steering clear of\s+({N})",
    rf"bypass(?:es|ing)?\s+({N})",
    rf"without ever passing\s+({N})",
    rf"with\s+({N})\s+forbidden",
    rf"detour around\s+({N})",
    rf"({N})\s+declared off-limits",
    rf"absent\s+({N})",
    rf"avoiding\s+({N})",
    rf"avoid(?:s|ing)?\s+(?:the\s+)?(?:net\s+)?({N})",
    rf"({N})-free",
]
PATH_ENDS = [                                     # (a, b)
    (rf"does\s+({N})\s+still connect to\s+({N})", False),
    (rf"does\s+({N})\s+have a way to\s+({N})", False),
    (rf"can\s+({N})\s+reach\s+({N})", False),
    (rf"does\s+({N})\s+still find its way to\s+({N})", False),
    (rf"would\s+({N})\s+and\s+({N})\s+remain linked", False),
    (rf"route check:\s*({N})\s+to\s+({N})", False),
    # No trailing \b here: a name ending in "]" has no word boundary after it,
    # which silently truncated "n117[1]" to "n117" and dropped the sentence.
    (rf"^\W*({N})\s+to\s+({N})", False),
    (rf"(?:from|connecting|between)\s+(?:input\s+|output\s+|net\s+|node\s+)?"
     rf"({N})\s+(?:to|and)\s+(?:input\s+|output\s+|net\s+|node\s+)?({N})", False),
    # "is B still within reach of A" names the destination first.
    (rf"is\s+({N})\s+still within reach of\s+({N})", True),
    (rf"({N})\s+(?:is\s+)?reachable from\s+({N})", True),
]


# ---------------------------------------------------------------------------
# Double entry.
#
# The patterns above were written by grouping the sentences into forms; the
# table below was written by reading each sentence on its own and saying what
# it means.  Two passes from opposite directions, and the labeller refuses to
# write the bank unless they agree everywhere -- a swapped pair that survives
# one reading is unlikely to survive both.  Keyed by a distinctive fragment,
# which must match exactly one sentence in the bank.
#
# The two "Same output if X and Y trade places?" sentences are deliberately
# absent: they name no observation point, so `symmetric` cannot be filled in
# from them at all.
# ---------------------------------------------------------------------------
HAND_CHECK = {
    # depends_on -- which net is perturbed (input) and which is watched (output)
    "have any influence over n30[0]": {"input": "n2", "output": "n30[0]"},
    "Will changing n4": {"input": "n4", "output": "n32[0]"},
    "dependence of n34[0]": {"output": "n34[0]", "input": "n6"},
    "wiggle n8": {"input": "n8", "output": "n36[0]"},
    "Is n38[0] sensitive": {"output": "n38[0]", "input": "n10"},
    "whether n3 is a function of": {"output": "n3", "input": "n0[2]"},
    "wiggle n12": {"input": "n12", "output": "n63[1]"},
    "does n2 matter to": {"input": "n2", "output": "n63[1]"},
    "influence flow from n3": {"input": "n3", "output": "n12"},
    "Does n0[0] have any influence": {"input": "n0[0]", "output": "n31[0]"},
    "truth table of n3 involve": {"output": "n3", "input": "n3"},
    "If n13 were frozen": {"input": "n13", "output": "n13[0]"},
    "part of the story n13[0] tells": {"input": "n5", "output": "n13[0]"},
    "Is n12 sensitive to n2": {"output": "n12", "input": "n2"},
    "cone of n475 give": {"output": "n475", "input": "n0[2]"},
    "hinge on what n12 carries": {"output": "n63[1]", "input": "n12"},
    "Is n30 even listening": {"output": "n30", "input": "n0"},
    "is n11[0] sensitive": {"output": "n11[0]", "input": "n12"},
    "Will changing n0 ever": {"input": "n0", "output": "n13[0]"},
    "dependence of n8 on n12": {"output": "n8", "input": "n12"},

    # symmetric -- which net is the observation point
    "swap n3 and n0[1]": {"output": "n31[0]", "a": "n3", "b": "n0[1]"},
    "Does n33[0] treat": {"output": "n33[0]", "a": "n5", "b": "n0[3]"},
    "Is n35[0] indifferent": {"output": "n35[0]", "a": "n7", "b": "n0[1]"},
    "Are n9 and n0[3] equal citizens": {"output": "n37[0]", "a": "n9", "b": "n0[3]"},
    "if n2 and n1 switched roles": {"output": "n13[0]", "a": "n2", "b": "n1"},
    "swap n1 and n0[3]": {"output": "n475", "a": "n1", "b": "n0[3]"},
    "on equal footing in the eyes of": {"output": "n13[0]", "a": "n0[1]", "b": "n1"},
    "Is n63[1] indifferent": {"output": "n63[1]", "a": "n13", "b": "n4[0]"},
    "Is n31[0] blind": {"output": "n31[0]", "a": "n1", "b": "n3"},
    "Does n15 treat": {"output": "n15", "a": "n0", "b": "n0[1]"},
    "Does permuting n1 and n0[3]": {"output": "n475", "a": "n1", "b": "n0[3]"},
    "Are n0[2] and n0[1] equal citizens": {"output": "n15", "a": "n0[2]", "b": "n0[1]"},
    "as seen from n63[1]": {"output": "n63[1]", "a": "n24[1]", "b": "n0[3]"},
    "If n13 and n1 exchanged": {"output": "n8", "a": "n13", "b": "n1"},
    "Test n16 for symmetry": {"output": "n16", "a": "n12", "b": "n4[0]"},
    "Trading places between n5 and n1": {"output": "n8", "a": "n5", "b": "n1"},
    "Symmetric with respect to n2": {"output": "n31[0]", "a": "n2", "b": "n4[0]"},
    "Does the function at n15 stay fixed": {"output": "n15", "a": "n0[1]", "b": "n9[0]"},

    # path_exists -- source, destination, and the net kept off the route
    "Steering clear of n703": {"a": "n5", "b": "n43", "avoid": "n703"},
    "bypasses n707": {"a": "n9", "b": "n47", "avoid": "n707"},
    "without ever passing n709": {"a": "n11", "b": "n49", "avoid": "n709"},
    "with n141 forbidden": {"a": "n24[2]", "b": "n117[1]", "avoid": "n141"},
    "Steering clear of n95": {"a": "n0", "b": "n26[1]", "avoid": "n95"},
    "bypasses n719": {"a": "n14", "b": "n31[1]", "avoid": "n719"},
    "mandatory detour around n719": {"a": "n0[1]", "b": "n30", "avoid": "n719"},
    "without ever passing n208": {"a": "n7", "b": "n26[1]", "avoid": "n208"},
    "n4156 declared off-limits": {"a": "n3", "b": "n31[1]", "avoid": "n4156"},
    "Absent n141": {"a": "n12", "b": "n13[0]", "avoid": "n141"},
    "Avoiding n14 at all costs": {"a": "n1", "b": "n117[1]", "avoid": "n14"},

    # delta_count -- a specific tag only when the sentence names one transform
    "buffer insertion just performed": {"kind": "buffers_added"},
    "eliminated by constant propagation": {"kind": "const_eliminated"},
    "inverter-pair collapse": {"kind": "collapsed"},
    "did the merge shave off": {"kind": "merged"},
    "How many dangling gates were removed": {"kind": "dangling"},
    "the last operation add or remove": {"kind": NEUTRAL_KIND},
    "how many gates disappeared": {"kind": NEUTRAL_KIND},
    "rewritten by the conversion": {"kind": NEUTRAL_KIND},

    # a/b sentences that name the destination first
    "hear from n61": {"a": "n61", "b": "n117[1]", "gate": "g454"},
    "still within reach of n1": {"a": "n1", "b": "n117[1]", "avoid": "n14"},
}


def run_hand_check(rows) -> int:
    """Compare every hand-read expectation against what extract() produced."""
    bad = 0
    for frag, want in HAND_CHECK.items():
        hits = [r for r in rows if frag in r["text"]]
        texts = {r["text"] for r in hits}
        if len(texts) != 1:
            print(f"  ✗ fragment {frag!r} matches {len(texts)} distinct sentences")
            bad += 1
            continue
        got = hits[0].get("params")
        if got is None:
            print(f"  ✗ {frag!r}: not labelled, expected {want}")
            bad += 1
        elif {k: got.get(k) for k in want} != want:
            print(f"  ✗ {frag!r}\n      want {want}\n      got  {got}")
            bad += 1
    print(f"  double entry: {len(HAND_CHECK) - bad}/{len(HAND_CHECK)} agree")
    return bad


# Phrasings that name the destination before the source.  Every other a/b
# sentence in the bank reads source-first, so taking sentence order is right
# except here -- "can n117[1] hear from n61" is n61 -> n117[1], not the reverse.
AB_REVERSED = [
    rf"({N})\s+(?:can\s+|could\s+|ever\s+)*hears?\s+from\s+({N})",
    rf"({N})\s+(?:is\s+)?reachable from\s+({N})",
    rf"({N})\s+(?:still\s+)?within reach of\s+({N})",
    rf"({N})\s+receives?\s+(?:\w+\s+)?from\s+({N})",
    rf"({N})\s+(?:is\s+)?(?:fed|driven)\s+(?:by|from)\s+({N})",
    rf"({N})\s+downstream of\s+({N})",
]


def _ab_order(text: str, nets):
    """(a, b) for a two-endpoint sentence: source first, however it is worded."""
    for pat in AB_REVERSED:
        m = re.search(pat, text, re.I)
        if m and m.group(1) in nets and m.group(2) in nets:
            return m.group(2), m.group(1)
    return nets[0], nets[1]


def _looks_like_net(tok: str) -> bool:
    """True for netlist identifiers (n5, n31[1], g868), false for English."""
    return bool(re.search(r"\d", tok) or "_" in tok)


def _match_roles(text: str, patterns):
    """First pattern that fires, as (first_role, second_role).

    An entry may be a bare pattern or a (pattern, swap) pair; swap marks a
    phrasing that names the roles in reverse.
    """
    for pat in patterns:
        rx, swap = pat if isinstance(pat, tuple) else (pat, False)
        m = re.search(rx, text, re.I)
        if not m:
            continue
        x, y = m.group(1), m.group(2)
        if not (_looks_like_net(x) and _looks_like_net(y)):
            continue
        return (y, x) if swap else (x, y)
    return None


def extract(text: str, op: str) -> Optional[Dict]:
    """Best-effort params for one (sentence, op), or None if not confident."""
    req = set(REQUIRED_PARAMS.get(op, ()))
    names = named_idents(text)
    p: Dict = {}

    # --- ops whose params are neither names nor enums ---------------------
    if op == "begin_case":
        cases = set(re.findall(r"\btest\d+\b", text, re.I))
        return {"name": cases.pop()} if len(cases) == 1 else None

    if op == "load_design":
        vs = re.findall(r"[\w./\-]*\b[\w\-]+\.v\b", text)
        bases = {os.path.basename(v) for v in vs}
        if len(bases) != 1:
            return None
        p = {"file": bases.pop()}
        d = os.path.dirname(vs[0])
        if not d:
            dm = re.search(r"\b((?:[\w.\-]+/)+)", text)
            d = dm.group(1) if dm else ""
        if d:
            p["dir"] = d
        return p

    if op == "write_design":
        vs = set(re.findall(r"\b[\w\-]+\.v\b", text))
        return {"file": vs.pop()} if len(vs) == 1 else None

    if op == "delta_count":
        return {"kind": _delta_kind(text)}

    if op == "insert_buffers":
        # "one BUF per load" is a different transform from "cap fanout at k":
        # it takes mode=dedicated and no k, which is why validate_intent_object
        # exempts it from the k requirement.
        dedicated = re.search(
            r"dedicated buffer|its own|their own|own private|private repeater|"
            r"personal (?:buf|repeater)|individual bufs|"
            r"one per (?:load|consumer|receiver|reader|sink)|"
            r"per (?:consumer|load|receiver)\b", text, re.I)
        if dedicated and len(names) == 1:
            return {"net": names[0], "mode": "dedicated"}
        k = _int_in(text)
        if k is None:
            return None
        p = {"k": k}
        if len(names) == 1:
            p["net"] = names[0]
        return p

    # --- name-valued -----------------------------------------------------
    name_req = sorted(req & NAME_KEYS)
    if name_req:
        if op == "rename":
            # Order does not decide this one: "rename n5 to alias" and "assign
            # alias as the new identifier of n5" say the same thing backwards.
            # What does decide it is that the OLD name is already in the
            # netlist (n1201, g5) and the NEW one is a fresh human label
            # (renamed_sig, stage2_out) -- a distinction the sentence cannot
            # blur, and the same prefix that says whether to rename a gate or
            # a wire.
            old = [n for n in names if re.fullmatch(r"[ng]\d+(?:\[\d+\])?", n)]
            new = [n for n in names if n not in old]
            if len(old) == 1 and len(new) == 1:
                p["old"], p["new"] = old[0], new[0]
                kind = "gate" if old[0][0].lower() == "g" else "wire"
                # Only claim the kind when the sentence does not contradict the
                # prefix; op_rename dispatches on it (rename_gate vs rename_net).
                noun_gate = re.search(r"\bgate\b|\bcell\b", text, re.I)
                noun_net = re.search(r"\bwire\b|\bnet\b|\bsignal\b", text, re.I)
                if not (kind == "gate" and noun_net and not noun_gate) and \
                   not (kind == "wire" and noun_gate and not noun_net):
                    p["kind"] = kind
            else:
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
                if m:
                    p["input"], p["output"] = m.group(1), m.group(2)
                else:
                    # Most of these never say "input" or "output" -- they say
                    # "does n2 influence n30" or "is n2 sensitive to n30",
                    # which put the same names in the same order and mean
                    # opposite things.  Each phrasing is enumerated instead.
                    roles = _match_roles(text, DEP_FORWARD)
                    if roles:
                        p["input"], p["output"] = roles
                    else:
                        roles = _match_roles(text, DEP_BACKWARD)
                        if not roles:
                            return None
                        p["output"], p["input"] = roles
        elif op == "symmetric":
            # Three names, and the odd one out is the observation point rather
            # than the swapped pair.  Positional order put the pair's first
            # member in `output` on every sentence, so require the explicit
            # "at OUTPUT ... inputs A and B" shape and skip the rest.
            m = re.search(
                rf"(?:at|of)\s+(?:output\s+)?({IDENT.pattern}).*?"
                rf"(?:inputs?|respect to)\s+(?:the\s+)?({IDENT.pattern})"
                rf"\s+and\s+({IDENT.pattern})", text, re.I)
            if m:
                p["output"], p["a"], p["b"] = m.group(1), m.group(2), m.group(3)
            else:
                # a and b are interchangeable by definition, so the only slot
                # that can be wrong is the observation point: find that one
                # explicitly and the leftovers are the swapped pair.
                if len(names) != 3:
                    return None
                out = None
                for pat in SYM_OUTPUT:
                    mm = re.search(pat, text, re.I)
                    if mm and mm.group(1) in names:
                        out = mm.group(1)
                        break
                if out is None:
                    return None
                rest = [n for n in names if n != out]
                if len(rest) != 2:
                    return None
                p["output"], p["a"], p["b"] = out, rest[0], rest[1]
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
                p["a"], p["b"] = _ab_order(text, nets)
                p["gate"] = gates[0]
            else:
                if gates or len(nets) not in (2, 3):
                    return None
                # The avoided net is not reliably last: "Is there a n701-free
                # route from n3 to n41?" names it first, and "is n117 still
                # within reach of n1?" names the destination before the source.
                # So the endpoints come from an explicit direction phrase and
                # the avoided net from its own marker -- never from position.
                ends = _match_roles(text, PATH_ENDS)
                if not ends:
                    return None
                a, b = ends
                if a not in nets or b not in nets or a == b:
                    return None
                p["a"], p["b"] = a, b
                rest = [n for n in nets if n not in (a, b)]
                if len(nets) == 3:
                    if len(rest) != 1:
                        return None
                    avoid = None
                    for pat in PATH_AVOID:
                        mm = re.search(pat, text, re.I)
                        if mm and mm.group(1) in nets:
                            avoid = mm.group(1)
                            break
                    # The marker and the leftover have to agree; if the
                    # sentence names an avoided net that is also an endpoint,
                    # one of the two readings is wrong and neither is used.
                    if avoid is not None and avoid != rest[0]:
                        return None
                    p["avoid"] = rest[0]
        elif len(name_req) == 1:
            # One slot, but several candidates -> which one is unclear.
            if len(names) != 1:
                return None
            p[name_req[0]] = names[0]
        elif name_req == ["a", "b"]:
            # Both roles are either symmetric (shared_cone, same_clock) or
            # written "from a to b", so sentence order is the right reading --
            # except for the handful of phrasings that name the destination
            # first, which _ab_order puts back the right way round.
            if len(names) != 2:
                return None
            p["a"], p["b"] = _ab_order(text, names)
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
    # Reached by every op, including the ones with no required params at all.
    # Those are exactly the ops whose whole payload is optional -- convert_basis
    # carries basis and scope, xor_to_* carry scope, verify_equivalence carries
    # against -- so skipping them left 270 examples demonstrating an empty
    # params object for sentences that state the value outright.
    opt = set(OPTIONAL_PARAMS.get(op, ()))
    if "basis" in opt:
        b = _basis(text)
        if b:
            p["basis"] = b
    # scope is two different types under one key: an enum for insert_buffers,
    # a net name everywhere else.  Only the net-valued form is quoted from the
    # sentence, and the enum form is left to the required-param path.
    if "scope" in opt and op != "insert_buffers":
        s = _cone_scope(text)
        if not _scope_is_decidable(text, s):
            return None
        if s:
            p["scope"] = s
    if "against" in opt:
        a = _equiv_target(text)
        if a:
            p["against"] = a
    if "type" in opt and "type" not in p:
        t = _gate_type(text)
        if t:
            p["type"] = t
    if "value" in opt:
        v = _const_value(text)
        if v:
            p["value"] = v

    return p


def verify(text: str, op: str, params: Dict) -> Optional[str]:
    """Return an error string if the labelled params are not trustworthy."""
    clean, err = validate_intent_object({"intent": op, "params": params})
    if err:
        return err
    for key, val in params.items():
        # "dir" is two different things: a filesystem path for load_design, an
        # enum for list_ports.  Only the former is quoted from the sentence.
        if key == "dir" and op != "load_design":
            if val not in LIST_PORT_DIRS:
                return f"bad dir {val!r}"
            continue
        if key in VERBATIM_KEYS and isinstance(val, str):
            if val not in text:
                return f"{key}={val!r} does not occur in the sentence"
        # scope is a net name for every op but insert_buffers, where it is the
        # gate|signal enum.  Quote the net-valued form; check the enum by value.
        if key == "scope":
            if op == "insert_buffers":
                if val not in ("gate", "signal"):
                    return f"bad buffer scope {val!r}"
            elif val not in text:
                return f"scope={val!r} does not occur in the sentence"
        if key == "kind" and op == "delta_count":
            tag = _kind_tag(val)
            if tag is not None and tag not in DELTA_TAGS:
                return f"kind={val!r} resolves to unknown delta tag {tag!r}"
            if val != NEUTRAL_KIND and tag is None:
                return f"kind={val!r} matches no delta tag and is not neutral"
        if key == "file" and not val.endswith(".v"):
            return f"file={val!r} is not a Verilog filename"
        if key == "type" and val not in GATE_TYPES:
            return f"bad type {val!r}"
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

    # Report each population against its own denominator.  Counting only the
    # required-param ops was how 270 systematically empty rows stayed invisible:
    # convert_basis, xor_to_*, verify_equivalence and friends have no required
    # params, so they were never in any ratio and the run printed 99.8%.
    def bucket(op: str) -> str:
        if REQUIRED_PARAMS.get(op):
            return "required"
        return "optional" if OPTIONAL_PARAMS.get(op) else "none"

    pop = {"required": [0, 0], "optional": [0, 0], "none": [0, 0]}  # [total, with values]
    for r in out:
        b = pop[bucket(r["op"])]
        b[0] += 1
        if r.get("params"):
            b[1] += 1

    pct = lambda n, d: f"{100*n/d:.1f}%" if d else "—"
    print(f"總計 {len(rows)} 筆")
    print(f"  必填 params 的 op        {pop['required'][0]:5d}   標到值 "
          f"{pop['required'][1]} = {pct(*reversed(pop['required']))}  (空 = 漏標)")
    print(f"  只有選填 params 的 op    {pop['optional'][0]:5d}   標到值 "
          f"{pop['optional'][1]} = {pct(*reversed(pop['optional']))}  "
          f"(空未必是錯:設計層級的請求本來就沒有 scope)")
    print(f"  完全沒有 params 的 op    {pop['none'][0]:5d}   (空才是對的)")
    print(f"  成功標註(含空 params)   {labelled}")
    print(f"  無法確信、留空不標        {skipped}")
    if reasons:
        print("\n未標註原因(前 12):")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {v:5d}  {k}")

    print("\n交叉檢查(手讀 vs 規則):")
    hand_bad = run_hand_check(out)

    if args.sample:
        import random
        random.seed(11)
        pool = [r for r in out if r.get("params")]
        print(f"\n抽樣 {args.sample} 筆已標註的:")
        for r in random.sample(pool, min(args.sample, len(pool))):
            print(f"  {r['op']:20s} {json.dumps(r['params'], ensure_ascii=False):46s} "
                  f"| {r['text'][:58]}")

    if hand_bad and not args.dry_run:
        print(f"\nABORT: {hand_bad} hand-check disagreement(s); bank left "
              f"untouched. Fix the rule or the expectation before writing.")
        return 1

    if not args.dry_run:
        with open(args.bank, "w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.bank}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
