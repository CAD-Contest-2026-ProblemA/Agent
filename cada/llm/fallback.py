"""LLM fallback: translate one NL line into a structured intent.

Used only when the deterministic regex router fails to recognise a line.  The
LLM is constrained to emit exactly one JSON object {"intent", "params"}; the
result is validated and cached so temperature noise does not make behaviour
non-deterministic within a run.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from .allowed_intents import INTENT_CATALOG, validate_intent_object

# How many retrieved examples to append.  Measured over the full 1420-sentence
# set: 25 -> 94.3%, 50 -> 95.3% (BM25).  More examples also mean more
# wrong-label distractors, so this is an empirical setting, not "bigger is
# better" — re-measure before changing it.
DEFAULT_TOP_K = 50


def parse_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict) and "intent" in obj:
            obj.setdefault("params", {})
            return obj
    except Exception:
        return None
    return None


class Fallback:
    def __init__(self, client, retriever=None, top_k: int = DEFAULT_TOP_K,
                 exclude_case: Optional[str] = None):
        self.client = client
        self.cache: Dict[str, dict] = {}
        self.last_error: Optional[str] = None
        # Optional: nearest previously-classified requests, appended after the
        # cached catalog.  None simply means the model works from the catalog
        # alone, which is the behaviour that predates retrieval.
        self.retriever = retriever
        self.top_k = top_k
        # Set during evaluation so a testcase never retrieves its own answers.
        self.exclude_case = exclude_case

    def translate(self, line: str) -> Optional[dict]:
        key = line.strip()
        self.last_error = None

        if key in self.cache:
            return self.cache[key]
        if self.client is None or not self.client.available:
            return None

        obj, err = self._ask(key)
        if err:
            # One corrective round-trip: the model sees why its first attempt
            # was rejected (unknown intent name, missing param, ...) and picks
            # again from the catalog.
            obj, err = self._ask(key, feedback=err)
        if err:
            self.last_error = err
            return None

        self.cache[key] = obj
        return obj

    def retranslate(self, line: str, feedback: str) -> Optional[dict]:
        """Re-classify a line whose first intent failed a semantic check
        downstream (e.g. a param names a net that does not exist).  The
        corrected object replaces the cached one on success."""
        key = line.strip()
        self.last_error = None
        if self.client is None or not self.client.available:
            return None
        obj, err = self._ask(key, feedback=feedback)
        if err:
            self.last_error = err
            return None
        self.cache[key] = obj
        return obj

    def _examples(self, line: str) -> Optional[str]:
        """The per-request half of the system prompt, or None if unavailable.

        Retrieval failing must never fail the request — without the block the
        model still has the full op list and every disambiguation rule.
        """
        if self.retriever is None:
            return None
        try:
            from .retrieval import format_block
            hits = self.retriever.top_k(line, self.top_k,
                                        exclude_case=self.exclude_case)
            return format_block(hits) or None
        except Exception as exc:
            import sys
            sys.stderr.write(f"[retrieval] skipped: {exc}\n")
            return None

    def _ask(self, key: str, feedback: Optional[str] = None):
        user = key
        if feedback:
            user = (key + "\n\nNOTE: a previous attempt to classify this request "
                    "was rejected: " + feedback +
                    "\nChoose again from the catalog. If the request refers to "
                    "endpoint CLASSES (any primary input, any primary output, "
                    "any DFF/register D-pin) rather than concrete net names, "
                    "use the class-level intent (pi_to_po_depth, pi_to_dff_depth, "
                    "reg_to_reg_depth, global_max_depth) with empty params.")
        out = self.client.complete(INTENT_CATALOG, user, dynamic=self._examples(key))
        obj = parse_json_object(out or "")
        if obj is None:
            return None, "LLM did not return a valid JSON intent object."
        return validate_intent_object(obj)
