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
    def __init__(self, client):
        self.client = client
        self.cache: Dict[str, dict] = {}
        self.last_error: Optional[str] = None

    def translate(self, line: str) -> Optional[dict]:
        key = line.strip()
        self.last_error = None

        if key in self.cache:
            return self.cache[key]
        if self.client is None or not self.client.available:
            return None

        out = self.client.complete(INTENT_CATALOG, key)
        obj = parse_json_object(out or "")
        if obj is None:
            self.last_error = "LLM did not return a valid JSON intent object."
            return None

        obj, err = validate_intent_object(obj)
        if err:
            self.last_error = err
            return None

        self.cache[key] = obj
        return obj
