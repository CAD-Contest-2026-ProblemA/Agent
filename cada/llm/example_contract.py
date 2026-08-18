"""Fast, provider-free consistency checks for routing example banks."""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from .allowed_intents import ALLOWED_INTENTS, validate_intent_object


class ExampleContractError(ValueError):
    """One or more example rows violate the canonical routing contract."""


INTENT_ONLY_EXCEPTIONS = frozenset({("test100", 68), ("test163", 6)})


def normalize_example_text(text: str) -> str:
    """Normalize only representation, never wording or net identifiers."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def validate_example_banks(
        banks: Mapping[str, Sequence[dict]],
        *,
        allow_intent_only: Iterable[Tuple[str, int]] = (),
) -> Dict[str, int]:
    """Validate stored canonical objects and cross-bank duplicate labels.

    ``allow_intent_only`` is reserved for explicitly reviewed legacy rows whose
    prompt lacks enough information to fill required params.  Every other row
    must contain a complete, already-canonical ``params`` object.
    """
    allowed_missing: Set[Tuple[str, int]] = set(allow_intent_only)
    used_missing: Set[Tuple[str, int]] = set()
    errors = []
    coords = set()
    by_text = defaultdict(list)
    row_count = 0

    for bank_name, rows in banks.items():
        for index, row in enumerate(rows, 1):
            row_count += 1
            where = f"{bank_name}:{index}"
            if not isinstance(row, dict):
                errors.append(f"{where}: row must be a JSON object")
                continue
            required_fields = {"case", "line", "text", "op", "kind"}
            missing_fields = sorted(required_fields - set(row))
            if missing_fields:
                errors.append(f"{where}: missing fields {missing_fields}")
                continue

            case = row["case"]
            line = row["line"]
            coord: Optional[Tuple[str, int]] = None
            if not isinstance(case, str) or not case.strip():
                errors.append(f"{where}: case must be a non-empty string")
            if not isinstance(line, int) or isinstance(line, bool) or line < 1:
                errors.append(f"{where}: line must be a positive integer")
            if isinstance(case, str) and case.strip() and isinstance(line, int) \
                    and not isinstance(line, bool) and line >= 1:
                coord = (case, line)
                if coord in coords:
                    errors.append(
                        f"{where}: duplicate case/line {coord[0]}:{coord[1]}")
                coords.add(coord)

            if not isinstance(row["text"], str) or not row["text"].strip():
                errors.append(f"{where}: text must be a non-empty string")
                continue
            op = row["op"]
            op_valid = isinstance(op, str) and op in ALLOWED_INTENTS
            if not op_valid:
                errors.append(f"{where}: invalid intent {op!r}")
            if not isinstance(row["kind"], str) \
                    or row["kind"] not in {"safe", "hard"}:
                errors.append(f"{where}: invalid example kind {row['kind']!r}")

            clean: Optional[dict] = None
            if "params" not in row:
                if coord not in allowed_missing:
                    label = (f"{coord[0]}:{coord[1]}" if coord is not None
                             else "invalid coordinate")
                    errors.append(f"{where}: missing params for {label}")
                elif coord is not None:
                    used_missing.add(coord)
            elif op_valid:
                clean, err = validate_intent_object({
                    "intent": op, "params": row["params"],
                })
                if err or clean is None:
                    errors.append(f"{where}: invalid routing object: {err}")
                elif clean["params"] != row["params"]:
                    errors.append(
                        f"{where}: params are not canonical; stored="
                        f"{json.dumps(row['params'], sort_keys=True)} canonical="
                        f"{json.dumps(clean['params'], sort_keys=True)}")

            if op_valid:
                by_text[normalize_example_text(row["text"])].append(
                    (where, row, clean))

    unused_missing = sorted(allowed_missing - used_missing)
    if unused_missing:
        detail = ", ".join(f"{case}:{line}" for case, line in unused_missing)
        errors.append(f"unused intent-only exception(s): {detail}")

    for entries in by_text.values():
        ops = {row["op"] for _where, row, _clean in entries}
        if len(ops) > 1:
            detail = ", ".join(f"{where}={row['op']}"
                               for where, row, _clean in entries)
            errors.append(f"same normalized text has different intents: {detail}")
            continue
        objects = {
            json.dumps(clean, sort_keys=True)
            for _where, _row, clean in entries
            if clean is not None
        }
        has_intent_only = any(clean is None for _where, _row, clean in entries)
        if len(objects) > 1 or (objects and has_intent_only):
            detail = ", ".join(
                f"{where}=" + (json.dumps(clean, sort_keys=True)
                                if clean is not None else "<intent-only>")
                for where, _row, clean in entries)
            errors.append(f"same normalized text has different objects: {detail}")

    if errors:
        shown = errors[:20]
        suffix = f"\n... and {len(errors) - len(shown)} more" if len(errors) > 20 else ""
        raise ExampleContractError("\n".join(shown) + suffix)

    return {
        "rows": row_count,
        "unique_texts": len(by_text),
        "duplicate_groups": sum(len(entries) > 1 for entries in by_text.values()),
    }
