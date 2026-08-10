#!/usr/bin/env python3
"""Export the labelled routing sentences from the result spreadsheets to JSONL.

The 2130 sentences live in ``testcase/test91-171/prompt.txt`` (already in git),
but the label saying which op each one should route to exists only inside the
``routing_test_*.xlsx`` result files.  Nothing in the repo can read those, so
the ground truth is invisible to code review and to any evaluation script.
This lifts it into a plain-text file that git can diff.

Reads xlsx with the stdlib (a spreadsheet is a zip of XML) so the export needs
no third-party package.

Usage:
    python3 scripts/export_examples.py            # writes cada/llm/examples.jsonl
    python3 scripts/export_examples.py --check    # verify only, write nothing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from xml.etree import ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cada.llm.allowed_intents import ALLOWED_INTENTS

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

# (workbook, sheet) pairs holding one labelled sentence per row.
SOURCES = [
    ("routing_test_91_100.xlsx", "Pure LLM 91-100"),
    ("routing_test_101_171.xlsx", "Pure LLM 101-171"),
]
OUT = os.path.join("cada", "llm", "examples.jsonl")


def _col_index(ref: str) -> int:
    m = re.match(r"([A-Z]+)", ref or "")
    if not m:
        return 0
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_sheet(path: str, want: str):
    """Yield rows (list of cell strings) from one sheet of an xlsx file."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))

    rels = {r.get("Id"): r.get("Target")
            for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    for sh in ET.fromstring(z.read("xl/workbook.xml")).iter(NS + "sheet"):
        if sh.get("name") != want:
            continue
        target = rels[sh.get(REL + "id")].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        for row in ET.fromstring(z.read(target)).iter(NS + "row"):
            cells = {}
            for c in row.iter(NS + "c"):
                v = c.find(NS + "v")
                if c.get("t") == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(NS + "t"))
                elif v is None:
                    val = ""
                elif c.get("t") == "s":
                    val = shared[int(v.text)]
                else:
                    val = v.text
                cells[_col_index(c.get("r"))] = val
            yield [cells.get(i, "") for i in range(max(cells) + 1)] if cells else []
        return
    raise SystemExit(f"error: sheet {want!r} not found in {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rows, unknown = [], set()
    for book, sheet in SOURCES:
        path = os.path.join(ROOT, book)
        if not os.path.isfile(path):
            raise SystemExit(f"error: missing {book}")
        n = 0
        for r in read_sheet(path, sheet):
            # Columns: Testcase | 行 | Prompt | Ground truth | LLM route | 句型 | 成功
            if len(r) < 6 or not r[0].startswith("test") or not r[3]:
                continue
            op = r[3].strip()
            if op not in ALLOWED_INTENTS:
                unknown.add(op)
            rows.append({
                "case": r[0].strip(),
                "line": int(r[1]) if str(r[1]).isdigit() else None,
                "text": r[2].strip(),
                "op": op,
                # 安全 = plainly worded, 刁鑽/對抗 = deliberate paraphrase
                "kind": "safe" if r[5].strip() == "安全" else "hard",
            })
            n += 1
        print(f"{book:28s} {sheet:20s} {n:5d} rows")

    cases = {r["case"] for r in rows}
    ops = {r["op"] for r in rows}
    print(f"\ntotal {len(rows)} sentences / {len(cases)} testcases / {len(ops)} ops")
    print(f"  safe {sum(1 for r in rows if r['kind']=='safe')}   "
          f"hard {sum(1 for r in rows if r['kind']=='hard')}")
    if unknown:
        print(f"\nWARNING: {len(unknown)} label(s) not in ALLOWED_INTENTS: "
              f"{', '.join(sorted(unknown))}", file=sys.stderr)

    if args.check:
        return 1 if unknown else 0

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}")
    return 1 if unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
