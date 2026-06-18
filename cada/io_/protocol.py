"""The stdin/stdout protocol harness.

For every line of natural-language request read from stdin, we emit one
response frame::

    #RESPONSE <id>
    <body>
    #END <id>

``<id>`` starts at 1 and increments once per non-EOF line; the
beginning-of-testcase line is response 1.  The contest harness only sends the
next line after it sees ``#END <id>``, so every frame MUST be flushed.  All
responses are mirrored into ``<case_name>.log``.

EOF discipline: a real EOF (empty string from ``readline``) breaks the loop
without emitting a frame (avoids id drift); a blank line ("\\n") is a normal
request.
"""

from __future__ import annotations

import re
import sys
from typing import Callable, Optional, TextIO

_CASE_NAME = re.compile(r"case\s+name\s+is\s+(\S+?)[\.\s]*$", re.IGNORECASE)
_CASE_NAME2 = re.compile(r"beginning\s+of\s+testcase\s+(\S+?)[\.\s]", re.IGNORECASE)


def extract_case_name(line: str) -> Optional[str]:
    m = _CASE_NAME.search(line.strip())
    if m:
        return m.group(1).rstrip(".")
    m = _CASE_NAME2.search(line)
    if m:
        return m.group(1).rstrip(".")
    return None


class Protocol:
    def __init__(self, handler: Callable[[str, int], str],
                 on_case_name: Optional[Callable[[str], None]] = None,
                 out: TextIO = None, log_dir: str = "."):
        self.handler = handler
        self.on_case_name = on_case_name
        self.out = out or sys.stdout
        self.log_dir = log_dir
        self.log_fh: Optional[TextIO] = None
        self.case_name: Optional[str] = None
        self.id = 0

    def open_log(self, case_name: str):
        self.case_name = case_name
        try:
            import os
            path = os.path.join(self.log_dir, f"{case_name}.log")
            self.log_fh = open(path, "w")
        except Exception:
            self.log_fh = None

    def _emit(self, ident: int, body: str):
        frame = f"#RESPONSE {ident}\n{body}\n#END {ident}\n"
        self.out.write(frame)
        self.out.flush()
        if self.log_fh is not None:
            try:
                self.log_fh.write(frame)
                self.log_fh.flush()
            except Exception:
                pass

    def run(self, stream: TextIO = None):
        stream = stream or sys.stdin
        while True:
            line = stream.readline()
            if line == "":            # genuine EOF
                break
            line = line.rstrip("\n")
            self.id += 1

            # On the very first line, open the per-case log.
            if self.id == 1:
                name = extract_case_name(line) or "output"
                self.open_log(name)
                if self.on_case_name:
                    try:
                        self.on_case_name(name)
                    except Exception:
                        pass
            else:
                # Defensive: a mid-stream begin line still (re)sets the case.
                name = extract_case_name(line)
                if name and name != self.case_name and "beginning" in line.lower():
                    self.open_log(name)
                    if self.on_case_name:
                        try:
                            self.on_case_name(name)
                        except Exception:
                            pass

            try:
                body = self.handler(line, self.id)
            except Exception as exc:  # never crash the harness loop
                body = f"Error while processing request: {exc}"
            self._emit(self.id, body)

        if self.log_fh is not None:
            try:
                self.log_fh.close()
            except Exception:
                pass
