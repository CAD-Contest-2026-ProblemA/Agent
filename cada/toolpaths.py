"""Central registry for external tool locations (abc, yosys, ...).

Lets you pin exact paths instead of relying on ``$PATH``.  Paths can come from
(in increasing priority — later overrides earlier):

  1. ``configs/tools.yaml`` auto-loaded next to the project
  2. environment variables (``ABC_BIN``, ``YOSYS_BIN``, or ``CADA_<NAME>_BIN``)
  3. a ``--tools <file>`` given on the command line
  4. a ``tools:`` section inside the ``-config`` file

``resolve()`` returns the registered path if it exists, then tries built-in
candidates, and only falls back to a ``$PATH`` lookup as a last resort.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterable, Optional

_REGISTRY: dict = {}


def _exedir() -> str:
    """Directory the running program lives in: the executable's directory for
    a frozen (PyInstaller) binary, the project root when running from source.
    Lets tools.yaml reference tools shipped *next to the binary* via
    ``${EXEDIR}`` — the whole submission folder can then be copied anywhere."""
    import sys
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bundledir() -> str:
    """The one-file bundle's extraction dir (sys._MEIPASS) when frozen, the
    project root otherwise.  ``${BUNDLE}`` lets tools.yaml reference tools
    *embedded inside the executable itself* (e.g. a statically-linked abc
    added with --add-binary) — a fully self-contained, zero-install program."""
    import sys
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", _exedir())
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def register(name: str, path: Optional[str]) -> None:
    if path:
        p = str(path)
        if "${BUNDLE}" in p:
            p = p.replace("${BUNDLE}", _bundledir())
        if "${EXEDIR}" in p:
            p = p.replace("${EXEDIR}", _exedir())
        p = os.path.expanduser(p)
        # an embedded tool may be extracted without its exec bit — restore it
        if os.path.isfile(p) and not os.access(p, os.X_OK):
            try:
                os.chmod(p, os.stat(p).st_mode | 0o755)
            except OSError:
                pass
        _REGISTRY[name] = p


def register_many(mapping: dict) -> None:
    for k, v in (mapping or {}).items():
        register(str(k).strip().lower(), v)


def registered(name: str) -> Optional[str]:
    return _REGISTRY.get(name)


def resolve(name: str, env_var: Optional[str] = None,
            candidates: Iterable[str] = (),
            require_exec: bool = True) -> Optional[str]:
    """Resolve a tool to an absolute path, honouring the registry first."""
    def ok(p: Optional[str]) -> bool:
        if not p or not os.path.exists(p):
            return False
        return (not require_exec) or os.access(p, os.X_OK) or os.path.isfile(p)

    # 1. explicit registry entry
    p = _REGISTRY.get(name)
    if ok(p):
        return p
    # 2. environment variable override (e.g. ABC_BIN, CADA_ABC_BIN)
    for ev in filter(None, (env_var, f"CADA_{name.upper()}_BIN")):
        e = os.environ.get(ev)
        if ok(e):
            return e
    # 3. built-in candidate locations
    for c in candidates:
        if ok(c):
            return c
    # 4. last resort: $PATH
    return shutil.which(name)
