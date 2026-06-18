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


def register(name: str, path: Optional[str]) -> None:
    if path:
        _REGISTRY[name] = os.path.expanduser(str(path))


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
