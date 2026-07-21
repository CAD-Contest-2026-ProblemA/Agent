"""yosys resynthesis bridge (optional candidate generator).

Round-trips the combinational core through yosys — ``read_blif`` → ``opt
-full`` → ``techmap`` → ``aigmap`` → ``write_blif`` — to produce a
differently-structured AIG seed for the ABC portfolio.  yosys's word-level
clean-ups (constant folding, share/opt_merge dedup) occasionally expose
restructurings ABC's local rewrites miss; the flow mirrors
ALS_Final_Project's ``rtl_to_output.py`` yosys front-ends (``aigmap`` instead
of yosys's internal ``abc``, which is unreliable in sandboxes).

Everything degrades gracefully: no yosys, a timeout, or a parse error just
returns None and the caller proceeds without this seed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

from ..equiv.yosys_bridge import find_yosys

# label prefixes used by abc_opt._opt_blif must survive the round trip; BLIF
# names are opaque tokens to yosys, so they do.
_FLOW = ("setundef -zero; opt -full; techmap; opt -full; "
         "aigmap; opt_clean")


def blif_roundtrip(blif_text: str, timeout: int = 120) -> Optional[str]:
    """Return a yosys-resynthesised BLIF of ``blif_text``, or None."""
    yosys = find_yosys()
    if yosys is None:
        return None
    d = tempfile.mkdtemp(prefix="cada_ys_")
    pin = os.path.join(d, "in.blif")
    pout = os.path.join(d, "out.blif")
    try:
        with open(pin, "w") as f:
            f.write(blif_text)
        script = f'read_blif "{pin}"; {_FLOW}; write_blif "{pout}"'
        try:
            p = subprocess.run([yosys, "-q", "-p", script],
                               capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            return None
        if p.returncode != 0 or not os.path.exists(pout):
            return None
        with open(pout) as f:
            out = f.read()
        return out if ".names" in out or ".gate" in out else None
    finally:
        shutil.rmtree(d, ignore_errors=True)
