"""yosys sequential-equivalence fallback (equiv_induct).

Used only when the register sets differ between the two designs or ABC is
unavailable.  We write both designs with our own structural writer plus a small
behavioural ``dff`` module yosys can elaborate, then run the standard
equiv_make / equiv_induct / equiv_status -assert flow.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

from ..netlist.ir import Netlist
from ..netlist.writer import to_string

_DFF_MODULE = """
module dff(CK, RN, SN, D, Q);
  input CK, RN, SN, D;
  output reg Q;
  always @(posedge CK or negedge RN or negedge SN)
    if (!RN) Q <= 1'b0;
    else if (!SN) Q <= 1'b1;
    else Q <= D;
endmodule
"""


def find_yosys() -> Optional[str]:
    from ..toolpaths import resolve
    return resolve("yosys", env_var="YOSYS_BIN",
                   candidates=["/usr/local/bin/yosys", "/usr/bin/yosys"])


def equivalent(before: Netlist, after: Netlist,
               timeout: int = 280) -> Optional[bool]:
    ys = find_yosys()
    if ys is None:
        return None
    d = tempfile.mkdtemp(prefix="cada_yosys_")
    try:
        with open(os.path.join(d, "gold.v"), "w") as f:
            f.write(to_string(before) + _DFF_MODULE)
        with open(os.path.join(d, "gate.v"), "w") as f:
            f.write(to_string(after) + _DFF_MODULE)
        script = f"""
read_verilog {os.path.join(d, 'gold.v')}
prep -top top
design -stash gold
read_verilog {os.path.join(d, 'gate.v')}
prep -top top
design -stash gate
design -copy-from gold -as gold top
design -copy-from gate -as gate top
equiv_make gold gate equiv
hierarchy -top equiv
clean -purge
equiv_simple
equiv_induct
equiv_status -assert
"""
        sp = os.path.join(d, "s.ys")
        with open(sp, "w") as f:
            f.write(script)
        p = subprocess.run([ys, "-q", "-s", sp], capture_output=True,
                           text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        if "Equivalence successfully proven" in out:
            return True
        if "Unproven" in out or "failed" in out.lower() or p.returncode != 0:
            return False
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)
