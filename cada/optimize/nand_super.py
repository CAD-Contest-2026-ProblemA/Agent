"""Generic NAND/NOT technology mapping with ABC supergates.

The supergate library used here describes *functions*, not benchmark
topologies.  It is generated lazily from the embedded five-line genlib on the
first call in a process, then reused by every later call.  No prepared solution,
fingerprint, testcase name, or offline-results path participates in mapping.

``map_netlist`` accepts an arbitrary register-cut :class:`Netlist`, including
standalone cones with several outputs.  ABC's mapped BLIF is rebuilt into our
IR conservatively and the result is returned only after a local CEC proof.
Subprocess failures, timeouts, unexpected cells/BLIF constructs, malformed
interfaces, cycles, and inconclusive CEC all fail closed by returning ``None``.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..equiv import abc_bridge
from ..netlist.blif_export import shared_synthetic_labels, to_blif
from ..netlist.ir import GATE_TYPES, ONE_INPUT, Gate, Netlist, is_const
from ..transform import rewrite


# Keep this text self-contained: a packaged runtime must never need an
# offline_results checkout merely to use the generic mapper.  The relative
# basename is intentional; ABC records it in nand.super.
NAND_GENLIB = (
    "GATE zero 0 O=CONST0;\n"
    "GATE one 0 O=CONST1;\n"
    "GATE buf 1 O=a; PIN * NONINV 1 999 1 0 1 0\n"
    "GATE inv 1 O=!a; PIN * INV 1 999 1 0 1 0\n"
    "GATE nand 2 O=!(a*b); PIN * INV 1 999 1 0 1 0\n"
)

_SUPER_COMMAND = "super -I 5 -L 4 -T 20 -s nand.genlib"
_DSD_RECIPE: Tuple[str, ...] = ("&get -n", "&dsd", "&put", "strash")

_cache_lock = threading.Lock()
_cache_dir: Optional[str] = None


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _cleanup_cache() -> None:
    global _cache_dir
    directory = _cache_dir
    _cache_dir = None
    if directory:
        shutil.rmtree(directory, ignore_errors=True)


atexit.register(_cleanup_cache)


def _abc_script(commands: Sequence[str], abc: str) -> str:
    script: List[str] = []
    rc = os.path.join(os.path.dirname(abc), "abc.rc")
    if os.path.isfile(rc):
        script.append('source "{}"'.format(rc))
    script.extend(commands)
    return "; ".join(script)


def _run_abc(commands: Sequence[str], cwd: str,
             deadline: float) -> Tuple[bool, str]:
    """Run ABC in ``cwd`` while respecting the caller's shared deadline."""
    abc = abc_bridge.find_abc()
    left = _remaining(deadline)
    if abc is None or left <= 0.0:
        return False, "ABC unavailable or NAND-super deadline exhausted"
    try:
        proc = subprocess.run(
            [abc, "-q", _abc_script(commands, abc)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=left,
        )
    except subprocess.TimeoutExpired:
        return False, "ABC NAND-super timeout"
    except (OSError, ValueError) as exc:
        return False, "ABC NAND-super error: {}".format(exc)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


def _cached_library(deadline: float) -> Optional[str]:
    """Return a process-cached directory containing nand.genlib/nand.super."""
    global _cache_dir

    left = _remaining(deadline)
    if left <= 0.0 or not _cache_lock.acquire(timeout=left):
        return None
    try:
        if _cache_dir:
            genlib = os.path.join(_cache_dir, "nand.genlib")
            superfile = os.path.join(_cache_dir, "nand.super")
            if (os.path.isfile(genlib) and os.path.isfile(superfile)
                    and os.path.getsize(superfile) > 0):
                return _cache_dir
            # A caller or temp-file cleaner removed part of the cache.  Drop
            # the stale directory and regenerate instead of using it partly.
            shutil.rmtree(_cache_dir, ignore_errors=True)
            _cache_dir = None

        try:
            work = tempfile.mkdtemp(prefix="cada_nand_super_")
        except OSError:
            return None
        try:
            genlib = os.path.join(work, "nand.genlib")
            with open(genlib, "w") as stream:
                stream.write(NAND_GENLIB)
            ok, _log = _run_abc([_SUPER_COMMAND], work, deadline)
            superfile = os.path.join(work, "nand.super")
            if (not ok or not os.path.isfile(superfile)
                    or os.path.getsize(superfile) == 0):
                shutil.rmtree(work, ignore_errors=True)
                return None
            _cache_dir = work
            return work
        except (OSError, ValueError):
            shutil.rmtree(work, ignore_errors=True)
            return None
    finally:
        _cache_lock.release()


def warm_library(timeout: float = 120.0) -> bool:
    """Generate the generic supergate cache now; normally generation is lazy."""
    if timeout <= 0:
        return False
    return _cached_library(time.monotonic() + float(timeout)) is not None


def _strict_mapped_blif(text: str) -> bool:
    """Reject mapped BLIF that the deliberately small parser could misread."""
    lines = text.splitlines()
    if not any(line.strip().startswith(".model") for line in lines):
        return False
    if not any(line.strip() == ".end" for line in lines):
        return False

    allowed_cells = {"zero", "one", "buf", "inv", "nand"}
    for pos, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(".gate"):
            tokens = line.split()[1:]
            if not tokens or tokens[0] not in allowed_cells:
                return False
            pins: Dict[str, str] = {}
            for token in tokens[1:]:
                if "=" not in token:
                    return False
                key, value = token.split("=", 1)
                if not key or not value or key in pins:
                    return False
                pins[key] = value
            cell = tokens[0]
            expected = ({"O"} if cell in {"zero", "one"}
                        else {"a", "O"} if cell in {"buf", "inv"}
                        else {"a", "b", "O"})
            if set(pins) != expected:
                return False
        elif line.startswith(".names"):
            nets = line.split()[1:]
            table: List[str] = []
            scan = pos + 1
            while scan < len(lines):
                row = lines[scan].strip()
                if row.startswith("."):
                    break
                if row and not row.startswith("#"):
                    table.append(row)
                scan += 1
            if not ((len(nets) == 1 and table in ([], ["1"]))
                    or (len(nets) == 2
                        and table in (["1 1"], ["0 1"]))):
                return False
        elif line.startswith("."):
            directive = line.split()[0]
            if directive not in {".model", ".inputs", ".outputs", ".end"}:
                return False
        # Wrapped .inputs/.outputs continuation lines are intentionally
        # ignored.  Logic table rows were already checked with their .names.
    return True


def _directive_names(text: str, directive: str) -> Optional[List[str]]:
    """Read one possibly backslash-wrapped BLIF header directive."""
    lines = text.splitlines()
    found: Optional[List[str]] = None
    pos = 0
    while pos < len(lines):
        line = lines[pos].strip()
        tokens = line.split()
        if tokens and tokens[0] == directive:
            if found is not None:
                return None
            names = tokens[1:]
            while names and names[-1] == "\\":
                names.pop()
                pos += 1
                if pos >= len(lines):
                    return None
                continuation = lines[pos].strip().split()
                names.extend(continuation)
            found = names
        pos += 1
    return found


def _resolve_constant(net: str, constants: Dict[str, str]) -> Optional[str]:
    seen: Set[str] = set()
    while net in constants:
        if net in seen:
            return None
        seen.add(net)
        net = constants[net]
    return net


def _has_comb_cycle(nl: Netlist) -> bool:
    """Detect a cycle without trusting Netlist's last-driver-wins index."""
    by_output = {gate.out: gate for gate in nl.gates}
    state: Dict[str, int] = {}

    def visit(net: str) -> bool:
        mark = state.get(net, 0)
        if mark == 1:
            return True
        if mark == 2:
            return False
        gate = by_output.get(net)
        if gate is None:
            state[net] = 2
            return False
        state[net] = 1
        if any(visit(source) for source in gate.ins):
            return True
        state[net] = 2
        return False

    return any(visit(gate.out) for gate in nl.gates)


def _valid_source(nl: Netlist) -> bool:
    """Reject an IR that BLIF export would otherwise silently weaken."""
    expected_arity = {kind: (1 if kind in ONE_INPUT else 2)
                      for kind in GATE_TYPES}
    if any(gate.type not in expected_arity
           or len(gate.ins) != expected_arity[gate.type]
           for gate in nl.gates):
        return False
    gate_outputs = [gate.out for gate in nl.gates]
    if len(gate_outputs) != len(set(gate_outputs)):
        return False
    q_nets = {ff.q for ff in nl.dffs}
    if set(gate_outputs) & (set(nl.pi) | q_nets):
        return False
    ff_names = [ff.name for ff in nl.dffs]
    if len(ff_names) != len(set(ff_names)):
        return False
    return not _has_comb_cycle(nl)


def _rebuild(source: Netlist, text: str,
             po_out: Dict[str, str],
             pin_out: Dict[str, Tuple[str, str]]) -> Optional[Netlist]:
    """Strict counterpart of abc_opt's normal mapped-BLIF reconstruction."""
    # Imported here to avoid a top-level abc_opt <-> nand_super cycle.
    from . import abc_opt

    if not _strict_mapped_blif(text):
        return None
    gates, constants = abc_opt._parse_gate_blif(text)
    if not gates and not constants:
        return None

    arity = {"nand": 2, "not": 1, "buf": 1}
    if any(kind not in arity or len(inputs) != arity[kind]
           for kind, _output, inputs in gates):
        return None
    raw_outputs = [output for _kind, output, _inputs in gates]
    if len(raw_outputs) != len(set(raw_outputs)):
        return None

    candidate = source.snapshot()
    candidate.gates = []
    rename: Dict[str, str] = dict(po_out)
    gate_by_out = {
        output: (kind, inputs) for kind, output, inputs in gates
    }
    skip_outputs: Set[str] = set()
    mapped_pins: Dict[Tuple[str, str], str] = {}

    used_nets = set(source.all_nets()) | set(raw_outputs)

    net_serial = 0

    def fresh_net() -> str:
        nonlocal net_serial
        while True:
            net_serial += 1
            name = "__cada_nsp{}".format(net_serial)
            if name not in used_nets:
                used_nets.add(name)
                return name

    for label, (ff_name, attr) in pin_out.items():
        resolved = _resolve_constant(label, constants)
        if resolved is None:
            return None
        if is_const(resolved):
            mapped_pins[(ff_name, attr)] = resolved
        elif (resolved in gate_by_out
              and gate_by_out[resolved][0] == "buf"
              and len(gate_by_out[resolved][1]) == 1):
            skip_outputs.add(resolved)
            pin_source = _resolve_constant(
                gate_by_out[resolved][1][0], constants)
            if pin_source is None:
                return None
            mapped_pins[(ff_name, attr)] = pin_source
        else:
            mapped = fresh_net()
            rename[label] = mapped
            mapped_pins[(ff_name, attr)] = mapped

    def remap(net: str) -> Optional[str]:
        resolved = _resolve_constant(net, constants)
        if resolved is None:
            return None
        return rename.get(resolved, resolved)

    used_instances = {ff.name for ff in candidate.dffs}
    serial = 0

    def fresh_instance() -> str:
        nonlocal serial
        while True:
            serial += 1
            name = "__cada_nsg{}".format(serial)
            if name not in used_instances:
                used_instances.add(name)
                return name

    rebuilt: List[Gate] = []
    for kind, output, inputs in gates:
        if output in skip_outputs:
            continue
        mapped_inputs = [remap(net) for net in inputs]
        if any(net is None for net in mapped_inputs):
            return None
        real_output = rename.get(output, output)
        rebuilt.append(Gate(kind, fresh_instance(), real_output,
                            [net for net in mapped_inputs if net is not None]))

    candidate.gates = rebuilt
    produced = {gate.out for gate in rebuilt}
    for label, output in po_out.items():
        if output in produced:
            continue
        direct = remap(label)
        # A missing pseudo-output is not silently tied low.  It is acceptable
        # only when ABC explicitly resolved it to a real source or constant.
        if direct is None or direct == label:
            return None
        candidate.gates.append(
            Gate("buf", fresh_instance(), output, [direct]))

    ff_by_name = {ff.name: ff for ff in candidate.dffs}
    if any(ff_name not in ff_by_name
           for ff_name, _attr in mapped_pins):
        return None
    for ff in candidate.dffs:
        for attr in ("d", "clk", "rn", "sn"):
            key = (ff.name, attr)
            if key not in mapped_pins:
                return None
            mapped_pin = remap(mapped_pins[key])
            if mapped_pin is None:
                return None
            setattr(ff, attr, mapped_pin)

    candidate.touch()
    try:
        rewrite.to_basis(candidate, "NAND_NOT")
    except (AssertionError, IndexError, KeyError, ValueError):
        return None
    candidate.touch()

    if any(gate.type not in {"nand", "not"} for gate in candidate.gates):
        return None
    outputs = [gate.out for gate in candidate.gates]
    if len(outputs) != len(set(outputs)):
        return None
    source_boundaries = (set(source.pi) | {ff.q for ff in source.dffs}
                         | {net for net in source.all_nets()
                            if source.driver(net)[0] == "undriven"})
    for gate in candidate.gates:
        for net in gate.ins:
            if (not is_const(net)
                    and candidate.driver(net)[0] == "undriven"
                    and net not in source_boundaries):
                return None
    if _has_comb_cycle(candidate) or not _valid_source(candidate):
        return None
    return candidate


def map_netlist(nl: Netlist, timeout: float = 120.0,
                dsd: bool = False, verify: bool = True) -> Optional[Netlist]:
    """Map ``nl`` to the cached four-level NAND/NOT supergate library.

    With ``dsd=True``, the generic ``&get -n; &dsd; &put; strash`` transform is
    applied before technology mapping.  It is useful for wide decomposable
    logic; callers can also run a direct second-stage mapping with ``dsd=False``.

    ``timeout`` is one wall-clock budget shared by lazy library generation,
    mapping, parsing and (by default) local CEC.  ``None`` is returned on every
    failure or inconclusive proof.
    """
    if timeout <= 0 or not _valid_source(nl):
        return None
    deadline = time.monotonic() + float(timeout)
    library = _cached_library(deadline)
    if library is None or _remaining(deadline) <= 0.0:
        return None

    # Imported lazily to avoid the module cycle described in _rebuild().
    from . import abc_opt

    try:
        blif, po_out, pin_out = abc_opt._opt_blif(nl)
    except Exception:
        return None

    try:
        work = tempfile.mkdtemp(prefix="cada_nand_map_")
    except OSError:
        return None
    source_path = os.path.join(work, "input.blif")
    output_path = os.path.join(work, "output.blif")
    try:
        with open(source_path, "w") as stream:
            stream.write(blif)
        commands: List[str] = [
            'read_blif "{}"'.format(source_path),
            "strash",
        ]
        if dsd:
            commands.extend(_DSD_RECIPE)
        commands.extend((
            "read_genlib nand.genlib",
            "read_super nand.super",
            "map",
            'write_blif "{}"'.format(output_path),
        ))
        ok, _log = _run_abc(commands, library, deadline)
        if not ok or not os.path.isfile(output_path):
            return None
        with open(output_path) as stream:
            mapped_text = stream.read()
    except (OSError, UnicodeError, ValueError):
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)

    expected_outputs = set(po_out) | set(pin_out)
    mapped_outputs = _directive_names(mapped_text, ".outputs")
    if (not expected_outputs or mapped_outputs is None
            or len(mapped_outputs) != len(set(mapped_outputs))
            or set(mapped_outputs) != expected_outputs):
        return None
    try:
        candidate = _rebuild(nl, mapped_text, po_out, pin_out)
    except Exception:
        candidate = None
    if candidate is None:
        return None
    if verify:
        try:
            labels = shared_synthetic_labels(nl, candidate)
            reference_blif = to_blif(nl, synthetic_labels=labels)
            candidate_blif = to_blif(candidate, synthetic_labels=labels)
        except Exception:
            return None
        left = _remaining(deadline)
        try:
            proved = (left > 0.0 and abc_bridge.cec_blif(
                reference_blif, candidate_blif, timeout=left) is True)
        except Exception:
            proved = False
        if not proved:
            return None
    return candidate
