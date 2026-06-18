# CADA — LLM-Assisted Netlist Exploration and Transformation

An entry for **ICCAD 2026 Contest Problem A**.  The system accepts
natural-language requests on stdin, interprets each one, runs the corresponding
analysis / transformation / optimization on a gate-level Verilog netlist, and
streams answers back as `#RESPONSE <id>` … `#END <id>` frames (mirrored to
`<case_name>.log`).

> 中文版說明請見 [README_ZH.md](README_ZH.md)。

## Design in one sentence

A **deterministic gate-level EDA engine** wrapped in a **thin natural-language
front end**: a rule-based regex router maps each (highly templated) request to
one structured intent, and an LLM is used *only* as a fallback translator that
turns an unrecognised line into one `{"intent", "params"}` object — it never
reasons about the circuit.  All correctness-critical work (parsing, analysis,
transforms, equivalence) is deterministic Python; ABC/yosys are used only as
equivalence / cost-ranked-synthesis oracles.

## Requirements

* A Python interpreter — **3.8+** (the project itself is pure-stdlib for the
  benchmark path).
* **`abc` (required)** and **`yosys` (fallback)** on `PATH` (or `ABC_BIN`
  pointing at the ABC binary) — equivalence / optimization back ends.  Note:
  `setup.sh` / pip / uv do **not** install these external binaries; make sure
  they exist on the machine.  ABC is used by 24/40 testcases (test01–16 are pure
  Python and never call it); yosys is only a fallback when ABC's cec can't
  decide — exercised by 0/40.
* Optional Python packages: `openai` / `anthropic` (LLM fallback only — the
  benchmark never calls the LLM) and `PyYAML` (config parsing; falls back to a
  built-in mini-parser if absent).

## Get the code

To clone the repository contents **directly into the current folder** (no extra
`Agent/` subdirectory), pass `.` as the target — the current directory must be
empty:

```bash
mkdir my-submission && cd my-submission
git clone https://github.com/CAD-Contest-2026-ProblemA/Agent.git .
```

Or clone into a folder of your choosing:

```bash
git clone https://github.com/CAD-Contest-2026-ProblemA/Agent.git <folder> && cd <folder>
```

(If the target directory is not empty, `git clone … .` refuses; clone into a temp
dir and move the contents instead.)

## Quick start (recommended: uv)

`uv` fetches a self-contained modern Python (built on old glibc, so it runs on
old contest machines too) independent of the system Python:

```bash
bash setup.sh        # one-time: uv creates .venv with Python 3.12 (+ optional deps)

# run exactly like the contest harness:
./cada0001_alpha -config configs/default.yaml < testcase/test01/prompt.txt
```

The `cada0001_alpha` launcher prefers `.venv/bin/python`; if `.venv` is absent
but `uv` is present it bootstraps once; otherwise it falls back to the system
`python3`.

### Without uv

```bash
pip install -r requirements.txt     # only needed for the LLM fallback
./cada0001_alpha -config configs/default.yaml < testcase/test01/prompt.txt
```

The benchmark runs fully **offline** with no third-party packages — the config
parser falls back to a built-in mini-parser, and the regex router covers all 40
testcases (the LLM is never invoked on them).

## Configuration (`-config`)

Same shape as the contest's Figure 6.  Drop your key into
`configs/default.yaml`:

```yaml
provider: "openai"          # or: "anthropic"
openai:
  api_key: <YOUR_API_KEY>
  model: "gpt-4o-mini"
anthropic:
  api_key: <YOUR_API_KEY>
  model: "claude-haiku-4-5"
generation:
  temperature: 0.2
  max_output_tokens: 4096
```

## Pre-flight check (doctor)

Before running an evaluation, check the environment:

```bash
./cada0001_alpha --doctor        # or:  python3 -m cada.doctor  /  python3 scripts/doctor.py
```

It verifies, in order: **Python** (uv installed → `.venv` present → packages
inside `.venv`; if uv or `.venv` is missing the section fails and the host Python
is *not* inspected); **external tools** (`abc` required, `yosys` fallback —
resolved the same way the agent resolves them, then actually executed, so a
present-but-broken binary, e.g. a glibc/arch mismatch, is caught); the **config
files** and optional LLM key; and the **agent package** itself (import + parse a
testcase).  Exit code is non-zero on hard failures.

## External tool locations (avoid relying on $PATH)

Pin `abc`/`yosys` (and any other tool) in **`configs/tools.yaml`**, which is
auto-loaded — no need to put them on `$PATH`:

```yaml
tools:
  abc: /home/me/abc/abc          # point at the real binary (abc.rc lives beside it)
  yosys: /usr/local/bin/yosys
```

Resolution order (later wins): `configs/tools.yaml` → env vars
(`ABC_BIN`, `YOSYS_BIN`, or `CADA_<NAME>_BIN`) → `-tools <file>` → a `tools:`
section inside the `-config` file → built-in candidates → `$PATH` (last resort).
If a configured path is missing it degrades gracefully to the next source.

## Local testing

```bash
python3 scripts/run_local.py testcase/test22          # one case to stdout
python3 scripts/run_local.py --all                    # every case
```

## Evaluator (local self-check)

The contest ships no golden answers, so the evaluator rigorously checks what
*can* be verified locally and snapshots the rest:

```bash
python3 evaluator/evaluate.py                  # all cases
python3 evaluator/evaluate.py test21 test40    # selected cases
python3 evaluator/evaluate.py --verbose        # show every check
python3 evaluator/evaluate.py --update-golden   # pin current responses as baseline
```

Per case it reports four groups:

* **HARD** — the binary 0-score gate, checked *independently* and auto-derived
  from each request's wording: functional equivalence to the original (ABC
  `cec`), structural bounds ("no gate/signal drives > K"), gate-basis purity
  (whole design or a cone), and a valid round-tripping output netlist.
* **DERIVED** — answers with an objective value computed a second way (gate-type
  counts and PI/PO counts straight from the `.v`).
* **GOLDEN** — every response diffed against a pinned baseline in
  `evaluator/golden/` (regression detection).
* **OPT** — the achieved optimize cost (max depth / gate count); lower ranks better.

Exit code is non-zero if any HARD requirement fails.  Note: the golden baseline
is the *current engine's* output, not official answers — it catches drift, not
correctness.  Drop the organizers' answers into `evaluator/golden/<case>.txt`
(one normalized response per line) to grade for real.

## Architecture

```
stdin ─► io_/protocol ─► agent/agent (regex router; LLM fallback)
                              │
            ┌─────────────────┼──────────────────────────┐
            ▼                 ▼                           ▼
       analysis/*        transform/*  ── guards ──►  optimize/abc_opt
   counts depth paths    rewrite constprop           (cost-ranked, ABC)
   cones connectivity    cleanup buffering naming           │
   functional sequential        │                           │
            │                    ▼                           ▼
            └────────►  netlist/ir  (single source of truth) ◄── equiv/gate
                         reader · writer · blif_export        (ABC cec / yosys)
                              │
                  #RESPONSE/#END ─► stdout + <case>.log  (flushed every frame)
```

| Area | Module | Purpose |
|------|--------|---------|
| IR | `netlist/ir.py` | Flat gate-level netlist; gates, named-port DFFs, driver/loads, snapshots |
| Parse/emit | `netlist/reader.py`, `netlist/writer.py` | Self-written Verilog parser + canonical structural writer (round-trips exactly) |
| Equivalence | `netlist/blif_export.py`, `equiv/*` | Register-cut BLIF → ABC `cec`; yosys fallback |
| Analysis | `analysis/*` | counts, cones, depth, connectivity, paths (DP, never materialised), functional (ABC/SAT), sequential |
| Transform | `transform/*` | basis remap, XOR/XNOR decomposition, constant propagation, dangling removal, fixpoint duplicate merge, buffer trees, renaming |
| Optimize | `optimize/abc_opt.py` | Depth/area minimisation via ABC + unit-delay genlib mapping, basis-preserving, cec-guarded |
| Agent | `agent/*` | rule router, request state + snapshots + transform deltas |
| LLM | `llm/*` | thin dual-provider client + cached fallback translator |

## Correctness model

* Every structural transform is **equivalence-preserving by construction** and
  is additionally checked with a register-cut combinational `cec` before
  commit; any violation rolls back to the pre-transform snapshot.
* Basis conversion uses **pure-Python templates** (no ABC technology mapping),
  so the classic ABC `zero`/`one` cell leak never reaches the output and basis
  purity is guaranteed.
* Paths are **counted with a topological DP** (exact big-integer counts) and
  enumerated only under a bounded threshold — path sets are never materialised.
* Depth is measured as the contest defines it (one gate = one level, inverters
  included); the optimizer maps to a unit-delay library so ABC minimises that
  same metric.

Validation: all 40 testcases run end-to-end with correct framing, every prompt
line maps to a specific EDA operation, and every output netlist is ABC-verified
functionally equivalent to its input.

## Notes / tunable semantics

A few "exactly-correct-or-zero" conventions (path-enumeration output format,
"wire is a cut" definition, enable/hold strict-vs-broad counting, Boolean
equation over register-state leaves) are centralised and documented; adjust them
against an official sample if the grader's wording differs.  The DFF model is the
general named-port form `dff(.RN,.SN,.CK,.D,.Q)` with both asynchronous
active-low reset (RN) and set (SN) handled per-instance (reset dominates set).
