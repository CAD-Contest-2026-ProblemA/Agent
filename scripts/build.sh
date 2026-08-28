#!/usr/bin/env bash
# Package the agent into a single standalone executable with PyInstaller.
#
#   scripts/build.sh [exec_name]        # pure-LLM; default name: cada0001_alpha
#   scripts/build.sh cada1125_alpha     # pure-LLM; use YOUR team number
#   NO_RULES=0 scripts/build.sh cada1125_alpha   # regex first, LLM fallback
#   WITH_BM25=0 scripts/build.sh cada1125_alpha  # no retrieved examples
#   NO_RULES=0 WITH_LLM=0 scripts/build.sh cada1125_alpha  # regex-only, smaller
#   FULL_OPT_VERIFY=1 scripts/build.sh cada1125_alpha  # run ABC performance regression
#
# By default the openai/anthropic clients ARE bundled and the regex table is
# skipped, so every request is routed by the LLM.  Set NO_RULES=0 to build the
# hybrid rules-first mode instead.  WITH_LLM=0 is valid only with NO_RULES=0.
#
# The default NO_RULES=1 bundles a marker the binary reads at startup, so it
# behaves as if --no-rules were always passed: the regex table is skipped and
# every request line is routed by the LLM. This is a MEASUREMENT build, not a
# faster one -- it answers at the model's routing accuracy and spends one to
# three API calls per request line. Missing SDK/key fails at startup;
# request-time provider errors are reported and may degrade a line to a no-op.
# The binary still accepts --rules to put the table back for one run, prints a
# line to stderr saying which mode it is in, and refuses to start rules-off with
# no reachable LLM.
#
# BM25 example retrieval is ON by default.  WITH_BM25=0 bakes in an off marker,
# so LLM requests contain only the static intent catalog.  The small public
# bank still ships, allowing --bm25 to turn retrieval back on for one run.
#
# Output:  dist/<exec_name>   (a single file; no Python/uv needed to run it)
#
# IMPORTANT — glibc: PyInstaller bundles the Python interpreter but NOT the C
# library. Build on a machine whose glibc is <= the target's (ideally the
# CONTEST machine itself, which has uv), or the binary will fail to start with
# a "GLIBC_2.xx not found" error.
#
# NOT bundled: abc / yosys (they stay external — put their paths in
# configs/tools.yaml next to the binary, in the -config tools: section, in
# ABC_BIN/YOSYS_BIN, or on $PATH).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
NAME="${1:-cada0001_alpha}"
WITH_LLM="${WITH_LLM:-1}"
NO_RULES="${NO_RULES:-1}"
WITH_BM25="${WITH_BM25:-1}"
FULL_OPT_VERIFY="${FULL_OPT_VERIFY:-0}"

if [ "$WITH_BM25" != 0 ] && [ "$WITH_BM25" != 1 ]; then
  echo ">> ERROR: WITH_BM25 must be 0 or 1 (got: $WITH_BM25)" >&2
  exit 1
fi

if [ "$FULL_OPT_VERIFY" != 0 ] && [ "$FULL_OPT_VERIFY" != 1 ]; then
  echo ">> ERROR: FULL_OPT_VERIFY must be 0 or 1 (got: $FULL_OPT_VERIFY)" >&2
  exit 1
fi

# The one combination that cannot work.  Without the provider SDK the LLM
# client never initialises, and a rules-off binary with no router left answers
# every line with the same "not mapped to a specific EDA operation" ack -- a
# complete, well-formed, entirely empty transcript that exits 0.
if [ "$NO_RULES" = 1 ] && [ "$WITH_LLM" != 1 ]; then
  echo ">> ERROR: NO_RULES=1 needs WITH_LLM=1 — a rules-off binary with no" >&2
  echo "          bundled provider SDK has nothing left to route with." >&2
  exit 1
fi

# throwaway build venv with PyInstaller (kept out of the runtime .venv)
bvroot="$(mktemp -d)"
trap 'rm -rf "$bvroot"' EXIT
bvenv="$bvroot/venv"

echo ">> creating build venv with PyInstaller ..."
if command -v uv >/dev/null 2>&1; then
  uv venv "$bvenv" >/dev/null
  uv pip install --python "$bvenv/bin/python" pyinstaller pyyaml >/dev/null
  [ "$WITH_LLM" = 1 ] && uv pip install --python "$bvenv/bin/python" openai anthropic >/dev/null
else
  python3 -m venv "$bvenv"
  "$bvenv/bin/pip" install -q --upgrade pip
  "$bvenv/bin/pip" install -q pyinstaller pyyaml
  [ "$WITH_LLM" = 1 ] && "$bvenv/bin/pip" install -q openai anthropic
fi

echo ">> verifying prepared solution assets ..."
PYTHONPATH="$repo" "$bvenv/bin/python" \
  "$repo/scripts/verify_prepared_registry.py"

if [ "$FULL_OPT_VERIFY" = 1 ]; then
  echo ">> verifying generic runtime NAND-super cone flow ..."
  PYTHONPATH="$repo" "$bvenv/bin/python" \
    "$repo/scripts/verify_nand_super_runtime.py"
else
  echo ">> skipping ABC performance regression (set FULL_OPT_VERIFY=1 to run)"
fi

extra=()
if [ "$WITH_LLM" = 1 ]; then
  extra+=(--collect-all openai --collect-all anthropic)
fi

# Rules-off default.  The marker goes at the bundle root, which is one of the
# directories cada/main.py already searches (sys._MEIPASS, then the directory
# holding the binary), so no new lookup path is introduced.  Written under the
# build venv's temp dir so nothing lands in the repo.
if [ "$NO_RULES" = 1 ]; then
  echo ">> building a PURE-LLM binary: --no-rules is the default (--rules overrides)"
  marker="$bvroot/no_rules.flag"
  printf 'built by scripts/build.sh with NO_RULES=1\n' > "$marker"
  extra+=(--add-data "$marker:.")
fi

# BM25-off default.  Keep the public bank in the bundle so --bm25 remains a
# useful runtime override; this marker controls whether Agent constructs the
# retriever and appends a dynamic example block to each LLM request.
if [ "$WITH_BM25" = 0 ]; then
  echo ">> building with BM25 DISABLED by default (--bm25 overrides)"
  bm25_marker="$bvroot/no_bm25.flag"
  printf 'built by scripts/build.sh with WITH_BM25=0\n' > "$bm25_marker"
  extra+=(--add-data "$bm25_marker:.")
fi

# Embed a statically-linked abc (fully self-contained executable).  Produce it
# once with scripts/build_static_abc.sh; if absent the binary still works but
# needs an external abc (tools.yaml / ABC_BIN / ~/abc/abc / $PATH).
if [ -f "$repo/dist/abc_static/abc" ]; then
  echo ">> embedding statically-linked abc into the binary"
  extra+=(--add-binary "$repo/dist/abc_static/abc:abc"
          --add-data "$repo/dist/abc_static/abc.rc:abc")
else
  echo ">> NOTE: dist/abc_static/abc not found — abc will NOT be embedded"
  echo "         (run scripts/build_static_abc.sh first for a self-contained build)"
fi

# Embed a statically-linked yosys (+ its share/ tree, found via yosys's
# binary-relative lookup).  Optional: an extra resynthesis seed — the agent
# degrades gracefully without it.  Produce with scripts/build_static_yosys.sh.
if [ -f "$repo/dist/yosys_static/yosys" ]; then
  echo ">> embedding statically-linked yosys into the binary"
  extra+=(--add-binary "$repo/dist/yosys_static/yosys:yosys"
          --add-data "$repo/dist/yosys_static/share:yosys/share")
else
  echo ">> NOTE: dist/yosys_static/yosys not found — yosys will NOT be embedded"
fi

# Example bank for retrieval.  It is bundled even with WITH_BM25=0 so a user
# can enable retrieval for one run with --bm25.  --collect-submodules gathers
# Python modules,
# not data files, so without this the frozen binary finds no
# public_examples.jsonl,
# silently disables retrieval and routes at the pre-retrieval accuracy with
# nothing to say why.  scripts/spec_assets.py is the single source of truth
# (the .spec files use the same module) so the two build paths cannot drift.
#   WITH_DENSE=1 also embeds the ONNX encoder (~94MB).  Off by default: it
#   measured within the noise of the stdlib BM25 retriever.
dense_flag=()
[ "${WITH_DENSE:-0}" = 1 ] && dense_flag+=(--dense)
mapfile -t retrieval_args < <(
  "$bvenv/bin/python" "$repo/scripts/spec_assets.py" "$repo" \
    --pyinstaller-args "${dense_flag[@]}" | tr ' ' '\n' | grep -v '^$'
)
if [ ! -f "$repo/cada/llm/public_examples.jsonl" ]; then
  echo ">> ERROR: no retrieval assets resolved — the binary would ship without" >&2
  echo "          the example bank. Run scripts/export_public_examples.py first." >&2
  exit 1
fi

echo ">> running PyInstaller (name: $NAME) ..."
"$bvenv/bin/pyinstaller" --onefile --clean --noconfirm \
  --name "$NAME" \
  --paths "$repo" \
  --collect-submodules cada \
  --hidden-import yaml \
  --add-data "$repo/configs:configs" \
  "${retrieval_args[@]}" \
  "${extra[@]}" \
  "$repo/scripts/_pyi_entry.py"

echo ""
echo "Built: $repo/dist/$NAME"
if [ "$NO_RULES" = 1 ]; then
  echo "Routing: PURE-LLM (pass --rules to use regex-first for one run)"
else
  echo "Routing: REGEX-FIRST (pass --no-rules to use pure LLM for one run)"
fi
if [ "$WITH_BM25" = 1 ]; then
  echo "BM25:    ENABLED (pass --no-bm25 to omit retrieved examples)"
else
  echo "BM25:    DISABLED (pass --bm25 to append retrieved examples)"
fi
echo "Test it:  ./dist/$NAME --doctor"
echo "          ./dist/$NAME -config configs/api_key.yaml < testcase/test01/prompt.txt"
echo "(remember: abc/yosys are resolved at runtime — keep configs/tools.yaml next"
echo " to the binary, or set ABC_BIN / put abc on \$PATH)"
