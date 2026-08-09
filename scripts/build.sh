#!/usr/bin/env bash
# Package the agent into a single standalone executable with PyInstaller.
#
#   scripts/build.sh [exec_name]        # default name: cada0001_alpha
#   scripts/build.sh cada1125_alpha     # use YOUR team number
#   WITH_LLM=0 scripts/build.sh cada1125_alpha   # lightweight: skip openai/anthropic
#   NO_RULES=1 scripts/build.sh cada1125_alpha   # pure-LLM: rules off by default
#
# By default the openai/anthropic LLM fallback IS bundled. Set WITH_LLM=0 for a
# smaller binary (the 40 public testcases never invoke the LLM, so the
# lightweight build still passes them).
#
# NO_RULES=1 bundles a marker the binary reads at startup, so it behaves as if
# --no-rules were always passed: the regex table is skipped and every request
# line is routed by the LLM. This is a MEASUREMENT build, not a faster one --
# it answers at the model's routing accuracy, spends one to three API calls per
# request line, and fails the whole run if the key or the network is unavailable.
# The binary still accepts --rules to put the table back for one run, prints a
# line to stderr saying which mode it is in, and refuses to start rules-off with
# no reachable LLM.
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
NO_RULES="${NO_RULES:-0}"

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

# Example bank for retrieval.  --collect-submodules gathers Python modules,
# not data files, so without this the frozen binary finds no examples.jsonl,
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
if [ "${#retrieval_args[@]}" -eq 0 ]; then
  echo ">> ERROR: no retrieval assets resolved — the binary would ship without" >&2
  echo "          the example bank. Run scripts/export_examples.py first." >&2
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
  echo "Mode:  PURE-LLM — this binary routes every request through the LLM."
  echo "       Pass --rules to use the regex table for a run."
fi
echo "Test it:  ./dist/$NAME --doctor"
echo "          ./dist/$NAME -config configs/api_key.yaml < testcase/test01/prompt.txt"
echo "(remember: abc/yosys are resolved at runtime — keep configs/tools.yaml next"
echo " to the binary, or set ABC_BIN / put abc on \$PATH)"
