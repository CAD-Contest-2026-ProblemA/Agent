#!/usr/bin/env bash
# One-time environment setup using uv.
#
# uv downloads a self-contained, modern Python (built on old glibc, so it runs
# even on an old contest machine) that is completely independent of the system
# Python.  This sidesteps "system Python too old" without PyInstaller's
# glibc-bundling fragility.
#
# The benchmark itself needs ZERO third-party packages (config parsing falls
# back to a built-in mini-parser); PyYAML/openai/anthropic are only installed so
# the optional LLM fallback works.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYVER="${CADA_PYTHON:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo ">> installing managed Python ${PYVER} via uv ..."
uv python install "${PYVER}" || true

echo ">> creating .venv (python ${PYVER}) ..."
uv venv --python "${PYVER}" .venv

echo ">> installing dependencies (optional; enables LLM fallback) ..."
uv pip install --python .venv/bin/python -r requirements.txt || \
  echo "   (dependency install skipped/failed — the benchmark still runs without them)"

echo ""
echo "Setup complete. Test with:"
echo "  ./cada0001_alpha -config configs/default.yaml < testcase/test01/prompt.txt"
