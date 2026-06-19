#!/usr/bin/env bash
# Run the local evaluator inside a throwaway sandbox directory, so the
# `testNN_out.v` files the agent writes never land in the repo.
#
#   scripts/eval_sandbox.sh                      # all cases
#   scripts/eval_sandbox.sh test32 test37        # selected cases
#   scripts/eval_sandbox.sh --update-golden      # refresh the golden baseline
#   scripts/eval_sandbox.sh --verbose
#
# How it works: only the working directory is changed to a temp dir (the prompts
# write `testNN_out.v` relative to cwd, so the outputs land there and are deleted
# on exit). The evaluator code, the testcases (via a symlink) and the golden
# baseline are all reached by ABSOLUTE paths, so nothing else is affected.
#
# This only sandboxes the EVALUATOR. It does NOT change how the contest
# executable (./cada0001_alpha) runs — that keeps its own working directory.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

py="$repo/.venv/bin/python"
[ -x "$py" ] || py="$(command -v python3 || command -v python)"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT          # always clean up, even on error / Ctrl-C

ln -s "$repo/testcase" "$tmp/testcase"   # so the prompts' load paths resolve

export PYTHONHASHSEED=0            # reproducible output
( cd "$tmp" && "$py" "$repo/evaluator/evaluate.py" "$@" )
