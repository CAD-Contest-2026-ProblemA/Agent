#!/usr/bin/env bash
# Build a statically-linked abc and install it to dist/abc_static/ so that
# scripts/build.sh can embed it into the one-file executable.
#
#   scripts/build_static_abc.sh [abc_source]     # default: ~/abc (git clone ok)
#
# Static linking removes the glibc version dependency, so the same abc binary
# runs on any x86_64 Linux (kernel >= 3.2) regardless of distro age — exactly
# what a contest machine with an older glibc needs.  The dlopen warning during
# linking is harmless (abc's optional plugin loader, never used here).
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${1:-$HOME/abc}"
work="$(mktemp -d /tmp/abc_static_build.XXXXXX)"
trap 'rm -rf "$work"' EXIT

[ -d "$src" ] || { echo "error: abc source not found at $src" >&2; exit 1; }

echo ">> cloning $src -> $work/abc"
git clone -q "$src" "$work/abc" 2>/dev/null || cp -r "$src" "$work/abc"

echo ">> building statically (this takes a while)"
make -C "$work/abc" -j"$(nproc)" ABC_USE_NO_READLINE=1 LDFLAGS="-static" abc

file "$work/abc/abc" | grep -q "statically linked" || {
  echo "error: result is not statically linked" >&2; exit 1; }

mkdir -p "$repo/dist/abc_static"
strip -o "$repo/dist/abc_static/abc" "$work/abc/abc"
cp "$work/abc/abc.rc" "$repo/dist/abc_static/"
"$repo/dist/abc_static/abc" -c "version" | head -2

echo ""
echo "Installed: $repo/dist/abc_static/{abc,abc.rc}"
echo "Now rebuild the agent to embed it:  scripts/build.sh cada1125_beta"
