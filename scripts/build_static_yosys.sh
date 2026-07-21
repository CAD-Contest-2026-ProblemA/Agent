#!/usr/bin/env bash
# Build a statically-linked, relocatable yosys and install it (binary +
# share/ tree) to dist/yosys_static/ so scripts/build.sh can embed it.
#
#   scripts/build_static_yosys.sh [version]      # default: v0.66
#
# Only the core passes the agent needs are kept (read_blif/read_verilog, opt,
# techmap, aigmap, equiv_*, write_blif): TCL, plugins, readline, pyosys and
# yosys's internal abc are disabled, which is what makes a clean static link
# possible.  PREFIX is deliberately bogus, so the binary can only find its
# share/ files through yosys's binary-relative lookup (<bindir>/share/) —
# proving the pair is relocatable anywhere.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ver="${1:-v0.66}"
work="$(mktemp -d /tmp/yosys_static_build.XXXXXX)"
trap 'rm -rf "$work"' EXIT

echo ">> cloning yosys $ver"
git clone -q --depth 1 --branch "$ver" https://github.com/YosysHQ/yosys.git "$work/yosys"
git -C "$work/yosys" submodule update --init --depth 1

zlib=0
[ -f /usr/lib/x86_64-linux-gnu/libz.a ] && zlib=1
cat > "$work/yosys/Makefile.conf" <<EOF
PREFIX := /nonexistent_yosys_prefix
ENABLE_TCL := 0
ENABLE_ABC := 0
ENABLE_PLUGINS := 0
ENABLE_READLINE := 0
ENABLE_EDITLINE := 0
ENABLE_PYOSYS := 0
ENABLE_ZLIB := $zlib
LINKFLAGS += -static
EOF

echo ">> building statically (this takes a while)"
make -C "$work/yosys" -j"$(nproc)" yosys

file "$work/yosys/yosys" | grep -q "statically linked" || {
  echo "error: result is not statically linked" >&2; exit 1; }

rm -rf "$repo/dist/yosys_static"
mkdir -p "$repo/dist/yosys_static"
strip -o "$repo/dist/yosys_static/yosys" "$work/yosys/yosys"
cp -r "$work/yosys/share" "$repo/dist/yosys_static/"
"$repo/dist/yosys_static/yosys" -V

echo ""
echo "Installed: $repo/dist/yosys_static/{yosys,share/}"
echo "Now rebuild the agent to embed it:  scripts/build.sh cada1125_beta"
