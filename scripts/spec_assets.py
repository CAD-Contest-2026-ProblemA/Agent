"""PyInstaller payload for example retrieval, kept in version control.

The .spec files are gitignored build artifacts, so anything written directly
into them is lost on the next regeneration and invisible to review.  This
module holds the logic; a spec only needs:

    import sys; sys.path.insert(0, '<project root>')
    from scripts.spec_assets import retrieval_assets
    d, b, h = retrieval_assets('<project root>')
    datas += d; binaries += b; hiddenimports += h

The encoder and its vectors are optional: they are gitignored artifacts
produced by scripts/fetch_embed_model.py and scripts/build_vectors.py, so a
fresh clone has neither.  When they are absent the build still succeeds and
the binary uses the stdlib BM25 retriever — a missing model degrades retrieval
quality rather than breaking the build.  The example bank always ships;
without it there is no retrieval at all.
"""

from __future__ import annotations

import os
from typing import List, Tuple

PKG = os.path.join("cada", "llm")
BANK = "examples.jsonl"
VECS = "example_vecs.npy"
MODEL_DIR = "embed_model"
DENSE_IMPORTS = ["numpy", "onnxruntime", "tokenizers"]


def _note(msg: str) -> None:
    # stderr, not stdout: --pyinstaller-args pipes stdout straight into a
    # PyInstaller command line, and a stray word there becomes a bad flag.
    import sys
    print(f"[spec_assets] {msg}", file=sys.stderr)


def retrieval_assets(root: str, dense: bool = False
                     ) -> Tuple[List[tuple], List[tuple], List[str]]:
    """Return (datas, binaries, hiddenimports) to merge into a .spec.

    ``dense`` bundles the ONNX encoder and its vectors (~94MB).  Off by
    default: measured end to end the dense retriever landed within the noise
    of the stdlib BM25 one, so the shipped binary carries only the example
    bank and stays at its previous size.
    """
    datas: List[tuple] = []
    binaries: List[tuple] = []
    hidden: List[str] = []

    bank = os.path.join(root, PKG, BANK)
    if os.path.isfile(bank):
        datas.append((bank, PKG))
    else:
        _note(f"WARNING: {bank} missing — the binary will have NO example "
              f"retrieval. Run scripts/export_examples.py first.")

    model = os.path.join(root, PKG, MODEL_DIR)
    vecs = os.path.join(root, PKG, VECS)
    if not dense:
        _note("example bank bundled; dense encoder skipped (BM25 is the default)")
    elif os.path.isdir(model) and os.path.isfile(vecs):
        datas.append((model, os.path.join(PKG, MODEL_DIR)))
        datas.append((vecs, PKG))
        hidden.extend(DENSE_IMPORTS)
        try:
            from PyInstaller.utils.hooks import collect_all
            for pkg in ("onnxruntime", "tokenizers"):
                d, b, h = collect_all(pkg)
                datas += d
                binaries += b
                hidden += h
        except Exception as exc:  # not running inside PyInstaller
            _note(f"collect_all unavailable ({exc})")
        _note(f"dense retrieval bundled ({_size_mb(model, vecs):.0f} MB)")
    else:
        _note("dense requested but encoder/vectors absent — BM25 only")

    return datas, binaries, hidden


def _size_mb(model_dir: str, vecs: str) -> float:
    total = os.path.getsize(vecs) if os.path.isfile(vecs) else 0
    for dirpath, _, names in os.walk(model_dir):
        for n in names:
            total += os.path.getsize(os.path.join(dirpath, n))
    return total / 1e6


def pyinstaller_args(root: str, dense: bool = False) -> List[str]:
    """The same payload as CLI flags, for builds that do not use a .spec file.

    scripts/build.sh drives PyInstaller with flags and regenerates the spec
    every run, so anything written into a .spec is dead code on that path.
    Both paths read this module instead, so they cannot drift.
    """
    datas, _, hidden = retrieval_assets(root, dense=dense)
    args: List[str] = []
    for src, dst in datas:
        args += ["--add-data", f"{src}:{dst}"]
    for name in hidden:
        args += ["--hidden-import", name]
    return args


if __name__ == "__main__":
    import sys
    flags = {"--pyinstaller-args", "--dense"}
    argv = [a for a in sys.argv[1:] if a not in flags]
    dense = "--dense" in sys.argv
    root = argv[0] if argv else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    if "--pyinstaller-args" in sys.argv:
        # Consumed by scripts/build.sh; keep stdout to the flags alone.
        print(" ".join(pyinstaller_args(root, dense=dense)))
    else:
        d, b, h = retrieval_assets(root, dense=dense)
        print(f"\ndatas {len(d)}  binaries {len(b)}  hiddenimports {len(h)}")
        for src, dst in d[:6]:
            print(f"  {src} -> {dst}")
