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


def retrieval_assets(root: str) -> Tuple[List[tuple], List[tuple], List[str]]:
    """Return (datas, binaries, hiddenimports) to merge into a .spec."""
    datas: List[tuple] = []
    binaries: List[tuple] = []
    hidden: List[str] = []

    bank = os.path.join(root, PKG, BANK)
    if os.path.isfile(bank):
        datas.append((bank, PKG))
    else:
        print(f"[spec_assets] WARNING: {bank} missing — no retrieval will be possible")

    model = os.path.join(root, PKG, MODEL_DIR)
    vecs = os.path.join(root, PKG, VECS)
    if os.path.isdir(model) and os.path.isfile(vecs):
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
            print(f"[spec_assets] collect_all unavailable ({exc})")
        print(f"[spec_assets] dense retrieval bundled ({_size_mb(model, vecs):.0f} MB)")
    else:
        print("[spec_assets] no encoder/vectors — binary will use the BM25 retriever")

    return datas, binaries, hidden


def _size_mb(model_dir: str, vecs: str) -> float:
    total = os.path.getsize(vecs) if os.path.isfile(vecs) else 0
    for dirpath, _, names in os.walk(model_dir):
        for n in names:
            total += os.path.getsize(os.path.join(dirpath, n))
    return total / 1e6


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    d, b, h = retrieval_assets(root)
    print(f"\ndatas {len(d)}  binaries {len(b)}  hiddenimports {len(h)}")
    for src, dst in d[:6]:
        print(f"  {src} -> {dst}")
