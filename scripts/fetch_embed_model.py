#!/usr/bin/env python3
"""Download the sentence encoder used for example retrieval.

Run once at setup; the result lands in ``cada/llm/embed_model/`` and is what
the .spec bundles.  Not needed at runtime, and not needed at all if you are
happy with the BM25 retriever.

Variant choice matters for portability.  The quantised ONNX exports of this
model are instruction-set specific (``*_qint8_avx512``, ``*_quint8_avx2``, …)
and fail with SIGILL on a CPU that lacks the extension — and that failure
shows up on the first inference, not at load.  The default here is the plain
fp32 graph, which runs anywhere at the cost of ~3x the size.  Switch with
--variant only once the target CPU is known.

Usage:
    python3 scripts/fetch_embed_model.py
    python3 scripts/fetch_embed_model.py --variant onnx/model_quint8_avx2.onnx
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "cada", "llm", "embed_model")

REPO = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ONNX = "onnx/model.onnx"
SIDECARS = ["tokenizer.json", "config.json"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--variant", default=DEFAULT_ONNX,
                    help="path of the .onnx file inside the repo")
    ap.add_argument("--dest", default=DEST)
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("error: pip install huggingface-hub (or uv pip install ...)")

    os.makedirs(args.dest, exist_ok=True)
    wanted = [(args.variant, "model.onnx")] + [(f, os.path.basename(f)) for f in SIDECARS]
    for remote, local in wanted:
        try:
            src = hf_hub_download(repo_id=args.repo, filename=remote)
        except Exception as exc:
            if local == "model.onnx":
                sys.exit(f"error: could not fetch {remote}: {exc}")
            print(f"  (skipped optional {remote}: {exc})")
            continue
        dst = os.path.join(args.dest, local)
        shutil.copyfile(src, dst)
        print(f"  {local:16s} {os.path.getsize(dst)/1e6:8.1f} MB")

    total = sum(os.path.getsize(os.path.join(args.dest, f))
                for f in os.listdir(args.dest))
    print(f"\n{args.dest}  ({total/1e6:.1f} MB total)")
    print("next: python3 scripts/build_vectors.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
