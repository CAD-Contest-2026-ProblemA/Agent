#!/usr/bin/env python3
"""Precompute the example bank's embeddings so runtime only encodes the query.

Run at build time, after scripts/fetch_embed_model.py.  The .npy this writes
ships beside the binary; encoding 2130 sentences on every start would cost
seconds and defeat the point.

Also verifies the round trip: it re-encodes a few sentences and checks each
one retrieves itself, which catches a tokenizer/model mismatch that would
otherwise show up only as quietly useless retrieval.

Usage:
    python3 scripts/build_vectors.py
    python3 scripts/build_vectors.py --check   # verify an existing .npy
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cada.llm.embed import Encoder, default_model_dir
from cada.llm.retrieval import bank_path, load_bank, VECS_NAME

OUT = os.path.join(ROOT, "cada", "llm", VECS_NAME)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=default_model_dir())
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    import numpy as np

    bank = load_bank()
    if not bank:
        sys.exit(f"error: empty bank at {bank_path()} — run scripts/export_examples.py")
    print(f"bank      {len(bank)} sentences")

    if args.check:
        if not os.path.isfile(args.out):
            sys.exit(f"error: {args.out} missing")
        vecs = np.load(args.out)
        print(f"vectors   {vecs.shape} {vecs.dtype}")
        if vecs.shape[0] != len(bank):
            sys.exit(f"error: {vecs.shape[0]} vectors for {len(bank)} sentences — stale")
        norms = np.linalg.norm(vecs, axis=1)
        print(f"norms     min={norms.min():.4f} max={norms.max():.4f} (want ~1.0)")
        return 0

    enc = Encoder(args.model_dir)
    print(f"model     {args.model_dir}  dim={enc.dim}")
    vecs = enc.encode([r["text"] for r in bank])
    np.save(args.out, vecs)
    print(f"wrote     {args.out}  {vecs.shape} {vecs.dtype} "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)")

    # Self-retrieval: a sentence must rank itself first against its own bank.
    # If the tokenizer and graph disagree this is where it surfaces.
    probes = [0, len(bank) // 3, 2 * len(bank) // 3, len(bank) - 1]
    bad = 0
    for i in probes:
        q = enc.encode([bank[i]["text"]])[0]
        if int((vecs @ q).argmax()) != i and bank[int((vecs @ q).argmax())]["text"] != bank[i]["text"]:
            print(f"  MISMATCH at {i}: {bank[i]['text'][:50]!r}")
            bad += 1
    print(f"self-retrieval {len(probes)-bad}/{len(probes)} ok")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
