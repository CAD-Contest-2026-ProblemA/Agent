"""Sentence encoder: ONNX graph + WordPiece tokenizer, mean-pooled and L2-normed.

Deliberately not sentence-transformers.  At inference the model is fixed, so
the exported ONNX graph produces the same vectors as the torch pipeline for a
fraction of the footprint — torch would add several hundred MB to a 45MB
one-file binary and buy no accuracy.  What decides retrieval quality is the
model, not the runtime.

Only the incoming query is encoded here; the bank's vectors are precomputed by
scripts/build_vectors.py and ship as a .npy.

Everything is imported lazily and every failure is the caller's to survive —
cada.llm.retrieval falls back to BM25, and then to no retrieval at all.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"
# MiniLM's trained position table stops at 512; longer input would index past it.
MAX_TOKENS = 256


class Encoder:
    """Encode sentences to unit-length vectors.

    ``model_dir`` must hold model.onnx and tokenizer.json (see
    scripts/fetch_embed_model.py).
    """

    def __init__(self, model_dir: str, threads: int = 1):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model = os.path.join(model_dir, MODEL_FILE)
        tok = os.path.join(model_dir, TOKENIZER_FILE)
        for p in (model, tok):
            if not os.path.isfile(p):
                raise FileNotFoundError(p)

        self._np = np
        self._tok = Tokenizer.from_file(tok)
        self._tok.enable_truncation(max_length=MAX_TOKENS)

        opts = ort.SessionOptions()
        # One request at a time on a machine already running EDA tools; letting
        # ORT spawn a thread pool per session just competes with them.
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = threads
        self._sess = ort.InferenceSession(
            model, sess_options=opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}

    @property
    def dim(self) -> int:
        return self._sess.get_outputs()[0].shape[-1]

    def encode(self, texts: Sequence[str], batch_size: int = 64):
        """Return an (n, dim) float32 array of unit-length embeddings."""
        np = self._np
        out = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            encs = self._tok.encode_batch(chunk)
            width = max((len(e.ids) for e in encs), default=1) or 1
            ids = np.zeros((len(encs), width), dtype=np.int64)
            mask = np.zeros((len(encs), width), dtype=np.int64)
            for i, e in enumerate(encs):
                n = len(e.ids)
                ids[i, :n] = e.ids
                mask[i, :n] = e.attention_mask
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)

            hidden = self._sess.run(None, feed)[0]           # (b, seq, dim)
            # Mean-pool over real tokens only: padding would otherwise drag
            # every short sentence toward the same vector.
            m = mask[:, :, None].astype(np.float32)
            summed = (hidden * m).sum(axis=1)
            counts = np.clip(m.sum(axis=1), 1e-9, None)
            vecs = summed / counts
            norms = np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
            out.append((vecs / norms).astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), "float32")


def default_model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_model")


def try_load(model_dir: Optional[str] = None) -> Optional["Encoder"]:
    """Build an Encoder, or None if anything is missing. Never raises."""
    try:
        return Encoder(model_dir or default_model_dir())
    except Exception:
        return None
