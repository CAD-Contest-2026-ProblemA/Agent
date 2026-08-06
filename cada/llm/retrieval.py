"""Retrieve previously-classified requests similar to the incoming line.

The catalog can only carry a few dozen hand-written examples, but
``examples.jsonl`` holds 2130 labelled (sentence, op) pairs — far more than
fits in a prompt.  Retrieving the nearest ones per request puts that whole
bank within reach of a fixed-size prompt.

This is deliberately *additive*: the full op list and every disambiguation
rule stay in the static prompt, so a bad retrieval costs nothing but a few
unhelpful examples.  Nothing here can remove the correct answer from the
model's reach — which is exactly why the retrieval unit is example sentences
and not the op list itself.

Two backends behind one interface.  ``OnnxRetriever`` is the intended one;
``Bm25Retriever`` needs no third-party package and is the offline baseline
that says whether the embedding model earns its footprint.  If neither can be
built, ``build_retriever`` returns None and the caller simply omits the block.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

BANK_NAME = "examples.jsonl"
VECS_NAME = "example_vecs.npy"
_TOKEN = re.compile(r"[a-z]+")


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def data_path(name: str) -> str:
    """Locate a bundled data file, frozen or from source.

    Under PyInstaller one-file the payload is unpacked to sys._MEIPASS, so the
    package directory that ``__file__`` points at is not where --add-data put
    these.  Check the extraction dir first, then fall back to the source tree.
    """
    import sys
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
        if base:
            cand = os.path.join(base, "cada", "llm", name)
            if os.path.exists(cand):
                return cand
            cand = os.path.join(base, name)
            if os.path.exists(cand):
                return cand
    return os.path.join(_here(), name)


def bank_path() -> str:
    return data_path(BANK_NAME)


def load_bank(path: Optional[str] = None) -> List[dict]:
    path = path or bank_path()
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def tokenize(text: str) -> List[str]:
    # Net and gate names (n14, g0, n31[1]) are pure noise for picking an op —
    # every bank entry uses different ones — so keep alphabetic tokens only.
    return _TOKEN.findall(text.lower())


class Retriever:
    """Return the bank entries most similar to a query sentence."""

    name = "none"

    def __init__(self, bank: Sequence[dict]):
        self.bank = list(bank)

    def scores(self, query: str) -> List[float]:
        raise NotImplementedError

    def top_k(self, query: str, k: int = 25,
              exclude_case: Optional[str] = None) -> List[dict]:
        if not self.bank:
            return []
        ranked = sorted(range(len(self.bank)), key=self.scores(query).__getitem__,
                        reverse=True)
        out = []
        for i in ranked:
            row = self.bank[i]
            # Leave-one-testcase-out: when evaluating, the query's own testcase
            # must not supply its own answer or the measurement is meaningless.
            if exclude_case and row.get("case") == exclude_case:
                continue
            out.append(row)
            if len(out) >= k:
                break
        return out


class Bm25Retriever(Retriever):
    """Okapi BM25 over the bank. Pure stdlib — no dependency, ships anywhere."""

    name = "bm25"
    K1 = 1.5
    B = 0.75

    def __init__(self, bank: Sequence[dict]):
        super().__init__(bank)
        docs = [tokenize(r["text"]) for r in self.bank]
        self._len = [len(d) for d in docs]
        self._avg = (sum(self._len) / len(docs)) if docs else 0.0
        n = len(docs)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        # Inverted index: only documents sharing a query term can score.
        self._post: Dict[str, List[tuple]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, freq in Counter(d).items():
                self._post[term].append((i, freq))

    def scores(self, query: str) -> List[float]:
        out = [0.0] * len(self.bank)
        for term in tokenize(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, freq in self._post[term]:
                norm = self.K1 * (1 - self.B + self.B * self._len[i] / self._avg)
                out[i] += idf * freq * (self.K1 + 1) / (freq + norm)
        return out


class OnnxRetriever(Retriever):
    """Dense retrieval: a sentence encoder over vectors precomputed at build time.

    Only the incoming query is encoded at runtime; the bank's vectors ship
    alongside the binary, which is the whole reason an encoder has to be
    bundled at all.
    """

    name = "onnx"

    def __init__(self, bank: Sequence[dict], model_dir: str, vecs_path: str):
        super().__init__(bank)
        import numpy as np  # noqa: F401  (imported for the type, used below)
        from .embed import Encoder

        self._np = np
        self._enc = Encoder(model_dir)
        vecs = np.load(vecs_path)
        if vecs.shape[0] != len(self.bank):
            raise ValueError(
                f"vector count {vecs.shape[0]} != bank size {len(self.bank)}; "
                "re-run scripts/build_vectors.py")
        # Vectors are stored L2-normalised, so cosine is a plain dot product.
        self._vecs = vecs.astype("float32")

    def scores(self, query: str) -> List[float]:
        q = self._enc.encode([query])[0]
        return (self._vecs @ q).tolist()


def build_retriever(prefer: str = "auto",
                    bank: Optional[Sequence[dict]] = None) -> Optional[Retriever]:
    """Build a retriever, or None if the bank is missing.

    ``prefer`` is "auto", "bm25", "onnx", or "none".

    "auto" means BM25.  The dense backend retrieves better in isolation
    (recall@25 95.0% vs 93.9% over the 1420 evaluation queries) but that did
    not survive to the answer: end to end the two scored 94.2% and 94.3%, one
    sentence apart out of 1420.  BM25 needs no third-party package, adds
    nothing to the 45MB binary, and avoids the glibc / libstdc++ / SIGILL
    portability surface an ONNX runtime brings — so the dense path has to be
    asked for explicitly, and is worth re-testing only if the bank grows or
    the phrasings drift further from the labelled set.
    """
    if prefer == "none":
        return None
    rows = list(bank) if bank is not None else load_bank()
    if not rows:
        return None
    if prefer == "onnx":
        model_dir = data_path("embed_model")
        vecs = data_path(VECS_NAME)
        if os.path.isdir(model_dir) and os.path.isfile(vecs):
            try:
                return OnnxRetriever(rows, model_dir, vecs)
            except Exception as exc:
                import sys
                sys.stderr.write(f"[retrieval] onnx unavailable ({exc}); using bm25\n")
        else:
            import sys
            sys.stderr.write(f"[retrieval] missing {model_dir} or {vecs}; using bm25\n")
    try:
        return Bm25Retriever(rows)
    except Exception:
        return None


HEADER = """━━━ SIMILAR PAST REQUESTS ━━━
Requests already classified, ordered by similarity to the one you must answer.
They fix the intent NAME for a phrasing; they do not show params, and they do
not override the rules above.  If none of them fits, ignore them.
"""


def format_block(hits: Sequence[dict]) -> str:
    """Render retrieved examples as the dynamic half of the system prompt."""
    if not hits:
        return ""
    lines = [HEADER]
    for h in hits:
        lines.append(f'"{h["text"]}" → {h["op"]}')
    return "\n".join(lines)
