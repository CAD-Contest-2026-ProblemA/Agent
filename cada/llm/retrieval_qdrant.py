"""Qdrant-backed retrievers, for comparison against the in-process ones.

Qdrant is a vector search engine, not an encoder: it stores and searches the
same vectors ``cada.llm.embed`` produces.  At 2130 vectors an exact cosine is
one 2130x384 matmul, so its HNSW index can only approximate what OnnxRetriever
already computes exactly — the interesting part is not the dense search but
what Qdrant adds around it:

  * sparse vectors, so BM25 and dense scores live in one engine
  * Reciprocal Rank Fusion, a principled merge instead of naive interleaving

Both are measurable claims, which is why this module exists.  It is opt-in and
never imported unless a Qdrant retriever is asked for; nothing here ships in
the default path.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .retrieval import Bm25Retriever, Retriever, tokenize

COLLECTION = "examples"
RRF_K = 60  # standard Reciprocal Rank Fusion damping constant


def rrf_merge(ranked_lists: Sequence[Sequence[int]], k: int,
              damping: int = RRF_K) -> List[int]:
    """Reciprocal Rank Fusion: score an item by 1/(damping+rank) in each list.

    Interleaving takes rank-1 from every list before any rank-2, so one list
    being confidently right about its top hit counts for no more than another
    list being vaguely right.  RRF lets agreement accumulate instead.
    """
    scores: dict = {}
    for lst in ranked_lists:
        for rank, idx in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (damping + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:k]


class _SparseIndex(Bm25Retriever):
    """BM25 whose per-document weights can be emitted as sparse vectors.

    The doc-side BM25 weight already folds in idf, term frequency and length
    normalisation, so a dot product against a query vector of ones reproduces
    the BM25 score exactly — Qdrant's sparse search is then not an
    approximation of BM25, it *is* BM25.
    """

    def vocab_id(self, term: str) -> Optional[int]:
        return self._vocab.get(term)

    def build_vocab(self) -> None:
        self._vocab = {t: i for i, t in enumerate(sorted(self._idf))}

    def doc_sparse(self, i: int):
        out = {}
        for term, freq in self._doc_tf[i].items():
            vid = self._vocab.get(term)
            if vid is None:
                continue
            norm = self.K1 * (1 - self.B + self.B * self._len[i] / self._avg)
            out[vid] = self._idf[term] * freq * (self.K1 + 1) / (freq + norm)
        return out

    def query_sparse(self, query: str):
        out = {}
        for term in tokenize(query):
            vid = self._vocab.get(term)
            if vid is not None:
                out[vid] = 1.0
        return out

    def __init__(self, bank):
        super().__init__(bank)
        from collections import Counter
        self._doc_tf = [Counter(tokenize(r["text"])) for r in self.bank]
        self.build_vocab()


class QdrantRetriever(Retriever):
    """Dense (and optionally hybrid) retrieval through a local Qdrant instance.

    ``:memory:`` keeps everything in-process — no server, no container — which
    is what makes this comparable to the other backends rather than a test of
    someone's deployment.
    """

    def __init__(self, bank: Sequence[dict], vecs, model_dir: str,
                 hybrid: bool = False, location: str = ":memory:"):
        super().__init__(bank)
        from qdrant_client import QdrantClient, models
        from .embed import Encoder

        self._models = models
        self._enc = Encoder(model_dir)
        self.hybrid = hybrid
        self.name = "qdrant-hybrid" if hybrid else "qdrant"
        self._sparse = _SparseIndex(bank) if hybrid else None

        dim = int(vecs.shape[1])
        self._client = QdrantClient(location)
        kwargs = dict(
            collection_name=COLLECTION,
            vectors_config={"dense": models.VectorParams(
                size=dim, distance=models.Distance.COSINE)},
        )
        if hybrid:
            kwargs["sparse_vectors_config"] = {"bm25": models.SparseVectorParams()}
        self._client.create_collection(**kwargs)

        points = []
        for i, row in enumerate(self.bank):
            vector = {"dense": vecs[i].tolist()}
            if hybrid:
                sp = self._sparse.doc_sparse(i)
                vector["bm25"] = models.SparseVector(
                    indices=list(sp.keys()), values=list(sp.values()))
            points.append(models.PointStruct(
                id=i, vector=vector, payload={"case": row.get("case", "")}))
        self._client.upsert(COLLECTION, points=points)

    def scores(self, query: str) -> List[float]:
        raise NotImplementedError("qdrant ranks server-side, not by score vector")

    def top_k(self, query: str, k: int = 50,
              exclude_case: Optional[str] = None) -> List[dict]:
        models = self._models
        flt = None
        if exclude_case:
            # Leave-one-testcase-out has to happen inside the engine: filtering
            # after the fact would silently return fewer than k.
            flt = models.Filter(must_not=[models.FieldCondition(
                key="case", match=models.MatchValue(value=exclude_case))])
        qv = self._enc.encode([query])[0].tolist()
        # Over-fetch: the bank repeats 354 of its 2130 sentences, so asking for
        # exactly k and then deduplicating would return fewer distinct examples
        # than every other backend and quietly bias the comparison.
        want = k * 2

        if not self.hybrid:
            res = self._client.query_points(
                COLLECTION, query=qv, using="dense", limit=want,
                query_filter=flt).points
            return self._dedup([self.bank[p.id] for p in res], k)

        sp = self._sparse.query_sparse(query)
        prefetch = [
            models.Prefetch(query=qv, using="dense", limit=want * 2, filter=flt),
            models.Prefetch(
                query=models.SparseVector(indices=list(sp.keys()),
                                          values=list(sp.values())),
                using="bm25", limit=want * 2, filter=flt),
        ]
        res = self._client.query_points(
            COLLECTION, prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=want, query_filter=flt).points
        return self._dedup([self.bank[p.id] for p in res], k)

    @staticmethod
    def _dedup(rows: Sequence[dict], k: int) -> List[dict]:
        out, seen = [], set()
        for r in rows:
            key = (r["text"], r["op"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
            if len(out) >= k:
                break
        return out


def build_qdrant(bank: Sequence[dict], hybrid: bool = False) -> QdrantRetriever:
    """Construct a Qdrant retriever over the shipped vectors. Raises if absent."""
    import numpy as np
    from .retrieval import VECS_NAME, data_path

    vecs = np.load(data_path(VECS_NAME))
    if vecs.shape[0] != len(bank):
        raise ValueError(f"{vecs.shape[0]} vectors for {len(bank)} sentences")
    return QdrantRetriever(bank, vecs, data_path("embed_model"), hybrid=hybrid)
