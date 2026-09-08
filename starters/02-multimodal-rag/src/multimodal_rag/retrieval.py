"""Hybrid retrieval: BM25 lexical ranking fused with dense cosine ranking
via Reciprocal Rank Fusion (RRF), over the unified text+image item list.

See 01-basic-rag/src/basic_rag/retrieval.py for the full rationale
(duplicated here on purpose). The only difference here is that a "document"
may be a text chunk or an image caption -- retrieval doesn't need to care
which, because both were embedded and tokenized identically as `item.text`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .bm25 import BM25, tokenize
from .embeddings import Embedder
from .index import Index


@dataclass(frozen=True)
class RetrievedItem:
    item_id: str
    rank: int
    rrf_score: float


def reciprocal_rank_fusion(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Fuse several rank-ordered id lists into one score per id.

    RRF score for an id = sum, over each ranking that contains it, of
    1 / (k + rank) where `rank` is 1-based. Returns ids sorted by fused
    score, best first.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + position)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    """Combines a BM25 index (built once) with an Embedder over a fixed Index."""

    def __init__(self, index: Index, embedder: Embedder) -> None:
        self.index = index
        self.embedder = embedder
        self._bm25 = BM25([tokenize(text) for text in index.item_texts])

    def retrieve(self, query: str, top_k: int, rrf_k: int) -> list[RetrievedItem]:
        item_ids = self.index.item_ids

        bm25_order = self._bm25.rank(tokenize(query))
        bm25_ranking = [item_ids[i] for i in bm25_order]

        query_vector = self.embedder.embed([query])[0]
        vectors = self.index.vectors
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1.0
        query_norm = np.linalg.norm(query_vector) or 1.0
        similarities = (vectors @ query_vector) / (norms * query_norm)
        dense_order = np.argsort(-similarities)
        dense_ranking = [item_ids[i] for i in dense_order]

        fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=rrf_k)
        return [
            RetrievedItem(item_id=item_id, rank=i + 1, rrf_score=score)
            for i, (item_id, score) in enumerate(fused[:top_k])
        ]
