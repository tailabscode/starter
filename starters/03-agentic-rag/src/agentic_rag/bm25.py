"""A small, dependency-free Okapi BM25 implementation.

This is the "real work" of offline retrieval: exact and near-exact term overlap, weighted by
how rare a term is across the corpus and normalized for document length. It has no notion of
synonyms or meaning, which is exactly why it is fused with dense retrieval in ``retrieval.py``.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization. Good enough for BM25 term matching."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over a fixed list of documents, identified by position."""

    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = [tokenize(doc) for doc in documents]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)) if self._doc_tokens else 0.0
        self._term_freqs: list[Counter[str]] = [Counter(toks) for toks in self._doc_tokens]
        self._doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            self._doc_freq.update(tf.keys())
        self._n_docs = len(documents)

    def _idf(self, term: str) -> float:
        n_q = self._doc_freq.get(term, 0)
        # BM25+ style idf: stays positive even for terms in every document.
        return math.log((self._n_docs - n_q + 0.5) / (n_q + 0.5) + 1)

    def score(self, query: str) -> list[float]:
        """Return one BM25 score per document, in the original document order."""
        query_terms = tokenize(query)
        scores = [0.0] * self._n_docs
        if self._n_docs == 0 or not query_terms:
            return scores
        for i, tf in enumerate(self._term_freqs):
            doc_len = self._doc_len[i]
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf(term)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_len or 1))
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores
